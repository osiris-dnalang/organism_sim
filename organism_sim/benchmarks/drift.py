"""
organism_sim.benchmarks.drift — Shifting-logic (non-stationary) 6-multiplexer
=============================================================================

The truth table is perturbed every ``shift_every`` trials, unannounced:

    invert     output flipped
    addr_swap  address bits read in the other order
    data_perm  data bits permuted

Two groups on identical seeds and identical engines; only the organism's
structural hooks differ:

    plain      ``LCSAgent(structural=False)`` — XCS adapts by weight updates only
    organism   ``LCSAgent(structural=True)``  — sustained error → noise → repair
               (compaction, erosion) and mutation (GP variants + shock: reset
               experience of high-error rules, boost exploration)

Per shift we record recovery time (trials until the probe accuracy is back to
``recovered_at``), area-under-error over the window, and counts of repairs /
mutations. ``compare()`` aggregates medians and IQRs over seeds and applies the
pre-registered kill criterion:

    PASS  ⇔  median recovery(organism) ≤ 0.75 · median recovery(plain)
             and the interquartile ranges do not overlap.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params
from ..spec import Triggers
from .mux import _inputs

SHIFT_KINDS = ("invert", "addr_swap", "data_perm")


class DriftEnvironment:
    """6-mux whose logic is perturbed every ``shift_every`` trials."""

    def __init__(self, shift_every: int = 3000, seed: int = 0,
                 kinds: Sequence[str] = SHIFT_KINDS):
        self.shift_every = shift_every
        self.rng = np.random.default_rng(seed + 777)
        self.kinds = list(kinds)
        self.invert = False
        self.addr_swap = False
        self.data_perm = [0, 1, 2, 3]
        self.shifts: List[Dict[str, Any]] = []
        self.trial = 0

    def truth(self, bits: str) -> str:
        b = [int(c) for c in bits]
        if self.addr_swap:
            b[0], b[1] = b[1], b[0]
        data = [b[2 + i] for i in self.data_perm]
        out = data[2 * b[0] + b[1]]
        return str(1 - out if self.invert else out)

    def maybe_shift(self, trial: int) -> Optional[Dict[str, Any]]:
        self.trial = trial
        if trial % self.shift_every != 0:
            return None
        kind = self.kinds[int(self.rng.integers(len(self.kinds)))]
        if kind == "invert":
            self.invert = not self.invert
        elif kind == "addr_swap":
            self.addr_swap = not self.addr_swap
        else:
            perm = list(self.data_perm)
            while perm == self.data_perm:
                perm = list(self.rng.permutation(4))
            self.data_perm = perm
        rec = {"trial": trial, "kind": kind, "invert": self.invert, "addr_swap": self.addr_swap,
               "data_perm": list(self.data_perm)}
        self.shifts.append(rec)
        return rec


@dataclass
class ShiftRecord:
    index: int
    trial: int
    kind: str
    acc_before: float
    acc_after: float            # first probe after the shift
    recovery_trials: Optional[int]   # None = not recovered before the next shift
    area_under_error: float     # Σ (1 − acc)·probe_every over the window
    repairs: int
    mutations: int
    shocks: int
    macro_end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftRow:
    trial: int
    acc: float
    shift_index: int
    since_shift: int
    macro: int
    micro: int
    noise_rate: float
    entropy_bits: float
    repairs: int
    mutations: int

    FIELDS = ("trial", "acc", "shift_index", "since_shift", "macro", "micro", "noise_rate",
              "entropy_bits", "repairs", "mutations")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftLog:
    group: str
    seed: int
    rows: List[DriftRow] = field(default_factory=list)
    shifts: List[ShiftRecord] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"group": self.group, "seed": self.seed,
                           "rows": [r.to_dict() for r in self.rows],
                           "shifts": [s.to_dict() for s in self.shifts], "final": self.final},
                          indent=indent)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(DriftRow.FIELDS))
        w.writeheader()
        for r in self.rows:
            w.writerow(r.to_dict())
        return buf.getvalue()

    def recoveries(self) -> List[Optional[int]]:
        return [s.recovery_trials for s in self.shifts]


def _probe(agent: LCSAgent, inputs: List[str], truth: Callable[[str], str]) -> float:
    ok = 0
    for bits in inputs:
        agent.engine.step(agent.register(bits), explore=False)
        out = agent.engine.last_ctx.emits[0] if agent.engine.last_ctx.emits else None
        ok += out == truth(bits)
    agent.engine.action_set = []
    return ok / len(inputs)


def run_drift(trials: int = 18000, shift_every: int = 3000, structural: bool = False,
              seed: int = 0, probe_every: int = 100, n_eval: int = 256,
              recovered_at: float = 0.95, params: Optional[Params] = None,
              kinds: Sequence[str] = SHIFT_KINDS,
              triggers: Optional[Triggers] = None) -> DriftLog:
    """One agent on the drifting multiplexer. The first shift happens at
    ``shift_every`` (after a warm-up equal to one period); no shift is applied in
    the final ``shift_every`` trials so every shift has a full window."""
    rng = np.random.default_rng(seed)
    inputs = _inputs(np.random.default_rng(seed + 10_000), n_eval)
    env = DriftEnvironment(shift_every=shift_every, seed=seed, kinds=kinds)
    agent = LCSAgent("D", input_len=6, actions=["(emit 0)", "(emit 1)"], params=params,
                     seed=seed, structural=structural, triggers=triggers)
    log = DriftLog(group="organism" if structural else "plain", seed=seed)
    cur: Optional[Dict[str, Any]] = None       # open shift window
    last_acc = 0.0
    counters_at_shift = (0, 0, 0)

    def counts():
        c = agent.organism.counters
        return c["repair"], c["mutate"], agent.engine.counters.get("shock", 0)

    for trial in range(1, trials + 1):
        shift = env.maybe_shift(trial) if trial + shift_every <= trials else None
        if shift is not None:
            if cur is not None:
                _close(cur, log, counts(), counters_at_shift, agent, recovered=False)
            cur = {"index": len(env.shifts), "trial": trial, "kind": shift["kind"],
                   "acc_before": last_acc, "acc_after": None, "aue": 0.0, "recovery": None}
            counters_at_shift = counts()
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            last_acc = acc
            st = agent.state()
            since = trial - cur["trial"] if cur else trial
            log.rows.append(DriftRow(trial=trial, acc=acc, shift_index=len(env.shifts),
                                     since_shift=since, macro=st["macro"], micro=st["micro"],
                                     noise_rate=st["noise_rate"],
                                     entropy_bits=st["entropy_bits"],
                                     repairs=st["repairs"], mutations=st["mutations"]))
            if cur is not None:
                if cur["acc_after"] is None:
                    cur["acc_after"] = acc
                if cur["recovery"] is None:
                    cur["aue"] += (1.0 - acc) * probe_every
                    if acc >= recovered_at:
                        cur["recovery"] = since
    if cur is not None:
        _close(cur, log, counts(), counters_at_shift, agent, recovered=cur["recovery"] is not None)
    recs = [s.recovery_trials for s in log.shifts]
    done = [r for r in recs if r is not None]
    log.final = {
        "group": log.group, "seed": seed, "trials": trials, "shift_every": shift_every,
        "shifts": len(log.shifts), "recovered": len(done),
        "median_recovery": float(np.median(done)) if done else None,
        "mean_aue": float(np.mean([s.area_under_error for s in log.shifts])) if log.shifts else None,
        "repairs": agent.organism.counters["repair"], "mutations": agent.organism.counters["mutate"],
        "shocks": agent.engine.counters.get("shock", 0), "final_acc": last_acc,
        "audit_chain_valid": agent.organism.chain.verify(),
    }
    return log


def _close(cur, log: DriftLog, now, at_shift, agent: LCSAgent, recovered: bool) -> None:
    log.shifts.append(ShiftRecord(
        index=cur["index"], trial=cur["trial"], kind=cur["kind"], acc_before=cur["acc_before"],
        acc_after=cur["acc_after"] if cur["acc_after"] is not None else 0.0,
        recovery_trials=cur["recovery"], area_under_error=cur["aue"],
        repairs=now[0] - at_shift[0], mutations=now[1] - at_shift[1], shocks=now[2] - at_shift[2],
        macro_end=agent.engine.macro_size()))


# ── comparison + kill criterion ──────────────────────────────────────────────

def _summ(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"median": None, "q1": None, "q3": None, "n": 0}
    a = np.asarray(vals, dtype=float)
    return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "n": int(a.size)}


def compare(seeds: Sequence[int] = range(30), trials: int = 18000, shift_every: int = 3000,
            probe_every: int = 100, recovered_at: float = 0.95, speedup: float = 0.75,
            params: Optional[Params] = None, kinds: Sequence[str] = SHIFT_KINDS,
            unrecovered_as: Optional[int] = None,
            triggers: Optional[Triggers] = None) -> Dict[str, Any]:
    """Run both groups on the same seeds and apply the kill criterion.

    Unrecovered shifts count as ``unrecovered_as`` trials (default: the full
    period ``shift_every``) so a group that never recovers is not rewarded.
    """
    cap = shift_every if unrecovered_as is None else unrecovered_as
    out: Dict[str, Any] = {"seeds": list(seeds), "trials": trials, "shift_every": shift_every,
                           "recovered_at": recovered_at, "speedup_required": speedup,
                           "triggers": (triggers or Triggers(noise_floor=0.05,
                                                             repair_threshold=0.45)).to_dict(),
                           "groups": {}, "per_seed": []}
    rec: Dict[str, List[float]] = {"plain": [], "organism": []}
    aue: Dict[str, List[float]] = {"plain": [], "organism": []}
    unrec: Dict[str, int] = {"plain": 0, "organism": 0}
    muts: Dict[str, List[float]] = {"plain": [], "organism": []}
    for seed in seeds:
        row = {"seed": seed}
        for group, structural in (("plain", False), ("organism", True)):
            log = run_drift(trials, shift_every, structural, seed, probe_every, 256, recovered_at,
                            params, kinds, triggers)
            r = [s.recovery_trials if s.recovery_trials is not None else cap for s in log.shifts]
            unrec[group] += sum(s.recovery_trials is None for s in log.shifts)
            rec[group].extend(r)
            aue[group].extend(s.area_under_error for s in log.shifts)
            muts[group].append(log.final["mutations"])
            row[group] = {"median_recovery": float(np.median(r)) if r else None,
                          "mutations": log.final["mutations"], "repairs": log.final["repairs"],
                          "shocks": log.final["shocks"], "final_acc": log.final["final_acc"]}
        out["per_seed"].append(row)
    for g in ("plain", "organism"):
        out["groups"][g] = {"recovery": _summ(rec[g]), "aue": _summ(aue[g]),
                            "unrecovered_shifts": unrec[g],
                            "mutations_per_run": _summ(muts[g])}
    p, o = out["groups"]["plain"]["recovery"], out["groups"]["organism"]["recovery"]
    faster = o["median"] is not None and p["median"] is not None and o["median"] <= speedup * p["median"]
    disjoint = o["q3"] is not None and p["q1"] is not None and o["q3"] < p["q1"]
    out["verdict"] = {"pass": bool(faster and disjoint), "faster": bool(faster),
                      "iqr_disjoint": bool(disjoint),
                      "ratio": (o["median"] / p["median"]) if (o["median"] and p["median"]) else None}
    return out


def compare_csv(result: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seed", "plain_median_recovery", "organism_median_recovery", "plain_mutations",
                "organism_mutations", "organism_shocks", "plain_final_acc", "organism_final_acc"])
    for r in result["per_seed"]:
        w.writerow([r["seed"], r["plain"]["median_recovery"], r["organism"]["median_recovery"],
                    r["plain"]["mutations"], r["organism"]["mutations"], r["organism"]["shocks"],
                    r["plain"]["final_acc"], r["organism"]["final_acc"]])
    return buf.getvalue()


__all__ = ["DriftEnvironment", "ShiftRecord", "DriftRow", "DriftLog", "run_drift", "compare",
           "compare_csv", "SHIFT_KINDS"]
