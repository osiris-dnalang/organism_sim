"""
organism_sim.benchmarks.expone — Experiment One: detector-triggered random injection
=====================================================================================

Follow-up to Experiment Zero, which found that *injecting fresh hypotheses at a shift*
speeds re-convergence while the *content* of the hypotheses does not matter. Here the
agent is not told when shifts happen.

Environment: the hidden generalised-multiplexer family of Experiment Zero, instance
switching every ``shift_every`` trials, schedule unknown to the agent.

Arms (identical seeds / engines / budgets):

    plain      XCS
    cusum      XCS + one-sided CUSUM on its own exploit error; on detection inject N random
               zero-experience rules (cooldown between injections)
    periodic   XCS + N random rules every ``period`` trials, no detector (control: is
               detection needed, or does blind diversity do the job?)
    oracle     XCS + N random rules at the true shift (upper bound; Experiment Zero's C arm)

Metrics: per shift, trials until probe accuracy ≥ 0.95 (primary) and ≥ 0.99 (secondary);
injections total and *false* (not within ``false_window`` trials after a shift).

The detector arms only after the agent has converged once (first probe ≥ 0.95), as in the
relay benchmark, so initial learning is never mistaken for a shift.

Pre-registered criterion (seeds 0–4; (μ₀, k, h, N) tuned on seeds 100–104 only; cooldown
fixed at 300):
    C1  cusum median recovery < plain on ≥ 4/5 seeds
    C2  cusum false injections ≤ 1 per 10k trials
    PASS ⇔ C1 ∧ C2. Exploratory (reported, not judged): cusum vs periodic, cusum vs oracle.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params
from .expzero import DriftingHidden, inject, random_rules
from .mux import _inputs

ARMS = ("plain", "cusum", "periodic", "oracle", "grn")


@dataclass
class OneLog:
    arm: str
    seed: int
    recoveries95: List[Optional[int]] = field(default_factory=list)
    recoveries99: List[Optional[int]] = field(default_factory=list)
    shifts: List[int] = field(default_factory=list)
    injections: List[int] = field(default_factory=list)
    false_injections: int = 0
    probes: List[Dict[str, Any]] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    def scores(self, cap: int, which: str = "95") -> List[int]:
        v = self.recoveries95 if which == "95" else self.recoveries99
        return [x if x is not None else cap for x in v]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _probe(agent: LCSAgent, inputs: List[str], truth) -> float:
    ok = 0
    for bits in inputs:
        agent.engine.step(agent.register(bits), explore=False)
        out = agent.engine.last_ctx.emits[0] if agent.engine.last_ctx.emits else None
        ok += out == truth(bits)
    agent.engine.action_set = []
    return ok / len(inputs)


def run_arm(arm: str, seed: int, trials: int = 15000, shift_every: int = 3000,
            probe_every: int = 100, n_inject: int = 20, cusum: Tuple[float, float, float] = (0.05, 0.10, 4.0),
            cooldown: int = 300, period: int = 500, false_window: int = 400,
            params: Optional[Params] = None, grn_genome=None) -> OneLog:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "grn" and grn_genome is None:
        raise ValueError("grn arm needs grn_genome")
    rng = np.random.default_rng(seed)
    inputs = _inputs(np.random.default_rng(seed + 10_000), 256)
    env = DriftingHidden(seed, shift_every)
    log = OneLog(arm=arm, seed=seed)
    inj_rng = np.random.default_rng(seed + 777)
    last_inj = -10 ** 9

    def do_inject(agent: LCSAgent) -> None:
        nonlocal last_inj
        now = agent.organism.tick + 1
        if now - last_inj < cooldown:
            return
        last_inj = now
        inject(agent, random_rules(n_inject, int(inj_rng.integers(1 << 30))))
        log.injections.append(now)

    agent = LCSAgent("O", input_len=6, actions=["(emit 0)", "(emit 1)"], params=params, seed=seed,
                     cusum=None, cusum_signal="error",
                     cusum_on=do_inject if arm == "cusum" else None)
    grn = None
    if arm == "grn":
        from ..grn import GRN
        grn = GRN(grn_genome, agent, seed=seed, cusum=cusum)
    # warm-up window is not scored (all arms start from nothing); shifts are scored.
    # The detector arms only once the agent has converged (first probe >= 0.95), so
    # initial learning cannot be mistaken for a shift.
    window_start: Optional[int] = None
    w95 = w99 = None
    armed_at: Optional[int] = None
    for trial in range(1, trials + 1):
        if env.maybe_shift(trial, trials):
            if window_start is not None:
                log.recoveries95.append(w95)
                log.recoveries99.append(w99)
            log.shifts.append(trial)
            window_start, w95, w99 = trial, None, None
            if arm == "oracle":
                do_inject(agent)
        if arm == "periodic" and trial % period == 0:
            do_inject(agent)
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        agent.act(bits)
        r = 1.0 if agent.emitted() == env.truth(bits) else 0.0
        agent.reward(r)
        if grn is not None and armed_at is not None:
            grn.observe(r, agent.engine.last_explore)
            grn.step()
            if len(grn.injections) > len(log.injections):
                log.injections.append(trial)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            log.probes.append({"trial": trial, "acc": acc})
            if armed_at is None and acc >= 0.95:
                armed_at = trial
                if arm == "cusum":
                    agent.cusum, agent.cusum_s = cusum, 0.0
            if window_start is not None:
                since = trial - window_start
                if w95 is None and acc >= 0.95:
                    w95 = since
                if w99 is None and acc >= 0.99:
                    w99 = since
    if window_start is not None:
        log.recoveries95.append(w95)
        log.recoveries99.append(w99)
    log.false_injections = sum(1 for t in log.injections
                               if not any(0 <= t - s < false_window for s in log.shifts))
    log.final = {"arm": arm, "seed": seed, "shifts": log.shifts, "armed_at": armed_at,
                 "injections": len(log.injections),
                 "false_injections": log.false_injections, "cusum_fires": agent.cusum_fires,
                 "recoveries95": log.recoveries95, "recoveries99": log.recoveries99,
                 "macro_end": agent.engine.macro_size(),
                 "grn_expressions": dict(grn.expressions) if grn is not None else None,
                 "audit_chain_valid": agent.organism.chain.verify()}
    return log


def compare(seeds: Sequence[int] = range(5), trials: int = 15000, shift_every: int = 3000,
            n_inject: int = 20, cusum: Tuple[float, float, float] = (0.05, 0.10, 4.0),
            cooldown: int = 300, period: int = 500, arms: Sequence[str] = ARMS,
            params: Optional[Params] = None) -> Dict[str, Any]:
    cap = shift_every
    rows = []
    for sd in seeds:
        row: Dict[str, Any] = {"seed": sd, "arms": {}}
        for arm in arms:
            lg = run_arm(arm, sd, trials, shift_every, n_inject=n_inject, cusum=cusum,
                         cooldown=cooldown, period=period, params=params)
            row["arms"][arm] = {"recoveries95": lg.recoveries95, "recoveries99": lg.recoveries99,
                                "median95": float(np.median(lg.scores(cap, "95"))),
                                "median99": float(np.median(lg.scores(cap, "99"))),
                                "injections": len(lg.injections),
                                "false_injections": lg.false_injections}
        rows.append(row)
    n = len(rows)
    c1 = sum(r["arms"]["cusum"]["median95"] < r["arms"]["plain"]["median95"] for r in rows) \
        if {"cusum", "plain"} <= set(arms) else 0
    false_per_10k = (1e4 * sum(r["arms"]["cusum"]["false_injections"] for r in rows) / (n * trials)
                     if "cusum" in arms else None)
    pooled = {a: {"median95": float(np.median([r["arms"][a]["median95"] for r in rows])),
                  "median99": float(np.median([r["arms"][a]["median99"] for r in rows])),
                  "injections": float(np.mean([r["arms"][a]["injections"] for r in rows]))}
              for a in arms}
    out = {"seeds": list(seeds), "trials": trials, "shift_every": shift_every, "n_inject": n_inject,
           "cusum": list(cusum), "cooldown": cooldown, "period": period, "pooled": pooled, "rows": rows,
           "verdict": {"C1_count": c1, "C1": c1 >= 4 if n >= 5 else c1 == n,
                       "false_per_10k": false_per_10k, "C2": (false_per_10k is not None and false_per_10k <= 1.0),
                       "n": n}}
    out["verdict"]["pass"] = bool(out["verdict"]["C1"] and out["verdict"]["C2"])
    if {"cusum", "periodic"} <= set(arms):
        out["verdict"]["exploratory_cusum_beats_periodic"] = sum(
            r["arms"]["cusum"]["median95"] < r["arms"]["periodic"]["median95"] for r in rows)
    if {"cusum", "oracle"} <= set(arms):
        out["verdict"]["exploratory_cusum_vs_oracle_ratio"] = (
            pooled["cusum"]["median95"] / pooled["oracle"]["median95"] if pooled["oracle"]["median95"] else None)
    return out


def compare_csv(res: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seed", "arm", "recoveries95", "median95", "median99", "injections", "false_injections"])
    for r in res["rows"]:
        for a, v in r["arms"].items():
            w.writerow([r["seed"], a, json.dumps(v["recoveries95"]), v["median95"], v["median99"],
                        v["injections"], v["false_injections"]])
    return buf.getvalue()


__all__ = ["ARMS", "OneLog", "run_arm", "compare", "compare_csv"]
