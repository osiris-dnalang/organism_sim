"""
organism_sim.benchmarks.m2b — Specificity prior: does injecting rules of the right
specificity keep XCS agile, and is it more than tuning the covering wildcard rate?
==================================================================================

Follow-up to M2's exploratory finding (shape-matched random rules beat none 5/5 at 16 bits).
Two families (6-bit: k=2, width 6; 16-bit: k=3, width 16), drifting instances, schedule
known to the harness (question is the search, not detection).

Arms (identical seeds / engines / budgets):

    covering          XCS, default covering wildcard rate P# = 0.33         (control 1)
    periodic          XCS + blind injection: random rules (p# 0.5) every 500  (control 2)
    shape             XCS + specificity-matched random rules (k+1 specified bits, random
                      positions/values/actions) injected at start and each shift  (treatment)
    covering-matched  XCS with P# set so covered rules have the family's specificity
                      (6-bit: 3/6 → P# 0.5; 16-bit: 4/16 → P# 0.75)             (interpretation
                      control: is the effect "inject" or "right P#"?)

PRE-REGISTERED CRITERION (committed before the first run; fresh seeds 10–14; nothing tuned):
    per family:  C1  shape median recovery < covering on ≥ 4/5 seeds
                 C2  shape median recovery < periodic on ≥ 4/5 seeds
    PASS ⇔ C1 ∧ C2 on the 16-bit family (the family that motivated it); the 6-bit family is
    reported alongside. Exploratory (reported, not judged): shape vs covering-matched — if
    covering-matched ≈ shape, the mechanism is the wildcard rate and M3's learner should
    simply set P# per family; if shape < covering-matched, injection adds something beyond it.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params
from ..spec import Gene
from .expzero import DriftingHidden, inject, random_rules

ARMS = ("covering", "periodic", "shape", "covering-matched")
FAMILIES = {"6bit": {"k": 2, "width": 6, "N": 400, "shift_every": 3000, "trials": 15000,
                     "probe_every": 100, "n_inject": 20},
            "16bit": {"k": 3, "width": 16, "N": 1000, "shift_every": 30000, "trials": 120000,
                      "probe_every": 250, "n_inject": 40}}


def matched_p_wild(k: int, width: int) -> float:
    return 1.0 - (k + 1) / width


def shape_rules(n: int, seed: int, k: int, width: int) -> List[Gene]:
    rng = np.random.default_rng(seed + 55555)
    out = []
    for i in range(n):
        pos = list(rng.permutation(width))[:k + 1]
        cond = ["#"] * width
        for p in pos:
            cond[p] = str(int(rng.integers(2)))
        out.append(Gene(id=f"H{i}", name=f"shape_{i}", condition="".join(cond),
                        action=f"(emit {int(rng.integers(2))})", cluster="seed"))
    return out


@dataclass
class BLog:
    arm: str
    family: str
    seed: int
    recoveries: List[Optional[int]] = field(default_factory=list)
    shifts: List[int] = field(default_factory=list)
    injections: List[int] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    def scores(self, cap: int) -> List[int]:
        return [v if v is not None else cap for v in self.recoveries]

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


def run_arm(arm: str, family: str, seed: int, recovered_at: float = 0.95, period: int = 500) -> BLog:
    if arm not in ARMS or family not in FAMILIES:
        raise ValueError((arm, family))
    F = FAMILIES[family]
    k, width, N = F["k"], F["width"], F["N"]
    trials, shift_every, probe_every, n_inject = F["trials"], F["shift_every"], F["probe_every"], F["n_inject"]
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 10_000)
    inputs = ["".join(map(str, prng.integers(0, 2, width))) for _ in range(256)]
    env = DriftingHidden(seed, shift_every, k=k, width=width)
    p_wild = matched_p_wild(k, width) if arm == "covering-matched" else 0.33
    agent = LCSAgent("B", input_len=width, actions=["(emit 0)", "(emit 1)"],
                     params=Params(N=N, p_explore=0.5, p_wild=p_wild), seed=seed)
    log = BLog(arm=arm, family=family, seed=seed)
    inj_rng = np.random.default_rng(seed + 777)

    def do_inject(trial: int) -> None:
        s = int(inj_rng.integers(1 << 30))
        if arm == "shape":
            genes = shape_rules(n_inject, s, k, width)
        else:
            genes = [Gene(id=g.id, name=g.name, condition=g.condition.ljust(width, "#")[:width],
                          action=g.action, cluster=g.cluster) for g in random_rules(n_inject, s)]
        inject(agent, genes)
        log.injections.append(trial)

    if arm == "shape":
        do_inject(0)
    window_start: Optional[int] = None
    waiting = False
    for trial in range(1, trials + 1):
        if env.maybe_shift(trial, trials):
            if window_start is not None and waiting:
                log.recoveries.append(None)
            log.shifts.append(trial)
            window_start, waiting = trial, True
            if arm == "shape":
                do_inject(trial)
        if arm == "periodic" and trial % period == 0:
            do_inject(trial)
        bits = "".join(map(str, rng.integers(0, 2, width)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            if waiting and acc >= recovered_at and trial > window_start:
                log.recoveries.append(trial - window_start)
                waiting = False
    if window_start is not None and waiting:
        log.recoveries.append(None)
    log.final = {"arm": arm, "family": family, "seed": seed, "p_wild": p_wild, "shifts": log.shifts,
                 "recoveries": log.recoveries, "injections": len(log.injections),
                 "macro_end": agent.engine.macro_size(), "audit_chain_valid": agent.organism.chain.verify()}
    return log


def compare(seeds: Sequence[int] = range(10, 15), families: Sequence[str] = ("6bit", "16bit"),
            arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seeds": list(seeds), "families": {}}
    for fam in families:
        cap = FAMILIES[fam]["shift_every"]
        rows = []
        for sd in seeds:
            row: Dict[str, Any] = {"seed": sd, "arms": {}}
            for arm in arms:
                lg = run_arm(arm, fam, sd)
                row["arms"][arm] = {"recoveries": lg.recoveries, "median": float(np.median(lg.scores(cap))),
                                    "injections": len(lg.injections), "p_wild": lg.final["p_wild"]}
            rows.append(row)
        n = len(rows)

        def wins(a, b):
            return sum(r["arms"][a]["median"] < r["arms"][b]["median"] for r in rows)
        c1, c2 = wins("shape", "covering"), wins("shape", "periodic")
        out["families"][fam] = {
            "config": FAMILIES[fam], "matched_p_wild": matched_p_wild(FAMILIES[fam]["k"], FAMILIES[fam]["width"]),
            "pooled_median": {a: float(np.median([r["arms"][a]["median"] for r in rows])) for a in arms},
            "rows": rows,
            "verdict": {"C1_shape_vs_covering": c1, "C2_shape_vs_periodic": c2, "n": n,
                        "C1": c1 >= 4 if n >= 5 else c1 == n, "C2": c2 >= 4 if n >= 5 else c2 == n,
                        "exploratory_shape_vs_covering_matched": wins("shape", "covering-matched"),
                        "exploratory_covering_matched_vs_covering": wins("covering-matched", "covering")}}
        out["families"][fam]["verdict"]["pass"] = bool(out["families"][fam]["verdict"]["C1"]
                                                       and out["families"][fam]["verdict"]["C2"])
    out["verdict"] = {"pass": bool(out["families"].get("16bit", {}).get("verdict", {}).get("pass", False)),
                      "judged_on": "16bit"}
    return out


def compare_csv(res: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["family", "seed", "arm", "p_wild", "recoveries", "median", "injections"])
    for fam, d in res["families"].items():
        for r in d["rows"]:
            for a, v in r["arms"].items():
                w.writerow([fam, r["seed"], a, v["p_wild"], json.dumps(v["recoveries"]), v["median"], v["injections"]])
    return buf.getvalue()


__all__ = ["ARMS", "FAMILIES", "matched_p_wild", "shape_rules", "run_arm", "compare", "compare_csv"]
