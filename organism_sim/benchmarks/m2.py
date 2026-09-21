"""
organism_sim.benchmarks.m2 — M2: does content matter when injection stops being free?
=====================================================================================

Wide hidden family: 16 inputs, 3 hidden address positions selecting one of 8 hidden data
positions, 5 irrelevant positions, optional inversion; a fresh instance every
``shift_every`` trials (schedule known to the harness, as in Experiment Zero — the
question here is content, not detection). XCS population N = 1000 (harness calibration on
tuning seed 100: ≥ 0.95 within ~15k trials; N = 400 needs ~40k).

Arms (identical seeds / engines / budgets; ``n_inject`` rules injected at start and at
each shift as zero-experience hypotheses):

    none          covering only
    informed      family-shaped priors from the family definition only: 3 guessed address
                  positions + 1 guessed data position specified, action = the data bit's
                  value (``m2_seed_rules.dna``, written once, sha recorded)
    shape         same specificity (4 specified bits), random positions/values, random
                  action — isolates the *coupling* of action to a specified bit
    random        Experiment Zero's control: random rules, p(#) = 0.5 (~8 specified bits)
    periodic      Experiment One's fastest 6-bit mechanism: random rules every 500 trials

Metric: per scored shift, trials to probe accuracy ≥ 0.95 (cap = shift_every).

PRE-REGISTERED CRITERION (committed before the first evaluation; seeds 0–4; nothing tuned):
    C1  informed median < shape median on ≥ 4/5 seeds     (action coupling carries signal)
    C2  informed median < random median on ≥ 4/5 seeds    (shape carries signal)
    PASS ⇔ C1 ∧ C2. Exploratory: informed vs none; periodic vs none (does blind
    diversity still pay at 16 bits?).
Interpretation: PASS → priors contain signal once the space is wide and the LLM-prior
branch (Vector 1) reopens under a random-injection-matched control. C2 only → shape
matters, content beyond shape does not. Neither → on this family the substrate's
contribution remains topology + diversity, and M3 is built on that base.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params
from ..spec import Gene, parse_dna
from .expzero import DriftingHidden, inject, random_rules

SEED_DNA = Path(__file__).with_name("m2_seed_rules.dna")
ARMS = ("none", "informed", "shape", "random", "periodic")
K, WIDTH = 3, 16


def informed_rules(n: int = 40, seed: int = 999, k: int = K, width: int = WIDTH) -> List[Gene]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        pos = list(rng.permutation(width))
        addr, d = pos[:k], pos[k]
        vals = [int(x) for x in rng.integers(0, 2, k)]
        v = int(rng.integers(2))
        cond = ["#"] * width
        for a, va in zip(addr, vals):
            cond[a] = str(va)
        cond[d] = str(v)
        key = "".join(cond)
        if key in seen:
            continue
        seen.add(key)
        out.append(Gene(id=f"S{len(out)}", name=f"shape_{'_'.join(map(str, addr))}_{d}",
                        condition=key, action=f"(emit {v})", cluster="seed"))
    return out


def shape_rules(n: int = 40, seed: int = 0, k: int = K, width: int = WIDTH) -> List[Gene]:
    """Same specificity as informed (k + 1 bits), random positions, values, and action."""
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


def write_seed_dna(path: Path = SEED_DNA, n: int = 40, seed: int = 999) -> Path:
    genes = informed_rules(n, seed)
    lines = ["// M2 — structural priors for the 16-bit generalised-multiplexer family (3 address",
             "// positions, 8 data positions, 5 irrelevant, optional inversion). Written once from the",
             f"// FAMILY definition only; positions and values are guesses. Seed {seed}; never edited.",
             "ORGANISM M2_PRIORS {", '    META { version: "1.0.0", domain: "benchmark" }', "    GENOME {"]
    for g in genes:
        lines.append(f'        GENE {g.name} {{ id: "{g.id}", condition: "{g.condition}", '
                     f'action: "{g.action}", expression: 1.0 }}')
    lines += ["    }", "}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def load_seed_rules() -> List[Gene]:
    return parse_dna(SEED_DNA.read_text()).genome.rules()


@dataclass
class M2Log:
    arm: str
    seed: int
    recoveries: List[Optional[int]] = field(default_factory=list)
    shifts: List[int] = field(default_factory=list)
    injections: List[int] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)
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


def run_arm(arm: str, seed: int, trials: int = 120000, shift_every: int = 30000,
            probe_every: int = 250, n_inject: int = 40, period: int = 500,
            recovered_at: float = 0.95, params: Optional[Params] = None) -> M2Log:
    if arm not in ARMS:
        raise ValueError(arm)
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 10_000)
    inputs = ["".join(map(str, prng.integers(0, 2, WIDTH))) for _ in range(256)]
    env = DriftingHidden(seed, shift_every, k=K, width=WIDTH)
    agent = LCSAgent("M2", input_len=WIDTH, actions=["(emit 0)", "(emit 1)"],
                     params=params or Params(N=1000, p_explore=0.5), seed=seed)
    log = M2Log(arm=arm, seed=seed)
    inj_rng = np.random.default_rng(seed + 777)

    def cohort() -> List[Gene]:
        if arm == "informed":
            return load_seed_rules()[:n_inject]
        if arm == "shape":
            return shape_rules(n_inject, int(inj_rng.integers(1 << 30)))
        if arm in ("random", "periodic"):
            return random_rules(n_inject, int(inj_rng.integers(1 << 30)))[:n_inject]
        return []

    def do_inject(trial: int) -> None:
        c = cohort()
        if c:
            c = [Gene(id=g.id, name=g.name, condition=g.condition.ljust(WIDTH, "#")[:WIDTH],
                      action=g.action, cluster=g.cluster) for g in c]
            inject(agent, c)
            log.injections.append(trial)

    if arm in ("informed", "shape", "random"):
        do_inject(0)
    window_start: Optional[int] = None
    waiting = False
    for trial in range(1, trials + 1):
        if env.maybe_shift(trial, trials):
            if window_start is not None and waiting:
                log.recoveries.append(None)
            log.shifts.append(trial)
            window_start, waiting = trial, True
            if arm in ("informed", "shape", "random"):
                do_inject(trial)
        if arm == "periodic" and trial % period == 0:
            do_inject(trial)
        bits = "".join(map(str, rng.integers(0, 2, WIDTH)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            log.probes.append({"trial": trial, "acc": acc, "macro": agent.engine.macro_size()})
            if waiting and acc >= recovered_at and trial > window_start:
                log.recoveries.append(trial - window_start)
                waiting = False
    if window_start is not None and waiting:
        log.recoveries.append(None)
    log.final = {"arm": arm, "seed": seed, "shifts": log.shifts, "recoveries": log.recoveries,
                 "injections": len(log.injections), "macro_end": agent.engine.macro_size(),
                 "audit_chain_valid": agent.organism.chain.verify()}
    return log


def compare(seeds: Sequence[int] = range(5), trials: int = 120000, shift_every: int = 30000,
            n_inject: int = 40, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    cap = shift_every
    rows = []
    for sd in seeds:
        row: Dict[str, Any] = {"seed": sd, "arms": {}}
        for arm in arms:
            lg = run_arm(arm, sd, trials, shift_every, n_inject=n_inject)
            row["arms"][arm] = {"recoveries": lg.recoveries, "median": float(np.median(lg.scores(cap))),
                                "unrecovered": sum(v is None for v in lg.recoveries),
                                "injections": len(lg.injections)}
        rows.append(row)
    n = len(rows)

    def wins(a, b):
        return sum(r["arms"][a]["median"] < r["arms"][b]["median"] for r in rows)
    c1, c2 = wins("informed", "shape"), wins("informed", "random")
    out = {"seeds": list(seeds), "trials": trials, "shift_every": shift_every, "n_inject": n_inject,
           "k": K, "width": WIDTH, "N": 1000,
           "seed_rules_sha": hashlib.sha256(SEED_DNA.read_bytes()).hexdigest()[:16],
           "pooled_median": {a: float(np.median([r["arms"][a]["median"] for r in rows])) for a in arms},
           "rows": rows,
           "verdict": {"C1_informed_vs_shape": c1, "C2_informed_vs_random": c2, "n": n,
                       "C1": c1 >= 4 if n >= 5 else c1 == n, "C2": c2 >= 4 if n >= 5 else c2 == n,
                       "exploratory_informed_vs_none": wins("informed", "none"),
                       "exploratory_periodic_vs_none": wins("periodic", "none")}}
    out["verdict"]["pass"] = bool(out["verdict"]["C1"] and out["verdict"]["C2"])
    return out


def compare_csv(res: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seed", "arm", "recoveries", "median", "injections"])
    for r in res["rows"]:
        for a, v in r["arms"].items():
            w.writerow([r["seed"], a, json.dumps(v["recoveries"]), v["median"], v["injections"]])
    return buf.getvalue()


__all__ = ["ARMS", "K", "WIDTH", "SEED_DNA", "informed_rules", "shape_rules", "write_seed_dna",
           "load_seed_rules", "run_arm", "compare", "compare_csv"]
