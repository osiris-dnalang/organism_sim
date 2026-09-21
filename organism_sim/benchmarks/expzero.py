"""
organism_sim.benchmarks.expzero — Experiment Zero: offline structural priors vs random init
===========================================================================================

Task family (hidden instance per seed): a *generalised* 6-input multiplexer. Two of the six
positions are address bits (unknown which), the other four are data bits in an unknown
order, and the output may be inverted. The instance switches to a fresh one every
``shift_every`` trials (non-stationary). A seeder that knows only the family can write rules
of the right *shape* (two address bits + one data bit specified, output = that data bit) but
cannot know positions, order or inversion — so priors can supply structure, not solutions.

Arms (identical seeds, engines and budgets):

    A  random      XCS from covering only
    B  informed    XCS + 20 family-shaped rules from ``expzero_seed_rules.dna`` (written once,
                   before any instance), injected at start and at each shift
    C  control     XCS + 20 random well-formed rules, same injection schedule (separates
                   "any seeding helps" from "informed seeding helps")

Seeded rules enter with the engine's normal initial prediction/error/fitness and zero
experience — they are hypotheses, not answers.

Metrics: trials to first probe ≥ 0.95 from start; per shift, trials to recover ≥ 0.95.
Pre-registered criterion (seeds 0–4; nothing tuned): B beats **both** A and C on the
median of (initial + recoveries) on ≥ 4/5 seeds → PASS; else the claim is dead.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params
from ..spec import Gene, parse_dna
from .mux import _inputs

SEED_DNA = Path(__file__).with_name("expzero_seed_rules.dna")
ARMS = ("random", "informed", "control")


class HiddenMux:
    """One instance of the family: ``k`` hidden address positions select one of ``2**k``
    hidden data positions out of ``width`` inputs (the rest are irrelevant); optional
    inversion. ``k=2, width=6`` is the 6-bit family; ``k=3, width=16`` the wide one."""

    def __init__(self, rng: np.random.Generator, k: int = 2, width: int = 6):
        assert width >= k + 2 ** k
        perm = list(rng.permutation(width))
        self.k, self.width = k, width
        self.addr = perm[:k]
        self.data = perm[k:k + 2 ** k]
        self.irrelevant = perm[k + 2 ** k:]
        self.invert = bool(rng.integers(2))

    def truth(self, bits: str) -> str:
        b = [int(c) for c in bits]
        idx = 0
        for a in self.addr:
            idx = 2 * idx + b[a]
        out = b[self.data[idx]]
        return str(1 - out if self.invert else out)

    def describe(self) -> Dict[str, Any]:
        return {"k": self.k, "width": self.width, "addr": [int(a) for a in self.addr],
                "data": [int(d) for d in self.data], "irrelevant": [int(i) for i in self.irrelevant],
                "invert": self.invert}


class DriftingHidden:
    def __init__(self, seed: int, shift_every: int, k: int = 2, width: int = 6):
        self.rng = np.random.default_rng(seed + 4242)
        self.shift_every = shift_every
        self.k, self.width = k, width
        self.instance = HiddenMux(self.rng, k, width)
        self.shifts: List[int] = []

    def maybe_shift(self, trial: int, trials: int) -> bool:
        if trial % self.shift_every == 0 and trial + self.shift_every <= trials:
            self.instance = HiddenMux(self.rng, self.k, self.width)
            self.shifts.append(trial)
            return True
        return False

    def truth(self, bits: str) -> str:
        return self.instance.truth(bits)


# ── seed rules ───────────────────────────────────────────────────────────────

def informed_rules(n: int = 20, seed: int = 999) -> List[Gene]:
    """Family-shaped hypotheses: two guessed address positions with values, one guessed data
    position with value v, everything else '#', action = emit v. Positions are guesses."""
    rng = np.random.default_rng(seed)
    out: List[Gene] = []
    seen = set()
    while len(out) < n:
        pos = list(rng.permutation(6))
        a0, a1, d = pos[0], pos[1], pos[2]
        va, vb, v = (int(x) for x in rng.integers(0, 2, 3))
        cond = ["#"] * 6
        cond[a0], cond[a1], cond[d] = str(va), str(vb), str(v)
        key = ("".join(cond), v)
        if key in seen:
            continue
        seen.add(key)
        out.append(Gene(id=f"S{len(out)}", name=f"shape_{a0}{a1}_{d}", condition="".join(cond),
                        action=f"(emit {v})", cluster="seed"))
    return out


def random_rules(n: int = 20, seed: int = 0) -> List[Gene]:
    rng = np.random.default_rng(seed + 31337)
    out = []
    for i in range(n):
        cond = "".join("#" if rng.random() < 0.5 else str(int(rng.integers(2))) for _ in range(6))
        out.append(Gene(id=f"R{i}", name=f"rand_{i}", condition=cond,
                        action=f"(emit {int(rng.integers(2))})", cluster="seed"))
    return out


def write_seed_dna(path: Path = SEED_DNA, n: int = 20, seed: int = 999) -> Path:
    genes = informed_rules(n, seed)
    lines = ["// Experiment Zero — structural priors for the generalised-multiplexer family.",
             "// Written once from the FAMILY definition only (two unknown address positions, four",
             "// unknown data positions, optional inversion). Positions and values are guesses.",
             f"// Generated deterministically with seed {seed}; never edited after any instance was seen.",
             "ORGANISM EXPZERO_PRIORS {", '    META { version: "1.0.0", domain: "benchmark" }', "    GENOME {"]
    for g in genes:
        lines.append(f'        GENE {g.name} {{ id: "{g.id}", condition: "{g.condition}", '
                     f'action: "{g.action}", expression: 1.0 }}')
    lines += ["    }", "}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def load_seed_rules() -> List[Gene]:
    return parse_dna(SEED_DNA.read_text()).genome.rules()


def inject(agent: LCSAgent, genes: Sequence[Gene]) -> int:
    """Insert rules as fresh hypotheses (engine defaults for p/ε/F, zero experience)."""
    eng = agent.engine
    n = 0
    for g in genes:
        eng._insert(eng._new_rule(g.condition, g.action))
        n += 1
    eng._delete_excess()
    return n


# ── run ──────────────────────────────────────────────────────────────────────

@dataclass
class ZeroLog:
    arm: str
    seed: int
    initial: Optional[int]
    recoveries: List[Optional[int]] = field(default_factory=list)
    shifts: List[int] = field(default_factory=list)
    instances: List[Dict[str, Any]] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    def scores(self, cap: int) -> List[int]:
        vals = [self.initial] + list(self.recoveries)
        return [v if v is not None else cap for v in vals]

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
            probe_every: int = 100, recovered_at: float = 0.95, n_seed_rules: int = 20,
            params: Optional[Params] = None) -> ZeroLog:
    if arm not in ARMS:
        raise ValueError(arm)
    rng = np.random.default_rng(seed)
    inputs = _inputs(np.random.default_rng(seed + 10_000), 256)
    env = DriftingHidden(seed, shift_every)
    agent = LCSAgent("Z", input_len=6, actions=["(emit 0)", "(emit 1)"], params=params, seed=seed)
    seeds_for_arm = (load_seed_rules()[:n_seed_rules] if arm == "informed"
                     else random_rules(n_seed_rules, seed) if arm == "control" else [])
    if seeds_for_arm:
        inject(agent, seeds_for_arm)
    log = ZeroLog(arm=arm, seed=seed, initial=None, instances=[env.instance.describe()])
    window_start, waiting = 0, True
    for trial in range(1, trials + 1):
        if env.maybe_shift(trial, trials):
            log.shifts.append(trial)
            log.instances.append(env.instance.describe())
            if waiting:                                     # previous window never recovered
                if window_start == 0:
                    log.initial = None
                else:
                    log.recoveries.append(None)
            window_start, waiting = trial, True
            if seeds_for_arm:
                inject(agent, seeds_for_arm)
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            log.probes.append({"trial": trial, "acc": acc, "macro": agent.engine.macro_size()})
            if waiting and acc >= recovered_at and trial > window_start:
                since = trial - window_start
                if window_start == 0:
                    log.initial = since
                else:
                    log.recoveries.append(since)
                waiting = False
    if waiting:
        if window_start == 0:
            log.initial = None
        else:
            log.recoveries.append(None)
    log.final = {"arm": arm, "seed": seed, "initial": log.initial, "recoveries": log.recoveries,
                 "shifts": log.shifts, "macro_end": agent.engine.macro_size(),
                 "audit_chain_valid": agent.organism.chain.verify()}
    return log


def compare(seeds: Sequence[int] = range(5), trials: int = 15000, shift_every: int = 3000,
            params: Optional[Params] = None, arms: Sequence[str] = ARMS) -> Dict[str, Any]:
    cap = shift_every
    rows = []
    for sd in seeds:
        row: Dict[str, Any] = {"seed": sd, "arms": {}}
        for arm in arms:
            lg = run_arm(arm, sd, trials, shift_every, params=params)
            sc = lg.scores(cap)
            row["arms"][arm] = {"initial": lg.initial, "recoveries": lg.recoveries,
                                "median": float(np.median(sc)), "mean": float(np.mean(sc)),
                                "unrecovered": sum(v is None for v in [lg.initial] + lg.recoveries)}
        rows.append(row)
    wins = sum(1 for r in rows if r["arms"]["informed"]["median"] < r["arms"]["random"]["median"]
               and r["arms"]["informed"]["median"] < r["arms"]["control"]["median"])
    n = len(rows)
    out = {"seeds": list(seeds), "trials": trials, "shift_every": shift_every,
           "seed_rules_sha": __import__("hashlib").sha256(SEED_DNA.read_bytes()).hexdigest()[:16],
           "pooled_median": {a: float(np.median([v for r in rows for v in
                                                  [r["arms"][a]["median"]]])) for a in arms},
           "rows": rows,
           "verdict": {"informed_wins": wins, "n": n, "pass": wins >= 4 if n >= 5 else wins == n}}
    return out


def compare_csv(res: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seed", "arm", "initial", "recoveries", "median"])
    for r in res["rows"]:
        for a, v in r["arms"].items():
            w.writerow([r["seed"], a, v["initial"], json.dumps(v["recoveries"]), v["median"]])
    return buf.getvalue()


__all__ = ["HiddenMux", "DriftingHidden", "informed_rules", "random_rules", "write_seed_dna",
           "load_seed_rules", "inject", "run_arm", "compare", "compare_csv", "ARMS", "SEED_DNA"]
