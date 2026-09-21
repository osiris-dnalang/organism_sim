"""
organism_sim.benchmarks.m1 — M1: regulatory genes vs flat learners under drift
==============================================================================

Question: does *regulation* — genes that fire on internal events and on each other's
outputs, deciding when hypotheses enter the population and how the learner explores — buy
anything beyond the best single mechanism measured so far (CUSUM-triggered injection,
Experiment One)?

Arms (Experiment One harness, hidden drifting generalised-mux family, shift schedule
unknown to the agent, detector/regulation armed after first convergence):

    plain        XCS
    cusum        XCS + CUSUM-triggered injection (Experiment One's PASS profile)
    periodic     XCS + blind periodic injection (fastest known, 30 injections)
    grn-hand     XCS + the hand-written regulatory genome ``m1_hand.dna``
    grn-evolved  XCS + a genome meta-evolved on tuning seeds 100–102 from the hand genome
                 (fitness = mean per-seed median recovery); saved as ``m1_evolved.dna``

PRE-REGISTERED CRITERION (committed before the first evaluation run; seeds 0–4):
    C1  grn-evolved median recovery (≥ 0.95) < cusum on ≥ 4/5 seeds
    C2  grn-evolved false injections ≤ 1 per 10k trials
    PASS ⇔ C1 ∧ C2. Exploratory: grn-evolved vs periodic and vs plain; grn-hand vs cusum;
    which genes the evolved genome expresses.
Interpretation if FAIL: on this family regulation is redundant with a single detector +
injection; the next attempt must use a task that *needs* structure (M2/M3), not a retune.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..grn import GRN, genome_to_dna
from ..spec import Genome, parse_dna
from .expone import run_arm

HAND_DNA = Path(__file__).with_name("m1_hand.dna")
EVOLVED_DNA = Path(__file__).with_name("m1_evolved.dna")
CUSUM = (0.05, 0.10, 8.0)
N_INJECT = 40


def load(path: Path) -> Genome:
    return parse_dna(path.read_text()).genome


def fitness(genome: Genome, seeds: Sequence[int], trials: int = 15000, shift_every: int = 3000) -> float:
    vals = []
    for sd in seeds:
        lg = run_arm("grn", sd, trials, shift_every, n_inject=N_INJECT, cusum=CUSUM, grn_genome=genome)
        vals.append(float(np.median(lg.scores(shift_every))))
    return float(np.mean(vals))


def evolve(seeds: Sequence[int] = (100, 101, 102), pop: int = 8, gens: int = 5, seed: int = 0,
           log: Optional[List[Dict[str, Any]]] = None) -> Genome:
    rng = np.random.default_rng(seed)
    base = load(HAND_DNA)
    population = [base] + [GRN.mutate(base, rng) for _ in range(pop - 1)]
    scores = [fitness(g, seeds) for g in population]
    for gen in range(gens):
        order = np.argsort(scores)
        elite = [population[i] for i in order[:2]]
        children = list(elite)
        while len(children) < pop:
            a, b = population[int(order[int(rng.integers(4))])], population[int(order[int(rng.integers(4))])]
            child = GRN.mutate(GRN.crossover(a, b, rng) if rng.random() < 0.5 else a, rng)
            children.append(child)
        population = children
        scores = [scores[int(order[0])], scores[int(order[1])]] + [fitness(g, seeds) for g in population[2:]]
        if log is not None:
            log.append({"gen": gen, "best": float(min(scores)), "mean": float(np.mean(scores))})
    best = population[int(np.argmin(scores))]
    return best


def compare(seeds: Sequence[int] = range(5), trials: int = 15000, shift_every: int = 3000,
            evolved: Optional[Genome] = None) -> Dict[str, Any]:
    hand = load(HAND_DNA)
    evolved = evolved or load(EVOLVED_DNA)
    cap = shift_every
    rows = []
    for sd in seeds:
        row: Dict[str, Any] = {"seed": sd, "arms": {}}
        for arm, kw in (("plain", {}), ("cusum", {}), ("periodic", {}),
                        ("grn-hand", {"grn_genome": hand}), ("grn-evolved", {"grn_genome": evolved})):
            lg = run_arm("grn" if arm.startswith("grn") else arm, sd, trials, shift_every,
                         n_inject=N_INJECT, cusum=CUSUM, **kw)
            row["arms"][arm] = {"recoveries95": lg.recoveries95, "median95": float(np.median(lg.scores(cap))),
                                "median99": float(np.median(lg.scores(cap, "99"))),
                                "injections": len(lg.injections), "false_injections": lg.false_injections,
                                "expressions": lg.final.get("grn_expressions")}
        rows.append(row)
    n = len(rows)
    c1 = sum(r["arms"]["grn-evolved"]["median95"] < r["arms"]["cusum"]["median95"] for r in rows)
    false = 1e4 * sum(r["arms"]["grn-evolved"]["false_injections"] for r in rows) / (n * trials)
    pooled = {a: float(np.median([r["arms"][a]["median95"] for r in rows])) for a in rows[0]["arms"]}
    out = {"seeds": list(seeds), "trials": trials, "shift_every": shift_every, "cusum": list(CUSUM),
           "n_inject": N_INJECT, "hand_sha": hashlib.sha256(HAND_DNA.read_bytes()).hexdigest()[:16],
           "evolved_sha": hashlib.sha256(EVOLVED_DNA.read_bytes()).hexdigest()[:16] if EVOLVED_DNA.exists() else None,
           "pooled_median95": pooled, "rows": rows,
           "verdict": {"C1_count": c1, "C1": c1 >= 4 if n >= 5 else c1 == n,
                       "false_per_10k": false, "C2": false <= 1.0, "n": n,
                       "exploratory_evolved_beats_periodic": sum(
                           r["arms"]["grn-evolved"]["median95"] < r["arms"]["periodic"]["median95"] for r in rows),
                       "exploratory_evolved_beats_plain": sum(
                           r["arms"]["grn-evolved"]["median95"] < r["arms"]["plain"]["median95"] for r in rows),
                       "exploratory_hand_beats_cusum": sum(
                           r["arms"]["grn-hand"]["median95"] < r["arms"]["cusum"]["median95"] for r in rows)}}
    out["verdict"]["pass"] = bool(out["verdict"]["C1"] and out["verdict"]["C2"])
    return out


__all__ = ["HAND_DNA", "EVOLVED_DNA", "CUSUM", "N_INJECT", "load", "fitness", "evolve", "compare", "genome_to_dna"]
