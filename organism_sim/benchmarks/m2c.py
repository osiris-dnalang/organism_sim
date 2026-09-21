"""
organism_sim.benchmarks.m2c — Specificity prior, second replication on fresh seeds
=================================================================================

Prerequisite check before any further open-ended-generation work (M3): does injecting
random rules of the *family's specificity* make XCS recover faster after drift in the wide
(16-bit) space than (1) covering alone and (2) blind injection? The record so far:

    M2  (seeds 0–4,   exploratory)      shape < none 5/5 at 16 bits (≈ 13 %)
    M2b (seeds 10–14, pre-registered)   FAIL: shape < covering 3/5, shape < periodic 4/5

This is a third, independent sample. The harness is ``m2b`` **unchanged** (same arms,
families, budgets, injection sizes, probe schedule); only the seeds are new, and one
criterion is added. Arms, in the vocabulary of the request that prompted this run:

    covering          baseline — XCS on its default covering wildcard rate P# = 0.33
    periodic          control  — blind periodic injection of random rules (p# 0.5, every 500).
                      ("zero-specificity" here means *no specificity matching*; an all-wildcard
                      rule would be a degenerate default rule, not an injection control.)
    shape             treatment — specificity-matched random rules (k+1 specified bits) at
                      start and at every shift
    covering-matched  interpretation control carried over from M2b (P# set to the family's
                      specificity), reported, not judged

PRE-REGISTERED CRITERION (committed before the first run; fresh seeds 30–34; nothing tuned):
    on the 16-bit family
        C1  shape median recovery < covering on ≥ 4/5 seeds                (as M2b)
        C2  shape median recovery < periodic on ≥ 4/5 seeds                (as M2b)
        C3  pooled per-shift recoveries (3 shifts × 5 seeds = 15 per arm): one-sided
            permutation rank-sum test, shape < covering AND shape < periodic, p < 0.05 each
            (20 000 permutations, permutation seed 0)
    PASS ⇔ C1 ∧ C2 ∧ C3. The 6-bit family is reported alongside, not judged.
    Interpretation if FAIL: three samples, one exploratory hit and two non-replications —
    the specificity prior is closed; M3 stays blocked on base-learner efficiency and the
    next pre-registration is XCS engineering, exactly as PROGRAM.md already states.
    Interpretation if PASS: M2b was a false negative; proceed to scaffold M3 with
    shape-matched injection as part of the learner.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .m2b import ARMS, FAMILIES, compare_csv, run_arm

SEEDS = tuple(range(30, 35))
N_PERM = 20_000
PERM_SEED = 0
ALPHA = 0.05


def rank_sum_perm_test(a: Sequence[float], b: Sequence[float], n_perm: int = N_PERM,
                       seed: int = PERM_SEED) -> Dict[str, float]:
    """One-sided permutation test of H1: values in *a* are stochastically smaller than in
    *b*. Statistic = mean rank of *a* in the pooled ranking (average ranks on ties). The
    p-value is the fraction of label permutations with a mean rank ≤ the observed one
    (+1 correction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n = len(a)

    def mean_rank_a(order_idx: np.ndarray) -> float:
        # average ranks with ties, then take the mean over the first n labels
        vals = pooled[order_idx]
        ranks = np.empty(len(vals))
        srt = np.argsort(vals, kind="mergesort")
        sv = vals[srt]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            ranks[srt[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return float(ranks[:n].mean())

    idx = np.arange(len(pooled))
    observed = mean_rank_a(idx)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        if mean_rank_a(idx) <= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"mean_rank_a": observed, "n_a": int(n), "n_b": int(len(b)), "p_one_sided": float(p),
            "median_a": float(np.median(a)), "median_b": float(np.median(b))}


def compare(seeds: Sequence[int] = SEEDS, families: Sequence[str] = ("6bit", "16bit"),
            arms: Sequence[str] = ARMS, log=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"experiment": "m2c", "seeds": list(seeds), "harness": "m2b (unchanged)",
                           "families": {}}
    for fam in families:
        cap = FAMILIES[fam]["shift_every"]
        rows: List[Dict[str, Any]] = []
        for sd in seeds:
            row: Dict[str, Any] = {"seed": sd, "arms": {}}
            for arm in arms:
                lg = run_arm(arm, fam, sd)
                row["arms"][arm] = {"recoveries": lg.recoveries, "scores": lg.scores(cap),
                                    "median": float(np.median(lg.scores(cap))),
                                    "injections": len(lg.injections), "p_wild": lg.final["p_wild"]}
            rows.append(row)
            if log:
                log(f"{fam} seed {sd} " + json.dumps({a: (v["scores"], v["median"]) for a, v in row["arms"].items()}))
        n = len(rows)

        def wins(a: str, b: str) -> int:
            return sum(r["arms"][a]["median"] < r["arms"][b]["median"] for r in rows)

        def pooled(a: str) -> List[float]:
            return [x for r in rows for x in r["arms"][a]["scores"]]

        c1, c2 = wins("shape", "covering"), wins("shape", "periodic")
        t_cov = rank_sum_perm_test(pooled("shape"), pooled("covering"))
        t_per = rank_sum_perm_test(pooled("shape"), pooled("periodic"))
        c3 = t_cov["p_one_sided"] < ALPHA and t_per["p_one_sided"] < ALPHA
        verdict = {"C1_shape_vs_covering": c1, "C2_shape_vs_periodic": c2, "n": n,
                   "C1": c1 >= 4 if n >= 5 else c1 == n, "C2": c2 >= 4 if n >= 5 else c2 == n,
                   "C3_tests": {"shape_vs_covering": t_cov, "shape_vs_periodic": t_per},
                   "C3": bool(c3), "alpha": ALPHA,
                   "exploratory_shape_vs_covering_matched": wins("shape", "covering-matched"),
                   "exploratory_covering_matched_vs_covering": wins("covering-matched", "covering")}
        verdict["pass"] = bool(verdict["C1"] and verdict["C2"] and verdict["C3"])
        out["families"][fam] = {
            "config": FAMILIES[fam],
            "pooled_median": {a: float(np.median([r["arms"][a]["median"] for r in rows])) for a in arms},
            "pooled_per_shift_median": {a: float(np.median(pooled(a))) for a in arms},
            "rows": rows, "verdict": verdict}
        if log:
            log(f"{fam} POOLED {json.dumps(out['families'][fam]['pooled_median'])} "
                f"VERDICT {json.dumps({k: v for k, v in verdict.items() if k != 'C3_tests'})} "
                f"p(shape<covering)={t_cov['p_one_sided']:.4f} p(shape<periodic)={t_per['p_one_sided']:.4f}")
    out["verdict"] = {"pass": bool(out["families"].get("16bit", {}).get("verdict", {}).get("pass", False)),
                      "judged_on": "16bit", "criterion": "C1 and C2 and C3"}
    return out


def main(argv: Sequence[str] = ()) -> int:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    tag = f"m2c_eval_seeds{SEEDS[0]}-{SEEDS[-1]}"
    logf = open(results / "m2c_run.log", "a")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    log(f"START m2c seeds={list(SEEDS)} harness=m2b arms={list(ARMS)}")
    res = compare(log=log)
    (results / f"{tag}.json").write_text(json.dumps(res, indent=2))
    (results / f"{tag}.csv").write_text(compare_csv(res))
    log(f"OVERALL {json.dumps(res['verdict'])}")
    log(f"DONE {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
