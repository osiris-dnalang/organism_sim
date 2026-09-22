"""
experiments/substrate_opt.py — Substrate-Opt-1: can XCS's own knobs break the 16-bit
recovery ceiling that no injection layer could move?
=====================================================================================

Every experiment above the learner (Zero, One, M1, M2, M2b, M2c, M3) left 16-bit drift
recovery at ≈ 9,500–12,000 trials per shift. This is the first experiment *on* the learner:
a grid over four XCS parameters, on the M2b/M2c 16-bit family with the same budgets, no
injection of any kind (the `covering` arm's loop).

Grid (54 configurations) + the current default as a 55th point:

    theta_ga         GA period (trials)                    [25, 100, 500]
    as_subsumption   action-set subsumption after update   [on, off]
    mu               per-bit mutation probability          [0.01, 0.05, 0.10]
    p_wild           covering wildcard probability         [0.33, 0.50, 0.75]

Protocol (two stages, disjoint seeds — a grid search is tuning, so it is judged elsewhere):

    sweep   every configuration × tuning seeds 110–114; score = pooled median of per-seed
            median recovery (cap = shift interval, 30,000, for an unrecovered shift)
    judge   the single best sweep configuration and the default, on fresh seeds 40–44

PRE-REGISTERED DECISION (committed before the first run; judge seeds never used before):
    metric  pooled median recovery of the winner on the judge seeds (M2b's metric)
    PASS      winner < 3,000       → adopt as the mandatory 16-bit default (Params16)
    PARTIAL   3,000 ≤ winner < 8,000 and winner < default on ≥ 4/5 judge seeds
              → real improvement, target missed; record, do not adopt as mandatory
    BOUNDED   winner ≥ 8,000 (or no 4/5 win over default) → the XCS substrate's
              sample-efficiency ceiling at 16 bits is empirically bounded on this family:
              none of GA period, action-set subsumption, mutation rate or wildcard rate
              moves it. The sweep minimum is reported alongside so a lucky tuning-seed
              draw cannot be mistaken for the judged result.
Nothing in the grid is changed after the first run. The metric, the family, the budgets and
the recovery threshold (0.95 probe accuracy) are M2b's, unchanged.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expzero import DriftingHidden  # noqa: E402
from organism_sim.benchmarks.m2b import FAMILIES, _probe  # noqa: E402
from organism_sim.lcs import Params  # noqa: E402

GRID = {"theta_ga": (25, 100, 500), "as_subsumption": (True, False),
        "mu": (0.01, 0.05, 0.10), "p_wild": (0.33, 0.50, 0.75)}
DEFAULT = {"theta_ga": 25, "as_subsumption": True, "mu": 0.04, "p_wild": 0.33}   # Params() today
TUNE_SEEDS = tuple(range(110, 115))
JUDGE_SEEDS = tuple(range(40, 45))
FAMILY = "16bit"
PASS_BELOW, BOUNDED_AT = 3000.0, 8000.0


def configs() -> List[Dict[str, Any]]:
    keys = list(GRID)
    out = [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]
    if DEFAULT not in out:
        out.append(dict(DEFAULT))
    return out


def config_id(c: Dict[str, Any]) -> str:
    if set(c) == set(GRID):                       # Substrate-Opt-1's ids, kept stable for its results
        return f"tga{c['theta_ga']}_as{'on' if c['as_subsumption'] else 'off'}_mu{c['mu']:g}_pw{c['p_wild']:g}"
    parts = []
    for k in sorted(c):
        v = c[k]
        parts.append(f"{k}{'on' if v is True else 'off' if v is False else v}")
    return "_".join(parts)


def make_params(c: Dict[str, Any], family: str = FAMILY) -> Params:
    return replace(Params(N=FAMILIES[family]["N"], p_explore=0.5), **c)


def run_config(c: Dict[str, Any], seed: int, family: str = FAMILY,
               recovered_at: float = 0.95) -> Dict[str, Any]:
    """One learner, one seed, the M2b drifting family, no injection. Returns per-shift
    recoveries (None = not recovered before the next shift) and the capped scores."""
    F = FAMILIES[family]
    k, width = F["k"], F["width"]
    trials, shift_every, probe_every = F["trials"], F["shift_every"], F["probe_every"]
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 10_000)
    inputs = ["".join(map(str, prng.integers(0, 2, width))) for _ in range(256)]
    env = DriftingHidden(seed, shift_every, k=k, width=width)
    agent = LCSAgent("S", input_len=width, actions=["(emit 0)", "(emit 1)"],
                     params=make_params(c, family), seed=seed)
    recoveries: List[Optional[int]] = []
    window_start: Optional[int] = None
    waiting = False
    for trial in range(1, trials + 1):
        if env.maybe_shift(trial, trials):
            if window_start is not None and waiting:
                recoveries.append(None)
            window_start, waiting = trial, True
        bits = "".join(map(str, rng.integers(0, 2, width)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(agent, inputs, env.truth)
            if waiting and acc >= recovered_at and trial > window_start:
                recoveries.append(trial - window_start)
                waiting = False
    if window_start is not None and waiting:
        recoveries.append(None)
    scores = [v if v is not None else shift_every for v in recoveries]
    return {"config": config_id(c), "seed": seed, "recoveries": recoveries, "scores": scores,
            "median": float(np.median(scores)), "macro_end": agent.engine.macro_size(),
            "ga": agent.engine.counters["ga"], "subsume": agent.engine.counters["subsume"],
            "audit_chain_valid": agent.organism.chain.verify()}


def _job(args: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    return run_config(*args)


def sweep(cfgs: Sequence[Dict[str, Any]], seeds: Sequence[int], workers: int,
          log=print) -> Dict[str, Any]:
    jobs = [(c, s) for c in cfgs for s in seeds]
    by_cfg: Dict[str, List[Dict[str, Any]]] = {config_id(c): [] for c in cfgs}
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            by_cfg[r["config"]].append(r)
            if i % 10 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} runs, {time.time() - t0:.0f}s")
    table = []
    for c in cfgs:
        rows = sorted(by_cfg[config_id(c)], key=lambda r: r["seed"])
        table.append({"config": config_id(c), "params": c,
                      "per_seed_median": [r["median"] for r in rows],
                      "pooled_median": float(np.median([r["median"] for r in rows])),
                      "pooled_per_shift_median": float(np.median([x for r in rows for x in r["scores"]])),
                      "rows": rows})
    table.sort(key=lambda t: (t["pooled_median"], t["pooled_per_shift_median"]))
    return {"seeds": list(seeds), "table": table}


def judge(winner: Dict[str, Any], seeds: Sequence[int], workers: int, log=print,
          default: Dict[str, Any] = DEFAULT, pass_below: float = PASS_BELOW,
          bounded_at: float = BOUNDED_AT, partial_needs_wins: int = 4,
          pass_needs_wins: int = 0) -> Dict[str, Any]:
    res = sweep([winner, default], seeds, workers, log)
    by = {t["config"]: t for t in res["table"]}
    w, d = by[config_id(winner)], by[config_id(default)]
    wins = sum(a < b for a, b in zip(w["per_seed_median"], d["per_seed_median"]))
    wm = w["pooled_median"]
    if wm < pass_below and wins >= pass_needs_wins:
        verdict = "PASS"
    elif wm < bounded_at and wins >= partial_needs_wins:
        verdict = "PARTIAL"
    else:
        verdict = "BOUNDED"
    return {"seeds": list(seeds), "winner": w, "default": d, "winner_beats_default": f"{wins}/{len(seeds)}",
            "verdict": verdict, "criteria": {"pass_below": pass_below, "bounded_at": bounded_at,
                                              "partial_needs_wins": partial_needs_wins,
                                              "pass_needs_wins": pass_needs_wins}}


def main(argv: Sequence[str] = ()) -> int:
    workers = int(argv[0]) if argv else max(1, mp.cpu_count() - 1)
    results = ROOT / "results"
    logf = open(results / "substrate_opt1_run.log", "a")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    cfgs = configs()
    log(f"START Substrate-Opt-1 family={FAMILY} configs={len(cfgs)} tune_seeds={list(TUNE_SEEDS)} "
        f"judge_seeds={list(JUDGE_SEEDS)} workers={workers}")
    sw = sweep(cfgs, TUNE_SEEDS, workers, log)
    (results / "substrate_opt1_sweep_seeds110-114.json").write_text(json.dumps(sw, indent=1))
    for t in sw["table"][:10]:
        log(f"SWEEP {t['config']:32s} pooled_median={t['pooled_median']:.0f} per_seed={t['per_seed_median']}")
    dflt = next(t for t in sw["table"] if t["config"] == config_id(DEFAULT))
    log(f"SWEEP default {dflt['config']} pooled_median={dflt['pooled_median']:.0f} rank={sw['table'].index(dflt) + 1}")
    winner = sw["table"][0]["params"]
    log(f"WINNER {config_id(winner)} (sweep min {sw['table'][0]['pooled_median']:.0f}); judging on {list(JUDGE_SEEDS)}")
    jd = judge(winner, JUDGE_SEEDS, workers, log)
    (results / "substrate_opt1_eval_seeds40-44.json").write_text(json.dumps(jd, indent=1))
    log(f"JUDGE winner {jd['winner']['config']} pooled_median={jd['winner']['pooled_median']:.0f} "
        f"per_seed={jd['winner']['per_seed_median']} | default pooled_median={jd['default']['pooled_median']:.0f} "
        f"per_seed={jd['default']['per_seed_median']} | winner<default {jd['winner_beats_default']}")
    log(f"VERDICT {jd['verdict']}")
    log(f"DONE {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
