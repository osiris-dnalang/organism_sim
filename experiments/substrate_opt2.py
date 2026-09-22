"""
experiments/substrate_opt2.py — Substrate-Opt-2: the XCS mechanisms the literature says
scale the multiplexer, tested where Substrate-Opt-1's knobs did not
=====================================================================================

Substrate-Opt-1 bounded 16-bit drift recovery at ≈ 10,000 trials/shift over GA period,
action-set subsumption, mutation and wildcard rate — the default was at the floor of the
grid. Three mechanisms were *not* in that grid, and each has prior art naming it as what
lets XCS scale past the 11-multiplexer:

    selection   tournament selection in the GA (Butz, Sastry & Goldberg 2003: XCS with
                roulette fails the 20-mux robustly; with tournament selection it solves it)
    specify     Lanzi's specify operator (1997): when the action set's error is far above
                the population's, specialise a rule toward the current input
    N           population size (Wilson 1995: 11-mux N=400/800, 20-mux N=2000; this
                engine runs the 16-bit family at N=1000)

All three are now switches on ``Params`` (defaults off / N unchanged — every earlier result
is untouched). Grid: selection ∈ {roulette, tournament} × specify ∈ {off, on} ×
N ∈ {1000, 2000, 4000} = 12 configurations, the default among them. Same family, budgets,
metric and recovery threshold as M2b / M2c / Substrate-Opt-1; no injection.

Protocol as Substrate-Opt-1: sweep on tuning seeds 120–124; the single best configuration
and the default judged on fresh seeds 50–54.

PRE-REGISTERED DECISION (committed before the first run; judge seeds never used before):
    metric   pooled median recovery of the winner on the judge seeds
    PASS     winner < 7,000 and winner < default on ≥ 4/5 judge seeds (≥ 30 % under the
             Opt-1 ceiling on fresh seeds; the 3,000 line of Opt-1 is reported, not required)
    PARTIAL  7,000 ≤ winner < 8,000 and ≥ 4/5
    BOUNDED  otherwise
    If PASS: the winning configuration becomes ``Params16`` — the pre-registered learner for
    any further 16-bit rung — and M3 at width is unblocked *for that learner*; the sweep
    minimum is reported next to the judged number. If BOUNDED: the literature's mechanisms
    do not move this family either, and the program's next step is a different learner.
    Cost is reported (N=4000 is ≈ 4.5× the wall time of N=1000); the criterion is trials.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for p_ in (ROOT, ROOT / "experiments"):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))

from substrate_opt import config_id, judge, sweep  # noqa: E402

GRID = {"selection": ("roulette", "tournament"), "specify": (False, True), "N": (1000, 2000, 4000)}
DEFAULT = {"selection": "roulette", "specify": False, "N": 1000}
TUNE_SEEDS = tuple(range(120, 125))
JUDGE_SEEDS = tuple(range(50, 55))
PASS_BELOW, BOUNDED_AT = 7000.0, 8000.0


def configs() -> List[Dict[str, Any]]:
    keys = list(GRID)
    out = [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]
    assert DEFAULT in out
    return out


def main(argv: Sequence[str] = ()) -> int:
    import multiprocessing as mp
    workers = int(argv[0]) if argv else max(1, mp.cpu_count() - 1)
    results = ROOT / "results"
    logf = open(results / "substrate_opt2_run.log", "a")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    cfgs = configs()
    log(f"START Substrate-Opt-2 configs={len(cfgs)} tune_seeds={list(TUNE_SEEDS)} judge_seeds={list(JUDGE_SEEDS)} workers={workers}")
    sw = sweep(cfgs, TUNE_SEEDS, workers, log)
    (results / "substrate_opt2_sweep_seeds120-124.json").write_text(json.dumps(sw, indent=1))
    for t in sw["table"]:
        log(f"SWEEP {t['config']:44s} pooled_median={t['pooled_median']:.0f} per_seed={t['per_seed_median']}")
    dflt = next(t for t in sw["table"] if t["config"] == config_id(DEFAULT))
    log(f"SWEEP default {dflt['config']} pooled_median={dflt['pooled_median']:.0f} rank={sw['table'].index(dflt) + 1}")
    winner = sw["table"][0]["params"]
    log(f"WINNER {config_id(winner)} (sweep min {sw['table'][0]['pooled_median']:.0f}); judging on {list(JUDGE_SEEDS)}")
    jd = judge(winner, JUDGE_SEEDS, workers, log, default=DEFAULT, pass_below=PASS_BELOW,
               bounded_at=BOUNDED_AT, partial_needs_wins=4, pass_needs_wins=4)
    jd["sweep_min"] = sw["table"][0]["pooled_median"]
    (results / "substrate_opt2_eval_seeds50-54.json").write_text(json.dumps(jd, indent=1))
    log(f"JUDGE winner {jd['winner']['config']} pooled_median={jd['winner']['pooled_median']:.0f} "
        f"per_seed={jd['winner']['per_seed_median']} | default pooled_median={jd['default']['pooled_median']:.0f} "
        f"per_seed={jd['default']['per_seed_median']} | winner<default {jd['winner_beats_default']}")
    log(f"VERDICT {jd['verdict']}")
    log(f"DONE {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
