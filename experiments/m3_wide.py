"""
experiments/m3_wide.py — M3b: open-ended task generation at 12–16 bits with the
Substrate-Opt-2 learner
=========================================================================================

M3 failed at 6–10 bits because nothing there was hard: every task fell in a few
iterations, so a random task stream out-generated the POET loop and the minimal criterion
was overhead. M3 was then blocked at width because the learner needed ~10k trials per
instance. Substrate-Opt-2 lifted that to ≈ 6,000 with ``lcs.PARAMS16``; this is the rung it
unblocked, run exactly once, pre-registered.

Harness: ``benchmarks.m3`` unchanged in mechanism (agent–task pairs, minimal criterion,
capped task population, transfer; random stream control at the same creation rate and
budget), generalised to a ``Space``: **k = 3 address bits, widths 12–16, PARAMS16**
(N 4000, tournament selection, specify). Budget per arm and seed: 30 iterations ×
3,000 training trials per task (M3 used 2,000 at N 400); 4 initial pairs, at most 8 tasks.
Every other harness parameter is M3's.

PRE-REGISTERED CRITERION (committed before the first run; fresh seeds 60–64; nothing tuned):
    C1  ANNECS(poet) > ANNECS(random) on ≥ 4/5 seeds                         (as M3)
    C2  pooled ANNECS ratio poet / random ≥ 1.25                              (as M3)
    C3  on the poet arm's *final task population* — the self-generated distribution —
        the best accuracy reachable by the poet arm's agents exceeds that reachable by
        the random arm's agents on ≥ 4/5 seeds  (competence over the self-generated
        distribution beats a fixed-curriculum learner of equal budget on that same
        distribution)
    PASS ⇔ C1 ∧ C2 ∧ C3. Reported, not judged: whether the ANNECS curve is still rising
    over the last 5 iterations (continuous growth), max width reached, transfers.
    If PASS: open-ended task generation does work at a scale where tasks are hard, and
    the program's central claim has its first positive wide-space result; the next rung
    is M4/M7 on this learner. If FAIL: with the learner no longer the excuse, the POET
    loop itself is what does not deliver on this family, and M3 closes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks.m3 import Space, compare_row, verdict  # noqa: E402
from organism_sim.lcs import PARAMS16  # noqa: E402

SEEDS = tuple(range(60, 65))
SPACE = Space(k=3, widths=(12, 16), params=PARAMS16, audit_window=2000)   # memory: 16 live agents/worker
KW: Dict[str, Any] = {"iterations": 30, "train_trials": 3000, "n_pairs": 4, "gen_every": 3,
                      "children_per_task": 2, "max_tasks": 8, "transfer_every": 5,
                      "mc_lo": 0.6, "mc_hi": 0.95, "solved_at": 0.95}


def _job(seed: int) -> Dict[str, Any]:
    t0 = time.time()
    row = compare_row(seed, final_distribution=True, space=SPACE, **KW)
    row["wall_s"] = round(time.time() - t0)
    return row


def main(argv: Sequence[str] = ()) -> int:
    workers = int(argv[0]) if argv else min(len(SEEDS), max(1, mp.cpu_count() - 1))
    results = ROOT / "results"
    logf = open(results / "m3b_run.log", "a")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    log(f"START M3b seeds={list(SEEDS)} space=k{SPACE.k}_w{SPACE.widths[0]}-{SPACE.widths[1]} "
        f"learner=PARAMS16(N={PARAMS16.N},{PARAMS16.selection},specify={PARAMS16.specify}) kw={KW} workers={workers}")
    rows = []
    with mp.Pool(workers) as pool:
        for row in pool.imap_unordered(_job, SEEDS):
            rows.append(row)
            fd = row["final_distribution"]
            log(f"seed {row['seed']} poet annecs {row['poet']['annecs']} w {row['poet']['max_width']} "
                f"tr {row['poet']['transfers']} | random annecs {row['random']['annecs']} w {row['random']['max_width']} "
                f"| final-dist poet {fd['poet_agents']:.3f} random {fd['random_agents']:.3f} n {fd['n_tasks']} "
                f"| {row['wall_s']}s")
    rows.sort(key=lambda r: r["seed"])
    params = dict(KW, space={"k": SPACE.k, "widths": list(SPACE.widths), "params": PARAMS16.to_dict()})
    out = verdict(rows, SEEDS, final_distribution=True, params=params)
    (results / "m3b_eval_seeds60-64.json").write_text(json.dumps(out, indent=1))
    v = out["verdict"]
    log(f"POOLED annecs {out['pooled_annecs']} ratio {out['ratio']:.3f} C1 {v['C1_count']}/5 "
        f"C3 {v['C3_count']}/5 still_rising {v['exploratory_still_rising']}/5")
    log(f"VERDICT {'PASS' if v['pass'] else 'FAIL'} {json.dumps({k: v[k] for k in ('C1', 'C2', 'C3')})}")
    log(f"DONE {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
