"""
experiments/m3_ablate.py — M3c: curriculum or inheritance?
==========================================================

M3b PASSED at 12–16 bits (ANNECS 35 vs 25, ratio 1.40, C1 4/5) but its ``poet`` arm bundles
two mechanisms, and the result does not say which carried it:

    curriculum    children are admitted only if some current agent scores in [0.6, 0.95)
                  on them before training — the environment is held at the edge of competence
    inheritance   an admitted child starts from a *copy* of its parent's agent, and every
                  ``transfer_every`` iterations a better agent is copied onto a task

This ablation separates them. Three arms, same harness, space (k = 3, widths 12–16,
``PARAMS16``), budgets and fresh seeds:

    poet         as M3b (curriculum + inheritance + transfer)
    poet-fresh   curriculum only: the minimal criterion is kept, but every admitted child
                 gets a fresh agent and no transfer happens  (clarification recorded before
                 the run: "curriculum only" must switch off *both* transfer paths, or the
                 ablation is partial)
    random       as M3b (uniform task stream, same creation rate and budget, fresh agents)

PRE-REGISTERED CRITERIA (committed before the first run; fresh seeds 70–74; nothing tuned):
    A  poet-fresh > random on ANNECS on ≥ 4/5 seeds AND pooled ratio ≥ 1.25
       → the minimal criterion alone produces the effect
    B  poet > poet-fresh on ANNECS on ≥ 4/5 seeds
       → inheritance adds to it

    A ∧ B    both mechanisms contribute
    A ∧ ¬B   the curriculum is the mechanism; inheritance is not needed
    ¬A ∧ B   inheritance is the mechanism — M3b's PASS was competence transfer, not task
             generation, and the program's claim about open-ended generation does not hold
    ¬A ∧ ¬B  M3b does not replicate on these seeds

All four outcomes are recorded as the result; there is no "PASS" for this run, only which
of the four it is. Exploratory, reported not judged: the final-distribution score of each
arm's agents on the poet arm's final tasks, **width-matched** (M3b's seed-61 confound: an
agent of the wrong width scores 0 for a reason that is not competence), ANNECS still rising,
max width, transfers.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks.m3 import Space, final_distribution_score, run_arm  # noqa: E402
from organism_sim.lcs import PARAMS16  # noqa: E402

ARMS = ("poet", "poet-fresh", "random")
SEEDS = tuple(range(70, 75))
SPACE = Space(k=3, widths=(12, 16), params=PARAMS16, audit_window=2000)
KW: Dict[str, Any] = {"iterations": 30, "train_trials": 3000, "n_pairs": 4, "gen_every": 3,
                      "children_per_task": 2, "max_tasks": 8, "transfer_every": 5,
                      "mc_lo": 0.6, "mc_hi": 0.95, "solved_at": 0.95}
RESULTS = ROOT / "results"
RATIO_MIN = 1.25
WINS_NEEDED = 4


def _checkpoint(seed: int) -> Path:
    return RESULTS / f"m3c_seed{seed}.json"


def _job(seed: int) -> Dict[str, Any]:
    """All three arms on one seed; checkpointed so a restart resumes rather than repeats."""
    cp = _checkpoint(seed)
    if cp.exists():
        return json.loads(cp.read_text())
    t0 = time.time()
    logs = {}
    for arm in ARMS:
        def progress(row: Dict[str, Any], arm: str = arm) -> None:
            print(f"  seed {seed} {arm} it {row['iter']}/{KW['iterations']} tasks {row['n_tasks']} "
                  f"archive {row['archive']} annecs {row['annecs']} acc {row['mean_acc']:.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        logs[arm] = run_arm(arm, seed, progress=progress, space=SPACE, **KW)
    row: Dict[str, Any] = {"seed": seed, **{a: logs[a].final for a in ARMS}}
    # exploratory: every arm's agents on the poet arm's final task population, width-matched
    frng = np.random.default_rng(seed + 2)
    ptasks, _ = logs["poet"].live
    row["final_distribution"] = {
        a: final_distribution_score(ptasks, logs[a].live[1], frng, width_matched=True)
        for a in ARMS}
    row["wall_s"] = round(time.time() - t0)
    cp.write_text(json.dumps(row))
    return row


def outcome(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    def wins(a: str, b: str) -> int:
        return sum(r[a]["annecs"] > r[b]["annecs"] for r in rows)

    pooled = {a: float(np.median([r[a]["annecs"] for r in rows])) for a in ARMS}
    ratio_fresh = pooled["poet-fresh"] / pooled["random"] if pooled["random"] else float("inf")
    ratio_poet = pooled["poet"] / pooled["random"] if pooled["random"] else float("inf")
    need = WINS_NEEDED if n >= 5 else n
    a_wins, b_wins = wins("poet-fresh", "random"), wins("poet", "poet-fresh")
    A = bool(a_wins >= need and ratio_fresh >= RATIO_MIN)
    B = bool(b_wins >= need)
    label = {(True, True): "A and B: curriculum and inheritance both contribute",
             (True, False): "A only: the curriculum is the mechanism; inheritance is not needed",
             (False, True): "B only: inheritance is the mechanism — M3b was competence transfer,"
                            " not task generation",
             (False, False): "neither: M3b does not replicate on these seeds"}[(A, B)]
    return {"pooled_annecs": pooled, "ratio_poet_fresh_vs_random": ratio_fresh,
            "ratio_poet_vs_random": ratio_poet,
            "A_poet_fresh_beats_random": f"{a_wins}/{n}", "B_poet_beats_poet_fresh": f"{b_wins}/{n}",
            "A": A, "B": B, "outcome": label,
            "criteria": {"wins_needed": need, "ratio_min": RATIO_MIN},
            "exploratory_still_rising": {a: sum(r[a]["annecs_curve"][-1] > r[a]["annecs_curve"][-6]
                                                for r in rows) for a in ARMS},
            "exploratory_max_width": {a: [r[a]["max_width"] for r in rows] for a in ARMS},
            "exploratory_transfers": [r["poet"]["transfers"] for r in rows],
            "exploratory_final_distribution_width_matched": {
                a: [round(r["final_distribution"][a]["mean"], 4) for r in rows] for a in ARMS},
            "exploratory_final_distribution_n_scored": {
                a: [r["final_distribution"][a]["n_scored"] for r in rows] for a in ARMS}}


def main(argv: Sequence[str] = ()) -> int:
    workers = int(argv[0]) if argv else min(len(SEEDS), max(1, mp.cpu_count() - 1))
    logf = open(RESULTS / "m3c_run.log", "a")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    log(f"START M3c arms={list(ARMS)} seeds={list(SEEDS)} space=k{SPACE.k}_w{SPACE.widths[0]}-{SPACE.widths[1]} "
        f"learner=PARAMS16(N={PARAMS16.N},{PARAMS16.selection},specify={PARAMS16.specify}) kw={KW} workers={workers}")
    rows: List[Dict[str, Any]] = []
    with mp.Pool(workers) as pool:
        for row in pool.imap_unordered(_job, SEEDS):
            rows.append(row)
            fd = row["final_distribution"]
            log("seed {} ".format(row["seed"]) + " | ".join(
                f"{a} annecs {row[a]['annecs']}/{row[a]['archive']} w {row[a]['max_width']} "
                f"fd {fd[a]['mean']:.3f} (n {fd[a]['n_scored']}/{fd[a]['n_tasks']})" for a in ARMS)
                + f" | {row['wall_s']}s")
    rows.sort(key=lambda r: r["seed"])
    out = {"experiment": "m3c", "seeds": list(SEEDS), "arms": list(ARMS),
           "params": dict(KW, space={"k": SPACE.k, "widths": list(SPACE.widths),
                                     "params": PARAMS16.to_dict()}),
           "rows": rows, "verdict": outcome(rows)}
    (RESULTS / "m3c_eval_seeds70-74.json").write_text(json.dumps(out, indent=1))
    for sd in SEEDS:
        _checkpoint(sd).unlink(missing_ok=True)
    v = out["verdict"]
    log(f"POOLED {v['pooled_annecs']} ratio(fresh/random) {v['ratio_poet_fresh_vs_random']:.3f} "
        f"A {v['A_poet_fresh_beats_random']} B {v['B_poet_beats_poet_fresh']}")
    log(f"OUTCOME A={v['A']} B={v['B']} — {v['outcome']}")
    log(f"DONE {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
