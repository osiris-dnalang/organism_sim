"""
organism_sim.benchmarks.ci_guard — regression guards run on every push
=====================================================================

    python -m organism_sim.benchmarks.ci_guard [--latency-us 600] [--relay-median 150]
                                                [--seeds 5]

1. Latency budget: steady-state µs per full LCS trial (match + select + act +
   credit + GA + organism tick + hash-chained telemetry row) at a 400-rule
   population must be ≤ --latency-us. Measured 167 µs on the dev machine after
   the condition-matrix cache; CI runners are slower, so the default budget is
   3.5× that. A change that pushes the loop past the budget fails the build.
2. Dead-relay recovery: the one pre-registered PASS. The organism arm must
   recover (≥ 0.95) on every seed with median recovery ≤ --relay-median trials
   and zero false reroutes, on --seeds seeds of the evaluation set.
Exit code 1 on any violation; prints a JSON line per guard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from ..agent import LCSAgent
from ..spec import Triggers
from .mux import mux6
from .relay import run_relay


def latency_us(warm: int = 1500, measure: int = 3000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = LCSAgent("A", 6, ["(emit 0)", "(emit 1)"], seed=seed)

    def trial():
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        a.act(bits)
        a.reward(1.0 if a.emitted() == mux6(bits) else 0.0)
        a.tick()
    for _ in range(warm):
        trial()
    t = time.perf_counter()
    for _ in range(measure):
        trial()
    return (time.perf_counter() - t) / measure * 1e6


def dead_relay(seeds: int) -> dict:
    tr = Triggers(noise_floor=0.05, repair_threshold=0.25, unrepaired_integral=10.0, window=100)
    recs, false, recovered = [], 0, 0
    for sd in range(seeds):
        f = run_relay("organism", 6000, 3000, "dead", sd, triggers=tr).final
        recs.append(f["recovery"] if f["recovery"] is not None else 3000)
        false += f["false_reroutes"]
        recovered += f["recovered"]
    return {"median_recovery": float(np.median(recs)), "recovered": recovered, "seeds": seeds,
            "false_reroutes": false}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-us", type=float, default=600.0)
    ap.add_argument("--relay-median", type=float, default=150.0)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    ok = True
    us = latency_us()
    print(json.dumps({"guard": "latency", "us_per_trial": round(us, 1), "budget_us": a.latency_us,
                      "pass": us <= a.latency_us}))
    ok &= us <= a.latency_us
    r = dead_relay(a.seeds)
    r_ok = (r["recovered"] == r["seeds"] and r["median_recovery"] <= a.relay_median
            and r["false_reroutes"] == 0)
    print(json.dumps({"guard": "dead_relay", **r, "budget_median": a.relay_median, "pass": r_ok}))
    ok &= r_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
