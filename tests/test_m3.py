"""M3 harness: task genome, mutation, minimal criterion, ANNECS bookkeeping, both arms."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks.m3 import ARMS, WIDTHS, Task, evaluate, run_arm, train  # noqa: E402


def test_task_sample_and_truth():
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = Task.sample(rng, 0)
        assert WIDTHS[0] <= t.width <= WIDTHS[1]
        assert len(set(t.addr + t.data)) == 6 and all(0 <= p < t.width for p in t.addr + t.data)
        if t.twist is not None:
            assert t.twist not in t.addr + t.data
        for _ in range(20):
            b = [int(x) for x in rng.integers(0, 2, t.width)]
            out = b[t.data[2 * b[t.addr[0]] + b[t.addr[1]]]]
            if t.twist is not None:
                out ^= b[t.twist]
            assert t.truth("".join(map(str, b))) == str(1 - out if t.invert else out)


def test_mutation_keeps_tasks_valid_and_links_parent():
    rng = np.random.default_rng(1)
    t = Task.sample(rng, 0)
    for i in range(1, 200):
        c = t.mutate(rng, i, it=3)
        assert c.parent == t.tid and c.created_iter == 3
        assert WIDTHS[0] <= c.width <= WIDTHS[1]
        assert len(set(c.addr + c.data)) == 6 and all(0 <= p < c.width for p in c.addr + c.data)
        if c.twist is not None:
            assert c.twist not in c.addr + c.data
        t = c


def test_agent_learns_a_task_and_evaluate_reflects_it():
    rng = np.random.default_rng(2)
    t = Task(0, 6, [0, 1], [2, 3, 4, 5], False, None)
    from organism_sim.benchmarks.m3 import _agent
    a = _agent(6, 0)
    before = evaluate(a, t, np.random.default_rng(9))
    train(a, t, rng, 3000)
    after = evaluate(a, t, np.random.default_rng(9))
    assert after > 0.9 and after > before
    assert evaluate(a, Task(1, 7, [0, 1], [2, 3, 4, 5], False, None), rng) == 0.0   # width mismatch


def test_arms_run_and_annecs_monotone():
    for arm in ARMS:
        lg = run_arm(arm, seed=0, iterations=6, train_trials=500, gen_every=2, transfer_every=3)
        curve = lg.final["annecs_curve"]
        assert len(curve) == 6 and all(b >= a for a, b in zip(curve, curve[1:]))
        assert lg.final["archive"] >= 4 and lg.final["audit_chains_valid"]
        assert all(set(r) >= {"iter", "n_tasks", "annecs", "max_width", "mean_acc"} for r in lg.rows)
        if arm == "random":
            assert lg.final["transfers"] == 0
