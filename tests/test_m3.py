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


# ── Space generalisation (M3b) ───────────────────────────────────────────────

def test_default_space_is_m3_and_k3_tasks_are_well_formed():
    import numpy as np

    from organism_sim.benchmarks.m3 import DEFAULT_SPACE, Space, Task
    from organism_sim.lcs import PARAMS16
    assert (DEFAULT_SPACE.k, DEFAULT_SPACE.widths, DEFAULT_SPACE.params.N) == (2, (6, 10), 400)
    sp = Space(k=3, widths=(12, 16), params=PARAMS16)
    rng = np.random.default_rng(3)
    for _ in range(50):
        t = Task.sample(rng, 0, space=sp)
        assert 12 <= t.width <= 16 and len(t.addr) == 3 and len(t.data) == 8
        used = set(t.addr) | set(t.data) | ({t.twist} if t.twist is not None else set())
        assert len(used) == 11 + (t.twist is not None)
        bits = "".join(map(str, rng.integers(0, 2, t.width)))
        b = [int(c) for c in bits]
        idx = 4 * b[t.addr[0]] + 2 * b[t.addr[1]] + b[t.addr[2]]
        expect = b[t.data[idx]] ^ (b[t.twist] if t.twist is not None else 0)
        assert t.truth(bits) == str(1 - expect if t.invert else expect)
        c = t.mutate(rng, 1, 1, sp)
        assert 12 <= c.width <= 16 and len(c.addr) == 3 and c.parent == 0
    import pytest
    with pytest.raises(ValueError):
        Task.sample(rng, 0, width=10, space=sp)


def test_m3b_driver_is_pre_registered_on_fresh_seeds():
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import m3_wide
    assert set(m3_wide.SEEDS) == set(range(60, 65))
    assert not set(m3_wide.SEEDS) & (set(range(0, 55)) | set(range(100, 125)))
    assert m3_wide.SPACE.k == 3 and m3_wide.SPACE.widths == (12, 16) and m3_wide.SPACE.params.N == 4000
    assert m3_wide.KW["train_trials"] == 3000 and m3_wide.KW["iterations"] == 30


# ── M3c ablation ─────────────────────────────────────────────────────────────

def test_poet_fresh_keeps_the_curriculum_and_drops_every_transfer_path():
    """poet-fresh must admit children by the same minimal criterion but never copy an agent."""
    from organism_sim.benchmarks.m3 import ALL_ARMS, Space, run_arm
    from organism_sim.lcs import Params
    assert ALL_ARMS == ("poet", "poet-fresh", "random")
    sp = Space(k=2, widths=(6, 8), params=Params(N=200, p_explore=0.5), audit_window=500)
    kw = dict(iterations=9, train_trials=300, n_pairs=3, gen_every=3, children_per_task=2,
              max_tasks=6, transfer_every=3, space=sp)
    poet = run_arm("poet", 5, **kw)
    fresh = run_arm("poet-fresh", 5, **kw)
    rand = run_arm("random", 5, **kw)
    assert fresh.final["transfers"] == 0                      # no transfer at all
    assert poet.final["transfers"] >= 0
    # the curriculum is intact: both poet arms create by mutation under the minimal
    # criterion, so they create far fewer tasks than the uniform stream
    assert fresh.final["archive"] <= rand.final["archive"]
    assert all(t.get("parent") is not None or t["created_iter"] == 0 for t in fresh.tasks)
    assert run_arm("poet-fresh", 5, **kw).final == fresh.final   # deterministic


def test_width_matched_final_distribution_reports_coverage():
    import numpy as np

    from organism_sim.benchmarks.m3 import Space, final_distribution_score, run_arm
    from organism_sim.lcs import Params
    sp = Space(k=2, widths=(6, 8), params=Params(N=200), audit_window=500)
    lg = run_arm("poet", 5, iterations=3, train_trials=200, n_pairs=3, gen_every=3, space=sp)
    tasks, agents = lg.live
    rng = np.random.default_rng(0)
    allt = final_distribution_score(tasks, agents, rng)
    matched = final_distribution_score(tasks, agents, rng, width_matched=True)
    assert allt["n_scored"] == len(tasks) and matched["n_scored"] <= len(tasks)
    assert set(matched["widths"]) == {a.input_len for a in agents.values()}
    empty = final_distribution_score(tasks, {}, rng, width_matched=True)
    assert empty["mean"] == 0.0 and empty["n_scored"] == 0      # no agents → nothing scored


def test_m3c_driver_is_pre_registered_on_fresh_seeds():
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import m3_ablate
    assert set(m3_ablate.SEEDS) == set(range(70, 75))
    assert not set(m3_ablate.SEEDS) & (set(range(0, 65)) | set(range(100, 125)))
    assert m3_ablate.ARMS == ("poet", "poet-fresh", "random")
    assert m3_ablate.KW == {"iterations": 30, "train_trials": 3000, "n_pairs": 4, "gen_every": 3,
                            "children_per_task": 2, "max_tasks": 8, "transfer_every": 5,
                            "mc_lo": 0.6, "mc_hi": 0.95, "solved_at": 0.95}   # M3b's, unchanged
    assert m3_ablate.SPACE.k == 3 and m3_ablate.SPACE.params.N == 4000

    def row(seed, p, f, r):
        curve = [0] * 5
        return {"seed": seed,
                **{a: {"annecs": v, "archive": 50, "max_width": 16, "transfers": 0,
                       "annecs_curve": curve + [v]}
                   for a, v in (("poet", p), ("poet-fresh", f), ("random", r))},
                "final_distribution": {a: {"mean": 0.5, "n_scored": 5, "n_tasks": 8} for a in m3_ablate.ARMS}}
    both = m3_ablate.outcome([row(i, 40, 35, 20) for i in range(5)])
    assert both["A"] and both["B"] and both["outcome"].startswith("A and B")
    curr = m3_ablate.outcome([row(i, 35, 35, 20) for i in range(5)])
    assert curr["A"] and not curr["B"] and "curriculum is the mechanism" in curr["outcome"]
    inh = m3_ablate.outcome([row(i, 40, 20, 20) for i in range(5)])
    assert not inh["A"] and inh["B"] and "inheritance is the mechanism" in inh["outcome"]
    none = m3_ablate.outcome([row(i, 20, 20, 25) for i in range(5)])
    assert not none["A"] and not none["B"] and none["outcome"].startswith("neither")
