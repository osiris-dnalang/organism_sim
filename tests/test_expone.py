"""Experiment One harness: own-error CUSUM with injection handler, arming, arms."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expone import ARMS, compare, run_arm  # noqa: E402


def test_own_error_cusum_calls_handler_and_skips_explore_trials():
    fired = []
    a = LCSAgent("O", 6, ["(emit 0)", "(emit 1)"], seed=0, cusum=(0.05, 0.1, 0.5),
                 cusum_signal="error", cusum_on=lambda ag: fired.append(ag.organism.tick))
    # explore trial: no update
    a.act("000000", explore=True)
    a.reward(0.0)
    a.tick()
    assert a.cusum_s == 0.0 and not fired
    # an exploit miss adds 1 − μ₀ − k = 0.85 > h: the handler fires (not a reroute) and S resets
    a.act("000000", explore=False)
    a.reward(0.0)
    a.tick()
    assert len(fired) == 1 and a.cusum_fires == 1 and a.cusum_s == 0.0
    assert a.routes == set() and a.reroutes == []


def test_arms_run_and_record():
    for arm in [a for a in ARMS if a != "grn"]:          # grn needs a genome; covered in test_grn
        lg = run_arm(arm, seed=0, trials=4000, shift_every=2000, probe_every=100,
                     cusum=(0.05, 0.1, 4.0), period=500)
        assert lg.shifts == [2000] and len(lg.recoveries95) == 1 and len(lg.recoveries99) == 1
        assert lg.final["audit_chain_valid"]
        if arm == "plain":
            assert lg.injections == []
        if arm == "oracle":
            assert lg.injections == [2000] and lg.false_injections == 0
        if arm == "periodic":
            assert len(lg.injections) == 8


def test_cusum_arms_only_after_convergence():
    lg = run_arm("cusum", seed=1, trials=3000, shift_every=1500, probe_every=100)
    assert lg.final["armed_at"] is not None
    assert all(t >= lg.final["armed_at"] for t in lg.injections)


def test_compare_fields():
    res = compare(seeds=[0], trials=4000, shift_every=2000, arms=("plain", "cusum"))
    assert {"C1", "C2", "false_per_10k", "pass"} <= set(res["verdict"])
    assert set(res["pooled"]) == {"plain", "cusum"}
