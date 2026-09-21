"""Substrate-Opt-1: grid, seed disjointness, parameter mapping, and a cheap deterministic run."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import substrate_opt as so  # noqa: E402

from organism_sim.lcs import Params, RuleEngine  # noqa: E402


def test_grid_has_54_points_plus_default_and_fresh_disjoint_seeds():
    cfgs = so.configs()
    assert len(cfgs) == 55 and so.DEFAULT in cfgs
    assert len({so.config_id(c) for c in cfgs}) == 55
    assert not set(so.TUNE_SEEDS) & set(so.JUDGE_SEEDS)
    used = set(range(0, 35)) | set(range(100, 110))
    assert not set(so.JUDGE_SEEDS) & used and not set(so.TUNE_SEEDS) & used


def test_default_config_is_the_engine_default():
    p = Params()
    assert (p.theta_ga, p.as_subsumption, p.mu, p.p_wild) == (25, True, 0.04, 0.33)
    mp_ = so.make_params(so.DEFAULT)
    assert (mp_.theta_ga, mp_.as_subsumption, mp_.mu, mp_.p_wild) == (25, True, 0.04, 0.33)
    assert mp_.N == so.FAMILIES["16bit"]["N"]


def test_as_subsumption_switch_changes_behaviour_only_when_off():
    def run(flag):
        eng = RuleEngine(6, ["(emit 0)", "(emit 1)"], Params(as_subsumption=flag), seed=0)
        import numpy as np
        rng = np.random.default_rng(0)
        for _ in range(3000):
            bits = "".join(map(str, rng.integers(0, 2, 6)))
            eng.step(bits)
            eng.reward(1.0 if bits[0] == "1" else 0.0, bits, terminal=True)
        return eng.counters["subsume"], eng.macro_size()
    on, off = run(True), run(False)
    assert on[0] > 0
    assert on != off


def test_run_config_is_deterministic_and_reports_every_shift():
    c = {"theta_ga": 25, "as_subsumption": True, "mu": 0.05, "p_wild": 0.5}
    a = so.run_config(c, seed=0, family="6bit")
    b = so.run_config(c, seed=0, family="6bit")
    assert a == b
    assert len(a["scores"]) == 4 and a["audit_chain_valid"] and a["config"] == "tga25_ason_mu0.05_pw0.5"
