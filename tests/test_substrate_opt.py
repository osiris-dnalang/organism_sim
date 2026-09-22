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


# ── Substrate-Opt-2: tournament selection, specify, N ────────────────────────

def test_opt2_grid_default_and_fresh_seeds():
    import substrate_opt2 as so2
    cfgs = so2.configs()
    assert len(cfgs) == 12 and so2.DEFAULT in cfgs
    assert len({so.config_id(c) for c in cfgs}) == 12
    used = set(range(0, 45)) | set(range(100, 115))
    assert not set(so2.TUNE_SEEDS) & used and not set(so2.JUDGE_SEEDS) & used
    assert not set(so2.TUNE_SEEDS) & set(so2.JUDGE_SEEDS)
    p = Params()
    assert (p.selection, p.specify, p.N) == ("roulette", False, 400)         # engine defaults untouched
    assert so.make_params(so2.DEFAULT).N == 1000 and so.config_id(so2.DEFAULT) == "N1000_selectionroulette_specifyoff"


def test_tournament_and_specify_learn_the_6mux_and_are_deterministic():
    import numpy as np

    def truth(b):
        b = [int(c) for c in b]
        return str(b[2 + 2 * b[0] + b[1]])

    def run(params, seed=0, trials=3000):
        eng = RuleEngine(6, ["(emit 0)", "(emit 1)"], params, seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(trials):
            bits = "".join(map(str, rng.integers(0, 2, 6)))
            eng.step(bits)
            eng.reward(1.0 if (eng.last_ctx.emits and eng.last_ctx.emits[0] == truth(bits)) else 0.0,
                       bits, terminal=True)
        ok = 0
        probe = np.random.default_rng(7)
        for _ in range(200):
            bits = "".join(map(str, probe.integers(0, 2, 6)))
            eng.step(bits, explore=False)
            ok += bool(eng.last_ctx.emits) and eng.last_ctx.emits[0] == truth(bits)
        return ok / 200, dict(eng.counters), [r.condition for r in eng.rules]

    both = Params(N=400, selection="tournament", specify=True)
    acc, counters, rules = run(both)
    assert acc >= 0.9 and counters["specify"] > 0 and counters["ga"] > 0
    assert run(both) == (acc, counters, rules)                       # seeded, deterministic
    assert all(r.numerosity > 0 for r in RuleEngine(6, ["(emit 0)"], both).rules)
    acc_t, c_t, _ = run(Params(N=400, selection="tournament"))
    assert acc_t >= 0.9 and c_t["specify"] == 0


def test_specify_tolerates_an_action_set_of_deleted_rules():
    from organism_sim.lcs import Rule
    eng = RuleEngine(6, ["(emit 0)", "(emit 1)"], Params(N=4, specify=True, selection="tournament"), seed=0)
    eng.step("010101")
    for r in eng.action_set:
        r.numerosity = 0                                   # as if deleted after matching
    eng.reward(0.0, "010101", terminal=True)               # must not raise
    eng.action_set = [Rule(condition="######", action="(emit 0)", prediction=0.5, error=0.5, fitness=0.1, numerosity=0)]
    eng.reward(0.0, "010101", terminal=True)


def test_params16_is_the_judged_opt2_winner():
    import json

    from organism_sim.lcs import PARAMS16
    d = json.loads((ROOT / "results" / "substrate_opt2_eval_seeds50-54.json").read_text())
    assert d["verdict"] == "PASS"
    w = d["winner"]["params"]
    assert (PARAMS16.N, PARAMS16.selection, PARAMS16.specify) == (w["N"], w["selection"], w["specify"])
    assert d["winner"]["pooled_median"] < 7000 and d["winner_beats_default"] == "5/5"
