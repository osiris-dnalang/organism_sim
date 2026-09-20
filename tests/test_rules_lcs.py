"""DSL interpreter, ternary conditions, and the rule engine (XCS + reinforce modes)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.lcs import Params, Rule, RuleEngine  # noqa: E402
from organism_sim.rules import (  # noqa: E402
    BudgetExceeded,
    DSLError,
    matches,
    parse_sexpr,
    run_action,
    specificity,
    subsumes,
    to_sexpr,
    validate,
)

# ── conditions ───────────────────────────────────────────────────────────────

def test_ternary_matching():
    assert matches("1#0#", "1101") and matches("####", "0000")
    assert not matches("1#0#", "1111") and not matches("1#0", "1101")
    assert subsumes("1###", "1#0#") and not subsumes("1#0#", "1###")
    assert not subsumes("1#0#", "1#0#")
    assert specificity("1#0#") == 2


# ── DSL ──────────────────────────────────────────────────────────────────────

def test_sexpr_roundtrip_and_validation():
    src = "(seq (set x (add (reg 0) (reg 1))) (if (eq (var x) 2) (emit 1) (emit 0)))"
    node = parse_sexpr(src)
    assert to_sexpr(node) == src
    validate(node)
    with pytest.raises(DSLError):
        validate(parse_sexpr("(launch missiles)"))
    with pytest.raises(DSLError):
        parse_sexpr("(emit 1")


def test_all_ops_execute():
    ctx = run_action(
        "(seq (emit a) (set v 3) (send B s1) (sever C) (route D) (adjust temp 0.5) "
        "(if (eq (var v) 3) (emit yes) (emit no)) (if (not (last)) (emit nolast)))",
        register="0101", last="")
    assert ctx.emits == ["a", "yes", "nolast"]
    assert ctx.sends == [("B", "s1")] and ctx.severs == ["C"] and ctx.routes == ["D"]
    assert ctx.adjusts == {"temp": 0.5} and ctx.vars == {"v": 3}


def test_register_access_and_budget():
    ctx = run_action("(emit (add (reg 0) (reg 3)))", "1001")
    assert ctx.emits == ["2.0"]
    with pytest.raises(BudgetExceeded):
        run_action("(seq " + "(emit 1) " * 100 + ")", "0", budget=64)


# ── engine mechanics ─────────────────────────────────────────────────────────

def test_covering_creates_one_rule_per_missing_action():
    eng = RuleEngine(4, ["(emit 0)", "(emit 1)"], seed=0)
    m = eng.match("1010")
    assert {r.key for r in m} == {"(emit 0)", "(emit 1)"}
    assert eng.counters["cover"] == 2 and eng.size == 2
    for r in m:
        assert matches(r.condition, "1010")


def test_step_is_deterministic_and_exploit_uses_argmax():
    a = RuleEngine(3, ["(emit 0)", "(emit 1)"], seed=1)
    b = RuleEngine(3, ["(emit 0)", "(emit 1)"], seed=1)
    for _ in range(20):
        ca, cb = a.step("101"), b.step("101")
        assert ca.emits == cb.emits
        a.reward(1.0 if ca.emits == ["1"] else 0.0, "101")
        b.reward(1.0 if cb.emits == ["1"] else 0.0, "101")
    assert a.step("101", explore=False).emits == ["1"]


def test_reward_updates_prediction_toward_reward():
    eng = RuleEngine(2, ["(emit 0)"], seed=0)
    eng.step("00", explore=False)
    rule = eng.action_set[0]
    for _ in range(30):
        eng.step("00", explore=False)
        eng.reward(1.0, "00")
    assert rule.prediction > 0.95 and rule.error < 0.05 and rule.experience >= 30


def test_bucket_brigade_pays_deferred_set():
    eng = RuleEngine(1, ["(emit 0)"], seed=0, params=Params(gamma=0.5))
    eng.step("0", explore=False)
    first = eng.action_set[0]
    eng.reward(0.0, "0", terminal=False)          # deferred
    assert eng._pending is not None and first.experience == 0
    eng.step("0", explore=False)                  # pays r + γ·max(PA)
    assert eng._pending is None and first.experience == 1


def test_population_bounded_by_N():
    eng = RuleEngine(6, ["(emit 0)", "(emit 1)"], params=Params(N=50), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        reg = "".join(map(str, rng.integers(0, 2, 6)))
        eng.step(reg)
        eng.reward(float(rng.integers(0, 2)), reg)
    assert eng.size <= 50 and eng.counters["delete"] > 0


def test_compact_dedupes_subsumes_and_prunes():
    eng = RuleEngine(4, ["(emit 1)"], seed=0)
    general = Rule("1###", "(emit 1)", 1.0, 0.0, 0.5, experience=50, id=1)
    specific = Rule("1#0#", "(emit 1)", 1.0, 0.0, 0.5, experience=50, id=2)
    dup = Rule("1#0#", "(emit 1)", 1.0, 0.0, 0.5, experience=50, id=3)
    weak = Rule("0000", "(emit 1)", 0.1, 0.5, 1e-6, experience=50, id=4)
    eng.rules = [general, specific, dup, weak]
    removed = eng.compact()
    assert removed == 3 and eng.rules == [general] and general.numerosity == 3


def test_gp_mutate_inserts_variants_and_donor_crossover():
    eng = RuleEngine(4, ["(emit 0)", "(emit 1)"], seed=0)
    donor = RuleEngine(4, ["(emit 0)", "(emit 1)"], seed=1)
    eng.step("1010")
    donor.step("0101")
    before = eng.macro_size()
    n = eng.gp_mutate(donor=donor, n=6)
    assert n == 6 and eng.macro_size() >= before and eng.counters["mutate"] == 6
    for r in eng.rules:
        assert len(r.condition) == 4 and r.key in eng.actions


def test_erode_decays_rules_outside_action_set():
    eng = RuleEngine(2, ["(emit 0)", "(emit 1)"], seed=0)
    eng.step("00", explore=False)
    inside = {id(r) for r in eng.action_set}
    before = {id(r): r.fitness for r in eng.rules}
    eng.erode(excess=1.0, erosion=0.5)
    for r in eng.rules:
        assert r.fitness == pytest.approx(before[id(r)] * (0.5 if id(r) not in inside else 1.0))


def test_reinforce_mode_accumulates_strength():
    eng = RuleEngine(2, ["(emit 0)", "(emit 1)"], seed=0,
                     params=Params(mode="reinforce", reinforce_decay=1.0, p_wild=0.0))
    for _ in range(10):
        eng.step("01", explore=True)
        eng.reward(1.0 if eng.last_action == "(emit 1)" else 0.0, "01")
    pa = eng.prediction_array(eng.match("01"))
    assert pa["(emit 1)"] > pa["(emit 0)"]
    assert eng.step("01", explore=False).emits == ["1"]


def test_engine_learns_6mux_standalone():
    def mux6(b):
        return b[2 + 2 * b[0] + b[1]]
    rng = np.random.default_rng(0)
    eng = RuleEngine(6, ["(emit 0)", "(emit 1)"], seed=0)
    for _ in range(4000):
        bits = rng.integers(0, 2, 6)
        reg = "".join(map(str, bits))
        ctx = eng.step(reg)
        eng.reward(1.0 if int(ctx.emits[0]) == mux6(bits) else 0.0, reg)
    ok = 0
    for _ in range(500):
        bits = rng.integers(0, 2, 6)
        ok += int(eng.step("".join(map(str, bits)), explore=False).emits[0]) == mux6(bits)
    assert ok / 500 >= 0.95
