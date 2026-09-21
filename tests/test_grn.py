"""Regulatory genes: trigger grammar, signal board / dependency semantics, control effects,
.dna round-trip, evolution operators, and the grn benchmark arm."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expone import run_arm  # noqa: E402
from organism_sim.benchmarks.m1 import HAND_DNA, load  # noqa: E402
from organism_sim.grn import GRN, Trigger, genome_to_dna  # noqa: E402
from organism_sim.spec import Gene, Genome, parse_dna  # noqa: E402


def _agent():
    return LCSAgent("G", 6, ["(emit 0)", "(emit 1)"], seed=0)


def test_trigger_grammar_and_render_roundtrip():
    for text in ("on_genesis", "continuous", "on_error", "after G1", "on_signal shift",
                 "when cusum > 8", "when error >= 0.3", "when since_inject < 200"):
        t = Trigger.parse(text)
        assert Trigger.parse(t.render()).__dict__ == t.__dict__
    with pytest.raises(ValueError):
        Trigger.parse("when consciousness > 1")
    with pytest.raises(ValueError):
        Trigger.parse("whenever")


def test_cascade_signal_dependency_after():
    genes = [Gene(id="G0", name="detect", trigger="when cusum > 8", action="(emit shift)",
                  outputs=["shift"], cluster="sense"),
             Gene(id="G1", name="inject", trigger="on_signal shift",
                  action="(seq (adjust inject 40) (adjust cusum_reset 1))", dependencies=["G0"],
                  cluster="repair"),
             Gene(id="G2", name="boost", trigger="after G1", action="(adjust explore 0.8)",
                  cluster="repair")]
    a = _agent()
    grn = GRN(Genome(genes=genes), a, seed=0)
    assert grn.order == ["G0", "G1", "G2"]
    fired = []
    for _ in range(14):
        grn.observe(0.0, False)
        fired.append(tuple(grn.step()))
    ticks = {gid: next(i + 1 for i, f in enumerate(fired) if gid in f) for gid in ("G0", "G1", "G2")}
    assert ticks["G0"] < ticks["G1"] < ticks["G2"]          # detect → inject → boost, in order
    assert len(grn.injections) == 1 and a.engine.p.p_explore == 0.8
    assert a.engine.macro_size() >= 30           # 40 injected minus dedupe of identical random rules


def test_dependency_blocks_without_upstream_output():
    genes = [Gene(id="G1", name="needs", trigger="continuous", action="(emit x)",
                  dependencies=["G0"], cluster="c"),
             Gene(id="G0", name="never", trigger="when error > 5", action="(emit y)",
                  outputs=["y"], cluster="c")]
    grn = GRN(Genome(genes=genes), _agent(), seed=0)
    for _ in range(3):
        grn.observe(1.0, False)
        assert "G1" not in grn.step()


def test_rule_gene_expression_injects_rule():
    genes = [Gene(id="R0", name="rule", trigger="on_genesis", condition="1#0###",
                  action="(emit 1)", cluster="prior")]
    a = _agent()
    grn = GRN(Genome(genes=genes), a, seed=0)
    grn.observe(1.0, False)
    assert grn.step() == ["R0"]
    assert any(r.condition == "1#0###" for r in a.engine.rules)
    grn.observe(1.0, False)
    assert grn.step() == []                                  # genesis fires once


def test_explore_boost_expires_and_injection_floor():
    genes = [Gene(id="B", name="boost", trigger="continuous", action="(adjust explore 0.9)", cluster="c"),
             Gene(id="I", name="inj", trigger="continuous", action="(adjust inject 10)", cluster="c")]
    a = _agent()
    grn = GRN(Genome(genes=genes), a, seed=0, boost_ticks=3)
    for _ in range(60):
        grn.observe(1.0, False)
        grn.step()
    assert a.engine.p.p_explore == 0.9                       # continuously re-boosted
    assert len(grn.injections) == 2                          # floor of 50 ticks between injections


def test_dna_roundtrip_and_hand_genome():
    g = load(HAND_DNA)
    assert len(g.genes) == 5 and all(Trigger.parse(x.trigger) for x in g.genes)
    text = genome_to_dna(g, "X")
    back = parse_dna(text).genome
    assert [(x.id, x.trigger, x.action, x.dependencies, x.outputs) for x in back.genes] == \
        [(x.id, x.trigger, x.action, x.dependencies, x.outputs) for x in g.genes]


def test_mutate_and_crossover_preserve_validity():
    rng = np.random.default_rng(0)
    g = load(HAND_DNA)
    for _ in range(30):
        m = GRN.mutate(g, rng, rate=1.0)
        GRN(m, _agent(), seed=0)                             # parses and validates
        assert len(m.genes) == len(g.genes)
    c = GRN.crossover(g, GRN.mutate(g, rng), rng)
    GRN(c, _agent(), seed=0)
    assert {x.cluster for x in c.genes} <= {x.cluster for x in g.genes}


def test_grn_arm_runs_in_harness():
    lg = run_arm("grn", seed=0, trials=4000, shift_every=2000, probe_every=100,
                 cusum=(0.05, 0.1, 8.0), grn_genome=load(HAND_DNA))
    assert lg.shifts == [2000] and len(lg.recoveries95) == 1
    assert lg.final["grn_expressions"] is not None and lg.final["audit_chain_valid"]
    with pytest.raises(ValueError):
        run_arm("grn", seed=0, trials=500, shift_every=250)
