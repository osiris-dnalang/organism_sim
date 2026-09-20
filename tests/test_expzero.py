"""Experiment Zero harness: hidden generalised-mux family, seed rules as a .dna artifact,
injection semantics, arm runs."""

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expzero import (  # noqa: E402
    ARMS,
    SEED_DNA,
    DriftingHidden,
    HiddenMux,
    informed_rules,
    inject,
    load_seed_rules,
    random_rules,
    run_arm,
)

ALL = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=6)]


def test_hidden_mux_is_a_multiplexer_on_hidden_positions():
    inst = HiddenMux(np.random.default_rng(3))
    a0, a1 = inst.addr
    for bits in ALL:
        b = [int(c) for c in bits]
        sel = inst.data[2 * b[a0] + b[a1]]
        expect = str((1 - b[sel]) if inst.invert else b[sel])
        assert inst.truth(bits) == expect
    # instances differ across seeds
    assert HiddenMux(np.random.default_rng(1)).describe() != HiddenMux(np.random.default_rng(2)).describe()


def test_drifting_family_shifts_on_schedule_not_at_end():
    env = DriftingHidden(seed=0, shift_every=100)
    d0 = env.instance.describe()
    assert not env.maybe_shift(50, 1000)
    assert env.maybe_shift(100, 1000) and env.instance.describe() != d0 or env.shifts == [100]
    assert not env.maybe_shift(1000, 1000)          # no shift inside the final window


def test_seed_rules_are_family_shaped_and_stable():
    genes = load_seed_rules()
    assert len(genes) == 20
    for g in genes:
        assert sum(c != "#" for c in g.condition) == 3       # two address bits + one data bit
        v = g.action.split()[-1].rstrip(")")
        assert v in ("0", "1")
    # the .dna is the artifact: regeneration with the fixed seed reproduces it byte for byte
    regenerated = [(g.condition, g.action) for g in informed_rules(20, 999)]
    assert regenerated == [(g.condition, g.action) for g in genes]
    assert hashlib.sha256(SEED_DNA.read_bytes()).hexdigest()[:16] == "819ac39930b0c4fc"


def test_random_rules_deterministic_and_well_formed():
    a, b = random_rules(20, 7), random_rules(20, 7)
    assert [(g.condition, g.action) for g in a] == [(g.condition, g.action) for g in b]
    assert all(len(g.condition) == 6 and set(g.condition) <= set("01#") for g in a)


def test_inject_adds_fresh_hypotheses():
    agent = LCSAgent("Z", 6, ["(emit 0)", "(emit 1)"], seed=0)
    n = inject(agent, load_seed_rules())
    assert n == 20 and agent.engine.macro_size() == 20
    r = agent.engine.rules[0]
    assert r.experience == 0 and r.prediction == agent.engine.p.p_init


def test_run_arm_records_windows():
    for arm in ARMS:
        lg = run_arm(arm, seed=0, trials=2500, shift_every=1000, probe_every=100)
        assert lg.shifts == [1000] and len(lg.recoveries) == 1
        assert lg.initial is None or lg.initial % 100 == 0
        assert lg.final["audit_chain_valid"] and len(lg.instances) == 2
