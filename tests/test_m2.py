"""M2 harness: wide hidden family, informed/shape/random cohorts, arm runs."""

import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks.expzero import HiddenMux  # noqa: E402
from organism_sim.benchmarks.m2 import (  # noqa: E402
    ARMS,
    SEED_DNA,
    WIDTH,
    K,
    informed_rules,
    load_seed_rules,
    run_arm,
    shape_rules,
)


def test_wide_family_is_a_multiplexer_with_irrelevant_bits():
    inst = HiddenMux(np.random.default_rng(5), k=K, width=WIDTH)
    assert len(inst.addr) == 3 and len(inst.data) == 8 and len(inst.irrelevant) == 5
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = [int(x) for x in rng.integers(0, 2, WIDTH)]
        idx = 4 * b[inst.addr[0]] + 2 * b[inst.addr[1]] + b[inst.addr[2]]
        expect = str((1 - b[inst.data[idx]]) if inst.invert else b[inst.data[idx]])
        assert inst.truth("".join(map(str, b))) == expect
        # irrelevant bits never change the output
        b2 = list(b)
        for i in inst.irrelevant:
            b2[i] ^= 1
        assert inst.truth("".join(map(str, b2))) == expect


def test_six_bit_family_unchanged():
    inst = HiddenMux(np.random.default_rng(3))
    assert inst.k == 2 and inst.width == 6 and inst.irrelevant == []
    for bits in itertools.islice(("".join(map(str, b)) for b in itertools.product([0, 1], repeat=6)), 64):
        b = [int(c) for c in bits]
        sel = inst.data[2 * b[inst.addr[0]] + b[inst.addr[1]]]
        assert inst.truth(bits) == str((1 - b[sel]) if inst.invert else b[sel])


def test_seed_rules_shape_and_artifact():
    genes = load_seed_rules()
    assert len(genes) == 40
    for g in genes:
        spec = [(i, c) for i, c in enumerate(g.condition) if c != "#"]
        assert len(spec) == K + 1 and len(g.condition) == WIDTH
        v = g.action.split()[-1].rstrip(")")
        assert v in ("0", "1") and any(c == v for _, c in spec)     # action = a specified bit's value
    assert [(g.condition, g.action) for g in informed_rules(40, 999)] == \
        [(g.condition, g.action) for g in genes]
    assert hashlib.sha256(SEED_DNA.read_bytes()).hexdigest()[:16] == "033cbaa28f10d83d"


def test_shape_rules_match_specificity_only():
    rs = shape_rules(40, 3)
    assert all(sum(c != "#" for c in g.condition) == K + 1 and len(g.condition) == WIDTH for g in rs)
    assert rs[0].condition != shape_rules(40, 4)[0].condition


def test_run_arm_short():
    for arm in ARMS:
        lg = run_arm(arm, seed=0, trials=3000, shift_every=1500, probe_every=500, n_inject=5)
        assert lg.shifts == [1500] and len(lg.recoveries) == 1 and lg.final["audit_chain_valid"]
        if arm == "none":
            assert lg.injections == []
        if arm in ("informed", "shape", "random"):
            assert lg.injections == [0, 1500]
