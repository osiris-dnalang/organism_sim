"""LCS agent + bus + 6-multiplexer benchmark (Phase A, Phase B tiers)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.mux import (  # noqa: E402
    PROTOCOL_DNA,
    mux6,
    run_single,
    run_split,
)
from organism_sim.bus import Payload, RoutedBus  # noqa: E402
from organism_sim.lcs import Params  # noqa: E402
from organism_sim.spec import parse_dna  # noqa: E402

# ── bus ──────────────────────────────────────────────────────────────────────

def test_payload_schema():
    p = Payload(content={"x": 1}, pressure=0.3, sender="A", recipient="B")
    d = p.to_dict()
    assert set(d) == {"content", "pressure", "sender", "recipient", "kind", "tick", "id",
                      "reply_to"}
    with pytest.raises(ValueError):
        Payload(content="", pressure=-1, sender="A")
    with pytest.raises(ValueError):
        Payload(content="", pressure=0, sender="A", kind="bogus")


def test_routed_delivery_and_sever():
    bus = RoutedBus()
    a = LCSAgent("A", 2, ["(send B s0)"], symbols=("s0",), bus=bus)
    b = LCSAgent("B", 2, ["(emit 0)"], symbols=("s0",), bus=bus)
    bus.add(a)
    bus.add(b)
    bus.connect("A", "B")
    bus.inject("hello", pressure=0.7)                 # env → every node
    assert bus.deliver() == 2 and a.last_symbol == "hello" and b.last_symbol == "hello"
    assert a.organism.external_noise == pytest.approx(0.7)
    a.act("01")                                       # (send B s0) → queued
    assert bus.deliver() == 1 and b.last_symbol == "s0" and b.last_symbol_from == "A"
    assert bus.sever("A", "B") and not bus.sever("A", "B")
    a.act("01")                                       # route gone → send not emitted
    assert a.outputs == [] and bus.deliver() == 0
    bus.send(Payload(content="x", pressure=0.0, sender="A", recipient="nobody"))
    assert bus.deliver() == 0 and bus.dropped == 1


def test_register_encodes_last_symbol():
    b = LCSAgent("B", 4, ["(emit 0)"], symbols=("s0", "s1", "s2", "s3"))
    assert b.symbol_bits == 3 and b.cond_len == 7
    assert b.register("1010") == "1010000"
    b.last_symbol = "s2"
    assert b.register("1010") == "1010011"
    with pytest.raises(ValueError):
        b.register("101")


def test_credit_crosses_bus_and_error_becomes_noise():
    bus = RoutedBus()
    a = LCSAgent("A", 2, ["(send B s0)"], bus=bus)
    b = LCSAgent("B", 1, ["(emit 0)", "(emit 1)"], symbols=("s0",), bus=bus)
    bus.add(a)
    bus.add(b)
    bus.connect("A", "B")
    a.act("00")
    bus.deliver()
    b.act("1", explore=False)
    exp_a = a.engine.action_set[0].experience
    b.reward(0.0)                                     # wrong → pressure 1 to A
    assert bus.deliver() == 1 and a.credits_received == 1
    assert a.engine.action_set[0].experience == exp_a + 1
    assert b.organism.external_noise == pytest.approx(1.0)
    assert a.organism.external_noise == pytest.approx(1.0)


def test_organism_processor_hooks_operate_on_rule_set():
    a = LCSAgent("A", 3, ["(emit 0)", "(emit 1)"], structural=True, seed=0)
    a.act("101")
    a.reward(1.0)
    a.tick()
    assert a.organism.processor is a
    assert a.organism.state.entropy_bits > 0
    n_before = a.engine.macro_size()
    a.mutate(a.organism, "noise")
    assert a.engine.counters["mutate"] > 0 and a.engine.macro_size() >= n_before
    a.repair(a.organism)
    assert a.engine.counters["compact"] == 1
    off = LCSAgent("B", 3, ["(emit 0)"], structural=False)
    off.mutate(off.organism, "noise")
    assert off.engine.counters["mutate"] == 0


def test_frozen_protocol_sender_from_dna():
    genome = parse_dna(PROTOCOL_DNA.read_text()).genome
    assert [g.condition for g in genome.rules()] == ["00", "01", "10", "11"]
    a = LCSAgent("A", 2, [f"(send B s{i})" for i in range(4)], rules=genome.rules(), frozen=True)
    a.routes.add("B")
    for addr, sym in (("00", "s0"), ("01", "s1"), ("10", "s2"), ("11", "s3")):
        a.act(addr)
        assert a.outputs[0].content == sym and a.outputs[0].recipient == "B"
    a.reward(0.0)
    assert a.engine.rules[0].experience == 0        # frozen: no learning


# ── 6-mux ────────────────────────────────────────────────────────────────────

def test_mux6_truth():
    assert mux6("000000") == "0" and mux6("001000") == "1"
    assert mux6("010100") == "1" and mux6("100010") == "1" and mux6("110001") == "1"
    assert mux6("110000") == "0" and mux6("011000") == "0"


def test_phase_a_single_organism_solves_6mux():
    log = run_single(3000, seed=0, log_every=1000)
    assert log.rows[-1].eval_acc >= 0.95
    assert log.final["audit_chain_valid"]
    rows = json.loads(log.to_json())["rows"]
    assert len(rows) == 3 and len(log.to_csv().splitlines()) == 4


def test_phase_a_deterministic():
    a = run_single(1000, seed=4, log_every=500)
    b = run_single(1000, seed=4, log_every=500)
    assert a.to_json() == b.to_json()


@pytest.mark.parametrize("seed", [0, 1])
def test_phase_b_protocol_sender_isolated_vs_coupled(seed):
    """Deterministic tier: A's protocol comes from protocol_sender.dna; B must learn
    the multiplexer over the bus. Isolated B is capped by the majority ceiling."""
    iso = run_split(6000, seed=seed, coupled=False, log_every=6000, params=Params(p_explore=0.3))
    cpl = run_split(6000, seed=seed, coupled=True, log_every=6000, params=Params(p_explore=0.3))
    assert iso.final["eval_acc"] <= 0.75
    assert cpl.final["eval_acc"] >= 0.95
    assert iso.rows[-1].credits == 0 and cpl.rows[-1].credits == 6000
    assert cpl.final["audit_chains_valid"]


def test_phase_b_coadaptation_from_scratch():
    """Exploratory tier: both agents learn. Seed 2 reaches a signalling system;
    other seeds may settle in partial pooling (see README numbers)."""
    cpl = run_split(10000, seed=2, coupled=True, log_every=10000, sender="learn",
                    params=Params(p_explore=0.3))
    iso = run_split(10000, seed=2, coupled=False, log_every=10000, sender="learn",
                    params=Params(p_explore=0.3))
    assert cpl.final["eval_acc"] >= 0.95
    assert iso.final["eval_acc"] <= 0.75
    # the learned protocol is injective: a distinct symbol per address
    assert len(set(cpl.final["A_policy"].values())) == 4


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "organism_sim.cli", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)


def test_cli_bench(tmp_path):
    out = tmp_path / "mux.csv"
    r = _cli("bench", "mux", "--phase", "A", "--trials", "1000", "--log-every", "500",
             "--csv", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["phase"] == "A" and d["eval_acc"] >= 0.9
    assert len(out.read_text().splitlines()) == 3
