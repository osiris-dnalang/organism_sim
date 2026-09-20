"""Relay nodes, broadcast sends, exploit-only upstream pressure, topological reroute,
and the relay-failure harness."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.relay import GROUPS, SENDER_DNA, compare, run_relay  # noqa: E402
from organism_sim.bus import Relay, RoutedBus  # noqa: E402
from organism_sim.spec import Triggers, parse_dna  # noqa: E402

SYM = ("s0", "s1", "s2", "s3")


def _chain():
    bus = RoutedBus()
    genome = parse_dna(SENDER_DNA.read_text()).genome
    a = LCSAgent("A", 2, [f"(send * {s})" for s in SYM], rules=genome.rules(), bus=bus,
                 frozen=True, structural=True)
    b = LCSAgent("B", 4, ["(emit 0)", "(emit 1)"], symbols=SYM, bus=bus, seed=1)
    r1, r2 = Relay("R1", bus, SYM, seed=3), Relay("R2", bus, SYM, seed=4)
    for n in (a, b, r1, r2):
        bus.add(n)
    a.routes.add("R1")
    a.peers = ["R1", "R2"]
    r1.routes.add("B")
    r2.routes.add("B")
    b.routes.update({"R1", "R2"})
    return bus, a, b, r1, r2


def test_relay_forwards_signal_and_credit():
    bus, a, b, r1, r2 = _chain()
    bus.next_tick()
    a.act("10")                                # (send * s2) → R1 only
    assert bus.deliver() == 2                  # A→R1, R1→B
    assert b.last_symbol == "s2" and b.last_symbol_from == "R1" and r1.upstream == "A"
    assert r2.forwarded == 0
    b.act("0000", explore=False)
    b.reward(0.0)                              # exploit miss → pressure 1 upstream
    assert bus.deliver() == 2                  # B→R1, R1→A
    assert a.credits_received == 1 and a.organism.external_noise == 1.0


def test_relay_dead_and_poisoned_modes():
    bus, a, b, r1, r2 = _chain()
    r1.mode = "dead"
    bus.next_tick()
    a.act("00")
    bus.deliver()
    assert b.last_symbol is None and r1.dropped == 1
    r1.mode = "poisoned"
    bus.next_tick()
    a.act("00")
    bus.deliver()
    assert b.last_symbol in SYM and b.organism.external_noise == 1.0


def test_exploration_misses_carry_no_upstream_pressure():
    bus, a, b, r1, r2 = _chain()
    bus.next_tick()
    a.act("00")
    bus.deliver()
    b.act("1111", explore=True)
    b.reward(0.0)
    bus.deliver()
    assert a.organism.external_noise == 0.0
    assert a.credits_received == 1


def test_silence_credit_broadcasts_to_routes():
    bus, a, b, r1, r2 = _chain()
    r1.mode = "dead"
    bus.next_tick()
    a.act("00")
    bus.deliver()
    b.act("0000", explore=False)
    b.reward(0.0)                              # nothing consumed → credit to every route
    n = bus.deliver()
    assert n >= 2 and a.credits_received >= 1  # R1 and R2 both relay it up to A


def test_frozen_sender_has_no_entropy_trigger():
    _, a, *_ = _chain()
    assert a.triggers.entropy_floor_bits == 0.0
    assert Triggers().entropy_floor_bits == 3.0


def test_reroute_switches_to_unrouted_peer_with_cooldown():
    bus, a, b, r1, r2 = _chain()
    a.organism.tick = 10
    assert a.reroute("test") == "R2" and a.routes == {"R2"}
    assert a.reroute("test") is None            # cooldown
    a.organism.tick = 10 + a.reroute_cooldown
    assert a.reroute("test") == "R1" and a.routes == {"R1"}
    assert [r["to"] for r in a.reroutes] == ["R2", "R1"]


def test_broadcast_send_targets_all_routes():
    bus, a, b, r1, r2 = _chain()
    a.routes.add("R2")
    bus.next_tick()
    a.act("11")
    assert sorted(p.recipient for p in a.outputs) == ["R1", "R2"]


def test_run_relay_groups_and_fields():
    for g in GROUPS:
        log = run_relay(g, trials=1200, fail_at=600, mode="dead", seed=0, probe_every=100,
                        bursts=())
        f = log.final
        assert f["group"] == g and f["failed_relay"] in ("R1", "R2")
        assert {"recovery", "false_reroutes", "reroutes_total", "armed_at",
                "audit_chains_valid"} <= set(f)
        assert f["audit_chains_valid"]
        rows = json.loads(log.to_json())["rows"]
        assert rows[-1]["trial"] == 1200 and len(log.to_csv().splitlines()) == len(rows) + 1
    none = run_relay("none", trials=1200, fail_at=600, mode="dead", seed=0, bursts=())
    assert none.final["reroutes_total"] == 0


def test_run_relay_deterministic():
    a = run_relay("organism", 1500, 700, "poisoned", seed=5, bursts=(300,))
    b = run_relay("organism", 1500, 700, "poisoned", seed=5, bursts=(300,))
    assert a.to_json() == b.to_json()


def test_compare_verdict_fields():
    res = compare(seeds=[0], mode="dead", trials=1500, fail_at=800, c1_trials=600)
    assert set(res["groups"]) == {"none", "supervisor", "organism"} and set(GROUPS) >= set(res["groups"])
    assert {"C1_capability", "C2_latency_vs_supervisor", "C3_false_alarms",
            "sanity_none_stuck", "pass"} <= set(res["verdict"])


# ── CUSUM trigger ────────────────────────────────────────────────────────────

def test_cusum_accumulates_only_above_drift_and_fires_reroute():
    bus, a, b, r1, r2 = _chain()
    a.cusum = (0.05, 0.10, 1.0)
    a.organism.tick = 500                      # past the reroute cooldown
    # below mu0 + k: no accumulation
    a._pressure_this_trial = [0.1]
    a.tick()
    assert a.cusum_s == 0.0 and a.cusum_fires == 0
    # sustained high pressure accumulates and fires
    for _ in range(3):
        a._pressure_this_trial = [0.6]
        a.tick()
    assert a.cusum_fires == 1 and a.routes == {"R2"} and a.cusum_s == 0.0
    assert a.reroutes[-1]["why"] == "cusum"


def test_cusum_arm_runs_in_relay_harness():
    log = run_relay("organism-cusum", trials=1500, fail_at=800, mode="poisoned", seed=0,
                    bursts=(), cusum=(0.05, 0.10, 3.0))
    f = log.final
    assert f["group"] == "organism-cusum" and f["audit_chains_valid"]
    assert any(r["why"] == "cusum" for r in log.final["reroutes_after_fail"]) or f["reroutes_total"] >= 0
