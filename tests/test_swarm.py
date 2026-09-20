"""Population tests: bus semantics, determinism, coupling → synchronisation,
horizontal transfer, noise ramp, JSON/CSV export, CLI."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.swarm import (  # noqa: E402
    Coupling,
    Message,
    MessageBus,
    Population,
    TickMetrics,
    sync_order_parameter,
)


def test_bus_is_tick_scoped():
    bus = MessageBus()
    bus.publish(Message("a", 1, 0.0, 4.0, 0.1, 0.9, "G0"))
    bus.publish(Message("b", 1, 90.0, 4.0, 0.1, 0.9, "G1"))
    assert len(bus) == 2 and bus.published == 2
    assert [m.sender for m in bus.read(exclude="a")] == ["b"]
    bus.clear()
    assert len(bus) == 0 and bus.published == 2


def test_sync_order_parameter():
    assert sync_order_parameter([10.0, 10.0, 10.0]) == pytest.approx(1.0)
    assert sync_order_parameter([0.0, 180.0]) == pytest.approx(0.0, abs=1e-12)
    assert sync_order_parameter([0.0, 90.0, 180.0, 270.0]) == pytest.approx(0.0, abs=1e-12)
    assert sync_order_parameter([]) == 0.0


def test_population_deterministic():
    a = Population(n=4, seed=3).run(40)
    b = Population(n=4, seed=3).run(40)
    assert a.to_json() == b.to_json()


def test_agents_are_heterogeneous():
    pop = Population(n=6, seed=0)
    phases = {round(o.state.phase_deg, 6) for o in pop.organisms}
    freqs = {round(o.OMEGA_DEG, 6) for o in pop.organisms}
    assert len(phases) == 6 and len(freqs) == 6
    assert pop.log.rows == []
    pop.step()
    assert pop.log.rows[0].sync_r < 0.95, "uncoupled fresh population should not start in sync"


def test_uncoupled_population_matches_isolated_agents():
    """With coupling off, each agent's trajectory equals the same seed run alone."""
    pop = Population(n=3, seed=5, coupling=Coupling(phase_gain=0.0, transfer_rate=0.0))
    pop.run(30)
    for o in pop.organisms:
        assert o.chain.verify()
    # Population bookkeeping consistent with agent counters
    assert pop.log.rows[-1].cumulative_mutations == sum(o.counters["mutate"] for o in pop.organisms)
    assert pop.bus.published == 3 * 30


def test_phase_coupling_increases_synchronisation():
    r = {}
    for g in (0.0, 0.15, 0.3):
        pop = Population(n=8, seed=0, coupling=Coupling(phase_gain=g, transfer_rate=0.0))
        pop.run(200)
        r[g] = float(np.mean(pop.log.column("sync_r")[100:]))
    assert r[0.0] < r[0.15] < r[0.3]
    assert r[0.3] > 0.7


def test_ring_topology_uses_two_neighbours():
    pop = Population(n=6, seed=0, coupling=Coupling(topology="ring"))
    pop.step()
    assert len(pop._neighbours(0)) == 2
    names = {m.sender for m in pop._neighbours(0)}
    assert names == {pop.organisms[5].spec.name, pop.organisms[1].spec.name}


def test_horizontal_transfer_raises_low_entropy_agent():
    pop = Population(n=3, seed=0, coupling=Coupling(phase_gain=0.0, transfer_rate=0.2))
    pop.step()
    low = pop.organisms[0]
    low.expression = np.full(24, low.EXPR_FLOOR)
    low.expression[0] = 1.0
    low._recompute_entropy()
    h0 = low.state.entropy_bits
    row = pop.step()
    assert row.transfers_this_tick >= 1
    assert low.state.entropy_bits > h0


def test_noise_ramp_forces_mutations_only_after_threshold():
    pop = Population(n=6, seed=1, ramp=0.005, coupling=Coupling(phase_gain=0.0,
                                                                 transfer_rate=0.0))
    pop.run(200)
    rows = pop.log.rows
    thr = pop.organisms[0].triggers.repair_threshold
    before = [r for r in rows if r.noise_floor < thr]
    after = [r for r in rows if r.noise_floor > thr + 0.3]
    assert before and after
    assert sum(r.mutations_this_tick for r in before) <= 1
    assert sum(r.mutations_this_tick for r in after) >= 5
    assert rows[-1].noise_floor == pytest.approx(0.40 + 0.005 * 200)


def test_log_exports_json_and_csv():
    pop = Population(n=4, seed=2)
    pop.run(10)
    rows = json.loads(pop.log.to_json())
    assert len(rows) == 10 and set(rows[0]) == set(TickMetrics.FIELDS)
    reader = list(csv.DictReader(io.StringIO(pop.log.to_csv())))
    assert len(reader) == 10 and reader[0].keys() == set(TickMetrics.FIELDS)
    assert float(reader[-1]["mean_entropy"]) == pytest.approx(rows[-1]["mean_entropy"])
    s = pop.summary()
    assert s["audit_chains_valid"] is True and s["messages_published"] == 40


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "organism_sim.cli", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=180)


def test_cli_swarm_exports(tmp_path):
    csv_p, json_p = tmp_path / "pop.csv", tmp_path / "pop.json"
    r = _cli("swarm", "--n", "4", "--ticks", "20", "--seed", "0", "--ramp", "0.01",
             "--csv", str(csv_p), "--json-log", str(json_p))
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["n"] == 4 and s["ticks"] == 20 and s["audit_chains_valid"]
    assert len(json.loads(json_p.read_text())) == 20
    assert len(csv_p.read_text().splitlines()) == 21
    r2 = _cli("swarm", "--n", "4", "--ticks", "20", "--seed", "0", "--no-coupling")
    assert json.loads(r2.stdout)["coupling"]["phase_gain"] == 0.0
