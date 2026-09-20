"""Single-agent tests: parameters, genome format, tick lifecycle, stability under the
noise floor, repair/mutation triggers, metrics, audit chain, CLI."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import organism_sim as osim  # noqa: E402
from organism_sim.audit import GENESIS_HASH, AuditChain  # noqa: E402
from organism_sim.noise import Environment, hostile_environment, quiet_environment  # noqa: E402
from organism_sim.organism import Organism, shannon_bits  # noqa: E402
from organism_sim.spec import OrganismSpec, Phase, Status, Triggers, parse_dna  # noqa: E402

FLOOR = Triggers.noise_floor
LEGACY_DNA = Path("/home/enki/organism")


# ── parameters ───────────────────────────────────────────────────────────────

def test_default_triggers_clear_the_floor():
    t = Triggers()
    assert t.repair_threshold > t.noise_floor
    assert t.entropy_floor_bits == 3.0 and t.window == 5 and t.unrepaired_integral == 1.0


def test_repair_threshold_below_floor_is_rejected():
    with pytest.raises(ValueError):
        Triggers(repair_threshold=0.3)
    with pytest.raises(ValueError):
        Triggers(repair_threshold=FLOOR)
    with pytest.raises(ValueError):
        Triggers(window=0)


def test_package_has_no_physics_vocabulary():
    """Guard: the sanitised names must not regress."""
    banned = ("THETA_LOCK", "LAMBDA_PHI", "CHI_PC", "consciousness", "phase_conjugate",
              "qiskit", "IBMQuantum", "51.843", "2.176435e-8", "0.3843", "CRSM")
    for py in (ROOT / "organism_sim").glob("*.py"):
        text = py.read_text()
        for word in banned:
            assert word not in text, f"{word!r} found in {py.name}"


# ── genome / spec ────────────────────────────────────────────────────────────

def test_alpha_spec_roundtrip():
    spec = osim.build_alpha_spec()
    assert len(spec.genome) == 24
    assert len(set(round(p, 6) for p in spec.genome.phases())) == 24
    back = OrganismSpec.from_json(spec.to_json())
    assert back.to_dict() == spec.to_dict()


def test_resonance_offset_is_a_plain_parameter():
    spec = osim.build_alpha_spec(resonance_offset_deg=30.0)
    assert spec.genome.phases()[0] == pytest.approx(30.0)
    assert spec.meta["resonance_offset_deg"] == 30.0


def test_parse_dna_dialect():
    text = """
    ORGANISM TEST_ORG {
        META { version: "1.2.3", domain: "unit_test" }
        DNA { purpose: "p", some_number: 1.5 }
        METRICS { lambda: 0.95 }
        GENOME {
            GENE Alpha { id: "G0", expression: 1.0, dependencies: [], outputs: ["a", "b"] }
            GENE Beta { id: "G1", expression: 0.5, dependencies: ["G0"] }
        }
    }
    """
    spec = parse_dna(text)
    assert spec.name == "TEST_ORG" and spec.domain == "unit_test"
    assert [g.id for g in spec.genome.genes] == ["G0", "G1"]
    assert spec.genome.genes[1].dependencies == ["G0"]
    assert spec.meta["raw_dna"]["some_number"] == 1.5     # preserved, not interpreted
    assert spec.meta["raw_metrics"]["lambda"] == 0.95


@pytest.mark.parametrize("fname", ["QBYTE_MINER_72GENE.dna", "SCIMITAR_SSE_128.dna",
                                   "MAXIMUS_OMEGA_120GENE.dna"])
def test_parse_legacy_genome_files(fname):
    p = LEGACY_DNA / fname
    if not p.exists():
        pytest.skip("legacy genome file not present")
    spec = parse_dna(p.read_text())
    assert len(spec.genome) >= 24
    org = Organism(spec, env=quiet_environment(len(spec.genome), seed=1), seed=1)
    assert org.run(10).audit_chain_valid


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_phase_order_per_tick():
    org = osim.build_alpha(seed=0)
    org.step()
    org.step()
    assert tuple(org.phase_trace) == Phase.sequence() * 2
    assert org.tick == 2 and len(org.chain) == 2


def test_deterministic_for_seed():
    a = osim.build_alpha(seed=7).run(50)
    b = osim.build_alpha(seed=7).run(50)
    assert a.final_state == b.final_state and a.genome_fingerprint == b.genome_fingerprint
    assert a.audit_head != GENESIS_HASH  # heads differ: records carry wall-clock timestamps


def test_idle_until_first_tick():
    org = osim.build_alpha()
    assert org.status is Status.IDLE
    org.step()
    assert org.status is not Status.IDLE


def test_state_definitions():
    org = osim.build_alpha(seed=3)
    org.run(5)
    s = org.state
    assert s.efficiency == pytest.approx(s.coherence * s.entropy_bits / max(s.noise_rate, s.EPS))
    assert 0.0 <= s.phase_deg < 360.0 and 0.0 <= s.coherence <= 1.0
    t = org.triggers
    assert math.exp(-t.repair_threshold * math.log(2) / t.repair_threshold) == pytest.approx(0.5)


# ── stability under the noise floor ──────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_stable_under_noise_floor(seed):
    org = osim.build_alpha(seed=seed, noise_floor=FLOOR)
    r = org.run(100)
    assert r.max_noise < 2.0 * org.triggers.repair_threshold
    assert 0.5 * FLOOR < r.mean_noise < 1.5 * FLOOR
    assert r.repairs > 0
    assert r.mutations < 5, "agent is thrashing on idle noise"
    assert r.ticks_stable >= 90
    assert r.min_entropy >= org.triggers.entropy_floor_bits
    assert r.audit_chain_valid


def test_silent_repair_keeps_status_stable():
    org = osim.build_alpha(seed=0)
    org.step()
    org.state.noise_rate = 0.6
    org._events = []
    org._maintain()
    assert "repair" in org._events and "mutation" not in org._events
    assert org.status is Status.STABLE


def test_environment_floor_enforced():
    env = Environment(n_genes=8, seed=0, noise_floor=FLOOR, jitter=0.0, burst_prob=0.0)
    ph = np.linspace(0, 360, 8, endpoint=False)
    assert all(env.sample(i, ph).noise >= FLOOR - 1e-12 for i in range(50))


def test_environment_ramp():
    env = Environment(n_genes=4, seed=0, noise_floor=0.1, jitter=0.0, burst_prob=0.0, ramp=0.01)
    ph = np.zeros(4)
    assert env.sample(10, ph).noise == pytest.approx(0.2)
    assert env.current_floor == pytest.approx(0.2)


def test_quiet_environment_no_repair_no_mutation():
    spec = osim.build_alpha_spec()
    r = Organism(spec, env=quiet_environment(24, seed=0), seed=0).run(100)
    assert r.repairs == 0 and r.mutations == 0 and r.ticks_stable >= 95


def test_hostile_environment_repair_suppresses_noise():
    spec = osim.build_alpha_spec()
    repaired = Organism(spec, env=hostile_environment(24, seed=0), seed=0)
    r = repaired.run(100)
    assert r.repairs >= 10 and r.mutations_noise >= 1 and r.audit_chain_valid
    unrepaired = Organism(osim.build_alpha_spec(Triggers(repair_threshold=float("inf"))),
                          env=hostile_environment(24, seed=0), seed=0)
    unrepaired.run(100)
    g_r = np.array(repaired.history()["noise"])
    g_u = np.array(unrepaired.history()["noise"])
    assert g_r.mean() < 0.75 * g_u.mean() and g_r.max() < g_u.max()


# ── triggers ─────────────────────────────────────────────────────────────────

def test_repair_fires_above_threshold_and_displaces_to_sink():
    org = osim.build_alpha(seed=0)
    org.state.noise_rate = 3.0
    org._cache = np.zeros(24)
    org._events = []
    org._maintain()
    factor = math.exp(-org.triggers.repair_gain)
    assert "repair" in org._events
    assert org.sink_load == pytest.approx(3.0 - 3.0 * factor)
    assert org._cache is None


def test_no_repair_below_threshold():
    org = osim.build_alpha(seed=0)
    org.state.noise_rate = 0.3
    org._events = []
    org._maintain()
    assert "repair" not in org._events


def test_mutation_on_entropy_collapse_is_independent_of_noise():
    org = osim.build_alpha(seed=0)
    org.expression = np.full(24, Organism.EXPR_FLOOR)
    org.expression[0] = 1.0
    org.state.noise_rate = 0.1
    org._recompute_entropy()
    assert org.state.entropy_bits < org.triggers.entropy_floor_bits
    h0, gen0 = shannon_bits(org.expression), org.generation
    org._events = []
    org._maintain()
    assert "mutation_reason=entropy" in org._events
    assert org.generation == gen0 + 1 and shannon_bits(org.expression) > h0
    assert org.expression.min() >= Organism.EXPR_FLOOR and org.expression.max() <= 1.0


def test_entropy_is_independent_of_noise():
    org = osim.build_alpha(seed=0)
    org._recompute_entropy()
    h = org.state.entropy_bits
    org.state.noise_rate = 5.0
    org._recompute_entropy()
    assert org.state.entropy_bits == h


def test_mutation_on_sustained_unrepaired_noise():
    org = osim.build_alpha(seed=0)
    org.step()
    t = org.triggers
    org.state.noise_rate = 1.0
    org._events = []
    org._maintain()
    assert "repair" in org._events and "mutation" not in org._events
    assert 0.0 < org.unrepaired_integral < t.unrepaired_integral
    for _ in range(t.window):
        org.state.noise_rate = 2.0
        org._events = []
        org._maintain()
        if "mutation" in org._events:
            break
    assert "mutation_reason=noise" in org._events
    assert org.counters["mutate_noise"] == 1 and org.unrepaired_integral == 0.0


def test_sink_flush():
    org = osim.build_alpha(seed=0, triggers=Triggers(sink_capacity=0.5))
    org.state.noise_rate = 5.0
    org._events = []
    org._maintain()
    assert "sink_flush" in org._events and org.sink_load == 0.0


def test_phase_reset_at_trigger_angle():
    org = osim.build_alpha(seed=0)
    org.step()
    org.K_ENV_COUPLING = 0.0
    org.state.phase_deg = (org.triggers.trigger_deg - org.OMEGA_DEG) % 360.0
    org._events = []
    org._sample = org.env.sample(org.tick + 1, org.gene_phase)
    org._update()
    assert "phase_reset" in org._events and org.state.phase_deg == 0.0


def test_metrics_bounded():
    org = osim.build_alpha(seed=1)
    org.run(30)
    d = org.metrics.to_dict()
    assert set(d) == set(osim.Metrics.NAMES)
    assert all(0.0 <= v <= 1.0 for v in d.values())
    assert d["cycle_position"] == pytest.approx((30 % Organism.PERIOD_TICKS) / Organism.PERIOD_TICKS)


# ── audit chain ──────────────────────────────────────────────────────────────

def test_audit_chain_links_verifies_and_detects_tampering():
    org = osim.build_alpha(seed=0)
    org.run(10)
    recs = org.chain.records
    assert recs[0].prev_hash == GENESIS_HASH
    assert all(b.prev_hash == a.hash for a, b in zip(recs, recs[1:]))
    assert org.chain.verify()
    recs[4].state["noise_rate"] = 0.0
    assert not org.chain.verify()


def test_audit_hmac_seal():
    org = osim.build_alpha(seed=0, audit_secret=b"k")
    org.run(5)
    seal = org.chain.seal()
    assert seal and org.chain.check_seal(seal)
    assert not AuditChain.from_records(org.chain.records, secret=b"x").check_seal(seal)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "organism_sim.cli", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=120)


def test_cli_run_json_and_telemetry(tmp_path):
    out = tmp_path / "t.jsonl"
    r = _cli("run", "--ticks", "20", "--seed", "1", "--json", "--telemetry", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["ticks"] == 20 and d["audit_chain_valid"] is True
    assert len(out.read_text().splitlines()) == 20


def test_cli_exposes_repair_threshold():
    r = _cli("run", "--ticks", "30", "--seed", "2", "--repair-threshold", "0.6",
             "--entropy-floor", "2.5", "--window", "3", "--integral", "0.8", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)["triggers"]
    assert d == {**d, "repair_threshold": 0.6, "entropy_floor_bits": 2.5, "window": 3,
                 "unrepaired_integral": 0.8}
    base = json.loads(_cli("run", "--ticks", "30", "--seed", "2", "--json").stdout)
    assert json.loads(r.stdout)["repairs"] < base["repairs"]


def test_cli_rejects_threshold_below_floor():
    r = _cli("run", "--repair-threshold", "0.3")
    assert r.returncode == 2 and "must exceed noise_floor" in r.stderr


def test_cli_dump_spec_reload(tmp_path):
    p = tmp_path / "alpha.json"
    p.write_text(_cli("dump-spec").stdout)
    r = _cli("run", "--spec", str(p), "--ticks", "5")
    assert r.returncode == 0 and "ALPHA" in r.stdout
