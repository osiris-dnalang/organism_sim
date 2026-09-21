"""M-Bridge-1: ``capture_frame`` exports the organism's actual state as schema-valid JSON.

Three layers are exercised: a bare ``Organism`` (no learner, no regulation → those blocks are
null), an ``LCSAgent`` (learner + topology, no regulation), and a ``GRN`` over an agent
(everything). Every mirrored value is checked against the live attribute it came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
jsonschema = pytest.importorskip("jsonschema")

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.alpha import build_alpha  # noqa: E402
from organism_sim.benchmarks.expzero import DriftingHidden  # noqa: E402
from organism_sim.grn import GRN  # noqa: E402
from organism_sim.spec import parse_dna  # noqa: E402
from organism_sim.telemetry import (  # noqa: E402
    FRAME_SCHEMA,
    capture_frame,
    frame_digest,
    frame_json,
)

SCHEMA = json.loads((ROOT / "schemas" / "telemetry_frame.schema.json").read_text())
HAND = ROOT / "organism_sim" / "benchmarks" / "m1_hand.dna"


def _validate(frame):
    jsonschema.Draft7Validator(SCHEMA).validate(frame)


def _grn_runtime(seed=0, trials=200):
    agent = LCSAgent("O", input_len=6, actions=["(emit 0)", "(emit 1)"], seed=seed)
    agent.peers = ["P1", "P2"]
    grn = GRN(parse_dna(HAND.read_text()).genome, agent, seed=seed)
    env = DriftingHidden(seed, shift_every=10 ** 6)
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        agent.act(bits)
        r = 1.0 if agent.emitted() == env.truth(bits) else 0.0
        agent.reward(r)
        grn.observe(r, agent.engine.last_explore)
        grn.step()
        agent.tick()
    return agent, grn


def test_bare_organism_frame_is_honest_about_missing_layers():
    org = build_alpha(seed=0)
    org.run(20)
    frame = capture_frame(org, tick=20)
    _validate(frame)
    assert frame["schema"] == FRAME_SCHEMA
    assert frame["learner"] is None and frame["topology"] is None and frame["regulation"] is None
    assert frame["cusum"] is None and frame["error"] is None and frame["active_genes"] == []
    o = frame["organism"]
    assert o["tick"] == org.tick == 20
    assert o["state"] == org.state.to_dict()
    assert o["metrics"] == org.metrics.to_dict()
    assert o["audit_head"] == org.chain.head
    assert o["chain_length"] == len(org.chain.records) == 20
    assert o["events"] == list(org.chain.records[-1].events)
    assert o["status"] == frame["status"] == org.status.value


def test_agent_frame_has_learner_and_topology_but_no_regulation():
    agent = LCSAgent("A", input_len=6, actions=["(emit 0)", "(emit 1)"], seed=1,
                     cusum=(0.05, 0.10, 4.0), cusum_signal="error")
    agent.peers = ["B"]
    for bits in ("010101", "111000", "000111"):
        agent.act(bits)
        agent.reward(0.0)
        agent.tick()
    frame = capture_frame(agent, tick=3)
    _validate(frame)
    assert frame["regulation"] is None
    le, tp = frame["learner"], frame["topology"]
    assert le["trial"] == agent.trial == 3
    assert le["macro"] == agent.engine.macro_size()
    assert le["micro"] == agent.engine.size
    assert le["p_explore"] == agent.engine.p.p_explore
    assert le["cusum"] == {"signal": "error", "s": agent.cusum_s, "h": 4.0, "mu0": 0.05,
                           "k": 0.10, "fires": agent.cusum_fires}
    assert frame["cusum"] == agent.cusum_s and frame["cusum_threshold"] == 4.0
    assert tp["peers"] == ["B"] and tp["routes"] == sorted(agent.routes) and tp["on_bus"] is False


def test_grn_frame_mirrors_live_regulation_state():
    agent, grn = _grn_runtime()
    frame = capture_frame(grn, tick=grn.tick)
    _validate(frame)
    r = frame["regulation"]
    assert r["tick"] == grn.tick == 200
    assert frame["cusum"] == r["cusum"]["s"] == grn.cusum_s
    assert frame["cusum_threshold"] == grn.cusum_h == 8.0
    assert frame["error"] == r["metrics"]["error"] == grn.err_ema
    assert frame["active_genes"] == r["active_genes"] == sorted(grn.expressed_last)
    assert r["board"] == sorted(grn.board)
    assert r["metrics"] == grn.metrics()
    assert [g["id"] for g in r["genes"]] == [g.id for g in grn.genome.genes]
    assert r["genes"][0]["trigger"] == "when cusum > 8"
    assert r["expressions"] == grn.expressions
    assert frame["organism"]["tick"] == agent.organism.tick
    assert frame["learner"]["trial"] == agent.trial
    assert frame["topology"]["peers"] == ["P1", "P2"]


def test_frame_from_any_layer_resolves_to_the_same_organism():
    agent, grn = _grn_runtime(trials=50)
    f_org = capture_frame(agent.organism, tick=50)
    f_agent = capture_frame(agent, tick=50)
    f_grn = capture_frame(grn, tick=50)
    assert f_org["organism"] == f_agent["organism"] == f_grn["organism"]
    assert f_org["learner"] is None and f_agent["learner"] == f_grn["learner"]
    assert f_agent["regulation"] is None and f_grn["regulation"] is not None


def test_frame_is_plain_json_and_digest_renders():
    _, grn = _grn_runtime(trials=30)
    frame = capture_frame(grn, tick=30)
    text = frame_json(frame)
    assert json.loads(text) == frame               # nothing lost in serialisation
    assert json.loads(json.dumps(frame)) == frame  # no numpy scalars leaked
    lines = frame_digest(frame)
    assert lines[0].startswith("tick 30") and any(ln.startswith("regulation:") for ln in lines)


def test_schema_rejects_fabricated_fields():
    org = build_alpha(seed=0)
    org.step()
    frame = capture_frame(org, tick=1)
    v = jsonschema.Draft7Validator(SCHEMA)
    v.validate(frame)
    for mutate in (lambda d: d.update(invented_metric=1.0),
                   lambda d: d.pop("regulation"),
                   lambda d: d.update(cusum=-1.0),
                   lambda d: d["organism"].update(consciousness=0.5)):
        bad = json.loads(json.dumps(frame))
        mutate(bad)
        with pytest.raises(jsonschema.ValidationError):
            v.validate(bad)


def test_capture_frame_rejects_unknown_targets():
    with pytest.raises(TypeError):
        capture_frame(object(), tick=0)
