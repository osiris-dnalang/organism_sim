"""
organism_sim.telemetry — structured snapshot of a running organism
==================================================================

``capture_frame(target, tick)`` serialises the *actual* runtime state of one organism
into a plain-JSON dict (the "frame"), for consumption by anything outside the
execution loop — the chat bridge, a dashboard, a log. Nothing here is computed; every
value is read off an object that already holds it.

The runtime is layered, and the frame follows the layers::

    Organism      state machine        → frame["organism"]   (always)
    LCSAgent      rule engine + bus    → frame["learner"], frame["topology"]  (if present)
    GRN           regulatory genome    → frame["regulation"] (if present)

``target`` may be any of the three; the others are found through ``GRN.agent`` and
``LCSAgent.organism``. Layers that do not exist are ``null`` — a bare ``Organism`` has
no CUSUM detector and no topology, and the frame says so rather than inventing them.

Top-level convenience keys (``cusum``, ``cusum_threshold``, ``error``, ``active_genes``,
``status``) mirror the deepest layer that defines them:

    cusum         GRN.cusum_s  (error-driven, threshold GRN.cusum_h)
                  else LCSAgent.cusum_s when the agent's detector is configured, else null
    error         GRN.err_ema  (EMA of exploit-trial error), else null
    active_genes  GRN.expressed_last — regulator genes expressed on the last GRN step

The contract is ``schemas/telemetry_frame.schema.json``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from .organism import Organism

FRAME_SCHEMA = "organism_sim.telemetry_frame/1"


def _f(x: Any) -> float:
    return float(x)


def _resolve(target: Any):
    """→ (organism, agent | None, grn | None) from any layer."""
    grn = agent = org = None
    if isinstance(target, Organism):
        org = target
    elif hasattr(target, "organism") and isinstance(target.organism, Organism):
        agent = target
        org = target.organism
    elif hasattr(target, "agent") and hasattr(target.agent, "organism"):
        grn = target
        agent = target.agent
        org = agent.organism
    else:
        raise TypeError(f"capture_frame: cannot resolve an Organism from {type(target).__name__}")
    return org, agent, grn


def _organism_block(org: Organism) -> Dict[str, Any]:
    last = org.chain.records[-1] if org.chain.records else None
    return {
        "name": org.spec.name,
        "tick": int(org.tick),
        "generation": int(org.generation),
        "status": org.status.value,
        "phase": org.phase.value,
        "state": {k: _f(v) for k, v in org.state.to_dict().items()},
        "metrics": {k: _f(v) for k, v in org.metrics.to_dict().items()},
        "counters": {k: int(v) for k, v in org.counters.items()},
        "sink_load": _f(org.sink_load),
        "unrepaired_integral": _f(org.unrepaired_integral),
        "events": list(last.events) if last is not None else [],
        "decision": org._decision,
        "confidence": _f(org._confidence),
        "n_genes": int(len(org.gene_ids)),
        "genome_fingerprint": org.genome_fingerprint(),
        "audit_head": org.chain.head,
        "chain_length": int(len(org.chain.records)),
        "triggers": {k: _f(v) for k, v in org.triggers.to_dict().items()},
    }


def _learner_block(agent: Any) -> Dict[str, Any]:
    eng = agent.engine
    cus: Optional[Dict[str, Any]] = None
    if agent.cusum is not None:
        mu0, k, h = agent.cusum
        cus = {"signal": agent.cusum_signal, "s": _f(agent.cusum_s), "h": _f(h),
               "mu0": _f(mu0), "k": _f(k), "fires": int(agent.cusum_fires)}
    return {
        "name": agent.name,
        "trial": int(agent.trial),
        "input_len": int(agent.input_len),
        "symbols": list(agent.symbols),
        "macro": int(eng.macro_size()),
        "micro": int(eng.size),
        "p_explore": _f(eng.p.p_explore),
        "last_action": eng.last_action,
        "last_explore": bool(eng.last_explore),
        "confidence": _f(eng.confidence()),
        "engine_counters": {k: int(v) for k, v in eng.counters.items()},
        "structural": bool(agent.structural),
        "frozen": bool(agent.frozen),
        "cusum": cus,
    }


def _topology_block(agent: Any) -> Dict[str, Any]:
    return {
        "routes": sorted(agent.routes),
        "peers": list(agent.peers),
        "bus_nodes": sorted(agent.bus.nodes) if agent.bus is not None else [],
        "on_bus": agent.bus is not None,
        "severed": int(len(agent.severed)),
        "reroutes": int(len(agent.reroutes)),
        "last_reroute": (dict(agent.reroutes[-1]) if agent.reroutes else None),
        "inbox": int(len(agent.inbox)),
        "credits_received": int(agent.credits_received),
    }


def _regulation_block(grn: Any) -> Dict[str, Any]:
    m = {k: _f(v) for k, v in grn.metrics().items()}
    return {
        "tick": int(grn.tick),
        "metrics": m,
        "cusum": {"s": _f(grn.cusum_s), "h": _f(grn.cusum_h), "mu0": _f(grn.cusum_mu0),
                  "k": _f(grn.cusum_k), "over": bool(grn.cusum_s > grn.cusum_h)},
        "active_genes": sorted(grn.expressed_last),
        "board": sorted(grn.board),
        "order": list(grn.order),
        "expressions": {k: int(v) for k, v in grn.expressions.items()},
        "injections": int(len(grn.injections)),
        "since_inject": int(grn.since_inject),
        "since_shift_est": int(grn.since_shift_est),
        "genes": [{"id": g.id, "name": g.name, "trigger": g.trigger, "action": g.action,
                   "kind": "rule" if g.is_rule else "regulator",
                   "dependencies": list(g.dependencies), "outputs": list(g.outputs),
                   "cluster": g.cluster}
                  for g in grn.genome.genes],
        "genome_version": int(grn.genome.version),
        "genome_fingerprint": grn.genome.fingerprint(),
    }


def capture_frame(target: Any, tick: int) -> Dict[str, Any]:
    """Snapshot *target* (``Organism`` | ``LCSAgent`` | ``GRN``) at bridge tick *tick*.

    The result is JSON-serialisable (``json.dumps(frame)`` succeeds) and validates against
    ``schemas/telemetry_frame.schema.json``. *tick* is the caller's clock (e.g. the
    bridge's trial counter); the organism's own tick is ``frame["organism"]["tick"]``.
    """
    org, agent, grn = _resolve(target)
    frame: Dict[str, Any] = {
        "schema": FRAME_SCHEMA,
        "tick": int(tick),
        "organism": _organism_block(org),
        "learner": _learner_block(agent) if agent is not None else None,
        "topology": _topology_block(agent) if agent is not None else None,
        "regulation": _regulation_block(grn) if grn is not None else None,
    }
    # convenience mirrors (deepest layer that defines each)
    if grn is not None:
        frame["cusum"] = _f(grn.cusum_s)
        frame["cusum_threshold"] = _f(grn.cusum_h)
        frame["error"] = _f(grn.err_ema)
        frame["active_genes"] = sorted(grn.expressed_last)
    elif agent is not None and agent.cusum is not None:
        frame["cusum"] = _f(agent.cusum_s)
        frame["cusum_threshold"] = _f(agent.cusum[2])
        frame["error"] = None
        frame["active_genes"] = []
    else:
        frame["cusum"] = None
        frame["cusum_threshold"] = None
        frame["error"] = None
        frame["active_genes"] = []
    frame["status"] = org.status.value
    return frame


def frame_json(frame: Dict[str, Any], indent: Optional[int] = None) -> str:
    """Canonical JSON text of a frame (sorted keys; numpy scalars coerced)."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")
    return json.dumps(frame, sort_keys=True, indent=indent, default=default)


def frame_digest(frame: Dict[str, Any]) -> List[str]:
    """Short human-readable lines for a chat surface (no interpretation, just values)."""
    o = frame["organism"]
    s = o["state"]
    lines = [f"tick {frame['tick']} · organism {o['name']} t={o['tick']} gen={o['generation']} "
             f"{o['status']} · noise={s['noise_rate']:.3f} H={s['entropy_bits']:.3f} "
             f"coh={s['coherence']:.3f}"]
    if frame["learner"] is not None:
        le = frame["learner"]
        lines.append(f"learner: trial {le['trial']} macro={le['macro']} micro={le['micro']} "
                     f"p_explore={le['p_explore']:.2f} last={le['last_action']}")
    if frame["regulation"] is not None:
        r = frame["regulation"]
        m = r["metrics"]
        lines.append(f"regulation: error={m['error']:.3f} acc={m['acc']:.3f} "
                     f"cusum={r['cusum']['s']:.2f}/{r['cusum']['h']:g} "
                     f"active={r['active_genes']} board={r['board']}")
    if frame["topology"] is not None:
        t = frame["topology"]
        lines.append(f"topology: routes={t['routes']} peers={t['peers']} severed={t['severed']} "
                     f"reroutes={t['reroutes']}")
    return lines


__all__ = ["FRAME_SCHEMA", "capture_frame", "frame_json", "frame_digest"]
