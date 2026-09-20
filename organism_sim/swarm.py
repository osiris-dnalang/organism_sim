"""
organism_sim.swarm — Multi-agent event loop
===========================================

A ``Population`` of N organisms sharing a ``MessageBus``. Each tick:

    1. every organism steps (independently, own environment stream);
    2. every organism publishes a ``Message`` (its state + decision) to the bus;
    3. every organism reads its neighbours' messages and derives next-tick
       coupling: a phase pull toward the neighbour mean (Kuramoto-style) and,
       if its own entropy is low and a neighbour's is high, a fractional
       horizontal transfer of that neighbour's expression profile.

Population metrics are recorded per tick into ``PopulationLog`` and exported as
JSON or CSV for plotting:

    mean_entropy, min_entropy, mean_noise, mean_coherence,
    sync_r  (Kuramoto order parameter of the agents' phases, 0..1),
    mutations_this_tick, repairs_this_tick, cumulative_mutations,
    stable_fraction, noise_floor (current environment floor, incl. ramp)

Everything is seeded; two runs with the same seed produce identical logs.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .alpha import build_alpha_spec
from .noise import Environment
from .organism import Organism, wrap_deg
from .spec import OrganismSpec, Triggers


@dataclass
class Message:
    sender: str
    tick: int
    phase_deg: float
    entropy_bits: float
    noise_rate: float
    coherence: float
    decision: Optional[str]
    expression: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("expression")
        return d


class MessageBus:
    """Tick-scoped broadcast bus. Messages published at tick t are readable
    until the bus is cleared at the start of tick t+1."""

    def __init__(self):
        self._current: Dict[str, Message] = {}
        self.published = 0

    def clear(self) -> None:
        self._current = {}

    def publish(self, msg: Message) -> None:
        self._current[msg.sender] = msg
        self.published += 1

    def read(self, exclude: Optional[str] = None) -> List[Message]:
        return [m for k, m in self._current.items() if k != exclude]

    def __len__(self) -> int:
        return len(self._current)


@dataclass
class Coupling:
    """Neighbour interaction strengths."""

    phase_gain: float = 0.15          # fraction of the phase error pulled per tick
    transfer_rate: float = 0.10       # fraction of neighbour profile copied
    transfer_entropy_gap: float = 0.5  # min (neighbour − self) entropy to trigger transfer
    topology: str = "all"             # "all" or "ring"


@dataclass
class TickMetrics:
    tick: int
    mean_entropy: float
    min_entropy: float
    mean_noise: float
    mean_coherence: float
    sync_r: float
    mutations_this_tick: int
    repairs_this_tick: int
    cumulative_mutations: int
    stable_fraction: float
    noise_floor: float
    transfers_this_tick: int

    FIELDS = ("tick", "mean_entropy", "min_entropy", "mean_noise", "mean_coherence", "sync_r",
              "mutations_this_tick", "repairs_this_tick", "cumulative_mutations",
              "stable_fraction", "noise_floor", "transfers_this_tick")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PopulationLog:
    def __init__(self):
        self.rows: List[TickMetrics] = []

    def append(self, row: TickMetrics) -> None:
        self.rows.append(row)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([r.to_dict() for r in self.rows], indent=indent)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(TickMetrics.FIELDS))
        w.writeheader()
        for r in self.rows:
            w.writerow(r.to_dict())
        return buf.getvalue()

    def column(self, name: str) -> List[float]:
        return [getattr(r, name) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


def sync_order_parameter(phases_deg: Sequence[float]) -> float:
    """Kuramoto r = |mean(e^{iθ})| ∈ [0, 1]; 1 = all phases equal."""
    if not phases_deg:
        return 0.0
    th = np.radians(np.asarray(phases_deg, dtype=float))
    return float(abs(np.mean(np.exp(1j * th))))


class Population:
    """N organisms + bus + per-tick metric log."""

    def __init__(self, n: int = 8, seed: int = 0, triggers: Optional[Triggers] = None,
                 coupling: Optional[Coupling] = None, noise_floor: Optional[float] = None,
                 burst_prob: float = 0.05, ramp: float = 0.0,
                 freq_spread_deg: float = 1.5, specs: Optional[List[OrganismSpec]] = None):
        self.seed = seed
        self.freq_spread_deg = freq_spread_deg
        self.coupling = coupling or Coupling()
        self.bus = MessageBus()
        self.log = PopulationLog()
        self.tick = 0
        self.transfers = 0
        triggers = triggers or Triggers()
        floor = triggers.noise_floor if noise_floor is None else noise_floor
        if floor >= triggers.repair_threshold:
            raise ValueError("noise_floor must be below repair_threshold")
        specs = specs or [build_alpha_spec(triggers) for _ in range(n)]
        rng = np.random.default_rng(seed)
        self.organisms: List[Organism] = []
        for i, spec in enumerate(specs):
            spec.name = f"{spec.name}-{i:02d}"
            env = Environment(n_genes=len(spec.genome), seed=seed * 1000 + i, noise_floor=floor,
                              burst_prob=burst_prob, ramp=ramp)
            org = Organism(spec, env=env, seed=seed * 1000 + i)
            # Heterogeneous agents: random initial phase and natural frequency, so
            # synchronisation is something coupling has to produce, not a given.
            org.state.phase_deg = float(rng.uniform(0.0, 360.0))
            org.OMEGA_DEG = float(Organism.OMEGA_DEG + rng.normal(0.0, self.freq_spread_deg))
            self.organisms.append(org)
        self._last_mutations = [0] * len(self.organisms)
        self._last_repairs = [0] * len(self.organisms)

    # ── loop ─────────────────────────────────────────────────────────────────

    def step(self) -> TickMetrics:
        self.tick += 1
        self.bus.clear()
        for org in self.organisms:
            org.step()
        for org in self.organisms:
            s = org.state
            self.bus.publish(Message(
                sender=org.spec.name, tick=self.tick, phase_deg=s.phase_deg,
                entropy_bits=s.entropy_bits, noise_rate=s.noise_rate, coherence=s.coherence,
                decision=org._decision, expression=org.expression.tolist()))
        transfers = self._apply_coupling()
        return self._record(transfers)

    def run(self, ticks: int = 100) -> PopulationLog:
        for _ in range(ticks):
            self.step()
        return self.log

    def _neighbours(self, i: int) -> List[Message]:
        me = self.organisms[i].spec.name
        msgs = self.bus.read(exclude=me)
        if self.coupling.topology == "ring" and len(self.organisms) > 2:
            n = len(self.organisms)
            names = {self.organisms[(i - 1) % n].spec.name, self.organisms[(i + 1) % n].spec.name}
            msgs = [m for m in msgs if m.sender in names]
        return msgs

    def _apply_coupling(self) -> int:
        transfers = 0
        c = self.coupling
        for i, org in enumerate(self.organisms):
            msgs = self._neighbours(i)
            if not msgs:
                continue
            # Phase pull toward the circular mean of neighbours
            th = np.radians([m.phase_deg for m in msgs])
            mean_deg = math.degrees(math.atan2(np.sin(th).mean(), np.cos(th).mean())) % 360.0
            org.phase_coupling_deg = c.phase_gain * wrap_deg(mean_deg - org.state.phase_deg)
            # Horizontal transfer from the highest-entropy neighbour when we are low
            best = max(msgs, key=lambda m: m.entropy_bits)
            if best.entropy_bits - org.state.entropy_bits >= c.transfer_entropy_gap:
                donor = np.asarray(best.expression, dtype=float)
                if donor.shape == org.expression.shape:
                    org.expression = np.clip(
                        (1 - c.transfer_rate) * org.expression + c.transfer_rate * donor,
                        org.EXPR_FLOOR, 1.0)
                    org._recompute_entropy()
                    transfers += 1
        self.transfers += transfers
        return transfers

    def _record(self, transfers: int) -> TickMetrics:
        orgs = self.organisms
        ent = [o.state.entropy_bits for o in orgs]
        muts = [o.counters["mutate"] for o in orgs]
        reps = [o.counters["repair"] for o in orgs]
        row = TickMetrics(
            tick=self.tick,
            mean_entropy=float(np.mean(ent)), min_entropy=float(np.min(ent)),
            mean_noise=float(np.mean([o.state.noise_rate for o in orgs])),
            mean_coherence=float(np.mean([o.state.coherence for o in orgs])),
            sync_r=sync_order_parameter([o.state.phase_deg for o in orgs]),
            mutations_this_tick=int(sum(m - prev for m, prev in zip(muts, self._last_mutations))),
            repairs_this_tick=int(sum(r - prev for r, prev in zip(reps, self._last_repairs))),
            cumulative_mutations=int(sum(muts)),
            stable_fraction=float(np.mean([o.status.value == "stable" for o in orgs])),
            noise_floor=float(orgs[0].env.current_floor),
            transfers_this_tick=transfers,
        )
        self._last_mutations, self._last_repairs = muts, reps
        self.log.append(row)
        return row

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        rows = self.log.rows
        return {
            "n": len(self.organisms), "ticks": self.tick, "seed": self.seed,
            "coupling": asdict(self.coupling),
            "final_mean_entropy": rows[-1].mean_entropy if rows else None,
            "final_sync_r": rows[-1].sync_r if rows else None,
            "mean_sync_r": float(np.mean(self.log.column("sync_r"))) if rows else None,
            "cumulative_mutations": rows[-1].cumulative_mutations if rows else 0,
            "mutations_per_agent_per_100": (
                100.0 * rows[-1].cumulative_mutations / (len(self.organisms) * self.tick)
                if rows else 0.0),
            "total_transfers": self.transfers,
            "messages_published": self.bus.published,
            "audit_chains_valid": all(o.chain.verify() for o in self.organisms),
            "final_noise_floor": rows[-1].noise_floor if rows else None,
        }


__all__ = ["Message", "MessageBus", "Coupling", "TickMetrics", "PopulationLog", "Population",
           "sync_order_parameter"]
