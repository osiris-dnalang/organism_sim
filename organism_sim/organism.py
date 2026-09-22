"""
organism_sim.organism — Single-agent state machine
==================================================

One tick = five phases:

    1. INGEST    environment → ``EnvSample``
    2. UPDATE    noise_rate ← EMA(noise + prediction error); coherence; phase advance
    3. DECIDE    score genes by expression · (forward + w·cached prediction) · resonance;
                 softmax → decision + confidence
    4. MAINTAIN  noise_rate > repair_threshold → silent repair (noise · e^{−repair_gain},
                 removed amount → sink, cache reset). Unrepaired excess erodes
                 off-resonance genes. Mutation only on structural collapse:
                 entropy_bits < entropy_floor_bits, or the rolling integral of
                 unrepaired excess over ``window`` ticks > ``unrepaired_integral``.
    5. LOG       telemetry row → hash chain

Update rules (all NumPy, all seeded):

    noise_{t+1} = (1 − κ)·noise_t + κ·(env_noise + η·‖prediction error‖)
    coherence   = exp(−noise · ln2 / repair_threshold)     (½ at the threshold)
    entropy     = H₂(expression / Σ expression)  [bits]
    phase_{t+1} = phase_t + ω + k·wrap(env_phase − phase_t) + coupling  (mod 360)
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from .audit import AuditChain
from .noise import Environment, EnvSample
from .spec import (
    Metrics,
    OrganismSpec,
    Phase,
    RunSummary,
    State,
    Status,
    TelemetryRecord,
    Triggers,
)

_LN2 = math.log(2.0)


def wrap_deg(x: float) -> float:
    """Wrap an angle difference into (−180, 180]."""
    return ((x + 180.0) % 360.0) - 180.0


def shannon_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


class Processor:
    """Optional plug-in that replaces the organism's decision/genome dynamics.

    strengths()          → 1-D array; the organism's entropy is computed over it
    decide(org)          → (decision label, confidence) during DECIDE
    erode(org, excess)   → react to unrepaired noise
    repair(org)          → silent compaction on repair
    mutate(org, reason)  → structural mutation
    """

    def strengths(self) -> np.ndarray: ...
    def decide(self, org: "Organism"): ...
    def erode(self, org: "Organism", excess: float) -> None: ...
    def repair(self, org: "Organism") -> None: ...
    def mutate(self, org: "Organism", reason: str) -> None: ...


class Organism:
    """Runtime for one ``OrganismSpec``. ``step()`` runs a tick; ``run(n)`` runs n."""

    # Tunable dynamics (class attributes so experiments can override per instance)
    KAPPA_TRACKING = 0.50     # EMA rate for noise_rate
    ETA_PREDICTION = 0.10     # weight of prediction error in noise_rate
    EROSION = 0.30            # entropy erosion per unit of unrepaired excess
    OMEGA_DEG = 11.0          # free-running phase advance per tick
    K_ENV_COUPLING = 0.10     # coupling of phase toward the environment phase
    PREDICTION_WEIGHT = 0.9   # weight of the cached prediction in decisions
    EXPR_USE_GAIN = 0.06      # use-it-or-lose-it drift on expression
    EXPR_DECAY = 0.02
    EXPR_FLOOR = 0.01
    MUTATION_RATE = 0.08
    PERIOD_TICKS = 47         # cycle_position metric period

    def __init__(self, spec: OrganismSpec, env: Optional[Environment] = None,
                 seed: Optional[int] = 0, audit_secret: Optional[bytes] = None,
                 audit_window: Optional[int] = None):
        self.spec = spec
        self.triggers: Triggers = spec.triggers
        self.rng = np.random.default_rng(seed)
        n = len(spec.genome)
        if n == 0:
            raise ValueError("organism needs at least one gene")
        self.env = env or Environment(n_genes=n, seed=seed, noise_floor=self.triggers.noise_floor)

        self.expression = np.array(spec.genome.expressions(), dtype=float)
        self.gene_phase = np.array(spec.genome.phases(), dtype=float)
        self.gene_ids = [g.id for g in spec.genome.genes]

        self.state = State(coherence=spec.initial_state.coherence,
                           noise_rate=spec.initial_state.noise_rate,
                           phase_deg=spec.initial_state.phase_deg)
        self.metrics = Metrics()
        self.tick = 0
        self.generation = 0
        self.sink_load = 0.0
        self.chain = AuditChain(audit_secret, max_records=audit_window)
        self.phase: Phase = Phase.LOG
        self.phase_trace: List[Phase] = []
        self.status = Status.IDLE

        self.phase_coupling_deg = 0.0   # set externally by a population bus each tick
        self.external_noise = 0.0       # injected by a bus/node before the next tick
        self.on_repair = None           # optional callbacks for wrappers
        self.on_mutate = None
        self.processor = None           # optional Processor (e.g. lcs.RuleEngine adapter)
        self._cache: Optional[np.ndarray] = None
        self._sample: Optional[EnvSample] = None
        self._decision: Optional[str] = None
        self._confidence = 0.0
        self._events: List[str] = []
        self._hist: Dict[str, List[float]] = {"entropy": [], "noise": [], "coherence": []}
        self._excess: deque = deque(maxlen=self.triggers.window)
        self.counters = {"repair": 0, "mutate": 0, "mutate_entropy": 0, "mutate_noise": 0,
                         "reset": 0, "flush": 0, "stable": 0}
        self._recompute_entropy()
        self.state.recompute_efficiency()

    # ── tick ─────────────────────────────────────────────────────────────────

    def step(self) -> TelemetryRecord:
        self.tick += 1
        self._events = []
        self._ingest()
        self._update()
        self._decide()
        self._maintain()
        return self._log()

    def inject_noise(self, amount: float) -> None:
        """Add *amount* to the next tick's EMA noise input (e.g. message pressure)."""
        self.external_noise += max(0.0, float(amount))

    def run(self, ticks: int = 100) -> RunSummary:
        for _ in range(ticks):
            self.step()
        return self.summary()

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_trace.append(phase)

    def _ingest(self) -> None:
        self._enter(Phase.INGEST)
        self._sample = self.env.sample(self.tick, self.gene_phase)

    def _update(self) -> None:
        self._enter(Phase.UPDATE)
        s, tr = self.state, self.triggers
        assert self._sample is not None
        pred_err = 0.0
        if self._cache is not None:
            pred_err = float(np.linalg.norm(self._sample.signal - self._cache)) / math.sqrt(
                self._cache.size)
        s.noise_rate = (1.0 - self.KAPPA_TRACKING) * s.noise_rate + self.KAPPA_TRACKING * (
            self._sample.noise + self.ETA_PREDICTION * pred_err + self.external_noise)
        self.external_noise = 0.0
        s.coherence = math.exp(-s.noise_rate * _LN2 / tr.repair_threshold)

        s.phase_deg = (s.phase_deg + self.OMEGA_DEG
                       + self.K_ENV_COUPLING * wrap_deg(self.env.phase_deg - s.phase_deg)
                       + self.phase_coupling_deg) % 360.0
        self.phase_coupling_deg = 0.0

        # Periodic phase reset: mild noise reduction, phase snaps to 0
        if abs(wrap_deg(s.phase_deg - tr.trigger_deg)) < tr.trigger_tol_deg:
            s.noise_rate *= math.exp(-tr.repair_gain / 2.0)
            s.phase_deg = 0.0
            self.counters["reset"] += 1
            self._events.append("phase_reset")

        self._recompute_entropy()
        s.recompute_efficiency()
        self._update_metrics()

    def _recompute_entropy(self) -> None:
        if self.processor is not None:
            self.expression = np.asarray(self.processor.strengths(), dtype=float)
        self.state.entropy_bits = shannon_bits(self.expression)

    def _update_metrics(self) -> None:
        s, tr, m = self.state, self.triggers, self.metrics
        m.repair_headroom = max(0.0, 1.0 - self.sink_load / tr.sink_capacity)
        m.coherence = s.coherence
        m.cycle_position = (self.tick % self.PERIOD_TICKS) / float(self.PERIOD_TICKS)
        m.input_quality = self._sample.signal_quality if self._sample else 0.0
        m.decision_confidence = self._confidence
        m.load_headroom = max(0.0, min(1.0, 1.0 - s.noise_rate / tr.repair_threshold))

    def _decide(self) -> None:
        self._enter(Phase.DECIDE)
        assert self._sample is not None
        if self.processor is not None:
            self._decision, self._confidence = self.processor.decide(self)
            self.metrics.decision_confidence = self._confidence
            return
        forward = self._sample.signal
        cached = self._cache if self._cache is not None else np.zeros_like(forward)
        resonance = np.cos(np.radians(self.state.phase_deg - self.gene_phase))
        score = self.expression * (forward + self.PREDICTION_WEIGHT * cached) * (
            0.5 + 0.5 * resonance)
        z = score - score.max()
        p = np.exp(z / 0.25)
        p /= p.sum()
        idx = int(np.argmax(p))
        self._decision = self.gene_ids[idx]
        self._confidence = float(p[idx])
        self.metrics.decision_confidence = self._confidence

        self.expression *= (1.0 - self.EXPR_DECAY)
        self.expression[idx] = min(1.0, self.expression[idx] + self.EXPR_USE_GAIN)
        self._cache = 0.7 * forward + 0.3 * resonance

    def _maintain(self) -> None:
        self._enter(Phase.MAINTAIN)
        s, tr = self.state, self.triggers

        if s.noise_rate > tr.repair_threshold:
            self._repair()

        excess = max(0.0, s.noise_rate - tr.repair_threshold)
        self._excess.append(excess)
        if excess > 0.0:
            if self.processor is not None:
                self.processor.erode(self, excess)
            else:
                resonance = np.cos(np.radians(s.phase_deg - self.gene_phase))
                self.expression *= 1.0 - self.EROSION * excess * (1.0 - resonance) / 2.0
                self.expression = np.clip(self.expression, self.EXPR_FLOOR, 1.0)
        self._recompute_entropy()

        reason = None
        if s.entropy_bits < tr.entropy_floor_bits:
            reason = "entropy"
        elif self.unrepaired_integral > tr.unrepaired_integral:
            reason = "noise"
        if reason:
            self._mutate(reason)
            self._recompute_entropy()
            if reason == "noise":
                self._excess.clear()

        s.coherence = math.exp(-s.noise_rate * _LN2 / tr.repair_threshold)
        s.recompute_efficiency()
        self._update_metrics()

        over = s.noise_rate > tr.repair_threshold
        if reason is None and not over:
            self.status = Status.STABLE
            self.counters["stable"] += 1
        elif reason and (s.entropy_bits < tr.entropy_floor_bits or over):
            self.status = Status.CRITICAL
        else:
            self.status = Status.DEGRADED

    def _repair(self) -> None:
        """Silent repair: noise_rate · e^{−repair_gain}; removed load goes to the sink;
        prediction cache is reset so stale error does not re-enter.

        With a processor, ``noise_rate`` is a *measured* error rate that compaction
        cannot lower, so it is left untouched: sustained error then accumulates as
        unrepaired excess and reaches the mutation trigger."""
        s, tr = self.state, self.triggers
        before = s.noise_rate
        if self.processor is None:
            s.noise_rate = max(0.0, before * math.exp(-tr.repair_gain))
            self.sink_load += before - s.noise_rate
        self._cache = None
        self.counters["repair"] += 1
        self._events.append("repair")
        if self.processor is not None:
            self.processor.repair(self)
        if self.on_repair is not None:
            self.on_repair(self)
        if self.sink_load > tr.sink_capacity:
            self.sink_load = 0.0
            self.counters["flush"] += 1
            self._events.append("sink_flush")

    def _mutate(self, reason: str = "entropy") -> None:
        """Gaussian perturbation + regression toward the mean (re-spreads a collapsed
        distribution, restoring entropy). With a processor: its structural mutation."""
        if self.processor is not None:
            self.processor.mutate(self, reason)
        else:
            e = self.expression + self.rng.normal(0.0, self.MUTATION_RATE,
                                                  size=self.expression.shape)
            e = 0.5 * e + 0.5 * e.mean()
            self.expression = np.clip(e, self.EXPR_FLOOR, 1.0)
        self.generation += 1
        self.counters["mutate"] += 1
        self.counters["mutate_" + reason] += 1
        self._events.append("mutation")
        self._events.append("mutation_reason=" + reason)
        if self.on_mutate is not None:
            self.on_mutate(self, reason)

    def _log(self) -> TelemetryRecord:
        self._enter(Phase.LOG)
        s = self.state
        self._hist["entropy"].append(s.entropy_bits)
        self._hist["noise"].append(s.noise_rate)
        self._hist["coherence"].append(s.coherence)
        rec = TelemetryRecord(
            tick=self.tick, timestamp=time.time(), state=s.to_dict(),
            metrics=self.metrics.to_dict(), decision=self._decision,
            confidence=self._confidence, events=list(self._events),
            generation=self.generation, sink_load=self.sink_load, prev_hash="",
        )
        return self.chain.append(rec)

    # ── reporting ────────────────────────────────────────────────────────────

    @property
    def unrepaired_integral(self) -> float:
        return float(sum(self._excess))

    def genome_fingerprint(self) -> str:
        return hashlib.sha256(np.round(self.expression, 6).tobytes()).hexdigest()[:16]

    def history(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._hist.items()}

    def summary(self) -> RunSummary:
        h = self._hist
        ent = np.array(h["entropy"] or [self.state.entropy_bits])
        noi = np.array(h["noise"] or [self.state.noise_rate])
        coh = np.array(h["coherence"] or [self.state.coherence])
        c = self.counters
        return RunSummary(
            name=self.spec.name, ticks=self.tick, generation=self.generation,
            status=self.status.value, final_state=self.state.to_dict(),
            final_metrics=self.metrics.to_dict(),
            mean_entropy=float(ent.mean()), min_entropy=float(ent.min()),
            mean_noise=float(noi.mean()), max_noise=float(noi.max()),
            coherence_std=float(coh.std()),
            repairs=c["repair"], mutations=c["mutate"], mutations_entropy=c["mutate_entropy"],
            mutations_noise=c["mutate_noise"], phase_resets=c["reset"], sink_flushes=c["flush"],
            ticks_stable=c["stable"], audit_chain_valid=self.chain.verify(),
            audit_head=self.chain.head, genome_fingerprint=self.genome_fingerprint(),
            triggers=self.triggers.to_dict(),
        )

    def telemetry(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.chain.records]

    def __repr__(self) -> str:
        s = self.state
        return (f"Organism({self.spec.name!r}, tick={self.tick}, gen={self.generation}, "
                f"coh={s.coherence:.3f} noise={s.noise_rate:.3f} H={s.entropy_bits:.3f} "
                f"phase={s.phase_deg:.1f}°, {self.status.value})")


__all__ = ["Organism", "Processor", "shannon_bits", "wrap_deg"]
