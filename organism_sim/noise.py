"""
organism_sim.noise — Simulated environment
==========================================

The *only* noise source in this package. Each tick yields a signal vector (one
component per gene) plus a scalar ``noise`` sample sitting on ``noise_floor``
with Gaussian jitter and occasional multiplicative bursts. Seeded and
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .spec import Triggers


@dataclass
class EnvSample:
    tick: int
    signal: np.ndarray
    noise: float
    burst: bool
    signal_quality: float

    def to_dict(self) -> Dict[str, float]:
        return {"tick": self.tick, "noise": self.noise, "burst": float(self.burst),
                "signal_quality": self.signal_quality}


@dataclass
class Environment:
    """Seeded stochastic environment.

    noise_floor   baseline noise injected every tick
    jitter        Gaussian σ around the floor
    burst_prob    per-tick probability of a burst
    burst_scale   multiplicative amplitude of a burst
    drift_deg     how far the driving signal's phase advances per tick
    ramp          optional per-tick additive increase of the floor (noise ramp)
    """

    n_genes: int
    seed: Optional[int] = 0
    noise_floor: float = Triggers.noise_floor
    jitter: float = 0.05
    burst_prob: float = 0.05
    burst_scale: float = 5.0
    drift_deg: float = 7.0
    ramp: float = 0.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _phase: float = field(default=0.0, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._phase = float(self._rng.uniform(0.0, 360.0))  # per-environment offset

    @property
    def current_floor(self) -> float:
        return self.noise_floor + self.ramp * self._tick

    def sample(self, tick: int, gene_phases_deg: np.ndarray) -> EnvSample:
        self._tick = tick
        self._phase = (self._phase + self.drift_deg) % 360.0
        clean = np.cos(np.radians(gene_phases_deg - self._phase))
        burst = bool(self._rng.random() < self.burst_prob)
        noise = max(0.0, self.current_floor + self._rng.normal(0.0, self.jitter))
        if burst:
            noise *= self.burst_scale
        sigma = 0.15 * (1.0 + noise)
        noisy = clean + self._rng.normal(0.0, sigma, size=clean.shape)
        p_signal = float(np.dot(clean, clean))
        p_noise = float(np.dot(noisy - clean, noisy - clean)) + 1e-9
        quality = 1.0 / (1.0 + p_noise / max(p_signal, 1e-9))
        return EnvSample(tick=tick, signal=noisy, noise=noise, burst=burst,
                         signal_quality=quality)

    @property
    def phase_deg(self) -> float:
        return self._phase


def quiet_environment(n_genes: int, seed: Optional[int] = 0) -> Environment:
    return Environment(n_genes=n_genes, seed=seed, noise_floor=0.05, jitter=0.01,
                       burst_prob=0.0)


def hostile_environment(n_genes: int, seed: Optional[int] = 0) -> Environment:
    return Environment(n_genes=n_genes, seed=seed, noise_floor=0.9, jitter=0.2,
                       burst_prob=0.25, burst_scale=5.0)


__all__ = ["EnvSample", "Environment", "quiet_environment", "hostile_environment"]
