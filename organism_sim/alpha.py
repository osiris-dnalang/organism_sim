"""
organism_sim.alpha — Reference 24-gene organism
===============================================

Four clusters (perception, maintenance, decision, repair), genes spread
uniformly around the circle from ``resonance_offset_deg``.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from .noise import Environment
from .organism import Organism
from .spec import Gene, Genome, OrganismSpec, RunSummary, State, Triggers, uniform_phases

_CLUSTERS = (
    ("perception", ("SignalIngest", "NoiseEstimator", "PhaseDetector",
                    "SpectralBinner", "SNRGate", "InputCoupler")),
    ("maintenance", ("PhaseLock", "FrameHold", "LatticeIndex",
                     "SeedRegister", "CoherenceIntegrator", "RateRegulator")),
    ("decision", ("ForwardScorer", "Predictor", "PredictionCache",
                  "SoftmaxSelector", "ConfidenceMonitor", "Router")),
    ("repair", ("NoiseSentinel", "Repairer", "NoiseSink",
                "SinkFlush", "EntropyMonitor", "GenomeMutator")),
)


def build_alpha_spec(triggers: Optional[Triggers] = None,
                     resonance_offset_deg: float = 0.0) -> OrganismSpec:
    names = [(c, n) for c, ns in _CLUSTERS for n in ns]
    phases = uniform_phases(len(names), resonance_offset_deg)
    genes = [Gene(id=f"G{i}", name=n, expression=1.0, action=f"{c}:{n}", phase_deg=ph, cluster=c)
             for i, ((c, n), ph) in enumerate(zip(names, phases))]
    return OrganismSpec(
        name="ALPHA", genome=Genome(genes=genes, purpose="reference 24-gene agent"),
        domain="reference", version="0.2.0", triggers=triggers or Triggers(),
        initial_state=State(coherence=1.0, noise_rate=0.0),
        meta={"resonance_offset_deg": resonance_offset_deg, "clusters": len(_CLUSTERS)},
    )


def build_alpha(seed: int = 0, noise_floor: float = Triggers.noise_floor,
                burst_prob: float = 0.05, triggers: Optional[Triggers] = None,
                repair_threshold: Optional[float] = None,
                audit_secret: Optional[bytes] = None, name: Optional[str] = None) -> Organism:
    if triggers is None:
        kw = {"noise_floor": noise_floor}
        if repair_threshold is not None:
            kw["repair_threshold"] = repair_threshold
        triggers = Triggers(**kw)
    spec = build_alpha_spec(triggers)
    if name:
        spec.name = name
    env = Environment(n_genes=len(spec.genome), seed=seed, noise_floor=triggers.noise_floor,
                      burst_prob=burst_prob)
    return Organism(spec, env=env, seed=seed, audit_secret=audit_secret)


def run_alpha(ticks: int = 100, seed: int = 0, noise_floor: float = Triggers.noise_floor,
              burst_prob: float = 0.05, repair_threshold: Optional[float] = None,
              verbose: bool = False) -> RunSummary:
    org = build_alpha(seed=seed, noise_floor=noise_floor, burst_prob=burst_prob,
                      repair_threshold=repair_threshold)
    for _ in range(ticks):
        rec = org.step()
        if verbose and (rec.events or rec.tick % 10 == 0):
            print(format_tick(rec), file=sys.stderr)
    return org.summary()


def format_tick(rec) -> str:
    s = rec.state
    return (f"  t{rec.tick:03d} coh={s['coherence']:.3f} noise={s['noise_rate']:.3f} "
            f"H={s['entropy_bits']:.3f} phase={s['phase_deg']:6.1f}° →{rec.decision:<4} "
            f"{' '.join(rec.events)}")


def format_summary(r: RunSummary) -> str:
    s, m, t = r.final_state, r.final_metrics, r.triggers
    return "\n".join([
        f"═══ {r.name} — run summary ═══",
        f"ticks={r.ticks} generation={r.generation} status={r.status.upper()}",
        f"final   coherence={s['coherence']:.4f} noise={s['noise_rate']:.4f} "
        f"entropy={s['entropy_bits']:.4f} bits phase={s['phase_deg']:.2f}° "
        f"efficiency={s['efficiency']:.3f}",
        f"metrics repair_headroom={m['repair_headroom']:.3f} input_quality={m['input_quality']:.3f} "
        f"confidence={m['decision_confidence']:.3f} load_headroom={m['load_headroom']:.3f}",
        f"entropy mean/min={r.mean_entropy:.3f}/{r.min_entropy:.3f}  "
        f"noise mean/max={r.mean_noise:.3f}/{r.max_noise:.3f}  σ(coh)={r.coherence_std:.4f}",
        f"events  repairs={r.repairs} (silent) mutations={r.mutations} "
        f"[entropy={r.mutations_entropy} noise={r.mutations_noise}] "
        f"resets={r.phase_resets} flushes={r.sink_flushes} stable={r.ticks_stable}/{r.ticks}",
        f"trigger repair>{t['repair_threshold']} floor={t['noise_floor']} "
        f"entropy<{t['entropy_floor_bits']} window={int(t['window'])} ∫excess>{t['unrepaired_integral']}",
        f"audit   valid={r.audit_chain_valid} head={r.audit_head[:16]}… "
        f"genome={r.genome_fingerprint}",
    ])


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    r = run_alpha(verbose="-v" in args)
    print(json.dumps(r.to_dict(), indent=2) if "--json" in args else format_summary(r))
    return 0 if r.audit_chain_valid else 1


if __name__ == "__main__":
    sys.exit(main())
