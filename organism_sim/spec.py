"""
organism_sim.spec — Data model: parameters, state, genome, telemetry
====================================================================

Pure dataclasses with JSON round-trip. Runtime behaviour lives in
``organism_sim.organism``; population behaviour in ``organism_sim.swarm``.

Every number here is a *tunable simulation parameter*. None of them is
claimed to be a physical constant or a measured hardware value.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Triggers:
    """Maintenance trigger points for one organism.

    repair_threshold   noise_rate above this → silent repair (no status change)
    noise_floor        ambient noise the environment injects every tick; must be
                       below ``repair_threshold`` or the agent repairs every tick
    entropy_floor_bits genome entropy below this → mutation (structural collapse)
    window             rolling window (ticks) for the unrepaired-noise integral
    unrepaired_integral Σ max(0, noise − repair_threshold) over window → mutation
    sink_capacity      how much displaced noise the sink holds before a flush
    trigger_deg        phase angle at which the periodic phase reset fires
    trigger_tol_deg    half-width of the reset window
    repair_gain        repair multiplies noise_rate by e^{−repair_gain}
    """

    repair_threshold: float = 0.45
    noise_floor: float = 0.40
    entropy_floor_bits: float = 3.0
    window: int = 5
    unrepaired_integral: float = 1.0
    sink_capacity: float = 10.0
    trigger_deg: float = 180.0
    trigger_tol_deg: float = 6.0
    repair_gain: float = 0.6

    def __post_init__(self):
        if self.repair_threshold <= self.noise_floor:
            raise ValueError(
                f"repair_threshold ({self.repair_threshold}) must exceed noise_floor "
                f"({self.noise_floor}); otherwise the agent repairs on idle noise every tick")
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.repair_gain <= 0:
            raise ValueError("repair_gain must be positive")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class State:
    """Agent-local state vector at one instant.

    coherence      ∈ [0, 1]; = ½ exactly when noise_rate == repair_threshold
    noise_rate     EMA of per-tick noise (≈ noise_floor when idle)
    entropy_bits   Shannon entropy of the genome's expression distribution
    phase_deg      internal oscillator phase ∈ [0, 360)
    efficiency     coherence · entropy_bits / max(noise_rate, EPS)
    """

    coherence: float = 1.0
    noise_rate: float = 0.0
    entropy_bits: float = 0.0
    phase_deg: float = 0.0
    efficiency: float = 0.0

    EPS = 1e-3

    def recompute_efficiency(self) -> float:
        self.efficiency = (self.coherence * self.entropy_bits) / max(self.noise_rate, self.EPS)
        return self.efficiency

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "State":
        return cls(**{k: float(d.get(k, getattr(cls, k, 0.0)))
                      for k in ("coherence", "noise_rate", "entropy_bits", "phase_deg",
                                "efficiency")})


@dataclass
class Metrics:
    """Derived per-tick metrics, each normalised to [0, 1]."""

    repair_headroom: float = 1.0     # 1 − sink_load / sink_capacity
    coherence: float = 1.0
    cycle_position: float = 0.0      # position within the oscillator period
    input_quality: float = 1.0       # SNR-like quality of the ingested signal
    decision_confidence: float = 0.0  # softmax probability of the chosen gene
    load_headroom: float = 1.0       # 1 − noise_rate / repair_threshold

    NAMES = ("repair_headroom", "coherence", "cycle_position", "input_quality",
             "decision_confidence", "load_headroom")

    def as_vector(self) -> Tuple[float, ...]:
        return tuple(getattr(self, n) for n in self.NAMES)

    def to_dict(self) -> Dict[str, float]:
        return dict(zip(self.NAMES, self.as_vector()))


# ─────────────────────────────────────────────────────────────────────────────
# Genome
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Gene:
    """One gene. ``phase_deg`` is the gene's preferred oscillator angle; decisions
    score a gene by ``expression · cos(phase − gene.phase_deg)``."""

    id: str
    name: str
    expression: float = 1.0
    trigger: str = "on_tick"
    action: str = ""
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    phase_deg: float = 0.0
    cluster: str = "default"
    condition: str = ""      # ternary pattern (0/1/#); non-empty → this gene is an LCS rule
                             # and ``action`` is DSL source, e.g. "(emit 1)"

    @property
    def is_rule(self) -> bool:
        return bool(self.condition)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Gene":
        return cls(
            id=str(d["id"]), name=str(d["name"]),
            expression=float(d.get("expression", 1.0)),
            trigger=str(d.get("trigger", "on_tick")), action=str(d.get("action", "")),
            dependencies=list(d.get("dependencies", [])), outputs=list(d.get("outputs", [])),
            phase_deg=float(d.get("phase_deg", 0.0)), cluster=str(d.get("cluster", "default")),
            condition=str(d.get("condition", "")),
        )


@dataclass
class Genome:
    genes: List[Gene] = field(default_factory=list)
    version: int = 1
    purpose: str = ""

    def expressions(self) -> List[float]:
        return [g.expression for g in self.genes]

    def rules(self) -> List["Gene"]:
        return [g for g in self.genes if g.is_rule]

    def phases(self) -> List[float]:
        return [g.phase_deg for g in self.genes]

    def fingerprint(self) -> str:
        payload = json.dumps([g.to_dict() for g in self.genes], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "purpose": self.purpose,
                "genes": [g.to_dict() for g in self.genes], "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(genes=[Gene.from_dict(g) for g in d.get("genes", [])],
                   version=int(d.get("version", 1)), purpose=str(d.get("purpose", "")))

    def __len__(self) -> int:
        return len(self.genes)


def uniform_phases(n: int, offset_deg: float = 0.0) -> List[float]:
    """Spread *n* genes evenly around the circle starting at *offset_deg*."""
    return [(offset_deg + 360.0 * i / n) % 360.0 for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Organism spec
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrganismSpec:
    name: str
    genome: Genome
    domain: str = "general"
    version: str = "0.1.0"
    triggers: Triggers = field(default_factory=Triggers)
    initial_state: State = field(default_factory=State)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "version": self.version,
                "triggers": self.triggers.to_dict(),
                "initial_state": self.initial_state.to_dict(),
                "genome": self.genome.to_dict(), "meta": dict(self.meta)}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrganismSpec":
        tr = d.get("triggers", {})
        return cls(
            name=str(d["name"]), genome=Genome.from_dict(d.get("genome", {})),
            domain=str(d.get("domain", "general")), version=str(d.get("version", "0.1.0")),
            triggers=Triggers(**tr) if tr else Triggers(),
            initial_state=State.from_dict(d.get("initial_state", {})),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "OrganismSpec":
        return cls.from_dict(json.loads(text))


# ─────────────────────────────────────────────────────────────────────────────
# .dna genome-file reader
# ─────────────────────────────────────────────────────────────────────────────
# Reads the legacy ``organism/*.dna`` files as plain ALife genomes. Numeric
# values in their META/DNA/METRICS sections are preserved in ``meta`` as raw
# data and are not interpreted.

_ORG_RE = re.compile(r"\borganism\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.I)
_SECTION_RE = re.compile(r"\b(meta|dna|metrics|genome)\s*\{", re.I)
_GENE_RE = re.compile(
    r"\bgene(?:\s*\[\s*(\d+)\s*\]\s*=\s*|\s+)([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.I)
_KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|[^,\n}]+)')


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _match_brace(text: str, open_idx: int) -> int:
    depth, i, in_str = 0, open_idx, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces in .dna source")


def _parse_value(raw: str) -> Any:
    raw = raw.strip().rstrip(",").strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("["):
        inner = raw[1:-1].strip()
        return [_parse_value(x) for x in inner.split(",")] if inner else []
    if raw in ("true", "false"):
        return raw == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _parse_kv(body: str) -> Dict[str, Any]:
    out, i = [], 0
    while i < len(body):
        if body[i] == "{":
            i = _match_brace(body, i) + 1
            continue
        out.append(body[i])
        i += 1
    return {k: _parse_value(v) for k, v in _KV_RE.findall("".join(out))}


def _sections(body: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    pos = 0
    while True:
        m = _SECTION_RE.search(body, pos)
        if not m:
            return found
        close = _match_brace(body, m.end() - 1)
        found[m.group(1).lower()] = body[m.end():close]
        pos = close + 1


def parse_dna(text: str) -> OrganismSpec:
    """Parse ``ORGANISM X { … GENE Y {…} }`` or ``organism x { genome { gene[001] = y {…} } }``."""
    src = _strip_comments(text)
    m = _ORG_RE.search(src)
    if not m:
        raise ValueError("no ORGANISM block found")
    name = m.group(1)
    body = src[m.end():_match_brace(src, m.end() - 1)]
    secs = _sections(body)
    meta = _parse_kv(secs.get("meta", ""))
    dna = _parse_kv(secs.get("dna", ""))
    metrics = _parse_kv(secs.get("metrics", ""))
    genome_body = secs.get("genome", body)

    genes: List[Gene] = []
    pos = 0
    while True:
        gm = _GENE_RE.search(genome_body, pos)
        if not gm:
            break
        close = _match_brace(genome_body, gm.end() - 1)
        kv = _parse_kv(genome_body[gm.end():close])
        default_id = f"G{int(gm.group(1)):03d}" if gm.group(1) else f"G{len(genes)}"
        genes.append(Gene(
            id=str(kv.get("id", default_id)), name=gm.group(2),
            expression=float(kv.get("expression", 1.0)),
            trigger=str(kv.get("trigger", "on_tick")), action=str(kv.get("action", "")),
            dependencies=[str(x) for x in kv.get("dependencies", [])],
            outputs=[str(x) for x in kv.get("outputs", [])],
            phase_deg=float(kv.get("phase_deg", 0.0)), cluster=str(kv.get("cluster", "default")),
            condition=str(kv.get("condition", "")),
        ))
        pos = close + 1
    if genes and not any(g.phase_deg for g in genes):
        for g, ph in zip(genes, uniform_phases(len(genes))):
            g.phase_deg = ph

    return OrganismSpec(
        name=name, genome=Genome(genes=genes, purpose=str(dna.get("purpose", ""))),
        domain=str(meta.get("domain", "general")), version=str(meta.get("version", "0.1.0")),
        meta={"raw_meta": meta, "raw_dna": dna, "raw_metrics": metrics},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle + telemetry
# ─────────────────────────────────────────────────────────────────────────────


class Phase(Enum):
    """The five steps of one tick, in order."""

    INGEST = "ingest"
    UPDATE = "update"
    DECIDE = "decide"
    MAINTAIN = "maintain"
    LOG = "log"

    @classmethod
    def sequence(cls) -> Tuple["Phase", ...]:
        return (cls.INGEST, cls.UPDATE, cls.DECIDE, cls.MAINTAIN, cls.LOG)


class Status(Enum):
    STABLE = "stable"       # no mutation this tick, noise under threshold after repair
    DEGRADED = "degraded"   # mutation fired and resolved, or noise still over threshold
    CRITICAL = "critical"   # mutation fired and the condition persists
    IDLE = "idle"           # never ticked


@dataclass
class TelemetryRecord:
    """One tick's telemetry row. ``prev_hash``/``hash`` form the audit chain."""

    tick: int
    timestamp: float
    state: Dict[str, float]
    metrics: Dict[str, float]
    decision: Optional[str]
    confidence: float
    events: List[str]
    generation: int
    sink_load: float
    prev_hash: str
    hash: str = ""

    def payload(self) -> str:
        d = asdict(self)
        d.pop("hash", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    name: str
    ticks: int
    generation: int
    status: str
    final_state: Dict[str, float]
    final_metrics: Dict[str, float]
    mean_entropy: float
    min_entropy: float
    mean_noise: float
    max_noise: float
    coherence_std: float
    repairs: int
    mutations: int
    mutations_entropy: int
    mutations_noise: int
    phase_resets: int
    sink_flushes: int
    ticks_stable: int
    audit_chain_valid: bool
    audit_head: str
    genome_fingerprint: str
    triggers: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


__all__ = ["Triggers", "State", "Metrics", "Gene", "Genome", "OrganismSpec", "uniform_phases",
           "parse_dna", "Phase", "Status", "TelemetryRecord", "RunSummary"]
