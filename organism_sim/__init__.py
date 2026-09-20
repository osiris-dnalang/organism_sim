"""
organism_sim — ALife simulation of autonomous state-machine agents
==================================================================

Two layers:

* substrate — ``Organism`` state machines (noise EMA, silent repair, mutation on
  structural collapse, hash-chained telemetry) and ``Population`` (phase
  coupling + horizontal transfer over a ``MessageBus``, JSON/CSV metrics);
* computation — a Learning Classifier System: ternary-condition rules with
  action bodies in a closed DSL (``rules``), an XCS-style ``RuleEngine``
  (``lcs``), ``LCSAgent`` binding an engine to an organism and a ``RoutedBus``
  (``agent``, ``bus``), and the 6-multiplexer benchmark (``benchmarks.mux``).

All parameters are simulation parameters. Nothing here models hardware.
"""

from .agent import LCSAgent
from .alpha import build_alpha, build_alpha_spec, run_alpha
from .audit import GENESIS_HASH, AuditChain
from .bus import Payload, RoutedBus
from .lcs import Params, Rule, RuleEngine
from .noise import Environment, hostile_environment, quiet_environment
from .organism import Organism, Processor, shannon_bits, wrap_deg
from .rules import Interpreter, matches, parse_sexpr, run_action, to_sexpr
from .spec import (
    Gene,
    Genome,
    Metrics,
    OrganismSpec,
    Phase,
    RunSummary,
    State,
    Status,
    TelemetryRecord,
    Triggers,
    parse_dna,
    uniform_phases,
)
from .swarm import (
    Coupling,
    Message,
    MessageBus,
    Population,
    PopulationLog,
    TickMetrics,
    sync_order_parameter,
)

__version__ = "0.3.0"

__all__ = [
    "__version__", "AuditChain", "GENESIS_HASH", "Organism", "Processor", "shannon_bits",
    "wrap_deg", "LCSAgent", "Payload", "RoutedBus", "Params", "Rule", "RuleEngine",
    "Interpreter", "matches", "parse_sexpr", "run_action", "to_sexpr",
    "Environment", "quiet_environment", "hostile_environment",
    "build_alpha", "build_alpha_spec", "run_alpha",
    "Gene", "Genome", "Metrics", "OrganismSpec", "Phase", "RunSummary", "State", "Status",
    "TelemetryRecord", "Triggers", "parse_dna", "uniform_phases",
    "Coupling", "Message", "MessageBus", "Population", "PopulationLog", "TickMetrics",
    "sync_order_parameter",
]
