"""organism_sim.terminal — local natural-language routing layer (no model in the loop).

Natural language is treated as *routing*, not generation: a deterministic keyword-and-slot
matcher maps a sentence to one of a closed set of intents; each intent is either a query
over the checked-in results/schemas/git state, a benchmark invocation, or a scaffold that
packages the exact system state into a prompt for an external reasoner.
"""
from .intent import Intent, parse
from .scaffold import scaffold
from .state import State

__all__ = ["Intent", "parse", "State", "scaffold"]
