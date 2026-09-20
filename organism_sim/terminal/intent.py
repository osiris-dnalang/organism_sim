"""Deterministic intent parsing: keywords → intent, regex → slots. Auditable and testable;
every sentence maps to exactly one intent or to HELP."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

INTENTS = ("RUN_EVAL", "RUN_GUARD", "QUERY_RESULTS", "STATUS", "SCAFFOLD", "SHOW_SCHEMA", "HELP")

# (intent, required keyword groups) — a group matches if any of its words appears
_RULES: List[Tuple[str, List[List[str]]]] = [
    ("SCAFFOLD", [["scaffold", "prompt", "hand off", "handoff", "package the context", "for claude"]]),
    ("RUN_GUARD", [["guard", "latency budget", "ci check", "regression"]]),
    ("SHOW_SCHEMA", [["schema", "contract"]]),
    ("QUERY_RESULTS", [["what", "how", "show", "report", "rate", "stats", "median", "result", "did"],
                       ["false", "recover", "relay", "drift", "signal", "mux", "bridge", "verdict",
                        "run", "result", "eval", "pooling", "cusum", "seed"]]),
    ("RUN_EVAL", [["run", "execute", "launch", "benchmark", "evaluate", "start", "sweep"]]),
    ("STATUS", [["status", "state", "commit", "where are we", "which version", "hash"]]),
]

_SLOT_PATTERNS: Dict[str, str] = {
    "seeds": r"\b(\d+)\s*seeds?\b",
    "trials": r"\b(\d+)\s*(?:trials|ticks|cycles)\b",
    "phase": r"\bphase\s*([ab])\b",
}


@dataclass
class Intent:
    name: str
    slots: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    confidence: float = 1.0          # 1.0 for a keyword hit, 0.0 for HELP fallback

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.name, "slots": dict(self.slots), "text": self.text,
                "confidence": self.confidence}


def _slots(t: str) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    for k, pat in _SLOT_PATTERNS.items():
        m = re.search(pat, t)
        if m:
            s[k] = int(m.group(1)) if k != "phase" else m.group(1).upper()
    if re.search(r"\bpoison", t):
        s["mode"] = "poisoned"
    elif re.search(r"\bdead\b|\bsilent\b", t):
        s["mode"] = "dead"
    if "cusum" in t or "change-point" in t or "changepoint" in t:
        s["trigger"] = "cusum"
    elif "threshold" in t:
        s["trigger"] = "threshold"
    for bench in ("relay", "drift", "mux", "swarm", "signal", "bridge", "tier 4", "tier4", "step 3"):
        if bench in t:
            s["benchmark"] = {"signal": "signalling", "tier 4": "tier4", "step 3": "step3"}.get(bench, bench)
            break
    if "isolated" in t:
        s["isolated"] = True
    if "supervisor" in t:
        s["group"] = "supervisor"
    elif "organism" in t:
        s["group"] = "organism-cusum" if s.get("trigger") == "cusum" else "organism"
    if "last" in t or "latest" in t or "most recent" in t:
        s["which"] = "last"
    return s


def parse(text: str) -> Intent:
    t = " ".join(text.lower().split())
    slots = _slots(t)
    for name, groups in _RULES:
        if all(any(w in t for w in g) for g in groups):
            return Intent(name, slots, text, 1.0)
    return Intent("HELP", slots, text, 0.0)


HELP_TEXT = """intents (deterministic keyword routing, no model):
  run <relay|drift|mux|swarm> [dead|poisoned] [cusum] [N seeds] [phase A|B] [isolated]
  guard                          — latency budget + dead-relay regression guard
  what/show ... <false alarms | recovery | signalling rate | drift | bridge | verdict>
  status                         — commits, results present, schemas, last guard numbers
  schema [telemetry|payload]     — print a contract
  scaffold: <your request>       — package exact system state into a prompt for an external reasoner
  help"""

__all__ = ["Intent", "parse", "INTENTS", "HELP_TEXT"]
