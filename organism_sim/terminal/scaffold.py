"""Package the exact system state into a deterministic prompt for an external reasoner.
The model is outside the execution loop: it receives commit hashes, the contract, the
latest measured numbers and the pre-registration rules as text, and returns text."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .state import State

CONSTRAINTS = """- Python 3.9+, NumPy only in organism_sim; no physical constants, hardware stubs or neural
  models in organism_sim/bridge (a test greps for the banned vocabulary).
- Steady-state latency of a full LCS trial at a 400-rule population: 167 µs measured; CI budget
  600 µs (`organism_sim.benchmarks.ci_guard`). Do not regress it.
- Every telemetry row must validate against the schema below and keep the hash chain
  (hash = SHA-256(prev_hash || canonical JSON)).
- Any performance claim must be pre-registered: criterion stated first, triggers tuned on seeds
  100+, evaluated once on seeds 0–29 (or 0–4), nothing re-tuned on evaluation seeds, negative
  results recorded.
- Neither organism_sim nor dnalang-core imports the other; bridge is the only place both appear."""


def scaffold(request: str, state: Optional[State] = None, out: Optional[Path] = None,
             copy: bool = True) -> str:
    st = state or State()
    commits = st.commits()
    card = st.scorecard()
    lines = [
        "# Context",
        "System: organism_sim — deterministic rule-evolving agents (XCS-lineage LCS, closed s-expression",
        "action DSL, routed Payload(content, pressure) bus, hash-chained telemetry) + bridge (organism",
        "controller over dnalang-core's dynamical-decoupling search space, Aer ground truth).",
        "",
        "Commits: " + ", ".join(f"{k}@{v}" for k, v in commits.items() if v),
        "",
        "# Measured state (pre-registered results, data under results/)",
        "| experiment | verdict | number |", "|---|---|---|",
    ]
    lines += [f"| {r['experiment']} | {r['verdict']} | {r['number']} |" for r in card]
    lines += ["", "# Hard constraints", CONSTRAINTS, "",
              "# Telemetry row contract (schemas/telemetry_row.schema.json)", "```json",
              st.schema("telemetry").strip(), "```", "",
              "# User intent", request.strip(), "",
              "# Task",
              "Respond to the user intent under the constraints above. If you propose a mechanism,",
              "also propose its pre-registered criterion and the seeds it will be judged on. Say what",
              "you are not sure of. Do not restate the context."]
    text = "\n".join(lines) + "\n"
    if out is not None:
        out.write_text(text)
    if copy:
        _clipboard(text)
    return text


def _clipboard(text: str) -> bool:
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip.exe"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
                return True
            except Exception:
                continue
    return False


__all__ = ["scaffold", "CONSTRAINTS"]
