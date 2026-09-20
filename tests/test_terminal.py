"""Terminal routing layer: deterministic intents, slot extraction, queries over checked-in
results, argv mapping, scaffold contents. No model, no network."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.terminal.intent import INTENTS, parse  # noqa: E402
from organism_sim.terminal.repl import argv_for, handle, render  # noqa: E402
from organism_sim.terminal.scaffold import scaffold  # noqa: E402
from organism_sim.terminal.state import State  # noqa: E402


@pytest.mark.parametrize("text,intent,slots", [
    ("what were the false alarm rates on the last dead relay run", "QUERY_RESULTS",
     {"mode": "dead", "benchmark": "relay", "which": "last"}),
    ("run the poisoned relay eval on 50 seeds and track the cusum threshold", "RUN_EVAL",
     {"seeds": 50, "mode": "poisoned", "trigger": "cusum", "benchmark": "relay"}),
    ("show the signalling rate", "QUERY_RESULTS", {"benchmark": "signalling"}),
    ("how did the bridge do", "QUERY_RESULTS", {"benchmark": "bridge"}),
    ("run mux phase b isolated for 3000 trials", "RUN_EVAL",
     {"phase": "B", "isolated": True, "trials": 3000, "benchmark": "mux"}),
    ("status", "STATUS", {}),
    ("run the guard", "RUN_GUARD", {}),
    ("show me the payload schema", "SHOW_SCHEMA", {}),
    ("scaffold: replace subsumption with a q-learning router", "SCAFFOLD", {}),
    ("make me a sandwich", "HELP", {}),
])
def test_intents_and_slots(text, intent, slots):
    it = parse(text)
    assert it.name == intent and it.name in INTENTS
    for k, v in slots.items():
        assert it.slots.get(k) == v, (k, it.slots)
    assert parse(text).to_dict() == it.to_dict()          # deterministic


def test_argv_mapping():
    assert argv_for(parse("run the poisoned relay eval on 4 seeds with cusum")) == \
        ["bench", "relay", "--mode", "poisoned", "--group", "organism-cusum", "--compare", "4"]
    assert argv_for(parse("run drift on 3 seeds")) == ["bench", "drift", "--compare", "3"]
    assert argv_for(parse("run the swarm for 50 ticks")) == ["swarm", "--ticks", "50"]
    assert argv_for(parse("guard")) == ["guard"]
    assert argv_for(parse("status")) is None


def test_queries_over_checked_in_results():
    st = State()
    fa = st.false_alarms("dead", "threshold")
    assert fa["organism_false_per_10k"] == 0.0 and fa["pass"] is True
    sig = st.signalling()
    assert sig["seeds"] == 50 and sig["full"] + sig["partial"] + sig["pooling"] == 50
    d = st.drift()
    assert d["verdict"] == "FAIL" and 0.5 < d["ratio"] < 1.0
    card = st.scorecard()
    names = {r["experiment"] for r in card}
    assert {"signalling (50 seeds)", "concept drift 6-mux", "relay dead (threshold)"} <= names
    assert all(set(r) == {"experiment", "verdict", "number"} for r in card)


def test_handle_dry_run_and_render():
    res = handle("run the dead relay eval on 2 seeds", dry=True)
    assert res["argv"][:4] == ["python", "-m", "organism_sim.cli", "bench"] and "exit" not in res
    assert "(dry run)" in render(res)
    res = handle("what is the signalling rate", dry=True)
    assert res["answer"]["seeds"] == 50
    assert "[QUERY_RESULTS]" in render(res)
    res = handle("nonsense words here", dry=True)
    assert res["intent"]["intent"] == "HELP" and "intents" in render(res)


def test_scaffold_contains_state_and_contract(tmp_path):
    text = scaffold("swap subsumption for a router", State(), out=tmp_path / "s.md", copy=False)
    assert (tmp_path / "s.md").read_text() == text
    assert "organism_sim@" in text and "# Hard constraints" in text
    assert "telemetry_row.schema.json" in text and '"$schema"' in text
    assert "| relay dead (threshold) | PASS |" in text
    assert "swap subsumption for a router" in text
    assert "pre-registered" in text
    again = scaffold("swap subsumption for a router", State(), copy=False)
    assert again == text                                   # deterministic


def test_cli_ask(tmp_path):
    r = subprocess.run([sys.executable, "-m", "organism_sim.cli", "ask", "--dry", "--json",
                        "show", "the", "signalling", "rate"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["intent"]["intent"] == "QUERY_RESULTS" and d["answer"]["seeds"] == 50
