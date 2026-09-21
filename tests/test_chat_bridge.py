"""M-Bridge-2 (gate + REPL) and M-Bridge-3 (proactive loop).

The model is a ``ScriptedClient``: every test is deterministic and offline. The invariant
under test is that nothing the model writes reaches the organism unless the gate accepts
it, and that a rejection leaves the runtime byte-for-byte where it was.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.chat_bridge import (  # noqa: E402
    AlertMonitor,
    Bridge,
    Runtime,
    ScriptedClient,
    extract_dna,
    gate,
    run_async,
)
from organism_sim.telemetry import capture_frame  # noqa: E402

SYSTEM = "test system prompt"


def fenced(src: str) -> str:
    return f"Here is the change.\n```dna\n{src}\n```\n"


GOOD = '''organism Patch {
  genome {
    gene calm { id: "B1", trigger: "when error < 0.02", action: "(adjust explore 0.1)", cluster: "bridge" }
    gene react { id: "B2", trigger: "after G1", action: "(adjust explore 0.9)", dependencies: ["G0"], cluster: "bridge" }
  }
}'''

BAD = {
    "syntax": 'organism X { genome { gene a { id: "Z1", trigger: "continuous", action: "(emit q)" } ',
    "illegal_action": 'organism X { genome { gene a { id: "Z1", trigger: "continuous", action: "(launch 1)" } } }',
    "cycle": ('organism X { genome { '
              'gene a { id: "Z1", trigger: "continuous", action: "(emit x)", dependencies: ["Z2"] } '
              'gene b { id: "Z2", trigger: "continuous", action: "(emit y)", dependencies: ["Z1"] } } }'),
    "unmapped_metric": 'organism X { genome { gene a { id: "Z1", trigger: "when vibes > 3", action: "(emit x)" } } }',
    "unmapped_param": 'organism X { genome { gene a { id: "Z1", trigger: "continuous", action: "(adjust warp 9)" } } }',
    "unknown_dependency": 'organism X { genome { gene a { id: "Z1", trigger: "continuous", action: "(emit x)", dependencies: ["NOPE"] } } }',
    "duplicate_id": 'organism X { genome { gene a { id: "G0", trigger: "continuous", action: "(emit x)" } } }',
    "rule_width": 'organism X { genome { gene a { id: "Z1", condition: "01#", action: "(emit 1)" } } }',
    "circuit": 'organism X { gene bell() on a, b { h a; cx a, b; } genome { bell() @ [0, 1]; } }',
    "bad_trigger": 'organism X { genome { gene a { id: "Z1", trigger: "whenever", action: "(emit x)" } } }',
}


def snapshot(rt: Runtime) -> dict:
    """Everything a rejected round must leave untouched."""
    return {
        "genome": rt.grn.genome.to_dict(),
        "order": list(rt.grn.order),
        "expressions": dict(rt.grn.expressions),
        "cusum": rt.grn.cusum_s,
        "board": set(rt.grn.board),
        "audit_head": rt.agent.organism.chain.head,
        "tick": rt.tick,
        "engine_macro": rt.agent.engine.macro_size(),
        "frame": capture_frame(rt.grn, rt.tick),
    }


@pytest.fixture
def rt() -> Runtime:
    r = Runtime.demo(seed=0)
    r.run(60)
    r.drain()
    return r


# ── extraction ───────────────────────────────────────────────────────────────

def test_extract_dna_finds_only_organism_blocks():
    assert extract_dna("prose only") is None
    assert extract_dna("```python\nprint(1)\n```") is None
    assert extract_dna(fenced(GOOD)).strip() == GOOD
    untagged = "```\n" + GOOD + "\n```"
    assert extract_dna(untagged).strip() == GOOD


# ── gate ─────────────────────────────────────────────────────────────────────

def test_gate_accepts_valid_genes_and_resolves_live_references(rt):
    res = gate(GOOD, rt.grn)
    assert res.ok and res.stage == "ok" and not res.errors
    assert [g.id for g in res.genes] == ["B1", "B2"]
    assert len(rt.grn.genome) == 5                    # gate is pure


@pytest.mark.parametrize("name", sorted(BAD))
def test_gate_rejects_each_invalid_proposal_without_touching_state(rt, name):
    before = snapshot(rt)
    res = gate(BAD[name], rt.grn)
    assert not res.ok, name
    assert res.errors, name
    assert res.stage in ("parse", "check", "bind"), res.stage
    assert snapshot(rt) == before


def test_gate_stage_and_message_per_failure(rt):
    assert gate(BAD["syntax"], rt.grn).stage == "parse"
    assert "unknown op 'launch'" in " ".join(gate(BAD["illegal_action"], rt.grn).errors)
    assert "cycle" in " ".join(gate(BAD["cycle"], rt.grn).errors)
    r = gate(BAD["unmapped_metric"], rt.grn)
    assert r.stage == "bind" and "unmapped metric 'vibes'" in r.errors[0]
    r = gate(BAD["unmapped_param"], rt.grn)
    assert r.stage == "bind" and "unmapped adjust parameter 'warp'" in r.errors[0]
    assert "duplicate gene id 'G0'" in gate(BAD["duplicate_id"], rt.grn).errors[0]
    assert "width 3 != register width 6" in gate(BAD["rule_width"], rt.grn).errors[0]


# ── bridge (reactive) ────────────────────────────────────────────────────────

def test_prose_reply_is_passed_through_and_nothing_is_injected(rt):
    llm = ScriptedClient(["The CUSUM is below threshold; no change needed."])
    b = Bridge(rt, llm, system_prompt=SYSTEM)
    before = snapshot(rt)
    reply = b.ask("how is the organism doing?")
    assert not reply.accepted and reply.dna is None and reply.rounds == 1
    assert "no change needed" in reply.text
    assert snapshot(rt) == before
    # the model saw the live frame, not a summary written by hand
    sent = llm.calls[0]["messages"][-1]["content"]
    assert "<telemetry tick=\"60\">" in sent and '"schema": "organism_sim.telemetry_frame/1"' in sent
    assert llm.calls[0]["system"] == SYSTEM


def test_invalid_dsl_is_bounced_back_and_never_injected(rt):
    llm = ScriptedClient([fenced(BAD["illegal_action"]), fenced(BAD["cycle"]), fenced(BAD["unmapped_metric"])])
    b = Bridge(rt, llm, system_prompt=SYSTEM, max_rounds=3)
    before = snapshot(rt)
    reply = b.ask("make it explore more when it fails")
    assert not reply.accepted and reply.rounds == 3 and len(reply.rejections) == 3
    assert [r.stage for r in reply.rejections] == ["check", "check", "bind"]
    assert snapshot(rt) == before
    # each bounce carried the exact diagnostics of the previous attempt
    second = llm.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and "launch" in second[-2]["content"]
    assert second[-1]["role"] == "user" and "REJECTED at check" in second[-1]["content"]
    assert "unknown op 'launch'" in second[-1]["content"]
    assert "REJECTED" in reply.render()


def test_self_correction_succeeds_on_second_round(rt):
    llm = ScriptedClient([fenced(BAD["unmapped_param"]), fenced(GOOD)])
    b = Bridge(rt, llm, system_prompt=SYSTEM)
    fp0 = rt.grn.genome.fingerprint()
    reply = b.ask("boost exploration after an injection")
    assert reply.accepted and reply.rounds == 2 and reply.injected == ["B1", "B2"]
    assert len(reply.rejections) == 1 and "warp" in reply.rejections[0].errors[0]
    assert len(rt.grn.genome) == 7 and rt.grn.genome.fingerprint() != fp0
    assert rt.grn.order[-2:] == ["B1", "B2"] and rt.grn.expressions["B2"] == 0
    assert rt.injections[-1]["ids"] == ["B1", "B2"] and rt.injections[-1]["tick"] == 60
    assert rt.frame["regulation"]["genome_version"] == 2
    # the injected genes are live: the same organism keeps ticking with them
    rt.run(20)
    assert "B1" in rt.frame["regulation"]["order"]
    # and the second injection of the same ids is refused (duplicate) — the gate is stateful
    # only through the genome it reads
    reply2 = Bridge(rt, ScriptedClient([fenced(GOOD)]), system_prompt=SYSTEM, max_rounds=1).ask("again")
    assert not reply2.accepted and "duplicate gene id 'B1'" in reply2.rejections[0].errors[0]


def test_history_is_kept_and_bounded(rt):
    llm = ScriptedClient(["a"] * 20)
    b = Bridge(rt, llm, system_prompt=SYSTEM, history_len=6)
    for i in range(10):
        b.ask(f"q{i}")
    assert len(b.history) == 6 and b.history[-2]["content"].endswith("q9")
    assert len(llm.calls[-1]["messages"]) == 7                    # 6 history + current


# ── proactive loop ───────────────────────────────────────────────────────────

def _force_errors(rt: Runtime) -> None:
    """Environment that always contradicts the agent → every exploit trial is an error."""
    agent = rt.agent
    rt.truth = lambda bits: "1" if agent.emitted() == "0" else "0"


def test_cusum_crossing_raises_alert_on_the_same_tick():
    rt = Runtime.demo(seed=1)
    rt.run(50)
    rt.drain()
    assert rt.grn.cusum_s <= rt.grn.cusum_h
    _force_errors(rt)
    crossed_at = None
    for _ in range(400):
        frame = rt.step()
        if frame["regulation"]["cusum"]["over"]:
            crossed_at = frame["tick"]
            break
    assert crossed_at is not None, "detector never crossed under forced error"
    shifts = [a for a in rt.pending if a.kind == "cusum_shift"]
    assert len(shifts) == 1
    assert shifts[0].tick == crossed_at                        # emitted on the crossing tick
    assert shifts[0].data["cusum"] > 8.0 and shifts[0].data["threshold"] == 8.0
    assert "CUSUM" in shifts[0].line()
    # Edge-triggered: exactly one alert per rising edge of the detector. Under sustained
    # forced error the hand genome resets the detector (G0 → shift → G1 cusum_reset) and it
    # climbs again, so several edges occur; each gets one alert, on its own tick.
    edges = [crossed_at]
    over = True
    for _ in range(400):
        f = rt.step()
        now = f["regulation"]["cusum"]["over"]
        if now and not over:
            edges.append(f["tick"])
        over = now
    assert len(edges) >= 2, "expected the detector to reset and re-cross"
    assert [a.tick for a in rt.alerts if a.kind == "cusum_shift"] == edges


def test_alert_monitor_is_edge_triggered_and_reads_only_frames():
    m = AlertMonitor()
    rt = Runtime.demo(seed=0)
    f = rt.step()
    assert not [a for a in m.check(f) if a.kind == "cusum_shift"]
    hi = dict(f)
    hi["regulation"] = dict(f["regulation"], cusum={"s": 9.0, "h": 8.0, "mu0": 0.05, "k": 0.1, "over": True})
    kinds = lambda alerts: [x.kind for x in alerts if x.kind == "cusum_shift"]  # noqa: E731
    assert kinds(m.check(hi)) == ["cusum_shift"]
    assert kinds(m.check(hi)) == []                           # still over → no repeat
    assert kinds(m.check(f)) == []                            # back under → nothing
    assert kinds(m.check(hi)) == ["cusum_shift"]              # crosses again → alerts again


def test_runtime_listeners_fire_synchronously_with_step():
    rt = Runtime.demo(seed=1)
    rt.run(50)
    seen = []
    rt.listeners.append(lambda a: seen.append((a.kind, a.tick, rt.tick)))
    _force_errors(rt)
    for _ in range(400):
        rt.step()
        if any(k == "cusum_shift" for k, _, _ in seen):
            break
    kind, alert_tick, runtime_tick_when_seen = next(s for s in seen if s[0] == "cusum_shift")
    assert alert_tick == runtime_tick_when_seen


def test_async_loop_interrupts_chat_with_alert_within_one_tick():
    rt = Runtime.demo(seed=1)
    rt.run(50)
    rt.drain()
    _force_errors(rt)
    llm = ScriptedClient(["Noted — I will watch the detector."])
    b = Bridge(rt, llm, system_prompt=SYSTEM)
    written = []                                   # (runtime tick at write time, line)
    reader = _Reader(written, rt)
    asyncio.run(run_async(b, reader.read_line, lambda s: written.append((rt.tick, s)),
                          ticks_per_slice=1, tick_delay=0.0, max_ticks=450))
    alert_lines = [(t, s) for t, s in written if "cusum_shift" in s]
    assert alert_lines, written[-5:]
    tick_written, line = alert_lines[0]
    alert = next(a for a in rt.alerts if a.kind == "cusum_shift")
    assert tick_written - alert.tick <= 1
    assert line == alert.line()
    # the chat reply also arrived, and the ticker kept running underneath it
    assert any("Noted" in s for _, s in written)
    assert rt.tick > 51


class _Reader:
    """Reader that asks one question, then waits for a cusum alert to be written, then quits."""

    def __init__(self, written, rt):
        self.written = written
        self.rt = rt
        self.asked = False

    async def read_line(self):
        if not self.asked:
            self.asked = True
            return "what is happening?"
        while not any("cusum_shift" in s for _, s in self.written):
            if self.rt.tick >= 450:
                return None
            await asyncio.sleep(0)
        return "/quit"


# ── system prompt ────────────────────────────────────────────────────────────

def test_system_prompt_is_pinned_to_the_runtime_vocabulary():
    """The prompt must name every metric / parameter / DSL op the runtime accepts, and its
    worked example must pass the gate — so prompt and runtime cannot drift apart silently."""
    from dnalang.action_dsl import EXPR_OPS, OPS

    from organism_sim.chat_bridge import PROMPT_PATH, load_system_prompt
    from organism_sim.grn import METRICS, PARAMS

    assert PROMPT_PATH.exists()
    text = load_system_prompt()
    for name in METRICS + PARAMS + OPS + EXPR_OPS:
        assert name in text, name
    for banned in ("THETA_LOCK", "LAMBDA_PHI", "consciousness", "51.843", "qiskit"):
        assert banned not in text
    rt = Runtime.demo(seed=0)
    rt.run(30)
    example = extract_dna(text.split("## Worked example")[1])
    assert example is not None
    res = gate(example, rt.grn)
    assert res.ok, res.errors
    # the default Bridge loads exactly this file
    b = Bridge(rt, ScriptedClient(["ok"]))
    assert b.system == text
