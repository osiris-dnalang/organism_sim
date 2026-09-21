"""
organism_sim.chat_bridge — neuro-symbolic bridge: English ↔ dna::}{::lang ↔ runtime
====================================================================================

A language model sits *outside* the execution loop. It sees a telemetry frame
(``telemetry.capture_frame``) and the user's sentence, and may answer in prose or propose
regulatory / rule genes as a ``dna`` code block. Nothing it writes reaches the organism
until it has passed the gate::

    LLM text ─▶ extract_dna ─▶ dnalang.parse ─▶ dnalang.check (merged with the live genome)
             ─▶ runtime binding (metrics ∈ grn.METRICS, adjust ∈ grn.PARAMS, rule width)
             ─▶ GRN.add_genes (atomic)                                  ─▶ organism

A rejection at any stage is returned to the model verbatim for self-correction, up to
``max_rounds`` times; the organism is not touched by a rejected round. The substrate stays
deterministic: the model can only add genes the language accepts, and every accepted gene
is data in the genome (serialisable, hashable, evolvable) like any other.

Proactive side: ``Runtime.step`` returns a frame and ``AlertMonitor`` derives edge-triggered
alerts from it (CUSUM crossing its threshold, status → critical, mutation, injection,
gene expression). ``run_async`` interleaves runtime ticks with user input and prints
alerts on the tick they occur.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import numpy as np
from dnalang import check as dl_check
from dnalang import parse as dl_parse
from dnalang.action_dsl import DSLError, parse_sexpr
from dnalang.parser import ParseError
from dnalang.regulation import Trigger

from .agent import LCSAgent
from .grn import GRN, METRICS, PARAMS, genome_to_dna
from .spec import Gene, Genome, _from_language, parse_dna
from .telemetry import capture_frame, frame_digest, frame_json

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "bridge_system.md"
HAND_DNA = Path(__file__).with_name("benchmarks") / "m1_hand.dna"


# ── LLM clients ──────────────────────────────────────────────────────────────

class LLMClient:
    """``complete(system, messages) -> str``; messages are ``{"role", "content"}`` dicts."""

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Local Ollama ``/api/chat`` (no streaming, no external dependency)."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: Optional[str] = None,
                 timeout: float = 600.0, temperature: float = 0.1, num_ctx: int = 8192):
        self.model = model
        self.num_ctx = num_ctx
        self.base_url = (base_url or os.environ.get("ORGANISM_OLLAMA")
                         or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        body = json.dumps({"model": self.model, "stream": False,
                           "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                           "messages": [{"role": "system", "content": system}] + messages}).encode()
        req = urllib.request.Request(self.base_url + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(f"ollama unreachable at {self.base_url}: {e}") from e
        return str(data.get("message", {}).get("content", ""))


class ScriptedClient(LLMClient):
    """Deterministic stand-in: returns canned responses in order and records every call."""

    def __init__(self, responses: Sequence[str]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.responses:
            raise RuntimeError("ScriptedClient: no responses left")
        return self.responses.pop(0)


# ── extraction ───────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)
_ORGANISM = re.compile(r"\borganism\b", re.I)


def extract_dna(text: str) -> Optional[str]:
    """The first fenced code block that looks like a dna organism (tag ``dna``/``dnalang``, or
    any tag whose body starts an ``organism`` block). ``None`` when the reply is prose."""
    for tag, body in _FENCE.findall(text):
        if tag.lower() in ("dna", "dnalang") or _ORGANISM.search(body):
            return body.strip() + "\n"
    return None


# ── validation gate ──────────────────────────────────────────────────────────

@dataclass
class GateResult:
    ok: bool
    stage: str                                  # parse | check | bind | ok
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    genes: List[Gene] = field(default_factory=list)
    source: str = ""
    merged_dna: str = ""

    def report(self) -> str:
        head = "ACCEPTED" if self.ok else f"REJECTED at {self.stage}"
        lines = [head] + [f"  error: {e}" for e in self.errors] + \
                [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _walk(node: Any, fn: Callable[[list], None]) -> None:
    if isinstance(node, list):
        fn(node)
        for x in node:
            _walk(x, fn)


def gate(source: str, grn: GRN) -> GateResult:
    """Validate *source* (a dna organism of rule/regulator genes) against the language and
    the live runtime. Pure: neither *grn* nor its agent is modified."""
    res = GateResult(ok=False, stage="parse", source=source)
    # 1. the language must parse the model's text on its own
    try:
        org = dl_parse(source)
    except ParseError as e:
        res.errors.append(f"syntax: {e}")
        return res
    if not org.kv_genes:
        res.errors.append("no rule or regulator genes found (a circuit organism cannot be injected)")
        return res
    if org.genes or org.genome:
        res.errors.append("circuit genes / genome instances are not injectable into a rule runtime")
        return res
    try:
        spec = _from_language(source)             # strict: no regex fallback for model text
    except Exception as e:
        res.errors.append(f"convert: {e}")
        return res
    new = list(spec.genome.genes)
    # 2. static checks on the merged genome (so refs to live genes resolve, dup ids show)
    res.stage = "check"
    merged = Genome(genes=list(grn.genome.genes) + new, version=grn.genome.version + 1)
    merged_src = genome_to_dna(merged, "MERGED", header=("gate: live genome + proposal",))
    res.merged_dna = merged_src
    try:
        diag = dl_check(dl_parse(merged_src))
    except ParseError as e:                       # would mean genome_to_dna emitted bad text
        res.errors.append(f"merged genome does not re-parse: {e}")
        return res
    res.warnings.extend(dict.fromkeys(diag.warnings))
    if diag.errors:
        res.errors.extend(dict.fromkeys(diag.errors))   # check_kv and lowering can both report
        return res
    # 3. runtime binding: things the language leaves to the consumer
    res.stage = "bind"
    width = grn.agent.cond_len
    for g in new:
        try:
            t = Trigger.parse(g.trigger)
        except ValueError as e:
            res.errors.append(f"gene {g.id}: {e}")
            continue
        if t.kind == "when" and t.ref not in METRICS:
            res.errors.append(f"gene {g.id}: unmapped metric {t.ref!r}; runtime metrics are {list(METRICS)}")
        if g.is_rule:
            if len(g.condition) != width:
                res.errors.append(f"gene {g.id}: rule condition width {len(g.condition)} != "
                                  f"register width {width} (input {grn.agent.input_len} + "
                                  f"symbol {grn.agent.symbol_bits} bits)")
            continue
        try:
            tree = parse_sexpr(g.action)
        except DSLError as e:
            res.errors.append(f"gene {g.id}: action: {e}")
            continue

        def bind(n: list, gid: str = g.id) -> None:
            if not n:
                return
            if n[0] == "adjust" and len(n) >= 2 and n[1] not in PARAMS:
                res.errors.append(f"gene {gid}: unmapped adjust parameter {n[1]!r}; "
                                  f"runtime parameters are {list(PARAMS)}")
            if n[0] == "var" and len(n) >= 2 and n[1] not in METRICS and n[1] != "last_error":
                res.errors.append(f"gene {gid}: (var {n[1]}) is not a runtime metric")
        _walk(tree, bind)
    if res.errors:
        return res
    res.ok, res.stage, res.genes = True, "ok", new
    return res


# ── alerts ───────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    tick: int
    kind: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"[ALERT t={self.tick} {self.kind}] {self.message}"


class AlertMonitor:
    """Edge-triggered alerts derived from consecutive frames. Stateless w.r.t. the runtime:
    it only reads frames."""

    def __init__(self) -> None:
        self._prev: Optional[Dict[str, Any]] = None
        self._cusum_over = False
        self._critical = False

    def check(self, frame: Dict[str, Any]) -> List[Alert]:
        out: List[Alert] = []
        t = frame["tick"]
        reg = frame["regulation"]
        if reg is not None:
            c = reg["cusum"]
            if c["over"] and not self._cusum_over:
                out.append(Alert(t, "cusum_shift",
                                 f"CUSUM {c['s']:.2f} > {c['h']:g}: detector reports a shift "
                                 f"(error EMA {reg['metrics']['error']:.3f})",
                                 {"cusum": c["s"], "threshold": c["h"]}))
            self._cusum_over = c["over"]
            if self._prev is not None and self._prev["regulation"] is not None:
                pr = self._prev["regulation"]
                if reg["injections"] > pr["injections"]:
                    out.append(Alert(t, "injection", "regulation injected fresh rules into the learner",
                                     {"injections": reg["injections"]}))
                newly = [g for g in reg["active_genes"] if reg["expressions"].get(g, 0) == 1]
                if newly:
                    out.append(Alert(t, "gene_first_expression",
                                     f"gene(s) expressed for the first time: {newly}",
                                     {"genes": newly}))
        org = frame["organism"]
        crit = org["status"] == "critical"
        if crit and not self._critical:
            out.append(Alert(t, "status_critical",
                             f"organism status CRITICAL (noise {org['state']['noise_rate']:.3f}, "
                             f"entropy {org['state']['entropy_bits']:.3f} bits)"))
        self._critical = crit
        if "mutation" in org["events"]:
            reason = next((e.split("=", 1)[1] for e in org["events"] if e.startswith("mutation_reason=")), "?")
            out.append(Alert(t, "mutation", f"structural mutation ({reason}); generation {org['generation']}"))
        self._prev = frame
        return out


# ── runtime ──────────────────────────────────────────────────────────────────

class Runtime:
    """One LCS agent + one regulatory genome on a binary task. ``step()`` runs one trial and
    returns the telemetry frame for it; alerts accumulate in ``pending`` until drained."""

    def __init__(self, agent: LCSAgent, grn: GRN, truth: Callable[[str], str], seed: int = 0,
                 before_trial: Optional[Callable[[int], None]] = None):
        self.agent = agent
        self.grn = grn
        self.truth = truth
        self.before_trial = before_trial
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.monitor = AlertMonitor()
        self.pending: List[Alert] = []
        self.alerts: List[Alert] = []
        self.frame: Dict[str, Any] = capture_frame(grn, 0)
        self.listeners: List[Callable[[Alert], None]] = []
        self.injections: List[Dict[str, Any]] = []

    @classmethod
    def demo(cls, seed: int = 0, shift_every: int = 3000, genome_path: Path = HAND_DNA) -> "Runtime":
        """Hidden drifting 6-mux (the M1 environment) with the hand-written regulatory genome."""
        from .benchmarks.expzero import DriftingHidden
        env = DriftingHidden(seed, shift_every)
        agent = LCSAgent("O", input_len=6, actions=["(emit 0)", "(emit 1)"], seed=seed)
        grn = GRN(parse_dna(Path(genome_path).read_text()).genome, agent, seed=seed)
        rt = cls(agent, grn, env.truth, seed=seed,
                 before_trial=lambda trial: env.maybe_shift(trial, 10 ** 9))
        rt.env = env
        return rt

    def step(self) -> Dict[str, Any]:
        self.tick += 1
        if self.before_trial is not None:
            self.before_trial(self.tick)
        bits = "".join(map(str, self.rng.integers(0, 2, self.agent.input_len)))
        self.agent.act(bits)
        r = 1.0 if self.agent.emitted() == self.truth(bits) else 0.0
        self.agent.reward(r)
        self.grn.observe(r, self.agent.engine.last_explore)
        self.grn.step()
        self.agent.tick()
        self.frame = capture_frame(self.grn, self.tick)
        for a in self.monitor.check(self.frame):
            self.pending.append(a)
            self.alerts.append(a)
            for fn in self.listeners:
                fn(a)
        return self.frame

    def run(self, ticks: int) -> Dict[str, Any]:
        for _ in range(ticks):
            self.step()
        return self.frame

    def drain(self) -> List[Alert]:
        out, self.pending = self.pending, []
        return out

    def inject(self, genes: Sequence[Gene], source: str = "") -> List[str]:
        """Add gate-accepted genes to the live genome (atomic; see ``GRN.add_genes``)."""
        ids = self.grn.add_genes(genes)
        self.injections.append({"tick": self.tick, "ids": ids, "source": source,
                                "genome_fingerprint": self.grn.genome.fingerprint()})
        self.frame = capture_frame(self.grn, self.tick)
        return ids


# ── bridge ───────────────────────────────────────────────────────────────────

@dataclass
class Reply:
    text: str                                   # the model's final prose (or last attempt)
    accepted: bool = False
    injected: List[str] = field(default_factory=list)
    dna: Optional[str] = None
    rejections: List[GateResult] = field(default_factory=list)
    rounds: int = 0
    tick: int = 0

    def render(self) -> str:
        parts = [self.text.strip()] if self.text.strip() else []
        if self.accepted:
            parts.append(f"[injected {self.injected} at tick {self.tick}]")
        elif self.rejections:
            parts.append(f"[rejected after {self.rounds} round(s)]")
            parts.append(self.rejections[-1].report())
        return "\n".join(parts)


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    return Path(path).read_text()


class Bridge:
    """Reactive side: ``ask(text)`` → ``Reply``. The gate runs on every dna block; rejections
    go back to the model with the exact diagnostics."""

    def __init__(self, runtime: Runtime, llm: LLMClient, system_prompt: Optional[str] = None,
                 max_rounds: int = 3, history_len: int = 12):
        self.runtime = runtime
        self.llm = llm
        self.system = system_prompt if system_prompt is not None else load_system_prompt()
        self.max_rounds = max_rounds
        self.history_len = history_len
        self.history: List[Dict[str, str]] = []
        self.log: List[Reply] = []

    # -- prompt assembly --------------------------------------------------------
    def user_message(self, text: str, frame: Optional[Dict[str, Any]] = None) -> str:
        frame = frame or self.runtime.frame
        digest = "\n".join(frame_digest(frame))
        return (f"<telemetry tick=\"{frame['tick']}\">\n{frame_json(frame, indent=None)}\n"
                f"</telemetry>\n<digest>\n{digest}\n</digest>\n\n{text}")

    def rejection_message(self, res: GateResult) -> str:
        return ("Your dna block was REJECTED by the validation gate and was NOT applied. "
                "Fix every item and resend the complete organism in one ```dna block, or "
                "answer in prose only if the request needs no genes.\n" + res.report())

    # -- one round --------------------------------------------------------------
    def _handle(self, out: str, messages: List[Dict[str, str]], reply: Reply) -> bool:
        """Process one model output. Returns True when the exchange is finished."""
        reply.rounds += 1
        reply.text = out
        src = extract_dna(out)
        if src is None:
            return True                                   # prose answer, nothing to gate
        reply.dna = src
        res = gate(src, self.runtime.grn)
        if res.ok:
            reply.injected = self.runtime.inject(res.genes, source=src)
            reply.accepted = True
            reply.tick = self.runtime.tick
            return True
        reply.rejections.append(res)
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": self.rejection_message(res)})
        return reply.rounds >= self.max_rounds

    def _finish(self, first_user: str, reply: Reply) -> Reply:
        self.history.append({"role": "user", "content": first_user})
        self.history.append({"role": "assistant", "content": reply.render()})
        self.history = self.history[-self.history_len:]
        self.log.append(reply)
        return reply

    def ask(self, text: str) -> Reply:
        first = self.user_message(text)
        messages = list(self.history) + [{"role": "user", "content": first}]
        reply = Reply(text="")
        while True:
            out = self.llm.complete(self.system, messages)
            if self._handle(out, messages, reply):
                return self._finish(text, reply)

    async def ask_async(self, text: str) -> Reply:
        """Same exchange; the model call runs in a thread so the event loop (and the runtime)
        keep ticking. Gating and injection happen on the loop thread, between ticks."""
        loop = asyncio.get_running_loop()
        first = self.user_message(text)
        messages = list(self.history) + [{"role": "user", "content": first}]
        reply = Reply(text="")
        while True:
            out = await loop.run_in_executor(None, self.llm.complete, self.system, messages)
            if self._handle(out, messages, reply):
                return self._finish(text, reply)


# ── async event loop ─────────────────────────────────────────────────────────

COMMANDS = """commands: /frame  /digest  /genes  /alerts  /pause  /resume  /step N  /quit
anything else is sent to the model with the current telemetry frame."""


def _command(bridge: Bridge, line: str) -> Optional[str]:
    rt = bridge.runtime
    parts = line.split()
    cmd = parts[0].lower()
    if cmd == "/frame":
        return frame_json(rt.frame, indent=2)
    if cmd == "/digest":
        return "\n".join(frame_digest(rt.frame))
    if cmd == "/genes":
        return genome_to_dna(rt.grn.genome, "LIVE")
    if cmd == "/alerts":
        return "\n".join(a.line() for a in rt.alerts[-20:]) or "(none)"
    if cmd == "/help":
        return COMMANDS
    return None


async def run_async(bridge: Bridge, read_line: Callable[[], Awaitable[Optional[str]]],
                    write: Callable[[str], None], ticks_per_slice: int = 25,
                    tick_delay: float = 0.0, max_ticks: Optional[int] = None) -> None:
    """Interleave runtime ticks and chat. Alerts are written on the tick they occur; user
    lines are answered without stopping the organism (``/pause`` stops it)."""
    rt = bridge.runtime
    paused = False
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            if not paused and (max_ticks is None or rt.tick < max_ticks):
                n = ticks_per_slice if max_ticks is None else min(ticks_per_slice, max_ticks - rt.tick)
                for _ in range(n):
                    rt.step()
                    for a in rt.drain():
                        write(a.line())
            await asyncio.sleep(tick_delay)

    async def reader() -> None:
        nonlocal paused
        while not stop.is_set():
            line = await read_line()
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if line.lower() in ("/quit", "/exit", "/q"):
                break
            if line.lower() == "/pause":
                paused = True
                write(f"[paused at tick {rt.tick}]")
                continue
            if line.lower() == "/resume":
                paused = False
                write("[resumed]")
                continue
            if line.lower().startswith("/step"):
                n = int(line.split()[1]) if len(line.split()) > 1 else 1
                for _ in range(n):
                    rt.step()
                    for a in rt.drain():
                        write(a.line())
                write(f"[tick {rt.tick}]")
                continue
            if line.startswith("/"):
                out = _command(bridge, line)
                write(out if out is not None else f"unknown command; {COMMANDS}")
                continue
            reply = await bridge.ask_async(line)
            write(reply.render())
        stop.set()

    tasks = [asyncio.ensure_future(ticker()), asyncio.ensure_future(reader())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            t.result()                            # re-raise a failure instead of hiding it
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="chat_bridge", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=os.environ.get("ORGANISM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--ollama", default=None, help="Ollama base URL (default $ORGANISM_OLLAMA or localhost:11434)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift-every", type=int, default=3000)
    ap.add_argument("--genome", type=Path, default=HAND_DNA)
    ap.add_argument("--ticks-per-slice", type=int, default=25)
    ap.add_argument("--tick-delay", type=float, default=0.02, help="seconds between slices")
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per model call (CPU inference is slow)")
    ap.add_argument("--warmup", type=int, default=0, help="ticks to run before the prompt appears")
    args = ap.parse_args(argv)

    rt = Runtime.demo(seed=args.seed, shift_every=args.shift_every, genome_path=args.genome)
    if args.warmup:
        rt.run(args.warmup)
        rt.drain()
    bridge = Bridge(rt, OllamaClient(args.model, args.ollama, timeout=args.timeout),
                    max_rounds=args.max_rounds)
    print(f"chat_bridge: {args.model} ↔ organism {rt.agent.name} (seed {args.seed}); {COMMANDS}")

    loop = asyncio.new_event_loop()

    async def read_line() -> Optional[str]:
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = await loop.run_in_executor(None, sys.stdin.readline)
        return line if line else None

    def write(s: str) -> None:
        sys.stdout.write("\r" + s + "\n> ")
        sys.stdout.flush()

    try:
        loop.run_until_complete(run_async(bridge, read_line, write, args.ticks_per_slice,
                                          args.tick_delay, args.max_ticks))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
    print(f"\nbye — {rt.tick} ticks, {len(rt.injections)} injection(s), {len(rt.alerts)} alert(s)")
    return 0


__all__ = ["LLMClient", "OllamaClient", "ScriptedClient", "extract_dna", "GateResult", "gate",
           "Alert", "AlertMonitor", "Runtime", "Reply", "Bridge", "run_async", "main",
           "load_system_prompt", "PROMPT_PATH"]
