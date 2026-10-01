"""
experiments/m7_evolve.py — M7a: LLM-guided evolution of the learner's own operators
===================================================================================

M7 on the ladder: "the learner's own operators are genomes; selection on learning speed,
with seeds disjoint from evaluation." M7a is the first experiment on that rung, using the
mechanism of AlphaEvolve (Novikov et al., Google DeepMind 2025; FunSearch, Romera-Paredes
et al. 2023): a language model proposes edits to the source of six learner operators
(``organism_sim/evolvable.py``), a fixed program evaluates each proposal, and an
evolutionary database decides what the model sees next. The model only proposes; the
evaluator decides; every proposal, response and score is written to a hash-chained ledger.

Question: at the scale available here — a 7B local model (qwen2.5:7b via Ollama, CPU) and
60 proposals — does LLM-guided operator evolution beat ``PARAMS16``, the pre-registered
16-bit learner (6,000 trials/shift on judge seeds 50–54, Substrate-Opt-2)?

Starting program: ``SEED_OPERATORS``, verified to reproduce ``RuleEngine`` with
``PARAMS16`` exactly (rules, counters, RNG state). Everything not in the six operators —
matching, credit assignment, subsumption, specify, the GA period, N — is fixed.

Evaluation cascade (AlphaEvolve §2.4), tuning seeds only (130–134, never used before):
    S1  6-bit family (k 2, width 6), N 400, 15,000 trials, shift every 3,000, probe every 100;
        seed 130. Gate: runs cleanly and its median recovery <= 2x the seed program's.
    S2  16-bit family (k 3, width 16), N 4000, 45,000 trials, shift every 15,000, probe every
        250; seeds 130–132. Score: median of the six recovery windows (cap 15,000). This is
        the database fitness.
    S3  the Substrate-Opt metric exactly: 16-bit, N 4000, 120,000 trials, shift every 30,000,
        probe every 250 on 256 inputs, recovery = 95 % probe accuracy, cap 30,000; seeds
        130–134; score = pooled median of per-seed medians. Run for the seed program and for
        any candidate whose S2 score ties or beats the best S2 so far; at most 10 S3 runs.
    Compute is priced: every candidate stage run is capped at 2x the seed program's CPU time
    on the same stage and seed (+10 s), and the population may never exceed N.

Budget: 60 proposals. A proposal is one model response, plus at most one repair round when
the response does not parse, fails the sandbox, or crashes in S1 (a slow S1 is not repaired).
Parent: with probability 0.6 one of the three best programs by S2, else any program that
passed S1; up to two inspirations from the five best, shown as diffs against the starting
program; the prompt also states PARAMS16's values and each object's attributes. Sampler seed
7130. Model qwen2.5:7b, temperature 0.8, context 8,192, at most 1,536 generated tokens; the
model digest is recorded and checked before every proposal. Measured: ~5 min per reply on
this CPU, ~13-15 h in all.

PRE-REGISTERED DECISION (committed before the first proposal; judge seeds 80–84 never used):
    winner   the evolved program (not the seed) with the lowest S3 score; ties by S2, then
             earliest. If no evolved program's S3 score is below the seed program's, the
             verdict is FAIL without judging: evolution found nothing on the tuning seeds.
    judge    winner and seed program (= PARAMS16), the S3 metric on seeds 80–84.
    PASS     winner pooled median <= 0.85 x seed pooled median, winner < seed on >= 4/5
             judge seeds, and winner CPU time <= 2.0 x seed CPU time (summed over seeds).
    PARTIAL  winner < seed on >= 4/5 and winner pooled median < seed's, but not PASS.
    FAIL     otherwise.
    If PASS: M7b (pre-registered separately) runs AlphaEvolve's "no evolution" ablation —
    the same budget with the parent always the seed program and no inspirations — before
    any claim that evolution, rather than repeated sampling, produced the gain. If FAIL: at
    this model size and budget, LLM-guided operator evolution does not improve the 16-bit
    learner. That is consistent with AlphaEvolve's own ablation (a small base model alone is
    markedly weaker) and closes this configuration, not M7.
    Reported, not judged: proposals that parsed, passed the sandbox, passed S1; the best-S2
    curve over proposals; the winner's diff; compute ratio; the judge-seed PARAMS16 number
    next to its Substrate-Opt-2 6,000.

Usage:
    python experiments/m7_evolve.py run [--workers 5]      # resumable; checkpoints per proposal
    python experiments/m7_evolve.py status
    python experiments/m7_evolve.py smoke --out DIR       # 1 proposal, tiny stages, not results/
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expzero import DriftingHidden  # noqa: E402
from organism_sim.benchmarks.m2b import _probe  # noqa: E402
from organism_sim.chat_bridge import LLMClient  # noqa: E402
from organism_sim.evolvable import (  # noqa: E402
    ALLOWED_ATTRS,
    ATTRIBUTE_GROUPS,
    EVOLVE_END,
    EVOLVE_START,
    MATH_FACADE,
    NP_FACADE,
    OPERATOR_SIGNATURES,
    SANDBOX_BUILTINS,
    SEED_OPERATORS,
    EvolvableEngine,
    OperatorError,
    SandboxViolation,
    check_source,
    load_operators,
)
from organism_sim.lcs import PARAMS16  # noqa: E402

# ── pre-registered constants ─────────────────────────────────────────────────

MODEL = "qwen2.5:7b"
TEMPERATURE, NUM_CTX, NUM_PREDICT = 0.8, 8192, 1536
PROPOSALS = 60
MAX_REPAIRS = 1
S3_BUDGET = 10
SAMPLER_SEED = 7130
P_EXPLOIT, TOP_PARENTS, TOP_INSPIRATIONS, N_INSPIRATIONS = 0.6, 3, 5, 2
CPU_CAP_RATIO, CPU_CAP_SLACK_S = 2.0, 10.0
S1_GATE_RATIO = 2.0
TUNE_SEEDS = {"S1": (130,), "S2": (130, 131, 132), "S3": (130, 131, 132, 133, 134)}
JUDGE_SEEDS = (80, 81, 82, 83, 84)
PASS_RATIO, PASS_WINS, PASS_COMPUTE = 0.85, 4, 2.0
RECOVERED_AT = 0.95
REPAIRABLE = ("EXCEPTION", "OPERATOR_ERROR", "INVARIANT")
N_PROBE = 256
AUDIT_WINDOW = 2000
MEMORY_CAP_BYTES = 3 * 1024 ** 3

STAGES: Dict[str, Dict[str, int]] = {
    "S1": {"k": 2, "width": 6, "N": 400, "trials": 15_000, "shift_every": 3_000, "probe_every": 100},
    "S2": {"k": 3, "width": 16, "N": 4000, "trials": 45_000, "shift_every": 15_000, "probe_every": 250},
    "S3": {"k": 3, "width": 16, "N": 4000, "trials": 120_000, "shift_every": 30_000, "probe_every": 250},
}

RESULTS = ROOT / "results"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "m7_system.md"
ACTIONS = ("(emit 0)", "(emit 1)")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── one stage run (in a child process) ───────────────────────────────────────

def run_stage(source: str, stage: Dict[str, int], seed: int,
              cpu_cap_s: Optional[float] = None) -> Dict[str, Any]:
    """One learner with the given operators on the drifting hidden-multiplexer family.

    The loop is Substrate-Opt's ``run_config`` with the engine's operators replaced: same
    random streams, probe set, recovery rule and caps. Status is OK or the reason it is not.
    """
    t0 = time.process_time()
    base = {"seed": seed, "recoveries": [], "scores": [], "median": None, "cpu_s": 0.0}
    try:
        ops = load_operators(source)
    except SandboxViolation as exc:
        return dict(base, status="SANDBOX", detail=exc.violations[:5])
    k, width, N = stage["k"], stage["width"], stage["N"]
    trials, shift_every, probe_every = stage["trials"], stage["shift_every"], stage["probe_every"]
    params = replace(PARAMS16, N=N)
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 10_000)
    inputs = ["".join(map(str, prng.integers(0, 2, width))) for _ in range(N_PROBE)]
    env = DriftingHidden(seed, shift_every, k=k, width=width)
    agent = LCSAgent("S", input_len=width, actions=list(ACTIONS), params=params, seed=seed,
                     audit_window=AUDIT_WINDOW)
    agent.engine = EvolvableEngine(agent.cond_len, list(ACTIONS), params, seed=seed, operators=ops)
    recoveries: List[Optional[int]] = []
    window_start: Optional[int] = None
    waiting = False
    try:
        for trial in range(1, trials + 1):
            if env.maybe_shift(trial, trials):
                if window_start is not None and waiting:
                    recoveries.append(None)
                window_start, waiting = trial, True
            bits = "".join(map(str, rng.integers(0, 2, width)))
            agent.act(bits)
            agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
            agent.tick()
            if trial % probe_every == 0 or trial == trials:
                if agent.engine.size > N:
                    return dict(base, status="INVARIANT", detail=f"population {agent.engine.size} > N {N}",
                                cpu_s=time.process_time() - t0)
                if cpu_cap_s is not None and time.process_time() - t0 > cpu_cap_s:
                    return dict(base, status="COMPUTE_CAP", detail=f"> {cpu_cap_s:.0f} s CPU at trial {trial}",
                                cpu_s=time.process_time() - t0)
                acc = _probe(agent, inputs, env.truth)
                if waiting and acc >= RECOVERED_AT and trial > window_start:
                    recoveries.append(trial - window_start)
                    waiting = False
    except OperatorError as exc:
        return dict(base, status="OPERATOR_ERROR", detail=str(exc)[:300], cpu_s=time.process_time() - t0)
    except Exception as exc:  # evolved code may raise anything; it is a failed candidate, not a crash
        return dict(base, status="EXCEPTION", detail=f"{type(exc).__name__}: {str(exc)[:240]}",
                    cpu_s=time.process_time() - t0)
    if window_start is not None and waiting:
        recoveries.append(None)
    scores = [v if v is not None else shift_every for v in recoveries]
    return dict(base, status="OK", recoveries=recoveries, scores=scores,
                median=float(np.median(scores)) if scores else None,
                cpu_s=time.process_time() - t0, ga=agent.engine.counters["ga"],
                macro_end=agent.engine.macro_size(), audit_chain_valid=agent.organism.chain.verify())


def _child(conn, source: str, stage: Dict[str, int], seed: int, cpu_cap_s: Optional[float]) -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))
    if cpu_cap_s is not None:
        hard = int(cpu_cap_s * 1.5 + 60)
        resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
    try:
        conn.send(run_stage(source, stage, seed, cpu_cap_s))
    except MemoryError:
        conn.send({"seed": seed, "status": "MEMORY_CAP", "scores": [], "median": None, "cpu_s": None})
    finally:
        conn.close()


def run_jobs(jobs: Sequence[Tuple[str, Dict[str, int], int, Optional[float]]],
             workers: int) -> List[Dict[str, Any]]:
    """Runs (source, stage, seed, cpu_cap) jobs, each in its own process, at most ``workers``
    at a time. A process that dies or outlives its wall budget is a failed run, not a crash."""
    ctx = mp.get_context("fork")
    results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    pending = list(range(len(jobs)))
    running: Dict[int, Tuple[Any, Any, float, float]] = {}
    while pending or running:
        while pending and len(running) < workers:
            i = pending.pop(0)
            source, stage, seed, cap = jobs[i]
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child, args=(child_conn, source, stage, seed, cap), daemon=True)
            proc.start()
            child_conn.close()
            wall = (cap * 3 + 300) if cap is not None else 6 * 3600
            running[i] = (proc, parent_conn, time.time(), wall)
        for i, (proc, conn, started, wall) in list(running.items()):
            seed = jobs[i][2]
            if conn.poll():
                try:
                    results[i] = conn.recv()
                except EOFError:
                    results[i] = {"seed": seed, "status": "DIED", "scores": [], "median": None, "cpu_s": None}
                proc.join(5)
                del running[i]
            elif not proc.is_alive():
                results[i] = {"seed": seed, "status": f"DIED(exit {proc.exitcode})", "scores": [],
                              "median": None, "cpu_s": None}
                del running[i]
            elif time.time() - started > wall:
                proc.kill()
                proc.join(5)
                results[i] = {"seed": seed, "status": "WALL_TIMEOUT", "scores": [], "median": None,
                              "cpu_s": None}
                del running[i]
        time.sleep(0.2)
    return [r for r in results if r is not None]


def stage_summary(rows: Sequence[Dict[str, Any]], stage_name: str) -> Dict[str, Any]:
    ok = all(r["status"] == "OK" for r in rows)
    out: Dict[str, Any] = {"ok": ok, "rows": sorted(rows, key=lambda r: r["seed"]),
                           "cpu_s": sum(r["cpu_s"] or 0.0 for r in rows)}
    if ok:
        if stage_name == "S3":
            out["score"] = float(np.median([r["median"] for r in rows]))
        else:
            out["score"] = float(np.median([s for r in rows for s in r["scores"]]))
    else:
        bad = next(r for r in rows if r["status"] != "OK")
        out["failure"] = f"{bad['status']}: {bad.get('detail', '')}"[:300]
    return out


# ── responses → programs ─────────────────────────────────────────────────────

class ParseError(ValueError):
    pass


_SEARCH_REPLACE = re.compile(r"<<<<<<< SEARCH[ \t]*\r?\n(.*?)\r?\n=======[ \t]*\r?\n(.*?)\r?\n?>>>>>>> REPLACE", re.S)
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def _replace_once(program: str, search: str, replacement: str) -> str:
    if program.count(search) == 1:
        return program.replace(search, replacement)
    # tolerate trailing whitespace differences only
    norm = lambda s: "\n".join(line.rstrip() for line in s.split("\n"))  # noqa: E731
    p_norm, s_norm = norm(program), norm(search)
    if s_norm and p_norm.count(s_norm) == 1:
        return p_norm.replace(s_norm, replacement)
    raise ParseError(f"SEARCH block not found exactly once: {search.strip().splitlines()[0][:80]!r}"
                     if search.strip() else "empty SEARCH block")


def _function_spans(program: str) -> Dict[str, Tuple[int, int]]:
    tree = ast.parse(program)
    return {n.name: (n.lineno, n.end_lineno) for n in tree.body if isinstance(n, ast.FunctionDef)}


def apply_response(program: str, response: str) -> Tuple[str, str]:
    """(new_program, mode). Mode "diff": SEARCH/REPLACE blocks; mode "functions": complete
    function definitions in a python block replace (or add) top-level functions by name."""
    blocks = _SEARCH_REPLACE.findall(response)
    if blocks:
        new = program
        for search, replacement in blocks:
            new = _replace_once(new, search, replacement)
        mode = "diff"
    else:
        code = "\n\n".join(_FENCE.findall(response)).strip("\n")
        if not code:
            raise ParseError("no SEARCH/REPLACE block and no python code block")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ParseError(f"python block does not parse: line {exc.lineno}: {exc.msg}") from None
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not defs:
            raise ParseError("python block defines no functions")
        code_lines = code.split("\n")
        new = program
        for fn in defs:
            text = "\n".join(code_lines[fn.lineno - 1 - len(fn.decorator_list):fn.end_lineno])
            spans = _function_spans(new)
            lines = new.split("\n")
            if fn.name in spans:
                a, b = spans[fn.name]
                lines[a - 1:b] = text.split("\n")
            else:
                end = next(i for i, line in enumerate(lines) if line.strip() == EVOLVE_END)
                lines[end:end] = [""] + text.split("\n")
            new = "\n".join(lines)
        mode = "functions"
    if EVOLVE_START not in new or EVOLVE_END not in new:
        raise ParseError("the EVOLVE-BLOCK markers must be kept")
    return new, mode


# ── ledger ───────────────────────────────────────────────────────────────────

class Ledger:
    """Append-only JSONL; each record carries the previous record's hash and its own
    (sha256 of the previous hash plus the canonical record without its hash)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.head = "0" * 64
        self.seq = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.head, self.seq = rec["hash"], rec["seq"] + 1

    @staticmethod
    def _digest(prev: str, rec: Dict[str, Any]) -> str:
        body = json.dumps({k: v for k, v in rec.items() if k != "hash"}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()

    def append(self, kind: str, **fields: Any) -> Dict[str, Any]:
        rec = {"seq": self.seq, "time": utc_now(), "kind": kind, "prev": self.head, **fields}
        rec["hash"] = self._digest(self.head, rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        self.head, self.seq = rec["hash"], self.seq + 1
        return rec

    @classmethod
    def verify(cls, path: Path) -> bool:
        prev = "0" * 64
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            rec = json.loads(line)
            if rec["seq"] != i or rec["prev"] != prev or rec["hash"] != cls._digest(prev, rec):
                return False
            prev = rec["hash"]
        return True


# ── model ────────────────────────────────────────────────────────────────────

class OllamaM7(LLMClient):
    """Ollama /api/chat with a generation cap, model unloaded after every call (frees RAM for
    the evaluators), and the model digest exposed for the ledger."""

    def __init__(self, model: str = MODEL, base_url: Optional[str] = None, timeout: float = 1800.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("ORGANISM_OLLAMA") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def digest(self) -> str:
        with urllib.request.urlopen(self.base_url + "/api/tags", timeout=30) as r:
            tags = json.loads(r.read().decode())
        for m in tags.get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
                return str(m.get("digest", ""))
        raise RuntimeError(f"model {self.model} is not installed in Ollama")

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        data = self._post("/api/chat", {
            "model": self.model, "stream": False, "keep_alive": 0,
            "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "messages": [{"role": "system", "content": system}] + messages})
        return str(data.get("message", {}).get("content", ""))


def system_prompt() -> str:
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    grouped = {a for attrs in ATTRIBUTE_GROUPS.values() for a in attrs}
    other = sorted(ALLOWED_ATTRS - grouped - set(vars(NP_FACADE)) - set(vars(MATH_FACADE)))
    fills = {
        "{{SIGNATURES}}": "\n".join(f"    {n}({', '.join(a)})" for n, a in OPERATOR_SIGNATURES.items()),
        "{{PARAMS}}": "\n".join(f"    {k} = {v!r}" for k, v in PARAMS16.to_dict().items()),
        "{{GROUPS}}": "\n".join(f"    {obj}: {', '.join(attrs)}" for obj, attrs in ATTRIBUTE_GROUPS.items()),
        "{{OTHER_METHODS}}": ", ".join(other),
        "{{BUILTINS}}": ", ".join(sorted(SANDBOX_BUILTINS)),
        "{{NUMPY}}": ", ".join(sorted(vars(NP_FACADE))),
        "{{MATH}}": ", ".join(sorted(vars(MATH_FACADE))),
    }
    for key, value in fills.items():
        text = text.replace(key, value)
    return text


def describe(prog: Dict[str, Any]) -> str:
    parts = []
    for name in ("S1", "S2", "S3"):
        s = prog.get(name)
        if s is None:
            continue
        if s["ok"]:
            parts.append(f"{name} score {s['score']:.0f} trials (CPU {s['cpu_s']:.0f} s)")
        else:
            parts.append(f"{name} failed: {s['failure'][:160]}")
    return "; ".join(parts) or "not evaluated"


def build_prompt(parent: Dict[str, Any], inspirations: Sequence[Dict[str, Any]]) -> str:
    out = ["## Current program (you will modify this one)",
           f"Evaluation: {describe(parent)}", "```python", parent["source"].strip("\n"), "```", ""]
    if inspirations:
        out.append("## Other programs from the database, for inspiration (do not modify these)")
        out.append("Each is shown as its difference from the starting program.")
        for j, prog in enumerate(inspirations, 1):
            diff = difflib.unified_diff(SEED_OPERATORS.splitlines(), prog["source"].splitlines(),
                                        "start", f"program_{j}", n=1, lineterm="")
            out += [f"### Program {j}", f"Evaluation: {describe(prog)}", "```diff",
                    "\n".join(diff) or "(identical to the starting program)", "```", ""]
    out += ["## Task",
            "Propose ONE idea that should make the learner reach 95% accuracy in fewer trials "
            "after the hidden task changes. State the idea in one or two sentences, then give "
            "the change: SEARCH/REPLACE blocks, or the complete new definitions of the functions "
            "you change in a single ```python block."]
    return "\n".join(out)


# ── the loop ─────────────────────────────────────────────────────────────────

class Experiment:
    def __init__(self, out_dir: Path, llm: LLMClient, workers: int = 5,
                 stages: Optional[Dict[str, Dict[str, int]]] = None,
                 proposals: int = PROPOSALS, s3_budget: int = S3_BUDGET,
                 log: Callable[[str], None] = print, model_digest: Optional[Callable[[], str]] = None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.llm, self.workers, self.log = llm, workers, log
        self.stages = stages or STAGES
        self.proposals, self.s3_budget = proposals, s3_budget
        self.model_digest = model_digest
        self.state_path = self.out / "state.json"
        self.ledger = Ledger(self.out / "ledger.jsonl")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"programs": [], "proposals_used": 0, "s3_used": 0, "sampler_state": None,
                          "digest": None, "config": json.loads(json.dumps(self.config()))}
        if self.state["config"] != json.loads(json.dumps(self.config())):
            raise RuntimeError("state.json was created under a different configuration "
                               "(prompt, seed program, stages or budgets); refusing to resume")
        self.rng = np.random.default_rng(SAMPLER_SEED)
        if self.state["sampler_state"] is not None:
            self.rng.bit_generator.state = self.state["sampler_state"]

    def config(self) -> Dict[str, Any]:
        return {"model": MODEL, "temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                "proposals": self.proposals, "max_repairs": MAX_REPAIRS, "s3_budget": self.s3_budget,
                "sampler_seed": SAMPLER_SEED, "tune_seeds": TUNE_SEEDS, "judge_seeds": JUDGE_SEEDS,
                "stages": self.stages, "cpu_cap_ratio": CPU_CAP_RATIO, "s1_gate_ratio": S1_GATE_RATIO,
                "system_prompt_sha256": sha256_text(system_prompt()),
                "seed_operators_sha256": sha256_text(SEED_OPERATORS)}

    def save(self) -> None:
        self.state["sampler_state"] = self.rng.bit_generator.state
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    @property
    def programs(self) -> List[Dict[str, Any]]:
        return self.state["programs"]

    @property
    def seed_program(self) -> Dict[str, Any]:
        return self.programs[0]

    def cap(self, stage: str, seed: int) -> float:
        row = next(r for r in self.seed_program[stage]["rows"] if r["seed"] == seed)
        return CPU_CAP_RATIO * row["cpu_s"] + CPU_CAP_SLACK_S

    def evaluate(self, source: str, stage: str, capped: bool = True) -> Dict[str, Any]:
        seeds = TUNE_SEEDS[stage]
        jobs = [(source, self.stages[stage], s, self.cap(stage, s) if capped else None) for s in seeds]
        return stage_summary(run_jobs(jobs, self.workers), stage)

    def best_s2(self) -> float:
        return min(p["S2"]["score"] for p in self.programs if p.get("S2") and p["S2"]["ok"])

    def ranked(self) -> List[Dict[str, Any]]:
        scored = [p for p in self.programs if p.get("S2") and p["S2"]["ok"]]
        return sorted(scored, key=lambda p: (p["S2"]["score"], p["id"]))

    def sample(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ranked = self.ranked()
        passed_s1 = [p for p in self.programs if p["status"] == "SEED" or p.get("S2") is not None]
        if self.rng.random() < P_EXPLOIT or len(passed_s1) <= 1:
            pool = ranked[:TOP_PARENTS]
        else:
            pool = passed_s1
        parent = pool[int(self.rng.integers(len(pool)))]
        others = [p for p in ranked[:TOP_INSPIRATIONS] if p["id"] != parent["id"]]
        take = min(N_INSPIRATIONS, len(others))
        picks = sorted(self.rng.choice(len(others), size=take, replace=False).tolist()) if take else []
        return parent, [others[i] for i in picks]

    def add_program(self, **fields: Any) -> Dict[str, Any]:
        prog = {"id": len(self.programs), "created": utc_now(), **fields}
        self.programs.append(prog)
        return prog

    def init_seed(self) -> None:
        if self.programs:
            return
        self.log(f"{utc_now()} seed program: S1/S2/S3 baselines on tuning seeds (uncapped)")
        prog = self.add_program(source=SEED_OPERATORS, sha256=sha256_text(SEED_OPERATORS), parent=None,
                                mode="seed", status="SEED")
        for stage in ("S1", "S2", "S3"):
            seeds = TUNE_SEEDS[stage]
            res = stage_summary(run_jobs([(SEED_OPERATORS, self.stages[stage], s, None) for s in seeds],
                                         self.workers), stage)
            if not res["ok"]:
                raise RuntimeError(f"seed program failed {stage}: {res['failure']}")
            prog[stage] = res
            self.log(f"{utc_now()} seed {stage} score {res['score']:.0f} cpu {res['cpu_s']:.0f}s")
        self.state["s3_used"] = 1
        self.ledger.append("SEED", program_sha256=prog["sha256"], config=self.state["config"],
                           S1=prog["S1"]["score"], S2=prog["S2"]["score"], S3=prog["S3"]["score"])
        self.save()

    def known(self, source: str) -> bool:
        return any(p.get("sha256") == sha256_text(source) for p in self.programs)

    def propose(self, parent: Dict[str, Any], inspirations: Sequence[Dict[str, Any]]
                ) -> Tuple[Optional[str], str, List[str], Optional[Dict[str, Any]]]:
        """At most 1 + MAX_REPAIRS model calls. A reply gets the repair round when it does not
        parse, fails the sandbox, or crashes in S1 (EXCEPTION, OPERATOR_ERROR, INVARIANT); a slow
        or capped S1 is a result, not a bug, and is not repaired.
        Returns (source, mode, responses, S1 summary); source None = rejected, S1 None with a
        source = duplicate of a known program."""
        system = system_prompt()
        messages = [{"role": "user", "content": build_prompt(parent, inspirations)}]
        responses: List[str] = []
        problems: List[str] = []
        for attempt in range(MAX_REPAIRS + 1):
            t0 = time.time()
            response = self.llm.complete(system, messages)
            responses.append(response)
            self.log(f"{utc_now()}   model reply {len(response)} chars in {time.time() - t0:.0f}s")
            new: Optional[str] = None
            try:
                new, mode = apply_response(parent["source"], response)
                problems = check_source(new)
            except ParseError as exc:
                mode, problems = "parse", [f"PARSE: {exc}"]
            if not problems and new is not None:
                if self.known(new):
                    return new, mode, responses, None
                s1 = self.evaluate(new, "S1")
                crashed = not s1["ok"] and s1["failure"].split(":")[0] in REPAIRABLE
                if not crashed or attempt == MAX_REPAIRS:
                    return new, mode, responses, s1
                problems = [f"RUNTIME: {s1['failure']}"]
            if attempt < MAX_REPAIRS:
                messages = messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": "That change was rejected:\n- " + "\n- ".join(problems[:8])
                     + "\nGive a corrected change in the same format. It is applied to the current "
                       "program shown first, not to your previous attempt."}]
        return None, "rejected:" + "; ".join(problems[:3])[:300], responses, None

    def step(self) -> None:
        n = self.state["proposals_used"]
        if self.model_digest is not None:
            digest = self.model_digest()
            if self.state["digest"] is None:
                self.state["digest"] = digest
            elif digest != self.state["digest"]:
                raise RuntimeError(f"model digest changed: {self.state['digest']} -> {digest}")
        parent, inspirations = self.sample()
        self.log(f"{utc_now()} proposal {n + 1}/{self.proposals}: parent #{parent['id']} "
                 f"inspirations {[p['id'] for p in inspirations]}")
        new, mode, responses, s1 = self.propose(parent, inspirations)
        fields = {"parent": parent["id"], "inspirations": [p["id"] for p in inspirations],
                  "proposal": n + 1, "responses": responses, "mode": mode}
        if new is None:
            prog = self.add_program(source=None, sha256=None, status="REJECTED", **fields)
        elif s1 is None:
            prog = self.add_program(source=new, sha256=sha256_text(new), status="DUPLICATE", **fields)
        else:
            prog = self.add_program(source=new, sha256=sha256_text(new), status="EVALUATING", **fields)
            prog["S1"] = s1
            gate = S1_GATE_RATIO * self.seed_program["S1"]["score"]
            if not s1["ok"]:
                prog["status"] = "S1_FAIL"
            elif s1["score"] > gate:
                prog["status"] = "S1_SLOW"
            else:
                prog["S2"] = s2 = self.evaluate(new, "S2")
                prog["status"] = "S2_FAIL" if not s2["ok"] else "S2_DONE"
                if s2["ok"] and s2["score"] <= self.best_s2_excluding(prog["id"]) \
                        and self.state["s3_used"] < self.s3_budget:
                    prog["S3"] = self.evaluate(new, "S3")
                    self.state["s3_used"] += 1
                    prog["status"] = "S3_DONE" if prog["S3"]["ok"] else "S3_FAIL"
        self.log(f"{utc_now()}   #{prog['id']} {prog['status']} ({mode}) {describe(prog)}")
        self.ledger.append("PROPOSAL", program_id=prog["id"], program_sha256=prog["sha256"],
                           status=prog["status"], mode=mode, parent=parent["id"],
                           inspirations=fields["inspirations"],
                           response_sha256=[sha256_text(r) for r in responses], responses=responses,
                           scores={s: (prog[s]["score"] if prog.get(s) and prog[s]["ok"] else None)
                                   for s in ("S1", "S2", "S3")})
        self.state["proposals_used"] = n + 1
        self.save()

    def best_s2_excluding(self, pid: int) -> float:
        return min(p["S2"]["score"] for p in self.programs
                   if p["id"] != pid and p.get("S2") and p["S2"]["ok"])

    def winner(self) -> Optional[Dict[str, Any]]:
        evolved = [p for p in self.programs[1:] if p.get("S3") and p["S3"]["ok"]]
        if not evolved:
            return None
        best = min(evolved, key=lambda p: (p["S3"]["score"], p["S2"]["score"], p["id"]))
        return best if best["S3"]["score"] < self.seed_program["S3"]["score"] else None

    def run(self) -> None:
        self.init_seed()
        while self.state["proposals_used"] < self.proposals:
            self.step()

    def judge(self) -> Dict[str, Any]:
        if self.state.get("judge") is not None:
            return self.state["judge"]
        win = self.winner()
        out: Dict[str, Any] = {"seeds": list(JUDGE_SEEDS), "proposals_used": self.state["proposals_used"],
                               "s3_used": self.state["s3_used"], "seed_s3_tuning": self.seed_program["S3"]["score"]}
        if win is None:
            out.update(verdict="FAIL", reason="no evolved program beat the seed program on tuning-seed S3")
        else:
            cap = 3.0 * max(r["cpu_s"] for r in self.seed_program["S3"]["rows"]) + 60
            jobs = [(win["source"], self.stages["S3"], s, cap) for s in JUDGE_SEEDS] + \
                   [(SEED_OPERATORS, self.stages["S3"], s, None) for s in JUDGE_SEEDS]
            rows = run_jobs(jobs, self.workers)
            w_rows, s_rows = rows[:len(JUDGE_SEEDS)], rows[len(JUDGE_SEEDS):]
            if not all(r["status"] == "OK" for r in w_rows + s_rows):
                bad = next(r for r in w_rows + s_rows if r["status"] != "OK")
                out.update(verdict="FAIL", reason=f"judge run failed: {bad['status']}", winner_id=win["id"],
                           winner_rows=w_rows, seed_rows=s_rows)
            else:
                wm = float(np.median([r["median"] for r in w_rows]))
                sm = float(np.median([r["median"] for r in s_rows]))
                wins = sum(a["median"] < b["median"] for a, b in zip(w_rows, s_rows))
                compute = sum(r["cpu_s"] for r in w_rows) / sum(r["cpu_s"] for r in s_rows)
                if wm <= PASS_RATIO * sm and wins >= PASS_WINS and compute <= PASS_COMPUTE:
                    verdict = "PASS"
                elif wins >= PASS_WINS and wm < sm:
                    verdict = "PARTIAL"
                else:
                    verdict = "FAIL"
                out.update(verdict=verdict, winner_id=win["id"], winner_sha256=win["sha256"],
                           winner_pooled_median=wm, seed_pooled_median=sm, winner_beats_seed=f"{wins}/{len(JUDGE_SEEDS)}",
                           compute_ratio=round(compute, 3), winner_rows=w_rows, seed_rows=s_rows,
                           winner_source=win["source"])
        out["criteria"] = {"pass_ratio": PASS_RATIO, "pass_wins": PASS_WINS, "pass_compute": PASS_COMPUTE}
        self.ledger.append("JUDGE", **{k: v for k, v in out.items() if k not in ("winner_rows", "seed_rows")})
        self.state["judge"] = out
        self.save()
        return out


def status_lines(state: Dict[str, Any]) -> List[str]:
    progs = state["programs"]
    counts: Dict[str, int] = {}
    for p in progs[1:]:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    lines = [f"proposals {state['proposals_used']}, S3 runs {state['s3_used']}, statuses {counts}"]
    for p in sorted([p for p in progs if p.get("S2") and p["S2"]["ok"]], key=lambda p: p["S2"]["score"])[:8]:
        lines.append(f"  #{p['id']:>3} parent {p.get('parent')} {describe(p)}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M7a: LLM-guided operator evolution")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=5)
    sub.add_parser("status")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        state = json.loads((RESULTS / "m7a" / "state.json").read_text())
        print("\n".join(status_lines(state)))
        print("ledger intact:", Ledger.verify(RESULTS / "m7a" / "ledger.jsonl"))
        return 0

    if args.cmd == "smoke":   # format and plumbing check on tiny stages; never writes results/
        if args.out.resolve().is_relative_to(RESULTS.resolve()):
            raise SystemExit("smoke output must not be inside results/")
        tiny = {"S1": dict(STAGES["S1"], trials=3000, shift_every=1000),
                "S2": dict(STAGES["S2"], trials=4000, shift_every=2000, N=1000),
                "S3": dict(STAGES["S3"], trials=4000, shift_every=2000, N=1000)}
        llm = OllamaM7()
        exp = Experiment(args.out, llm, workers=args.workers, stages=tiny, proposals=1, s3_budget=2,
                         model_digest=llm.digest)
        exp.run()
        print("\n".join(status_lines(exp.state)))
        print("ledger intact:", Ledger.verify(args.out / "ledger.jsonl"))
        return 0

    out = RESULTS / "m7a"
    logf = open(RESULTS / "m7a_run.log", "a", encoding="utf-8")

    def log(s: str) -> None:
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    llm = OllamaM7()
    exp = Experiment(out, llm, workers=args.workers, log=log, model_digest=llm.digest)
    log(f"{utc_now()} START M7a proposals={exp.proposals} used={exp.state['proposals_used']} "
        f"workers={args.workers} model={MODEL}")
    t0 = time.time()
    exp.run()
    result = exp.judge()
    (RESULTS / "m7a_eval_seeds80-84.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log(f"{utc_now()} VERDICT {result['verdict']} " + json.dumps({k: result.get(k) for k in (
        "winner_id", "winner_pooled_median", "seed_pooled_median", "winner_beats_seed", "compute_ratio",
        "reason")}))
    log(f"{utc_now()} DONE {time.time() - t0:.0f}s ledger intact {Ledger.verify(out / 'ledger.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
