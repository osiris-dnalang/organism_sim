"""M7a: evolvable operators, sandbox, evaluation stages, response parsing, ledger, and the loop
end to end with a scripted model (never Ollama)."""
import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import m7_evolve as m7  # noqa: E402
import substrate_opt as so  # noqa: E402

from organism_sim.agent import LCSAgent  # noqa: E402
from organism_sim.benchmarks.expzero import DriftingHidden  # noqa: E402
from organism_sim.chat_bridge import ScriptedClient  # noqa: E402
from organism_sim.evolvable import (  # noqa: E402
    SEED_OPERATORS,
    EvolvableEngine,
    SandboxViolation,
    check_source,
    load_operators,
)
from organism_sim.lcs import PARAMS16  # noqa: E402

TINY = {"S1": {"k": 2, "width": 6, "N": 200, "trials": 2000, "shift_every": 1000, "probe_every": 100},
        "S2": {"k": 3, "width": 16, "N": 200, "trials": 2000, "shift_every": 1000, "probe_every": 250},
        "S3": {"k": 3, "width": 16, "N": 200, "trials": 2000, "shift_every": 1000, "probe_every": 250}}


def _engine_state(evolvable, width, k, params, trials=1500, seed=130):
    rng = np.random.default_rng(seed)
    env = DriftingHidden(seed, 500, k=k, width=width)
    agent = LCSAgent("S", input_len=width, actions=["(emit 0)", "(emit 1)"], params=params, seed=seed)
    if evolvable:
        agent.engine = EvolvableEngine(agent.cond_len, ["(emit 0)", "(emit 1)"], params, seed=seed)
    for trial in range(1, trials + 1):
        env.maybe_shift(trial, trials)
        bits = "".join(map(str, rng.integers(0, 2, width)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == env.truth(bits) else 0.0)
        agent.tick()
    e = agent.engine
    return ([(r.condition, r.action, r.prediction, r.error, r.fitness, r.experience, r.numerosity,
              r.time_stamp, r.as_size, r.id) for r in e.rules],
            dict(e.counters), e.rng.bit_generator.state["state"]["state"])


@pytest.mark.parametrize("width,k,params", [
    (16, 3, replace(PARAMS16, N=600)),                                   # tournament + specify
    (6, 2, replace(PARAMS16, N=120)),                                    # deletion-heavy
    (16, 3, replace(PARAMS16, N=300, selection="roulette", specify=False)),
], ids=["params16", "small-N", "roulette"])
def test_seed_operators_reproduce_rule_engine_exactly(width, k, params):
    base = _engine_state(False, width, k, params)
    evo = _engine_state(True, width, k, params)
    assert base[1]["ga"] > 0 and base[1]["delete"] > 0
    assert evo == base


def test_stage_runner_is_substrate_opts_metric(monkeypatch):
    family = {"k": 3, "width": 16, "N": 300, "shift_every": 1000, "trials": 3000, "probe_every": 250}
    monkeypatch.setitem(so.FAMILIES, "tiny16", dict(family, n_inject=0))
    ref = so.run_config({"N": 300, "selection": "tournament", "specify": True}, 131, family="tiny16")
    ours = m7.run_stage(SEED_OPERATORS, family, 131)
    assert ours["status"] == "OK"
    assert ours["recoveries"] == ref["recoveries"] and ours["scores"] == ref["scores"]


def test_seed_ranges_are_fresh_and_disjoint():
    used = set(range(0, 5)) | set(range(10, 15)) | set(range(20, 25)) | set(range(30, 35)) \
        | set(range(40, 45)) | set(range(50, 55)) | set(range(60, 65)) | set(range(70, 75)) \
        | set(range(100, 125))
    tune = {s for seeds in m7.TUNE_SEEDS.values() for s in seeds}
    assert not tune & used and not set(m7.JUDGE_SEEDS) & used and not tune & set(m7.JUDGE_SEEDS)


def test_pre_registered_s3_stage_is_the_substrate_opt_family():
    s3, fam = m7.STAGES["S3"], so.FAMILIES["16bit"]
    assert (s3["k"], s3["width"], s3["trials"], s3["shift_every"], s3["probe_every"]) == \
        (fam["k"], fam["width"], fam["trials"], fam["shift_every"], fam["probe_every"])
    assert replace(PARAMS16, N=s3["N"]) == PARAMS16


# ── sandbox ──────────────────────────────────────────────────────────────────

def _with(fn_body_line, fn="cover_condition"):
    """SEED_OPERATORS with one line inserted as the first statement after an operator's
    docstring (located with ast, so multi-line docstrings are skipped correctly)."""
    node = next(n for n in ast.parse(SEED_OPERATORS).body
                if isinstance(n, ast.FunctionDef) and n.name == fn)
    first = node.body[1] if isinstance(node.body[0], ast.Expr) else node.body[0]
    lines = SEED_OPERATORS.split("\n")
    lines.insert(first.lineno - 1, "    " + fn_body_line)
    return "\n".join(lines)


@pytest.mark.parametrize("line,code", [
    ("import os", "FORBIDDEN_SYNTAX"),
    ("x = __import__('os')", "FORBIDDEN_NAME"),
    ("engine._insert(None)", "FORBIDDEN_ATTRIBUTE"),
    ("k = register.__class__", "FORBIDDEN_ATTRIBUTE"),
    ("f = getattr(engine, 'rules')", "FORBIDDEN_NAME"),
    ("h = open('/etc/passwd')", "FORBIDDEN_NAME"),
    ("z = np.load('x.npy')", "FORBIDDEN_ATTRIBUTE"),
    ("np.zeros(3).tofile('x')", "FORBIDDEN_ATTRIBUTE"),
    ("engine.rules.append(engine.rules[0])", "STATE_WRITE"),
    ("engine.rules[0] = None", "STATE_WRITE"),
    ("engine.p.mu = 0.5", "STATE_WRITE"),
    ("g = (x for x in register); f = g.gi_frame", "FORBIDDEN_ATTRIBUTE"),
    ("s = f'{register.__class__}'", "FORBIDDEN_ATTRIBUTE"),
    ("s = '{0.__class__}'.format(register)", "FORBIDDEN_ATTRIBUTE"),
    ("with open('x') as h: pass", "FORBIDDEN_SYNTAX"),
    ("global q", "FORBIDDEN_SYNTAX"),
])
def test_sandbox_rejects(line, code):
    problems = check_source(_with(line))
    assert any(p.startswith(code) for p in problems), problems
    with pytest.raises(SandboxViolation):
        load_operators(_with(line))


def test_sandbox_structure_rules():
    assert check_source(SEED_OPERATORS) == []
    missing = SEED_OPERATORS.replace("def deletion_votes(engine):", "def votes(engine):")
    assert any(p.startswith("MISSING") for p in check_source(missing))
    wrong = SEED_OPERATORS.replace("def mutate(engine, condition, action, register):",
                                   "def mutate(engine, condition, action):")
    assert any(p.startswith("SIGNATURE") for p in check_source(wrong))
    assert any(p.startswith("TOP_LEVEL") for p in check_source(SEED_OPERATORS + "\nX = 1\n"))
    assert any(p.startswith("SYNTAX") for p in check_source(SEED_OPERATORS + "\ndef (:\n"))


def test_local_lists_may_be_mutated():
    ok = _with("tmp = []; tmp.append(1); tmp.sort()")
    assert check_source(ok) == []


@pytest.mark.parametrize("line,fn,detail", [
    ('return "0" * len(register) if register[0] == "1" else "1" * len(register)', "cover_condition",
     "must match the register"),
    ("return [1.0]", "deletion_votes", "one finite"),
    ("return (2.0, 0.0, 0.0)", "child_estimates", "[0, 1]"),
    ('return ("x" * len(condition), action)', "mutate", "condition must be"),
    ("return (aset[0], aset[0], aset[0])", "select_parents", "two rules"),
])
def test_invalid_operator_outputs_fail_the_candidate(line, fn, detail):
    res = m7.run_stage(_with(line, fn), TINY["S1"], 130)
    assert res["status"] == "OPERATOR_ERROR" and detail in res["detail"]


def test_compute_cap_and_exceptions_fail_the_candidate():
    slow = _with("w = sum(range(300000))", "deletion_votes")
    assert m7.run_stage(slow, TINY["S1"], 130, cpu_cap_s=0.2)["status"] == "COMPUTE_CAP"
    boom = _with("q = 1 // 0", "select_parents")
    res = m7.run_stage(boom, TINY["S1"], 130)
    assert res["status"] == "EXCEPTION" and "ZeroDivisionError" in res["detail"]


def test_run_jobs_isolates_and_preserves_order():
    rows = m7.run_jobs([(SEED_OPERATORS, TINY["S1"], 130, None),
                        (_with("q = 1 // 0", "select_parents"), TINY["S1"], 131, 30.0)], workers=2)
    assert [r["seed"] for r in rows] == [130, 131]
    assert rows[0]["status"] == "OK" and rows[1]["status"] == "EXCEPTION"
    direct = m7.run_stage(SEED_OPERATORS, TINY["S1"], 130)
    assert (rows[0]["recoveries"], rows[0]["scores"]) == (direct["recoveries"], direct["scores"])


# ── responses ────────────────────────────────────────────────────────────────

DIFF = """Raise offspring fitness.
<<<<<<< SEARCH
            0.1 * (p1.fitness + p2.fitness) / 2)
=======
            0.2 * (p1.fitness + p2.fitness) / 2)
>>>>>>> REPLACE
"""
FUNCS = """Cover more generally, via a helper.
```python
def wild_rate(engine):
    return 0.5

def cover_condition(engine, register, action):
    return "".join("#" if engine.rng.random() < wild_rate(engine) else c for c in register)
```
"""


def test_apply_response_diff_and_functions():
    new, mode = m7.apply_response(SEED_OPERATORS, DIFF)
    assert mode == "diff" and "0.2 * (p1.fitness" in new and check_source(new) == []
    new, mode = m7.apply_response(SEED_OPERATORS, FUNCS)
    assert mode == "functions" and "def wild_rate(engine)" in new and check_source(new) == []
    assert new.count("def cover_condition") == 1 and "def deletion_votes" in new
    assert new.index("def wild_rate") < new.index(m7.EVOLVE_END)


@pytest.mark.parametrize("response", [
    "no code at all",
    "<<<<<<< SEARCH\nnot in the program\n=======\nx\n>>>>>>> REPLACE\n",
    "```python\ndef broken(:\n```",
])
def test_apply_response_rejects(response):
    with pytest.raises(m7.ParseError):
        m7.apply_response(SEED_OPERATORS, response)


# ── ledger ───────────────────────────────────────────────────────────────────

def test_ledger_chain_and_tamper_detection(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = m7.Ledger(path)
    led.append("A", x=1)
    led.append("B", y=[1, 2])
    assert m7.Ledger.verify(path)
    assert m7.Ledger(path).seq == 2                     # resumes at the head
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["x"] = 2
    path.write_text(json.dumps(rec, sort_keys=True) + "\n" + lines[1] + "\n")
    assert not m7.Ledger.verify(path)


# ── the loop, end to end ─────────────────────────────────────────────────────

CRASH = """Normalise offspring fitness.
```python
def child_estimates(engine, p1, p2):
    q = 1 // 0
    return (p1.prediction, p1.error, p1.fitness)
```
"""
FIX = """Fixed: keep the parents' mean, start offspring fitness higher.
```python
def child_estimates(engine, p1, p2):
    return ((p1.prediction + p2.prediction) / 2, (p1.error + p2.error) / 2,
            0.3 * (p1.fitness + p2.fitness) / 2)
```
"""


def test_system_prompt_states_parameters_and_attributes_per_object():
    text = m7.system_prompt()
    assert "selection = 'tournament'" in text and "N = 4000" in text
    assert "engine.p (parameters): N, beta" in text and "rule: condition, action" in text
    assert "{{" not in text


def test_loop_end_to_end_with_scripted_model(tmp_path):
    # a top-level import is not spliced in (only function definitions are), so the banned
    # import sits inside the function, where the sandbox must catch it
    bad = "Use os.\n```python\ndef cover_condition(engine, register, action):\n    import os\n    return register\n```"
    llm = ScriptedClient([DIFF, bad, FUNCS, CRASH, FIX, "nothing useful", "still nothing"])
    exp = m7.Experiment(tmp_path, llm, workers=3, stages=TINY, proposals=4, s3_budget=3, log=lambda s: None)
    exp.run()
    progs = exp.state["programs"]
    evaluated = ("S1_SLOW", "S2_DONE", "S2_FAIL", "S3_DONE", "S3_FAIL")
    assert progs[0]["status"] == "SEED" and exp.state["proposals_used"] == 4
    assert progs[1]["mode"] == "diff" and progs[1]["status"] in evaluated
    assert progs[2]["mode"] == "functions" and len(progs[2]["responses"]) == 2     # sandbox repair
    assert progs[3]["status"] in evaluated and len(progs[3]["responses"]) == 2     # runtime repair
    assert "RUNTIME: EXCEPTION: ZeroDivisionError" in llm.calls[4]["messages"][-1]["content"]
    assert progs[4]["status"] == "REJECTED" and len(progs[4]["responses"]) == 2
    assert m7.Ledger.verify(tmp_path / "ledger.jsonl")
    assert len(llm.calls) == 7 and "EVOLVE-BLOCK-START" in llm.calls[0]["messages"][0]["content"]
    assert "cover_condition(engine, register, action)" in llm.calls[0]["system"]

    again = m7.Experiment(tmp_path, ScriptedClient([]), workers=3, stages=TINY, proposals=4,
                          s3_budget=3, log=lambda s: None)
    again.run()                                       # resumes: nothing left to propose
    assert again.state["proposals_used"] == 4
    verdict = again.judge()
    assert verdict["verdict"] in ("PASS", "PARTIAL", "FAIL")
    assert again.judge() == verdict                   # judged once
    with pytest.raises(RuntimeError, match="different configuration"):
        m7.Experiment(tmp_path, ScriptedClient([]), workers=3, stages=TINY, proposals=5, s3_budget=3)
