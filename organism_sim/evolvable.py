"""
organism_sim.evolvable — learner operators as evolvable, sandboxed source (M7a)
===============================================================================

M7 on the milestone ladder treats the learner's own operators as genomes. This module makes
six of ``RuleEngine``'s operators replaceable by source code, so a search process (M7a: an
LLM proposing edits, AlphaEvolve-style) can change them while everything else stays fixed:

    cover_condition(engine, register, action) -> condition      covering
    select_parents(engine, aset) -> (rule, rule)                 GA parent selection
    crossover(engine, p1, p2) -> (cond1, cond2, act1, act2)      GA recombination
    mutate(engine, condition, action, register) -> (cond, act)   GA mutation
    child_estimates(engine, p1, p2) -> (prediction, error, fitness)   offspring initialisation
    deletion_votes(engine) -> [vote per macro-rule]              population control

What stays fixed: matching, the prediction array, action selection, the Widrow-Hoff /
accuracy / fitness update, action-set and GA subsumption, the specify operator, the GA
period, N. ``SEED_OPERATORS`` is a refactor of ``RuleEngine``'s own code that makes the same
random draws in the same order, so ``EvolvableEngine(SEED_OPERATORS)`` reproduces
``RuleEngine`` exactly (tests/test_m7_evolve.py checks rules, counters and RNG state).

Sandbox. Evolved source is checked before it is executed. The guarantee is narrow and
deliberate: evolved code cannot reach the task's ground truth or the probe set (no imports,
no names or attributes beginning with "_", an attribute allowlist, restricted builtins, numpy
and math only through whitelisted facades), and cannot assign to attributes or call mutating
methods on engine state directly. It is *not* a guarantee that evolved code is sensible;
output validation (``OperatorError``), the population-size invariant and the CPU caps in
``experiments/m7_evolve.py`` handle that.
"""

from __future__ import annotations

import ast
import builtins
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .lcs import Params, Rule, RuleEngine

EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"

OPERATOR_SIGNATURES: Dict[str, Tuple[str, ...]] = {
    "cover_condition": ("engine", "register", "action"),
    "select_parents": ("engine", "aset"),
    "crossover": ("engine", "p1", "p2"),
    "mutate": ("engine", "condition", "action", "register"),
    "child_estimates": ("engine", "p1", "p2"),
    "deletion_votes": ("engine",),
}

SEED_OPERATORS = '''\
# EVOLVE-BLOCK-START
def cover_condition(engine, register, action):
    """Condition for a new rule covering `register`: each bit kept or generalised to '#'."""
    return "".join("#" if engine.rng.random() < engine.p.p_wild else c for c in register)


def select_parents(engine, aset):
    """Two parents from the action set: tournament on fitness per micro-rule, or roulette."""
    p = engine.p
    if p.selection == "tournament":
        def pick():
            cands = [r for r in aset if r.numerosity > 0 and engine.rng.random() < p.tau]
            if not cands:
                cands = [aset[int(engine.rng.integers(len(aset)))]]
            return max(cands, key=lambda r: (r.fitness / max(r.numerosity, 1), -r.id))
    else:
        fit = np.array([max(r.fitness, 1e-12) for r in aset])

        def pick():
            return aset[int(engine.rng.choice(len(aset), p=fit / fit.sum()))]
    return pick(), pick()


def crossover(engine, p1, p2):
    """Two-point crossover of the conditions; actions swapped with probability 0.5."""
    c1, c2 = list(p1.condition), list(p2.condition)
    a1, a2 = p1.action, p2.action
    if engine.rng.random() < engine.p.chi and engine.cond_len > 1:
        x, y = sorted(engine.rng.integers(0, engine.cond_len + 1, size=2))
        c1[x:y], c2[x:y] = c2[x:y], c1[x:y]
        if engine.rng.random() < 0.5:
            a1, a2 = a2, a1
    return "".join(c1), "".join(c2), a1, a2


def mutate(engine, condition, action, register):
    """Per-bit mutation toward the register (bit <-> '#'); random action with probability mu."""
    out = []
    for c, b in zip(condition, register):
        if engine.rng.random() < engine.p.mu:
            out.append("#" if c != "#" else b)
        else:
            out.append(c)
    if engine.rng.random() < engine.p.mu:
        action = engine.actions[int(engine.rng.integers(len(engine.actions)))]
    return "".join(out), action


def child_estimates(engine, p1, p2):
    """Initial prediction, error and fitness of an offspring."""
    return ((p1.prediction + p2.prediction) / 2, (p1.error + p2.error) / 2,
            0.1 * (p1.fitness + p2.fitness) / 2)


def deletion_votes(engine):
    """One deletion vote per macro-rule (same order as engine.rules); one is drawn
    proportionally each time the population exceeds N."""
    p = engine.p
    total = engine.size
    avg_f = sum(r.fitness for r in engine.rules) / total
    votes = []
    for r in engine.rules:
        v = r.as_size * r.numerosity
        fpm = r.fitness / r.numerosity
        if r.experience > p.theta_del and fpm < p.delta * avg_f and fpm > 0:
            v *= avg_f / fpm
        votes.append(v)
    return votes
# EVOLVE-BLOCK-END
'''


# ── sandbox ──────────────────────────────────────────────────────────────────

_NP_NAMES = ("array", "asarray", "zeros", "ones", "full", "arange", "exp", "log", "log1p", "sqrt",
             "mean", "median", "std", "var", "sum", "prod", "cumsum", "argmax", "argmin", "argsort",
             "sort", "where", "clip", "maximum", "minimum", "abs", "isfinite", "isnan", "nan_to_num",
             "dot", "power", "floor", "ceil", "round", "percentile", "quantile", "count_nonzero",
             "unique", "concatenate", "stack", "tanh", "float64", "int64", "inf", "pi")
_MATH_NAMES = ("exp", "log", "log1p", "sqrt", "floor", "ceil", "tanh", "fabs", "isfinite", "inf",
               "pi")
NP_FACADE = SimpleNamespace(**{n: getattr(np, n) for n in _NP_NAMES})
MATH_FACADE = SimpleNamespace(**{n: getattr(math, n) for n in _MATH_NAMES})

SANDBOX_BUILTINS = {n: getattr(builtins, n)
             for n in ("abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
                       "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next",
                       "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
                       "ValueError", "ZeroDivisionError")}

_PARAM_FIELDS = tuple(Params.__dataclass_fields__)
_RULE_FIELDS = ("condition", "action", "prediction", "error", "fitness", "experience",
                "numerosity", "time_stamp", "as_size", "id", "key")
_ENGINE_ATTRS = ("rng", "p", "rules", "t", "cond_len", "actions", "size", "match_set",
                 "action_set")
_RNG_METHODS = ("random", "integers", "choice", "permutation", "normal", "uniform",
                "standard_normal", "binomial", "exponential", "poisson")
_ARRAY_METHODS = ("astype", "tolist", "copy", "shape", "item", "nonzero", "reshape", "all", "any",
                  "max", "min", "argmax", "argmin", "argsort", "mean", "std", "cumsum", "sum")
_STR_METHODS = ("join", "count", "replace", "startswith", "endswith", "find", "index", "split",
                "strip", "upper", "lower")
_MUTATING = ("append", "extend", "insert", "pop", "remove", "sort", "reverse", "clear", "update",
             "setdefault", "add", "discard")
_CONTAINER_METHODS = _MUTATING + ("copy", "get", "items", "keys", "values", "union",
                                  "intersection")
ATTRIBUTE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "engine": _ENGINE_ATTRS, "engine.p (parameters)": _PARAM_FIELDS, "rule": _RULE_FIELDS,
    "engine.rng": _RNG_METHODS}
ALLOWED_ATTRS = frozenset(_PARAM_FIELDS + _RULE_FIELDS + _ENGINE_ATTRS + _RNG_METHODS
                          + _ARRAY_METHODS + _STR_METHODS + _CONTAINER_METHODS + _NP_NAMES
                          + _MATH_NAMES)
FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "open", "globals", "locals", "vars",
                             "getattr", "setattr", "delattr", "hasattr", "type", "object",
                             "super", "print", "input", "breakpoint", "help", "dir", "id",
                             "memoryview", "classmethod", "staticmethod", "property"})
MAX_SOURCE_CHARS = 12_000
_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                    ast.AsyncFunctionDef, ast.Await, ast.With, ast.AsyncWith, ast.AsyncFor)


class SandboxViolation(ValueError):
    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        super().__init__("; ".join(self.violations))


class OperatorError(RuntimeError):
    """An evolved operator returned something the engine cannot use."""


def _rooted_in_attribute(node: ast.AST) -> bool:
    """True for receivers like ``engine.rules`` or ``engine.rules[0]`` (state reached through an
    attribute), False for plain local names."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Attribute)


def check_source(source: str) -> List[str]:
    """Every sandbox violation in ``source`` (empty list = acceptable)."""
    if len(source) > MAX_SOURCE_CHARS:
        return [f"TOO_LONG: {len(source)} chars > {MAX_SOURCE_CHARS}"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SYNTAX: line {exc.lineno}: {exc.msg}"]
    out: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            out.append(f"TOP_LEVEL: line {node.lineno}: only function definitions are allowed")
    defined = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, args in OPERATOR_SIGNATURES.items():
        fn = defined.get(name)
        if fn is None:
            out.append(f"MISSING: {name}{args}")
        elif tuple(a.arg for a in fn.args.args) != args or fn.args.vararg or fn.args.kwarg:
            out.append(f"SIGNATURE: {name} must take exactly {args}")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, _FORBIDDEN_NODES):
            out.append(f"FORBIDDEN_SYNTAX: line {line}: {type(node).__name__}")
        elif isinstance(node, ast.Name):
            if (node.id.startswith("_") and node.id != "_") or node.id in FORBIDDEN_NAMES:
                out.append(f"FORBIDDEN_NAME: line {line}: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr not in ALLOWED_ATTRS:
                out.append(f"FORBIDDEN_ATTRIBUTE: line {line}: .{node.attr}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) or (
                            isinstance(sub, ast.Subscript) and _rooted_in_attribute(sub.value)):
                        out.append(f"STATE_WRITE: line {line}: operators return values; "
                                   "they do not assign to engine or rule state")
                        break
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _MUTATING and _rooted_in_attribute(node.func.value):
                out.append(f"STATE_WRITE: line {line}: .{node.func.attr}() on engine or rule state")
    return out


def load_operators(source: str) -> SimpleNamespace:
    """Checks ``source`` and executes it in a restricted namespace; returns the six operators."""
    violations = check_source(source)
    if violations:
        raise SandboxViolation(violations)
    namespace: Dict[str, Any] = {"__builtins__": dict(SANDBOX_BUILTINS), "np": NP_FACADE,
                                 "math": MATH_FACADE}
    exec(compile(source, "<evolved-operators>", "exec"), namespace)   # checked above
    return SimpleNamespace(**{name: namespace[name] for name in OPERATOR_SIGNATURES})


def evolve_block(program: str) -> str:
    """The text between the EVOLVE-BLOCK markers (markers included)."""
    start, end = program.index(EVOLVE_START), program.index(EVOLVE_END) + len(EVOLVE_END)
    return program[start:end]


# ── engine ───────────────────────────────────────────────────────────────────

_COND_CHARS = frozenset("01#")


def _check_condition(cond: Any, length: int, where: str) -> str:
    if not isinstance(cond, str) or len(cond) != length or not set(cond) <= _COND_CHARS:
        raise OperatorError(f"{where}: condition must be a {length}-char string over 0/1/#, "
                            f"got {cond!r:.60}")
    return cond


def _matches(cond: str, register: str) -> bool:
    return all(c == "#" or c == b for c, b in zip(cond, register))


def _check_unit_float(x: Any, where: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise OperatorError(f"{where}: not a number: {x!r:.40}") from None
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise OperatorError(f"{where}: must lie in [0, 1], got {v}")
    return v


class EvolvableEngine(RuleEngine):
    """``RuleEngine`` with the six operators delegated to (possibly evolved) source code."""

    def __init__(self, cond_len: int, actions: Sequence[str], params: Params = None,
                 seed: int = 0, rules: Sequence[Rule] = (), operators: SimpleNamespace = None):
        self.ops = operators if operators is not None else load_operators(SEED_OPERATORS)
        super().__init__(cond_len, actions, params, seed=seed, rules=rules)

    def _check_action(self, action: Any, where: str) -> str:
        if action not in self.actions:
            raise OperatorError(f"{where}: action must be one of {self.actions}, got {action!r:.40}")
        return action

    def _cover(self, register: str, action: str) -> Rule:
        cond = _check_condition(self.ops.cover_condition(self, register, action), self.cond_len,
                                "cover_condition")
        if not _matches(cond, register):
            raise OperatorError("cover_condition: the covering condition must match the register")
        r = self._new_rule(cond, action)
        self._insert(r)
        self._delete_excess()
        self.counters["cover"] += 1
        return r

    def _delete_excess(self) -> None:
        p = self.p
        while self.size > p.N and self.rules:
            try:
                votes = np.asarray(self.ops.deletion_votes(self), dtype=float)
            except (TypeError, ValueError) as exc:
                raise OperatorError(f"deletion_votes: {exc}") from None
            if votes.shape != (len(self.rules),) or not np.all(np.isfinite(votes)) \
                    or (votes < 0).any() or votes.sum() <= 0:
                raise OperatorError("deletion_votes: need one finite, non-negative vote per "
                                    f"rule ({len(self.rules)}) with a positive sum")
            i = int(self.rng.choice(len(self.rules), p=votes / votes.sum()))
            self.rules[i].numerosity -= 1
            if self.rules[i].numerosity <= 0:
                self.rules.pop(i)
                self._bump()
            self.counters["delete"] += 1

    def _run_ga(self, aset: List[Rule], register: str) -> None:
        p = self.p
        num = sum(r.numerosity for r in aset)
        if num == 0:
            return
        mean_ts = sum(r.time_stamp * r.numerosity for r in aset) / num
        if self.t - mean_ts <= p.theta_ga:
            return
        for r in aset:
            r.time_stamp = self.t
        self.counters["ga"] += 1
        parents = self.ops.select_parents(self, aset)
        if not (isinstance(parents, (tuple, list)) and len(parents) == 2
                and all(any(x is q for q in aset) for x in parents)):
            raise OperatorError("select_parents: must return two rules from the action set")
        p1, p2 = parents
        out = self.ops.crossover(self, p1, p2)
        if not (isinstance(out, (tuple, list)) and len(out) == 4):
            raise OperatorError("crossover: must return (cond1, cond2, action1, action2)")
        c1, c2, a1, a2 = out
        for c, a in ((c1, a1), (c2, a2)):
            _check_condition(c, self.cond_len, "crossover")
            self._check_action(a, "crossover")
            mutated = self.ops.mutate(self, c, a, register)
            if not (isinstance(mutated, (tuple, list)) and len(mutated) == 2):
                raise OperatorError("mutate: must return (condition, action)")
            cond = _check_condition(mutated[0], self.cond_len, "mutate")
            act = self._check_action(mutated[1], "mutate")
            est = self.ops.child_estimates(self, p1, p2)
            if not (isinstance(est, (tuple, list)) and len(est) == 3):
                raise OperatorError("child_estimates: must return (prediction, error, fitness)")
            pe, ee, fe = (_check_unit_float(v, "child_estimates") for v in est)
            child = self._new_rule(cond, act, p=pe, e=ee, f=fe)
            if self._ga_subsumes(p1, child) or self._ga_subsumes(p2, child):
                continue
            self._insert(child)
        self._delete_excess()


__all__ = ["ALLOWED_ATTRS", "ATTRIBUTE_GROUPS", "EVOLVE_END", "EVOLVE_START", "EvolvableEngine", "MATH_FACADE",
           "NP_FACADE", "OPERATOR_SIGNATURES", "OperatorError", "SANDBOX_BUILTINS", "SEED_OPERATORS",
           "SandboxViolation", "check_source", "evolve_block", "load_operators"]
