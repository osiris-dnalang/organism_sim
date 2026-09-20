"""
organism_sim.rules — Ternary conditions and the action DSL
==========================================================

Condition: a string over ``{0, 1, #}``; ``#`` matches either bit. Matched
against a register string of the same length (input bits ⊕ state bits ⊕ last
received symbol bits).

Action DSL: closed s-expression language, tree-walked by ``Interpreter`` under a
hard step budget. No I/O, no side effects beyond the ``Context`` it is given.

    (emit SYM)             append SYM to ctx.emits
    (set VAR EXPR)         ctx.vars[VAR] = eval(EXPR)
    (send NEIGHBOR SYM)    append (NEIGHBOR, SYM) to ctx.sends
    (sever NEIGHBOR)       append NEIGHBOR to ctx.severs
    (route NEIGHBOR)       append NEIGHBOR to ctx.routes
    (adjust PARAM DELTA)   ctx.adjusts[PARAM] += DELTA
    (if EXPR THEN [ELSE])  conditional
    (seq A B ...)          sequence

Expressions: literals (int/float/symbol), (reg I), (var NAME), (last), (not E),
(eq A B), (and A B), (or A B), (add A B).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

Node = Union[int, float, str, list]

OPS = ("emit", "set", "send", "sever", "route", "adjust", "if", "seq")
EXPR_OPS = ("reg", "var", "last", "not", "eq", "and", "or", "add")


class BudgetExceeded(RuntimeError):
    pass


class DSLError(ValueError):
    pass


# ── conditions ───────────────────────────────────────────────────────────────

def matches(condition: str, register: str) -> bool:
    if len(condition) != len(register):
        return False
    return all(c == "#" or c == r for c, r in zip(condition, register))


def subsumes(general: str, specific: str) -> bool:
    """``general`` matches everything ``specific`` matches, and is strictly more general."""
    if len(general) != len(specific) or general == specific:
        return False
    return all(g == "#" or g == s for g, s in zip(general, specific))


def specificity(condition: str) -> int:
    return sum(1 for c in condition if c != "#")


# ── s-expressions ────────────────────────────────────────────────────────────

_TOK = re.compile(r"\(|\)|[^\s()]+")


def parse_sexpr(text: str) -> Node:
    toks = _TOK.findall(text)
    if not toks:
        raise DSLError("empty action")
    pos = 0

    def read() -> Node:
        nonlocal pos
        if pos >= len(toks):
            raise DSLError("unexpected end of action")
        t = toks[pos]
        pos += 1
        if t == "(":
            out: List[Node] = []
            while pos < len(toks) and toks[pos] != ")":
                out.append(read())
            if pos >= len(toks):
                raise DSLError("missing ')'")
            pos += 1
            return out
        if t == ")":
            raise DSLError("unexpected ')'")
        for cast in (int, float):
            try:
                return cast(t)
            except ValueError:
                pass
        return t

    node = read()
    if pos != len(toks):
        raise DSLError("trailing tokens")
    return node


def to_sexpr(node: Node) -> str:
    if isinstance(node, list):
        return "(" + " ".join(to_sexpr(n) for n in node) + ")"
    return str(node)


# ── interpreter ──────────────────────────────────────────────────────────────

@dataclass
class Context:
    register: str = ""
    last: str = ""                      # last received symbol
    vars: Dict[str, Any] = field(default_factory=dict)
    emits: List[str] = field(default_factory=list)
    sends: List[Tuple[str, str]] = field(default_factory=list)
    severs: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    adjusts: Dict[str, float] = field(default_factory=dict)
    steps: int = 0


class Interpreter:
    def __init__(self, step_budget: int = 64):
        self.step_budget = step_budget

    def run(self, node: Node, ctx: Context) -> Context:
        ctx.steps = 0
        self._exec(node, ctx)
        return ctx

    def _tick(self, ctx: Context) -> None:
        ctx.steps += 1
        if ctx.steps > self.step_budget:
            raise BudgetExceeded(f"action exceeded {self.step_budget} steps")

    def _exec(self, node: Node, ctx: Context) -> None:
        self._tick(ctx)
        if not isinstance(node, list) or not node:
            raise DSLError(f"not a statement: {node!r}")
        op = node[0]
        args = node[1:]
        if op == "emit":
            ctx.emits.append(str(self._eval(args[0], ctx)))
        elif op == "set":
            ctx.vars[str(args[0])] = self._eval(args[1], ctx)
        elif op == "send":
            ctx.sends.append((str(args[0]), str(self._eval(args[1], ctx))))
        elif op == "sever":
            ctx.severs.append(str(args[0]))
        elif op == "route":
            ctx.routes.append(str(args[0]))
        elif op == "adjust":
            ctx.adjusts[str(args[0])] = ctx.adjusts.get(str(args[0]), 0.0) + float(args[1])
        elif op == "if":
            if self._truthy(self._eval(args[0], ctx)):
                self._exec(args[1], ctx)
            elif len(args) > 2:
                self._exec(args[2], ctx)
        elif op == "seq":
            for a in args:
                self._exec(a, ctx)
        else:
            raise DSLError(f"unknown op {op!r}")

    def _eval(self, node: Node, ctx: Context) -> Any:
        self._tick(ctx)
        if not isinstance(node, list):
            return node
        if not node:
            raise DSLError("empty expression")
        op, args = node[0], node[1:]
        if op == "reg":
            i = int(args[0])
            return int(ctx.register[i]) if 0 <= i < len(ctx.register) and ctx.register[i] in "01" else 0
        if op == "var":
            return ctx.vars.get(str(args[0]), 0)
        if op == "last":
            return ctx.last
        if op == "not":
            return 0 if self._truthy(self._eval(args[0], ctx)) else 1
        if op == "eq":
            return 1 if self._eval(args[0], ctx) == self._eval(args[1], ctx) else 0
        if op == "and":
            return 1 if all(self._truthy(self._eval(a, ctx)) for a in args) else 0
        if op == "or":
            return 1 if any(self._truthy(self._eval(a, ctx)) for a in args) else 0
        if op == "add":
            return sum(float(self._eval(a, ctx)) for a in args)
        raise DSLError(f"unknown expression op {op!r}")

    @staticmethod
    def _truthy(v: Any) -> bool:
        return v not in (0, 0.0, "", "0", None, False)


def validate(node: Node) -> None:
    """Static check: every statement/expression op is in the closed DSL."""
    if not isinstance(node, list):
        return
    if not node:
        raise DSLError("empty node")
    op = node[0]
    if op in OPS or op in EXPR_OPS:
        for a in node[1:]:
            validate(a)
        return
    raise DSLError(f"unknown op {op!r}")


def emitted_symbols(node: Node) -> List[str]:
    """Symbols an action can emit/send (static), used to group rules by action."""
    out: List[str] = []
    if isinstance(node, list) and node:
        if node[0] == "emit" and not isinstance(node[1], list):
            out.append(str(node[1]))
        elif node[0] == "send" and not isinstance(node[2], list):
            out.append(f"{node[1]}:{node[2]}")
        for a in node[1:]:
            out.extend(emitted_symbols(a))
    return out


def action_key(node_or_src: Union[Node, str]) -> str:
    """Canonical string for grouping identical actions."""
    node = parse_sexpr(node_or_src) if isinstance(node_or_src, str) else node_or_src
    return to_sexpr(node)


def run_action(src: str, register: str, last: str = "", budget: int = 64) -> Context:
    node = parse_sexpr(src)
    validate(node)
    return Interpreter(budget).run(node, Context(register=register, last=last))


__all__ = ["matches", "subsumes", "specificity", "parse_sexpr", "to_sexpr", "Context",
           "Interpreter", "BudgetExceeded", "DSLError", "validate", "emitted_symbols",
           "action_key", "run_action", "OPS", "EXPR_OPS"]
