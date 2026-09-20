"""
organism_sim.lcs — Learning Classifier System rule engine
=========================================================

An accuracy-based LCS in the XCS lineage (Wilson 1995), used as the organism's
processor. Each rule = ternary ``condition`` + DSL ``action`` + prediction /
error / fitness / experience / numerosity.

Per input (one *trial*):

    match    [M] = rules whose condition matches the register; cover if some
             action is missing from [M]
    select   prediction array PA[a] = Σ p·F / Σ F over rules advocating a;
             explore → random action, exploit → argmax
    act      run the winning action AST → emits / sends
    credit   ``reward(r)``: Widrow–Hoff on p, ε; accuracy κ; relative-accuracy
             fitness. Bucket brigade: the previous action set receives
             γ·max(PA) so multi-step chains get credit. Cross-agent chains are
             closed by the bus forwarding the reward upstream (see ``agent``).
    GA       on [A] when its mean timestamp is stale: fitness-roulette parents,
             two-point condition crossover, condition mutation (bit ↔ #),
             action swap, GA subsumption, roulette deletion when |P| > N.

Repair (silent) = rule-set compaction: dedupe, subsume specific-into-general
with equal accuracy, prune at the fitness floor. Mutation (structural) =
extra GP operators on random rules: generalise / specialise / point-mutate,
crossover actions with a donor rule set (horizontal transfer).

Reward is in [0, 1]; the bus passes ``pressure`` = 1 − reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rules import (
    Context,
    Interpreter,
    action_key,
    parse_sexpr,
    subsumes,
    to_sexpr,
    validate,
)

WILD = 2  # encoding of '#'


@dataclass
class Params:
    N: int = 400               # max population (sum of numerosities)
    beta: float = 0.2          # learning rate
    alpha: float = 0.1         # accuracy falloff
    eps0: float = 0.01         # error below which a rule is "accurate"
    nu: float = 5.0            # accuracy exponent
    gamma: float = 0.71        # bucket-brigade discount
    theta_ga: int = 25         # GA period (trials)
    chi: float = 0.8           # crossover probability
    mu: float = 0.04           # per-bit mutation probability
    theta_del: int = 20        # deletion experience threshold
    delta: float = 0.1         # deletion fitness fraction
    theta_sub: int = 20        # subsumption experience threshold
    p_wild: float = 0.33       # covering wildcard probability
    p_init: float = 0.01
    eps_init: float = 0.01
    f_init: float = 0.01
    p_explore: float = 0.5
    step_budget: int = 64
    mode: str = "xcs"          # "xcs": accuracy-based (Wilson); "reinforce": cumulative
                               # strength (Roth–Erev), selection ∝ strength — breaks the
                               # pooling equilibrium in signalling games
    reinforce_decay: float = 0.999   # per-update forgetting in reinforce mode
    fitness_floor: float = 1e-4   # compaction prune level

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Rule:
    condition: str
    action: str                     # DSL source
    prediction: float
    error: float
    fitness: float
    experience: int = 0
    numerosity: int = 1
    time_stamp: int = 0
    as_size: float = 1.0
    id: int = 0
    key: str = field(default="", repr=False)

    def __post_init__(self):
        self.key = action_key(self.action)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "condition": self.condition, "action": self.action,
                "prediction": round(self.prediction, 4), "error": round(self.error, 4),
                "fitness": round(self.fitness, 4), "experience": self.experience,
                "numerosity": self.numerosity}


def _enc(cond: str) -> np.ndarray:
    return np.array([WILD if c == "#" else int(c) for c in cond], dtype=np.int8)


class RuleEngine:
    """XCS-style engine over a fixed-length register."""

    def __init__(self, cond_len: int, actions: Sequence[str], params: Optional[Params] = None,
                 seed: int = 0, rules: Sequence[Rule] = ()):
        self.cond_len = cond_len
        self.actions = [to_sexpr(parse_sexpr(a)) for a in actions]
        for a in self.actions:
            validate(parse_sexpr(a))
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        self.interp = Interpreter(self.p.step_budget)
        self.rules: List[Rule] = []
        self._next_id = 0
        self.t = 0
        self.match_set: List[Rule] = []
        self.action_set: List[Rule] = []
        self._pending: Optional[Tuple[List[Rule], float, str]] = None  # deferred (aset, r, reg)
        self.last_action: Optional[str] = None
        self.last_pa: Dict[str, float] = {}
        self.last_explore = False
        self.last_ctx: Optional[Context] = None
        self.counters = {"cover": 0, "ga": 0, "subsume": 0, "delete": 0, "compact": 0,
                         "mutate": 0, "budget_exceeded": 0}
        for r in rules:
            self._insert(r)

    # ── population ───────────────────────────────────────────────────────────

    def _new_rule(self, condition: str, action: str, p=None, e=None, f=None) -> Rule:
        self._next_id += 1
        return Rule(condition=condition, action=to_sexpr(parse_sexpr(action)),
                    prediction=self.p.p_init if p is None else p,
                    error=self.p.eps_init if e is None else e,
                    fitness=self.p.f_init if f is None else f,
                    time_stamp=self.t, id=self._next_id)

    def _insert(self, r: Rule) -> None:
        for q in self.rules:
            if q.condition == r.condition and q.key == r.key:
                q.numerosity += r.numerosity
                return
        self.rules.append(r)

    @property
    def size(self) -> int:
        return sum(r.numerosity for r in self.rules)

    def _delete_excess(self) -> None:
        p = self.p
        while self.size > p.N and self.rules:
            total = self.size
            avg_f = sum(r.fitness for r in self.rules) / total
            votes = []
            for r in self.rules:
                v = r.as_size * r.numerosity
                fpm = r.fitness / r.numerosity
                if r.experience > p.theta_del and fpm < p.delta * avg_f and fpm > 0:
                    v *= avg_f / fpm
                votes.append(v)
            votes = np.asarray(votes)
            i = int(self.rng.choice(len(self.rules), p=votes / votes.sum()))
            self.rules[i].numerosity -= 1
            if self.rules[i].numerosity <= 0:
                self.rules.pop(i)
            self.counters["delete"] += 1

    # ── trial ────────────────────────────────────────────────────────────────

    def match(self, register: str) -> List[Rule]:
        if len(register) != self.cond_len:
            raise ValueError(f"register length {len(register)} != {self.cond_len}")
        if self.rules:
            conds = np.stack([_enc(r.condition) for r in self.rules])
            reg = _enc(register)
            ok = ((conds == WILD) | (conds == reg)).all(axis=1)
            m = [r for r, k in zip(self.rules, ok) if k]
        else:
            m = []
        present = {r.key for r in m}
        for a in self.actions:
            if a not in present:
                m.append(self._cover(register, a))
        return m

    def _cover(self, register: str, action: str) -> Rule:
        cond = "".join("#" if self.rng.random() < self.p.p_wild else c for c in register)
        r = self._new_rule(cond, action)
        self._insert(r)
        self._delete_excess()
        self.counters["cover"] += 1
        return r

    def prediction_array(self, m: Sequence[Rule]) -> Dict[str, float]:
        if self.p.mode == "reinforce":
            out: Dict[str, float] = {}
            for r in m:
                out[r.key] = out.get(r.key, 0.0) + r.prediction * r.numerosity
            return out
        num: Dict[str, float] = {}
        den: Dict[str, float] = {}
        for r in m:
            num[r.key] = num.get(r.key, 0.0) + r.prediction * r.fitness
            den[r.key] = den.get(r.key, 0.0) + r.fitness
        return {k: (num[k] / den[k] if den[k] > 0 else 0.0) for k in num}

    def step(self, register: str, last: str = "", explore: Optional[bool] = None) -> Context:
        """Match → select → act. Returns the executed action's Context."""
        self.t += 1
        self.match_set = self.match(register)
        pa = self.prediction_array(self.match_set)
        if explore is None:
            p_exp = self.p.p_explore
            if getattr(self, "_explore_boost", 0) > 0:
                self._explore_boost -= 1
                p_exp = max(p_exp, 0.8)
            explore = bool(self.rng.random() < p_exp)
        self.last_explore = explore
        keys = sorted(pa)
        if self.p.mode == "reinforce" and explore:
            w = np.array([max(pa[k], 1e-6) for k in keys])
            key = keys[int(self.rng.choice(len(keys), p=w / w.sum()))]   # ∝ strength
        elif explore:
            key = keys[int(self.rng.integers(len(keys)))]
        else:
            best = max(pa.values())
            key = min(k for k in keys if pa[k] == best)   # deterministic tie-break
        # Bucket brigade: a deferred (non-terminal) action set is paid r + γ·max(PA) now
        if self._pending is not None:
            aset, r_prev, reg_prev = self._pending
            self._pending = None
            self._update(aset, r_prev + self.p.gamma * max(pa.values()))
            self._run_ga(aset, reg_prev)
        self.action_set = [r for r in self.match_set if r.key == key]
        self.last_pa = pa
        self.last_action = key
        ctx = Context(register=register, last=last)
        try:
            self.interp.run(parse_sexpr(key), ctx)
        except Exception:
            self.counters["budget_exceeded"] += 1
            ctx = Context(register=register, last=last)
        self.last_ctx = ctx
        return ctx

    def confidence(self) -> float:
        if not self.last_pa:
            return 0.0
        vals = np.array(list(self.last_pa.values()), dtype=float)
        vals = vals / max(vals.max(), 1e-9) if self.p.mode == "reinforce" else vals
        z = np.exp((vals - vals.max()) / 0.25)
        return float(z.max() / z.sum())

    # ── credit ───────────────────────────────────────────────────────────────

    def reward(self, r: float, register: Optional[str] = None, terminal: bool = True) -> None:
        """Credit the current action set with reward r ∈ [0, 1].

        ``terminal=True``: single-step — update now.
        ``terminal=False``: defer; the set is paid ``r + γ·max(PA)`` at the next
        ``step()`` (bucket brigade for multi-step chains).
        """
        r = float(min(1.0, max(0.0, r)))
        if not self.action_set:
            return
        if not terminal:
            self._pending = (self.action_set, r, register or "")
            return
        self._update(self.action_set, r)
        if register is not None:
            self._run_ga(self.action_set, register)

    def _update(self, aset: List[Rule], reward: float) -> None:
        p = self.p
        if p.mode == "reinforce":
            for r in aset:
                r.experience += 1
                r.prediction = r.prediction * p.reinforce_decay + reward
                r.error += p.beta * (abs(reward - min(1.0, r.prediction)) - r.error)
                r.fitness = r.prediction
            return
        as_num = sum(r.numerosity for r in aset)
        for r in aset:
            r.experience += 1
            lr = 1.0 / r.experience if r.experience < 1.0 / p.beta else p.beta
            r.prediction += lr * (reward - r.prediction)
            r.error += lr * (abs(reward - r.prediction) - r.error)
            r.as_size += lr * (as_num - r.as_size)
        acc = [1.0 if r.error < p.eps0 else p.alpha * (r.error / p.eps0) ** (-p.nu) for r in aset]
        tot = sum(a * r.numerosity for a, r in zip(acc, aset))
        for a, r in zip(acc, aset):
            r.fitness += p.beta * ((a * r.numerosity / tot if tot > 0 else 0.0) - r.fitness)
        self._action_set_subsumption(aset)

    def _action_set_subsumption(self, aset: List[Rule]) -> None:
        p = self.p
        cands = [r for r in aset if r.experience > p.theta_sub and r.error < p.eps0]
        if not cands:
            return
        best = max(cands, key=lambda r: (sum(1 for c in r.condition if c == "#"), -r.id))
        for r in list(aset):
            if r is not best and subsumes(best.condition, r.condition) and r.key == best.key:
                best.numerosity += r.numerosity
                aset.remove(r)
                if r in self.rules:
                    self.rules.remove(r)
                self.counters["subsume"] += 1

    # ── GA ───────────────────────────────────────────────────────────────────

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
        fit = np.array([max(r.fitness, 1e-12) for r in aset])
        pick = lambda: aset[int(self.rng.choice(len(aset), p=fit / fit.sum()))]  # noqa: E731
        p1, p2 = pick(), pick()
        c1, c2 = list(p1.condition), list(p2.condition)
        a1, a2 = p1.action, p2.action
        if self.rng.random() < p.chi and self.cond_len > 1:
            x, y = sorted(self.rng.integers(0, self.cond_len + 1, size=2))
            c1[x:y], c2[x:y] = c2[x:y], c1[x:y]
            if self.rng.random() < 0.5:
                a1, a2 = a2, a1
        for c, a, parent in ((c1, a1, p1), (c2, a2, p2)):
            cond = self._mutate_condition("".join(c), register)
            act = a if self.rng.random() >= p.mu else self.actions[int(self.rng.integers(len(self.actions)))]
            child = self._new_rule(cond, act,
                                   p=(p1.prediction + p2.prediction) / 2,
                                   e=(p1.error + p2.error) / 2,
                                   f=0.1 * (p1.fitness + p2.fitness) / 2)
            if self._ga_subsumes(p1, child) or self._ga_subsumes(p2, child):
                continue
            self._insert(child)
        self._delete_excess()

    def _mutate_condition(self, cond: str, register: str) -> str:
        out = []
        for c, r in zip(cond, register):
            if self.rng.random() < self.p.mu:
                out.append("#" if c != "#" else r)
            else:
                out.append(c)
        return "".join(out)

    def _ga_subsumes(self, parent: Rule, child: Rule) -> bool:
        p = self.p
        if (parent.experience > p.theta_sub and parent.error < p.eps0 and parent.key == child.key
                and (parent.condition == child.condition or subsumes(parent.condition, child.condition))):
            parent.numerosity += 1
            self.counters["subsume"] += 1
            return True
        return False

    # ── repair / mutation (organism hooks) ───────────────────────────────────

    def compact(self) -> int:
        """Silent repair: dedupe, subsume equal-accuracy specifics into generals, prune."""
        p = self.p
        before = len(self.rules)
        # dedupe
        merged: Dict[Tuple[str, str], Rule] = {}
        for r in self.rules:
            k = (r.condition, r.key)
            if k in merged:
                merged[k].numerosity += r.numerosity
            else:
                merged[k] = r
        self.rules = list(merged.values())
        # subsume
        accurate = [r for r in self.rules if r.experience > p.theta_sub and r.error < p.eps0]
        for g in accurate:
            for s in list(self.rules):
                if s is not g and s.key == g.key and subsumes(g.condition, s.condition) \
                        and s.error < p.eps0 and s in self.rules:
                    g.numerosity += s.numerosity
                    self.rules.remove(s)
        # prune
        self.rules = [r for r in self.rules
                      if r.fitness >= p.fitness_floor or r.experience <= p.theta_del]
        self.counters["compact"] += 1
        return before - len(self.rules)

    def gp_mutate(self, donor: Optional["RuleEngine"] = None, n: int = 4) -> int:
        """Structural mutation: on n random rules apply generalise / specialise /
        point-mutate / action crossover with a donor's rule set."""
        if not self.rules:
            return 0
        applied = 0
        for _ in range(n):
            r = self.rules[int(self.rng.integers(len(self.rules)))]
            op = int(self.rng.integers(4 if donor and donor.rules else 3))
            c = list(r.condition)
            if op == 0:      # generalise one specific bit
                idx = [i for i, ch in enumerate(c) if ch != "#"]
                if idx:
                    c[int(self.rng.choice(idx))] = "#"
            elif op == 1:    # specialise one wildcard
                idx = [i for i, ch in enumerate(c) if ch == "#"]
                if idx:
                    c[int(self.rng.choice(idx))] = str(int(self.rng.integers(2)))
            elif op == 2:    # point flip
                i = int(self.rng.integers(len(c)))
                c[i] = "#" if c[i] != "#" else str(int(self.rng.integers(2)))
            act = r.action
            if op == 3 and donor is not None and donor.rules:
                d = donor.rules[int(donor.rng.integers(len(donor.rules)))]
                if d.key in {a for a in self.actions}:
                    act = d.action
            child = self._new_rule("".join(c), act, p=r.prediction, e=r.error, f=0.1 * r.fitness)
            self._insert(child)
            applied += 1
        self._delete_excess()
        self.counters["mutate"] += applied
        return applied

    def shock(self, error_floor: Optional[float] = None, explore_boost: int = 200) -> int:
        """Structural shock response: rules whose error exceeds ``error_floor`` have
        their experience reset (so Widrow–Hoff re-learns at rate 1/exp) and fitness
        cut; exploration is boosted for ``explore_boost`` trials. Returns rules reset."""
        floor = self.p.eps0 * 10 if error_floor is None else error_floor
        n = 0
        for r in self.rules:
            if r.experience > 0 and r.error > floor:
                r.experience = 0
                r.fitness *= 0.1
                n += 1
        self._explore_boost = explore_boost
        self.counters["shock"] = self.counters.get("shock", 0) + 1
        return n

    def strengths(self) -> np.ndarray:
        """Per-rule fitness (used by the organism substrate as its expression vector)."""
        if not self.rules:
            return np.array([1.0])
        return np.array([max(r.fitness, 1e-9) for r in self.rules], dtype=float)

    def erode(self, excess: float, erosion: float) -> None:
        """Unrepaired noise decays fitness of rules outside the last action set."""
        keep = {id(r) for r in self.action_set}
        f = 1.0 - erosion * excess
        for r in self.rules:
            if id(r) not in keep:
                r.fitness *= max(0.0, f)

    # ── reporting ────────────────────────────────────────────────────────────

    def macro_size(self) -> int:
        return len(self.rules)

    def best_rules(self, k: int = 10) -> List[Dict[str, Any]]:
        rs = sorted(self.rules, key=lambda r: (-r.numerosity, r.error, -r.fitness))[:k]
        return [r.to_dict() for r in rs]

    def snapshot(self) -> Dict[str, Any]:
        return {"t": self.t, "macro": self.macro_size(), "micro": self.size,
                "counters": dict(self.counters), "actions": list(self.actions)}


__all__ = ["Params", "Rule", "RuleEngine"]
