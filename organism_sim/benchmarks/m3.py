"""
organism_sim.benchmarks.m3 — M3: open-ended task generation (POET-style) at small scale
=======================================================================================

Task space (small on purpose — the layer's mechanisms are measured to work at 6–10 bits):
hidden generalised-multiplexer instances with k = 2 address bits, width 6–10, a data
permutation, optional output inversion, and an optional *parity twist* (output XOR one
irrelevant bit). Tasks have a genome and mutate (width ± 1, re-draw positions, toggle
invert/twist, permute data).

Arms (identical seeds and total training budget):

    poet    agent–task pairs; every ``gen_every`` iterations each task spawns children;
            a child is admitted only if it passes the minimal criterion — some current
            agent scores in [mc_lo, mc_hi) on it *before training* (neither trivial nor
            impossible); the task population is capped (oldest dropped). Transfer: every
            ``transfer_every`` iterations, if another agent beats a task's own agent on it,
            the better agent is copied in.
    random  the same number of tasks, drawn uniformly from the task space with no
            minimal criterion, no transfer, same per-task training budget.

Metric: ANNECS — the number of tasks that were *unsolved* (< solved_at by every agent) when
created and later *solved* (≥ solved_at by some agent). Also: max width reached, fraction
twisted, mean accuracy.

PRE-REGISTERED CRITERION (committed before the first run; fresh seeds 20–24; nothing tuned):
    C1  ANNECS(poet) > ANNECS(random) on ≥ 4/5 seeds
    C2  pooled ratio ANNECS(poet) / ANNECS(random) ≥ 1.25
    PASS ⇔ C1 ∧ C2. Exploratory: max width reached; whether transfers happened; the ANNECS
    curve shape (still rising at the end?).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agent import LCSAgent
from ..lcs import Params

ARMS = ("poet", "random")
WIDTHS = (6, 10)


@dataclass(frozen=True)
class Space:
    """A task space: k address bits select one of 2^k data positions inside ``widths``; the
    learner every agent is built from. The default is M3's (k=2, 6–10 bits, N=400) and
    reproduces it exactly; M3b uses k=3, 12–16 bits, ``lcs.PARAMS16``."""
    k: int = 2
    widths: Tuple[int, int] = WIDTHS
    params: Params = field(default_factory=lambda: Params(N=400, p_explore=0.5))
    audit_window: Optional[int] = None   # bound each agent's retained telemetry (memory only)

    @property
    def n_data(self) -> int:
        return 1 << self.k

    @property
    def min_width(self) -> int:
        return self.k + self.n_data


DEFAULT_SPACE = Space()


@dataclass
class Task:
    tid: int
    width: int
    addr: List[int]
    data: List[int]
    invert: bool
    twist: Optional[int]                 # irrelevant position XORed into the output, or None
    parent: Optional[int] = None
    created_iter: int = 0
    solved_iter: Optional[int] = None
    unsolved_at_creation: bool = True

    def truth(self, bits: str) -> str:
        b = [int(c) for c in bits]
        idx = 0
        for a in self.addr:                         # k=2: 2*b[a0] + b[a1], as before
            idx = (idx << 1) | b[a]
        out = b[self.data[idx]]
        if self.twist is not None:
            out ^= b[self.twist]
        return str(1 - out if self.invert else out)

    def describe(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def sample(rng: np.random.Generator, tid: int, it: int = 0, width: Optional[int] = None,
               space: Space = DEFAULT_SPACE) -> "Task":
        k, nd = space.k, space.n_data
        w = int(width or rng.integers(space.widths[0], space.widths[1] + 1))
        if w < space.min_width:
            raise ValueError(f"width {w} < k + 2^k = {space.min_width}")
        perm = list(map(int, rng.permutation(w)))
        irrelevant = perm[k + nd:]
        twist = int(rng.choice(irrelevant)) if irrelevant and rng.random() < 0.5 else None
        return Task(tid, w, perm[:k], perm[k:k + nd], bool(rng.integers(2)), twist, None, it)

    def mutate(self, rng: np.random.Generator, tid: int, it: int,
               space: Space = DEFAULT_SPACE) -> "Task":
        w = self.width
        r = rng.random()
        if r < 0.3:
            w = int(min(space.widths[1], max(space.widths[0], w + int(rng.choice([-1, 1])))))
            child = Task.sample(rng, tid, it, w, space)
            child.invert = self.invert
        else:
            child = Task(tid, w, list(self.addr), list(self.data), self.invert, self.twist, None, it)
            if r < 0.5:
                child.invert = not child.invert
            elif r < 0.7:
                used = set(child.addr) | set(child.data)
                free = [i for i in range(w) if i not in used]
                child.twist = None if child.twist is not None or not free else int(rng.choice(free))
            elif r < 0.85:
                child.data = list(map(int, rng.permutation(child.data)))
            else:
                used = set(child.addr) | set(child.data) | ({child.twist} if child.twist is not None else set())
                free = [i for i in range(w) if i not in used]
                if free:
                    child.addr[int(rng.integers(len(child.addr)))] = int(rng.choice(free))
        child.parent = self.tid
        return child


def _agent(width: int, seed: int, space: Space = DEFAULT_SPACE) -> LCSAgent:
    return LCSAgent(f"A{width}", input_len=width, actions=["(emit 0)", "(emit 1)"],
                    params=space.params, seed=seed, audit_window=space.audit_window)


def evaluate(agent: LCSAgent, task: Task, rng: np.random.Generator, n: int = 128) -> float:
    if agent.input_len != task.width:
        return 0.0
    ok = 0
    for _ in range(n):
        bits = "".join(map(str, rng.integers(0, 2, task.width)))
        agent.engine.step(agent.register(bits), explore=False)
        out = agent.engine.last_ctx.emits[0] if agent.engine.last_ctx.emits else None
        ok += out == task.truth(bits)
    agent.engine.action_set = []
    return ok / n


def train(agent: LCSAgent, task: Task, rng: np.random.Generator, trials: int) -> None:
    for _ in range(trials):
        bits = "".join(map(str, rng.integers(0, 2, task.width)))
        agent.act(bits)
        agent.reward(1.0 if agent.emitted() == task.truth(bits) else 0.0)
        agent.tick()


@dataclass
class M3Log:
    arm: str
    seed: int
    rows: List[Dict[str, Any]] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)
    live: Optional[Tuple[List[Task], Dict[int, LCSAgent]]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("live", None)
        return d


def run_arm(arm: str, seed: int, iterations: int = 30, train_trials: int = 2000, n_pairs: int = 4,
            gen_every: int = 3, children_per_task: int = 2, max_tasks: int = 8, transfer_every: int = 5,
            mc_lo: float = 0.6, mc_hi: float = 0.95, solved_at: float = 0.95,
            space: Space = DEFAULT_SPACE) -> M3Log:
    if arm not in ARMS:
        raise ValueError(arm)
    rng = np.random.default_rng(seed)
    erng = np.random.default_rng(seed + 1)
    log = M3Log(arm=arm, seed=seed)
    next_tid = 0

    def new_tid() -> int:
        nonlocal next_tid
        next_tid += 1
        return next_tid - 1

    tasks: List[Task] = [Task.sample(rng, new_tid(), 0, space=space) for _ in range(n_pairs)]
    agents: Dict[int, LCSAgent] = {t.tid: _agent(t.width, seed * 100 + t.tid, space) for t in tasks}
    archive: List[Task] = list(tasks)             # every task ever created (for ANNECS)
    transfers = 0
    created_random = 0

    def best_score(task: Task) -> Tuple[float, Optional[int]]:
        best, who = -1.0, None
        for tid, ag in agents.items():
            s = evaluate(ag, task, erng)
            if s > best:
                best, who = s, tid
        return best, who

    for it in range(1, iterations + 1):
        # training: each task's agent trains on its task (equal budget across arms)
        for t in tasks:
            train(agents[t.tid], t, rng, train_trials)
        # task generation
        if it % gen_every == 0:
            if arm == "poet":
                children = []
                for t in tasks:
                    for _ in range(children_per_task):
                        c = t.mutate(rng, new_tid(), it, space)
                        s, _ = best_score(c)
                        if mc_lo <= s < mc_hi:                 # minimal criterion
                            c.unsolved_at_creation = True
                            children.append(c)
                for c in children:
                    if len(tasks) >= max_tasks:
                        oldest = min(tasks, key=lambda x: x.created_iter)
                        tasks.remove(oldest)
                        agents.pop(oldest.tid, None)
                    tasks.append(c)
                    archive.append(c)
                    # the child inherits (a copy of) its parent's agent when widths match
                    parent_agent = agents.get(c.parent)
                    if parent_agent is not None and parent_agent.input_len == c.width:
                        agents[c.tid] = copy.deepcopy(parent_agent)
                    else:
                        agents[c.tid] = _agent(c.width, seed * 100 + c.tid, space)
            else:
                for _ in range(len(tasks) * children_per_task // 2):   # matched creation rate
                    c = Task.sample(rng, new_tid(), it, space=space)
                    s, _ = best_score(c)
                    c.unsolved_at_creation = s < solved_at
                    created_random += 1
                    if len(tasks) >= max_tasks:
                        oldest = min(tasks, key=lambda x: x.created_iter)
                        tasks.remove(oldest)
                        agents.pop(oldest.tid, None)
                    tasks.append(c)
                    archive.append(c)
                    agents[c.tid] = _agent(c.width, seed * 100 + c.tid, space)
        # transfer (poet only)
        if arm == "poet" and it % transfer_every == 0:
            for t in tasks:
                own = evaluate(agents[t.tid], t, erng)
                best, who = own, t.tid
                for tid, ag in agents.items():
                    if tid != t.tid and ag.input_len == t.width:
                        s = evaluate(ag, t, erng)
                        if s > best + 0.05:
                            best, who = s, tid
                if who != t.tid:
                    agents[t.tid] = copy.deepcopy(agents[who])
                    transfers += 1
        # solved bookkeeping over the archive (any current agent)
        for t in archive:
            if t.solved_iter is None:
                s, _ = best_score(t)
                if s >= solved_at:
                    t.solved_iter = it
        annecs = sum(1 for t in archive if t.unsolved_at_creation and t.solved_iter is not None)
        accs = [evaluate(agents[t.tid], t, erng) for t in tasks]
        log.rows.append({"iter": it, "n_tasks": len(tasks), "archive": len(archive), "annecs": annecs,
                         "max_width": max(t.width for t in tasks), "twisted": sum(t.twist is not None for t in tasks),
                         "mean_acc": float(np.mean(accs)), "transfers": transfers})
    log.tasks = [t.describe() for t in archive]
    log.final = {"arm": arm, "seed": seed, "annecs": log.rows[-1]["annecs"], "archive": len(archive),
                 "max_width": max(r["max_width"] for r in log.rows), "transfers": transfers,
                 "annecs_curve": [r["annecs"] for r in log.rows],
                 "audit_chains_valid": all(a.organism.chain.verify() for a in agents.values())}
    log.live = (list(tasks), dict(agents))
    return log


def final_distribution_score(tasks: Sequence[Task], agents: Dict[int, LCSAgent],
                             rng: np.random.Generator) -> float:
    """Mean over *tasks* of the best accuracy any agent in *agents* reaches on it (0 when no
    agent has the task's width). Used to judge one arm's agents on the other arm's final
    task population."""
    if not tasks:
        return 0.0
    return float(np.mean([max([evaluate(a, t, rng) for a in agents.values()] or [0.0]) for t in tasks]))


def compare_row(seed: int, final_distribution: bool = False, **kw) -> Dict[str, Any]:
    """Both arms on one seed (and, for M3b, both arms' agents on poet's final tasks)."""
    p = run_arm("poet", seed, **kw)
    r = run_arm("random", seed, **kw)
    row: Dict[str, Any] = {"seed": seed, "poet": p.final, "random": r.final}
    if final_distribution:
        frng = np.random.default_rng(seed + 2)
        ptasks, pagents = p.live
        _, ragents = r.live
        row["final_distribution"] = {"n_tasks": len(ptasks),
                                     "poet_agents": final_distribution_score(ptasks, pagents, frng),
                                     "random_agents": final_distribution_score(ptasks, ragents, frng)}
    return row


def verdict(rows: List[Dict[str, Any]], seeds: Sequence[int], final_distribution: bool = False,
            params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n = len(rows)
    c1 = sum(r["poet"]["annecs"] > r["random"]["annecs"] for r in rows)
    pooled_p = float(np.median([r["poet"]["annecs"] for r in rows]))
    pooled_r = float(np.median([r["random"]["annecs"] for r in rows]))
    ratio = pooled_p / pooled_r if pooled_r > 0 else (float("inf") if pooled_p > 0 else 1.0)
    out = {"seeds": list(seeds), "params": params or {}, "rows": rows,
           "pooled_annecs": {"poet": pooled_p, "random": pooled_r}, "ratio": ratio,
           "verdict": {"C1_count": c1, "C1": c1 >= 4 if n >= 5 else c1 == n, "C2": ratio >= 1.25, "n": n,
                       "exploratory_max_width": {a: [r[a]["max_width"] for r in rows] for a in ARMS},
                       "exploratory_transfers": [r["poet"]["transfers"] for r in rows],
                       "exploratory_still_rising": sum(
                           r["poet"]["annecs_curve"][-1] > r["poet"]["annecs_curve"][-6] for r in rows)}}
    if final_distribution:
        c3 = sum(r["final_distribution"]["poet_agents"] > r["final_distribution"]["random_agents"] for r in rows)
        out["verdict"]["C3_count"] = c3
        out["verdict"]["C3"] = c3 >= 4 if n >= 5 else c3 == n
        out["verdict"]["pass"] = bool(out["verdict"]["C1"] and out["verdict"]["C2"] and out["verdict"]["C3"])
    else:
        out["verdict"]["pass"] = bool(out["verdict"]["C1"] and out["verdict"]["C2"])
    return out


def compare(seeds: Sequence[int] = range(20, 25), final_distribution: bool = False,
            **kw) -> Dict[str, Any]:
    rows = [compare_row(sd, final_distribution, **kw) for sd in seeds]
    params = {k: (v if not isinstance(v, Space) else {"k": v.k, "widths": list(v.widths),
                                                       "params": v.params.to_dict()})
              for k, v in kw.items()}
    return verdict(rows, seeds, final_distribution, params)


__all__ = ["Task", "Space", "DEFAULT_SPACE", "ARMS", "WIDTHS", "evaluate", "train", "run_arm",
           "compare", "compare_row", "verdict", "final_distribution_score"]
