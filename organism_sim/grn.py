"""
organism_sim.grn — Regulatory genes (dnalang v0.2)
==================================================

A regulatory genome is a set of ``Gene``s with

    trigger       when the gene is *expressed* this tick:
                  ``on_genesis`` | ``continuous`` | ``on_error`` | ``after <gene_id>`` |
                  ``on_signal <name>`` | ``when <metric> <op> <number>``
                  (metrics: error, acc, cusum, noise_rate, entropy_bits, since_inject,
                  since_shift_est, tick, explore; ops < <= > >= ==)
    dependencies  gene ids whose outputs must be on the signal board this tick
    outputs       signal names the gene puts on the board when expressed
    body          either an LCS rule (``condition`` + ``action``) — expression *injects*
                  that rule into the engine as a fresh hypothesis — or a control program
                  in the closed DSL (``action`` only), executed with the metrics as
                  variables; ``(adjust <param> <delta>)`` acts on the learner:
                      inject <n>      inject n random zero-experience rules now
                      explore <p>     set exploration probability to p for `boost_ticks`
                      cusum_reset 1   reset the detector
                      compact 1       compact the rule set
                  and ``(emit <signal>)`` puts a signal on the board.

Each tick: compute metrics → evaluate triggers → express active genes in dependency
order → outputs go on the board for the *next* tick (so ``after G`` means "G was
expressed last tick"). Regulation therefore decides *when hypotheses enter the
population and how the learner explores*, not merely which rule fires.

The genome is data (``spec.Gene``), so it is serialisable to ``.dna``, hashable, and
evolvable: ``mutate`` perturbs trigger thresholds/metrics/references, wiring, and control
parameters; ``crossover`` swaps whole clusters.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .agent import LCSAgent
from .rules import Context, Interpreter, parse_sexpr, validate
from .spec import Gene, Genome

METRICS = ("error", "acc", "cusum", "noise_rate", "entropy_bits", "since_inject",
           "since_shift_est", "tick", "explore")
PARAMS = ("inject", "explore", "cusum_reset", "compact")

# The trigger grammar is part of the language.
from dnalang.regulation import OPS as _TRIG_OPS  # noqa: E402
from dnalang.regulation import Trigger  # noqa: E402

OPS = _TRIG_OPS


class GRN:
    """Interpreter for one regulatory genome over one LCSAgent."""

    MIN_INJECT_GAP = 50

    def __init__(self, genome: Genome, agent: LCSAgent, seed: int = 0, boost_ticks: int = 100,
                 cusum: Tuple[float, float, float] = (0.05, 0.10, 8.0)):
        self.genome = genome
        self.agent = agent
        self.rng = np.random.default_rng(seed + 9001)
        self.triggers: Dict[str, Trigger] = {g.id: Trigger.parse(g.trigger) for g in genome.genes}
        for gid, t in self.triggers.items():
            if t.kind == "when" and t.ref not in METRICS:
                raise ValueError(f"gene {gid}: unknown metric {t.ref!r}; runtime metrics are {METRICS}")
        self.by_id = {g.id: g for g in genome.genes}
        self.order = self._topo()
        self.interp = Interpreter(64)
        self.tick = 0
        self.board: Set[str] = set()
        self.expressed_last: Set[str] = set()
        self.err_ema = 0.0
        self.acc_ema = 0.5
        self.last_error = 0.0
        self.cusum_mu0, self.cusum_k, self.cusum_h = cusum
        self.cusum_s = 0.0
        self.since_inject = 10 ** 6
        self.since_shift_est = 10 ** 6
        self.boost_ticks = boost_ticks
        self._boost_left = 0
        self._base_explore = agent.engine.p.p_explore
        self.injections: List[int] = []
        self.expressions: Dict[str, int] = {g.id: 0 for g in genome.genes}
        for g in genome.genes:
            if not g.is_rule:
                validate(parse_sexpr(g.action))

    def _topo(self) -> List[str]:
        ids = [g.id for g in self.genome.genes]
        deps = {g.id: [d for d in g.dependencies if d in ids] for g in self.genome.genes}
        out, seen = [], set()

        def visit(i, stack=()):
            if i in seen or i in stack:
                return
            for d in deps[i]:
                visit(d, stack + (i,))
            seen.add(i)
            out.append(i)
        for i in ids:
            visit(i)
        return out

    # ── per-trial update ─────────────────────────────────────────────────────

    def observe(self, reward: float, explore: bool) -> None:
        """Called after the agent's reward on a trial, before ``step``."""
        if not explore:
            e = 1.0 - reward
            self.last_error = e
            self.err_ema = 0.9 * self.err_ema + 0.1 * e
            self.acc_ema = 0.9 * self.acc_ema + 0.1 * reward
            self.cusum_s = max(0.0, self.cusum_s + (e - self.cusum_mu0 - self.cusum_k))
            if self.cusum_s > self.cusum_h and self.since_shift_est > 50:
                self.since_shift_est = 0            # estimate only; genes decide what to do
        else:
            self.last_error = 0.0

    def metrics(self) -> Dict[str, float]:
        s = self.agent.organism.state
        return {"error": self.err_ema, "acc": self.acc_ema, "cusum": self.cusum_s,
                "noise_rate": s.noise_rate, "entropy_bits": s.entropy_bits,
                "since_inject": float(self.since_inject), "since_shift_est": float(self.since_shift_est),
                "tick": float(self.tick), "explore": float(self.agent.engine.p.p_explore),
                "last_error": self.last_error}

    def step(self) -> List[str]:
        """Evaluate triggers, express genes, apply effects. Returns expressed gene ids."""
        self.tick += 1
        self.since_inject += 1
        self.since_shift_est += 1
        if self._boost_left > 0:
            self._boost_left -= 1
            if self._boost_left == 0:
                self.agent.engine.p.p_explore = self._base_explore
        m = self.metrics()
        board_now: Set[str] = set()
        expressed: List[str] = []
        for gid in self.order:
            g = self.by_id[gid]
            if not self.triggers[gid].fires(self.tick, m, self.board, self.expressed_last):
                continue
            if any(d not in self.board and d not in expressed for d in g.dependencies):
                continue
            expressed.append(gid)
            self.expressions[gid] += 1
            board_now.update(g.outputs)
            board_now.add(gid)
            if g.is_rule:
                eng = self.agent.engine
                eng._insert(eng._new_rule(g.condition, g.action))
                eng._delete_excess()
            else:
                ctx = Context(register="", vars=dict(m))
                try:
                    self.interp.run(parse_sexpr(g.action), ctx)
                except Exception:
                    continue
                board_now.update(ctx.emits)
                self._apply(ctx.adjusts)
        self.expressed_last = set(expressed)
        self.board = board_now
        return expressed

    def _apply(self, adjusts: Dict[str, float]) -> None:
        eng = self.agent.engine
        for k, v in adjusts.items():
            if k == "inject" and v >= 1:
                if self.since_inject < self.MIN_INJECT_GAP:
                    continue                          # hard floor so a degenerate genome cannot inject every tick
                from .benchmarks.expzero import inject, random_rules
                inject(self.agent, random_rules(int(v), int(self.rng.integers(1 << 30))))
                self.injections.append(self.tick)
                self.since_inject = 0
            elif k == "explore":
                eng.p.p_explore = float(min(1.0, max(0.0, v)))
                self._boost_left = self.boost_ticks
            elif k == "cusum_reset":
                self.cusum_s = 0.0
            elif k == "compact":
                eng.compact()

    # ── evolution over genomes ───────────────────────────────────────────────

    @staticmethod
    def mutate(genome: Genome, rng: np.random.Generator, rate: float = 0.3) -> Genome:
        genes = []
        for g in genome.genes:
            g2 = Gene.from_dict(g.to_dict())
            if rng.random() < rate:
                t = Trigger.parse(g2.trigger)
                if t.kind == "when":
                    r = rng.random()
                    if r < 0.5:
                        t.value = float(t.value * float(rng.uniform(0.5, 1.5)) + rng.normal(0, 0.02))
                    elif r < 0.75:
                        t.op = str(rng.choice(list(OPS)))
                    else:
                        t.ref = str(rng.choice([x for x in METRICS if x != "tick"]))
                    g2.trigger = t.render()
                elif rng.random() < 0.3:
                    g2.trigger = str(rng.choice(["continuous", "on_error",
                                                 f"when error > {rng.uniform(0.05, 0.5):.2f}",
                                                 f"when cusum > {rng.uniform(1, 12):.1f}",
                                                 f"when since_inject > {int(rng.integers(100, 1500))}"]))
            if not g2.is_rule and rng.random() < rate:
                g2.action = _mutate_action(g2.action, rng)
            genes.append(g2)
        return Genome(genes=genes, version=genome.version + 1, purpose=genome.purpose)

    @staticmethod
    def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
        clusters = sorted({g.cluster for g in a.genes} | {g.cluster for g in b.genes})
        take_a = {c: bool(rng.random() < 0.5) for c in clusters}
        genes = [Gene.from_dict(g.to_dict()) for g in a.genes if take_a[g.cluster]] + \
                [Gene.from_dict(g.to_dict()) for g in b.genes if not take_a[g.cluster]]
        return Genome(genes=genes, version=max(a.version, b.version) + 1, purpose=a.purpose)


_ADJ = re.compile(r"\(adjust (\w+) (-?[0-9.]+)\)")


def _mutate_action(src: str, rng: np.random.Generator) -> str:
    def rep(m):
        k, v = m.group(1), float(m.group(2))
        if k == "inject":
            v = float(max(5, min(80, int(v * rng.uniform(0.5, 1.6)))))
        elif k == "explore":
            v = float(min(1.0, max(0.05, v + rng.normal(0, 0.15))))
        return f"(adjust {k} {v:g})"
    return _ADJ.sub(rep, src)


def genome_to_dna(genome: Genome, name: str, header: Sequence[str] = ()) -> str:
    lines = [f"// {h}" for h in header] + [f"ORGANISM {name} {{",
                                            f'    META {{ version: "{genome.version}", domain: "grn" }}', "    GENOME {"]
    for g in genome.genes:
        fields = [f'id: "{g.id}"', f'trigger: "{g.trigger}"']
        if g.condition:
            fields.append(f'condition: "{g.condition}"')
        fields.append(f'action: "{g.action}"')
        if g.dependencies:
            fields.append("dependencies: [" + ", ".join(f'"{d}"' for d in g.dependencies) + "]")
        if g.outputs:
            fields.append("outputs: [" + ", ".join(f'"{o}"' for o in g.outputs) + "]")
        fields.append(f'cluster: "{g.cluster}"')
        lines.append(f"        GENE {g.name} {{ {', '.join(fields)} }}")
    lines += ["    }", "}"]
    return "\n".join(lines) + "\n"


__all__ = ["Trigger", "GRN", "METRICS", "PARAMS", "genome_to_dna"]
