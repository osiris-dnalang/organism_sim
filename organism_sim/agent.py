"""
organism_sim.agent — LCS agent: organism + rule engine + bus adapter
====================================================================

``LCSAgent`` binds an ``Organism`` (state machine, triggers, telemetry) to a
``RuleEngine`` (the computation) and speaks ``Payload`` on a ``RoutedBus``.

Register = input bits ⊕ encoding of the last received symbol (``symbol_bits``
wide; all-zero = none). Emitted symbols become ``result`` payloads; ``send``
becomes a ``signal`` payload to a neighbour.

Noise wiring: the agent's error (1 − reward) is injected into the organism's
noise EMA, so ``noise_rate`` tracks the running error rate; incoming payload
``pressure`` is injected too. Cross-agent credit: on ``reward()`` the agent
forwards a ``credit`` payload (pressure = 1 − r) to whichever neighbour's
signal it consumed this trial, closing the bucket brigade across the bus.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Any, Deque, Dict, List, Optional, Sequence, Set

import numpy as np

from .bus import Payload, RoutedBus
from .lcs import Params, Rule, RuleEngine
from .noise import Environment
from .organism import Organism, Processor
from .spec import Gene, Genome, OrganismSpec, Triggers

QUIET = dict(noise_floor=0.05, jitter=0.01, burst_prob=0.0)


def _symbol_bits(n_symbols: int) -> int:
    return max(1, math.ceil(math.log2(n_symbols + 1))) if n_symbols else 0


class LCSAgent(Processor):
    def __init__(self, name: str, input_len: int, actions: Sequence[str],
                 symbols: Sequence[str] = (), params: Optional[Params] = None,
                 triggers: Optional[Triggers] = None, seed: int = 0,
                 rules: Sequence[Gene] = (), bus: Optional[RoutedBus] = None,
                 pressure_window: int = 20, sever_pressure: float = 0.8,
                 structural: bool = False, frozen: bool = False):
        self.name = name
        self.structural = structural    # False → organism repair/mutation leave the rule set alone
        self.frozen = frozen            # True → never learns; always exploits (fixed protocol)
        self.input_len = input_len
        self.symbols = list(symbols)
        self.symbol_bits = _symbol_bits(len(self.symbols))
        self.cond_len = input_len + self.symbol_bits
        self.engine = RuleEngine(self.cond_len, actions, params, seed=seed,
                                 rules=[Rule(condition=g.condition, action=g.action,
                                             prediction=g.expression, error=0.0, fitness=g.expression)
                                        for g in rules if g.is_rule])
        self.triggers = triggers or Triggers(noise_floor=QUIET["noise_floor"], repair_threshold=0.45)
        if frozen and self.triggers.entropy_floor_bits > 0:
            # a frozen genome cannot collapse; only the noise-integral trigger applies
            self.triggers = replace(self.triggers, entropy_floor_bits=0.0)
        genes = [Gene(id=g.id, name=g.name, expression=g.expression, condition=g.condition,
                      action=g.action) for g in rules] or [Gene(id="G0", name="engine")]
        spec = OrganismSpec(name=name, genome=Genome(genes=genes), domain="lcs",
                            triggers=self.triggers)
        env = Environment(n_genes=len(genes), seed=seed, **QUIET)
        self.organism = Organism(spec, env=env, seed=seed)
        self.organism.processor = self
        self.bus = bus
        self.routes: Set[str] = set()
        self.donor: Optional["LCSAgent"] = None
        self.inbox: Deque[Payload] = deque()
        self.last_symbol: Optional[str] = None
        self.last_symbol_from: Optional[str] = None
        self.last_register: Optional[str] = None
        self.consumed_from: Optional[str] = None
        self.outputs: List[Payload] = []
        self.trial = 0
        self.pressure_in: Dict[str, Deque[float]] = {}
        self.pressure_window = pressure_window
        self.sever_pressure = sever_pressure
        self.severed: List[Dict[str, Any]] = []
        self.credits_received = 0
        self.peers: List[str] = []              # known alternative targets for rerouting
        self.reroutes: List[Dict[str, Any]] = []
        self.reroute_cooldown = 100             # trials before another reroute may fire
        self._last_reroute = -10 ** 9

    # ── bus side ─────────────────────────────────────────────────────────────

    def receive(self, payload: Payload) -> None:
        self.inbox.append(payload)
        if payload.kind == "signal":
            self.last_symbol = str(payload.content)
            self.last_symbol_from = payload.sender
            self.organism.inject_noise(payload.pressure)
            self.pressure_in.setdefault(payload.sender, deque(maxlen=self.pressure_window)).append(
                payload.pressure)
        elif payload.kind == "credit":
            self.credits_received += 1
            reward = payload.content.get("reward", 1.0 - payload.pressure) \
                if isinstance(payload.content, dict) else 1.0 - payload.pressure
            if not self.frozen:
                self.engine.reward(float(reward), self.last_register, terminal=True)
            self.organism.inject_noise(payload.pressure)

    def encode_symbol(self, sym: Optional[str]) -> str:
        if self.symbol_bits == 0:
            return ""
        idx = self.symbols.index(sym) + 1 if sym in self.symbols else 0
        return format(idx, f"0{self.symbol_bits}b")

    def register(self, input_bits: str) -> str:
        if len(input_bits) != self.input_len:
            raise ValueError(f"expected {self.input_len} input bits, got {len(input_bits)}")
        return input_bits + self.encode_symbol(self.last_symbol)

    # ── trial ────────────────────────────────────────────────────────────────

    def act(self, input_bits: str, explore: Optional[bool] = None) -> List[Payload]:
        """Match → select → act on this trial's input. Returns emitted payloads
        (also queued on the bus if one is attached)."""
        self.trial += 1
        reg = self.register(input_bits)
        self.last_register = reg
        self.consumed_from = self.last_symbol_from
        if self.frozen:
            explore = False
        ctx = self.engine.step(reg, last=self.last_symbol or "", explore=explore)
        self.last_symbol, self.last_symbol_from = None, None   # consumed
        self.outputs = []
        for sym in ctx.emits:
            self.outputs.append(Payload(content=sym, pressure=0.0, sender=self.name,
                                        recipient="env", kind="result"))
        for target, sym in ctx.sends:
            targets = sorted(self.routes) if target == "*" else (
                [target] if target in self.routes else [])
            for t in targets:
                self.outputs.append(Payload(content=sym, pressure=0.0, sender=self.name,
                                            recipient=t, kind="signal"))
        for target in ctx.severs:
            self._sever(target, "action")
        for target in ctx.routes:
            self.routes.add(target)
        if self.bus is not None:
            for p in self.outputs:
                if p.recipient != "env":
                    self.bus.send(p)
        return self.outputs

    def emitted(self) -> Optional[str]:
        for p in self.outputs:
            if p.kind == "result":
                return str(p.content)
        return None

    def reward(self, r: float) -> None:
        """Credit this trial; inject error as noise; forward credit upstream.
        Only exploit-trial error is injected — exploration errors are deliberate."""
        if not self.frozen:
            self.engine.reward(r, self.last_register, terminal=True)
        if not self.engine.last_explore:
            self.organism.inject_noise(1.0 - r)
        if self.bus is not None:
            if self.consumed_from and self.consumed_from in self.bus.nodes:
                targets = [self.consumed_from]
            else:
                targets = [t for t in sorted(self.routes) if t in self.bus.nodes]
            # pressure carries only *exploit* error: exploration misses are this
            # agent's deliberate choice, not a signal about the upstream link
            pressure = 0.0 if self.engine.last_explore else 1.0 - r
            for t in targets:
                self.bus.send(Payload(content={"reward": r, "explore": self.engine.last_explore},
                                      pressure=pressure, sender=self.name, recipient=t,
                                      kind="credit"))

    def tick(self):
        """Advance the organism state machine one tick (after reward)."""
        return self.organism.step()

    # ── Processor interface (called by the organism) ─────────────────────────

    def strengths(self) -> np.ndarray:
        return self.engine.strengths()

    def decide(self, org: Organism):
        return self.engine.last_action, self.engine.confidence()

    def erode(self, org: Organism, excess: float) -> None:
        if self.structural:
            self.engine.erode(excess, 0.05)

    def repair(self, org: Organism) -> None:
        if self.structural:
            self.engine.compact()

    def mutate(self, org: Organism, reason: str) -> None:
        if self.structural and not self.frozen:
            self.engine.gp_mutate(donor=self.donor.engine if self.donor else None)
            self.engine.shock()
        if self.structural and self.peers:
            self.reroute(reason)
        for src, hist in list(self.pressure_in.items()):
            if len(hist) >= self.pressure_window and float(np.mean(hist)) > self.sever_pressure:
                self._sever(src, "high_pressure")

    def reroute(self, why: str = "manual") -> Optional[str]:
        """Topological mutation: drop the current relay route, route to an unrouted peer."""
        alternatives = [p for p in self.peers if p not in self.routes]
        now = self.organism.tick
        if not alternatives or now - self._last_reroute < self.reroute_cooldown:
            return None
        self._last_reroute = now
        for old in sorted(self.routes):
            if old in self.peers:
                self.routes.discard(old)
        new = alternatives[int(self.organism.rng.integers(len(alternatives)))]
        self.routes.add(new)
        self.reroutes.append({"trial": now, "to": new, "why": why})
        self._excess_reset()
        return new

    def _excess_reset(self) -> None:
        self.organism._excess.clear()

    def _sever(self, target: str, why: str) -> None:
        if self.bus is not None and target in self.bus.nodes:
            self.bus.sever(target, self.name)      # stop them sending to us
        self.routes.discard(target)
        self.severed.append({"trial": self.trial, "target": target, "why": why})

    # ── reporting ────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        s = self.organism.state
        c = self.organism.counters
        return {"name": self.name, "trial": self.trial, "noise_rate": round(s.noise_rate, 4),
                "entropy_bits": round(s.entropy_bits, 4), "status": self.organism.status.value,
                "macro": self.engine.macro_size(), "micro": self.engine.size,
                "repairs": c["repair"], "mutations": c["mutate"],
                "engine": self.engine.counters, "routes": sorted(self.routes),
                "severed": len(self.severed), "credits": self.credits_received}


__all__ = ["LCSAgent"]
