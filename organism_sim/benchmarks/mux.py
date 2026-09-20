"""
organism_sim.benchmarks.mux — 6-multiplexer benchmark
=====================================================

mux6(b) = b[2 + 2·b0 + b1]: address bits (b0, b1) select one of four data bits.

Phase A  ``run_single``: one agent sees all 6 bits, emits 0/1.
Phase B  ``run_split``:  agent A sees the 2 address bits and can only *send* a
         symbol s0..s3 to B; agent B sees the 4 data bits ⊕ the received symbol
         and emits 0/1. Reward reaches B from the evaluator and A via B's
         credit payload. With the A→B route severed (``coupled=False``) B is
         limited to the majority-of-data-bits ceiling (11/16 ≈ 0.69).

Each trial is one bus tick. ``MuxLog`` rows are deterministic for a seed.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..agent import LCSAgent
from ..bus import RoutedBus
from ..lcs import Params
from ..spec import parse_dna

PROTOCOL_DNA = Path(__file__).with_name("protocol_sender.dna")


def mux6(bits: str) -> str:
    b = [int(c) for c in bits]
    return str(b[2 + 2 * b[0] + b[1]])


@dataclass
class MuxRow:
    trial: int
    exploit_acc: float          # accuracy over exploit trials in the last window
    eval_acc: float             # pure-exploit, no-learning probe (n_eval inputs)
    macro: int
    micro: int
    noise_rate: float
    entropy_bits: float
    repairs: int
    mutations: int
    ga: int
    covers: int
    delivered: int
    credits: int

    FIELDS = ("trial", "exploit_acc", "eval_acc", "macro", "micro", "noise_rate", "entropy_bits",
              "repairs", "mutations", "ga", "covers", "delivered", "credits")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MuxLog:
    def __init__(self):
        self.rows: List[MuxRow] = []
        self.final: Dict[str, Any] = {}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"rows": [r.to_dict() for r in self.rows], "final": self.final},
                          indent=indent)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(MuxRow.FIELDS))
        w.writeheader()
        for r in self.rows:
            w.writerow(r.to_dict())
        return buf.getvalue()

    def column(self, name: str) -> List[float]:
        return [getattr(r, name) for r in self.rows]


def _inputs(rng: np.random.Generator, n: int) -> List[str]:
    return ["".join(map(str, rng.integers(0, 2, 6))) for _ in range(n)]


# ── Phase A ──────────────────────────────────────────────────────────────────

def _eval_single(agent: LCSAgent, inputs: List[str]) -> float:
    ok = 0
    for bits in inputs:
        agent.engine.step(agent.register(bits), explore=False)
        out = agent.engine.last_ctx.emits[0] if agent.engine.last_ctx.emits else None
        ok += out == mux6(bits)
    agent.engine.action_set = []     # probe leaves no credit target
    return ok / len(inputs)


def run_single(trials: int = 5000, seed: int = 0, log_every: int = 250, window: int = 200,
               n_eval: int = 256, params: Optional[Params] = None) -> MuxLog:
    rng = np.random.default_rng(seed)
    eval_inputs = _inputs(np.random.default_rng(seed + 10_000), n_eval)
    agent = LCSAgent("A", input_len=6, actions=["(emit 0)", "(emit 1)"], params=params, seed=seed)
    log = MuxLog()
    recent: List[float] = []
    for trial in range(1, trials + 1):
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        agent.act(bits)
        r = 1.0 if agent.emitted() == mux6(bits) else 0.0
        agent.reward(r)
        agent.tick()
        if not agent.engine.last_explore:
            recent.append(r)
        if trial % log_every == 0 or trial == trials:
            st = agent.state()
            log.rows.append(MuxRow(
                trial=trial, exploit_acc=float(np.mean(recent[-window:])) if recent else 0.0,
                eval_acc=_eval_single(agent, eval_inputs), macro=st["macro"], micro=st["micro"],
                noise_rate=st["noise_rate"], entropy_bits=st["entropy_bits"],
                repairs=st["repairs"], mutations=st["mutations"], ga=st["engine"]["ga"],
                covers=st["engine"]["cover"], delivered=0, credits=0))
    log.final = {"phase": "A", "seed": seed, "trials": trials, "eval_acc": log.rows[-1].eval_acc,
                 "agent": agent.state(), "best_rules": agent.engine.best_rules(12),
                 "audit_chain_valid": agent.organism.chain.verify()}
    return log


# ── Phase B ──────────────────────────────────────────────────────────────────

SYMBOLS = ("s0", "s1", "s2", "s3")


def _split_trial(bus: RoutedBus, a: LCSAgent, b: LCSAgent, bits: str,
                 explore: Optional[bool], learn: bool) -> float:
    bus.next_tick()
    a.act(bits[:2], explore=explore)
    bus.deliver()                       # A's symbol (if routed) lands in B's inbox
    b.act(bits[2:], explore=explore)
    r = 1.0 if b.emitted() == mux6(bits) else 0.0
    if learn:
        b.reward(r)                     # queues credit to A
        bus.deliver()                   # A receives credit → its action set is paid
        a.tick()
        b.tick()
    else:
        a.engine.action_set = []
        b.engine.action_set = []
    return r


def run_split(trials: int = 8000, seed: int = 0, coupled: bool = True, log_every: int = 250,
              window: int = 200, n_eval: int = 256, params: Optional[Params] = None,
              sender: str = "protocol", sender_params: Optional[Params] = None) -> MuxLog:
    """``sender="protocol"``: A's rules come from ``protocol_sender.dna`` and are frozen;
    only B learns (deterministic tier). ``sender="learn"``: A co-adapts from scratch with
    cumulative (Roth–Erev) reinforcement — exploratory tier; may settle in a partial
    pooling equilibrium depending on seed."""
    rng = np.random.default_rng(seed)
    eval_inputs = _inputs(np.random.default_rng(seed + 10_000), n_eval)
    bus = RoutedBus()
    actions_a = [f"(send B {s})" for s in SYMBOLS]
    if sender == "protocol":
        genome = parse_dna(PROTOCOL_DNA.read_text()).genome
        a = LCSAgent("A", input_len=2, actions=actions_a, rules=genome.rules(), seed=seed,
                     bus=bus, structural=False, frozen=True)
    elif sender == "learn":
        # Accuracy-based fitness alone settles into a pooling equilibrium (one symbol
        # for every address); cumulative strength with state-specific rules can break it.
        sender_params = sender_params or Params(mode="reinforce", p_explore=1.0, N=32,
                                               p_wild=0.0, mu=0.0, chi=0.0,
                                               reinforce_decay=1.0, theta_ga=10 ** 9)
        a = LCSAgent("A", input_len=2, actions=actions_a, params=sender_params, seed=seed,
                     bus=bus, structural=False)
    else:
        raise ValueError("sender must be 'protocol' or 'learn'")
    b = LCSAgent("B", input_len=4, actions=["(emit 0)", "(emit 1)"], symbols=SYMBOLS,
                 params=params, seed=seed + 1, bus=bus)
    bus.add(a)
    bus.add(b)
    bus.connect("A", "B")               # A→B signals, B→A credit
    if not coupled:
        bus.sever("A", "B")             # isolate: B never receives A's symbol
    a.donor, b.donor = b, a
    log = MuxLog()
    recent: List[float] = []
    for trial in range(1, trials + 1):
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        r = _split_trial(bus, a, b, bits, None, True)
        if not b.engine.last_explore:
            recent.append(r)
        if trial % log_every == 0 or trial == trials:
            ev = float(np.mean([_split_trial(bus, a, b, x, False, False) for x in eval_inputs]))
            sa, sb = a.state(), b.state()
            log.rows.append(MuxRow(
                trial=trial, exploit_acc=float(np.mean(recent[-window:])) if recent else 0.0,
                eval_acc=ev, macro=sa["macro"] + sb["macro"], micro=sa["micro"] + sb["micro"],
                noise_rate=(sa["noise_rate"] + sb["noise_rate"]) / 2,
                entropy_bits=(sa["entropy_bits"] + sb["entropy_bits"]) / 2,
                repairs=sa["repairs"] + sb["repairs"], mutations=sa["mutations"] + sb["mutations"],
                ga=sa["engine"]["ga"] + sb["engine"]["ga"],
                covers=sa["engine"]["cover"] + sb["engine"]["cover"],
                delivered=bus.delivered, credits=a.credits_received))
    policy = {}
    for addr in ("00", "01", "10", "11"):
        a.engine.step(a.register(addr), explore=False)
        policy[addr] = a.engine.last_action
    a.engine.action_set = []
    log.final = {"phase": "B", "sender": sender, "coupled": coupled, "seed": seed,
                 "trials": trials, "A_policy": policy,
                 "eval_acc": log.rows[-1].eval_acc, "A": a.state(), "B": b.state(),
                 "bus": bus.describe(), "A_rules": a.engine.best_rules(8),
                 "B_rules": b.engine.best_rules(12),
                 "audit_chains_valid": a.organism.chain.verify() and b.organism.chain.verify()}
    return log


__all__ = ["mux6", "MuxRow", "MuxLog", "run_single", "run_split", "SYMBOLS"]
