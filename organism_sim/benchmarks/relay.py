"""
organism_sim.benchmarks.relay — Relay-failure recovery (resilient coordination)
===============================================================================

Split 6-multiplexer over a two-hop path:

    A (frozen protocol sender) ──► R1 (relay) ──► B (XCS receiver)
                                   R2 (spare relay, unrouted)

At ``fail_at`` the relay R1 fails: ``dead`` drops every signal, ``poisoned``
forwards a random symbol at pressure 1.0. Transient reward-corruption bursts
(``burst_len`` trials of random reward) occur at ``bursts`` — these are *not*
failures and must not trigger a reroute.

Groups on identical seeds:

    none        no rerouting ever (floor: majority ceiling ≈ 0.69 after failure)
    supervisor  centralized monitor with global probe access: reroutes A→R2
                after ``sup_patience`` consecutive probes below ``sup_threshold``
    organism    A's local trigger only: credit pressure → noise → sustained
                unrepaired excess → topological mutation (sever, route to peer)

Pre-registered criteria (30 seeds, profile tuned on seeds 100–104 only):

    C1  organism restores ≥ recovered_at within ``c1_trials`` of failure on
        ≥ 90 % of seeds
    C2  organism median recovery ≤ 1.25 × supervisor median
    C3  organism false reroutes ≤ 1 per 10k trials
    sanity: `none` final accuracy ≤ 0.75
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..agent import LCSAgent
from ..bus import Relay, RoutedBus
from ..lcs import Params
from ..spec import Triggers, parse_dna
from .mux import SYMBOLS, _inputs, mux6

SENDER_DNA = Path(__file__).with_name("relay_sender.dna")
GROUPS = ("none", "supervisor", "organism")


@dataclass
class RelayRow:
    trial: int
    acc: float
    failed: bool
    route: str
    reroutes: int
    a_noise: float
    b_macro: int

    FIELDS = ("trial", "acc", "failed", "route", "reroutes", "a_noise", "b_macro")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelayLog:
    group: str
    seed: int
    mode: str
    rows: List[RelayRow] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"group": self.group, "seed": self.seed, "mode": self.mode,
                           "rows": [r.to_dict() for r in self.rows], "final": self.final},
                          indent=indent)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(RelayRow.FIELDS))
        w.writeheader()
        for r in self.rows:
            w.writerow(r.to_dict())
        return buf.getvalue()


def _probe(bus: RoutedBus, a: LCSAgent, b: LCSAgent, inputs: List[str]) -> float:
    ok = 0
    for bits in inputs:
        bus.next_tick()
        a.act(bits[:2], explore=False)
        bus.deliver()
        b.act(bits[2:], explore=False)
        ok += b.emitted() == mux6(bits)
        a.engine.action_set = []
        b.engine.action_set = []
    return ok / len(inputs)


def run_relay(group: str = "organism", trials: int = 8000, fail_at: int = 3000,
              mode: str = "dead", seed: int = 0, probe_every: int = 100, n_eval: int = 128,
              recovered_at: float = 0.95, bursts: Sequence[int] = (1500, 5500),
              burst_len: int = 20, params: Optional[Params] = None,
              triggers: Optional[Triggers] = None, sup_threshold: float = 0.8,
              sup_patience: int = 2) -> RelayLog:
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    rng = np.random.default_rng(seed)
    inputs = _inputs(np.random.default_rng(seed + 10_000), n_eval)
    bus = RoutedBus()
    genome = parse_dna(SENDER_DNA.read_text()).genome
    triggers = triggers or Triggers(noise_floor=0.05, repair_threshold=0.25,
                                    unrepaired_integral=1.0, window=5)
    a = LCSAgent("A", input_len=2, actions=[f"(send * {s})" for s in SYMBOLS],
                 rules=genome.rules(), seed=seed, bus=bus, frozen=True,
                 structural=(group == "organism"), triggers=triggers)
    b = LCSAgent("B", input_len=4, actions=["(emit 0)", "(emit 1)"], symbols=SYMBOLS,
                 params=params or Params(p_explore=0.3), seed=seed + 1, bus=bus)
    r1 = Relay("R1", bus, SYMBOLS, seed=seed + 11)
    r2 = Relay("R2", bus, SYMBOLS, seed=seed + 12)
    for n in (a, b, r1, r2):
        bus.add(n)
    a.routes.add("R1")
    a.peers = ["R1", "R2"]
    r1.routes.add("B")
    r2.routes.add("B")
    b.routes.add("R1")            # credit path B→relay→A; broadcast if nothing consumed
    b.routes.add("R2")

    log = RelayLog(group=group, seed=seed, mode=mode)
    failed = False
    armed_at: Optional[int] = None          # first probe at which B has converged
    sup_low = 0
    recovery: Optional[int] = None
    first_probe_after_fail = True
    burst_set = {t for b0 in bursts for t in range(b0, b0 + burst_len)}
    relays = {"R1": r1, "R2": r2}
    failed_relay: Optional[str] = None

    for trial in range(1, trials + 1):
        if trial == fail_at:
            failed_relay = sorted(a.routes)[0]      # the relay A is actually using dies
            relays[failed_relay].mode = mode
            failed = True
        bits = "".join(map(str, rng.integers(0, 2, 6)))
        bus.next_tick()
        a.act(bits[:2])
        bus.deliver()                       # A→R→B (relay re-sends inside deliver)
        b.act(bits[2:])
        r = 1.0 if b.emitted() == mux6(bits) else 0.0
        if trial in burst_set:
            r = float(rng.integers(0, 2))   # transient corruption, not a failure
        b.reward(r)
        bus.deliver()                       # credit B→R→A
        a.tick()
        b.tick()
        if trial % probe_every == 0 or trial == trials:
            acc = _probe(bus, a, b, inputs)
            if armed_at is None and acc >= recovered_at:
                armed_at = trial
            if group == "supervisor" and armed_at is not None:
                # global view, one rule for failures and transients alike
                sup_low = sup_low + 1 if acc < sup_threshold else 0
                if sup_low >= sup_patience:
                    a.reroute("supervisor")
                    sup_low = 0
            if failed and recovery is None and not first_probe_after_fail and acc >= recovered_at:
                recovery = trial - fail_at
            if failed:
                first_probe_after_fail = False
            log.rows.append(RelayRow(trial=trial, acc=acc, failed=failed,
                                     route=",".join(sorted(a.routes)), reroutes=len(a.reroutes),
                                     a_noise=round(a.organism.state.noise_rate, 4),
                                     b_macro=b.engine.macro_size()))
    arm = armed_at if armed_at is not None else trials + 1
    false_reroutes = sum(1 for rr in a.reroutes
                         if arm <= rr["trial"] < fail_at
                         or (rr["trial"] >= fail_at and rr["to"] == failed_relay))
    log.final = {
        "group": group, "seed": seed, "mode": mode, "trials": trials, "fail_at": fail_at,
        "recovery": recovery, "recovered": recovery is not None,
        "final_acc": log.rows[-1].acc, "acc_before_fail": max(
            (r.acc for r in log.rows if not r.failed), default=0.0),
        "armed_at": armed_at, "failed_relay": failed_relay,
        "reroutes_total": len(a.reroutes),
        "reroutes_after_fail": [dict(rr) for rr in a.reroutes if rr["trial"] >= fail_at][:5],
        "false_reroutes": false_reroutes,
        "route_end": sorted(a.routes), "r1": {"forwarded": r1.forwarded, "dropped": r1.dropped},
        "r2": {"forwarded": r2.forwarded}, "credits_to_A": a.credits_received,
        "audit_chains_valid": a.organism.chain.verify() and b.organism.chain.verify(),
    }
    return log


def compare(seeds: Sequence[int] = range(30), mode: str = "dead", trials: int = 8000,
            fail_at: int = 3000, c1_trials: int = 1500, c1_rate: float = 0.9,
            c2_ratio: float = 1.25, c3_per_10k: float = 1.0, recovered_at: float = 0.95,
            triggers: Optional[Triggers] = None, params: Optional[Params] = None,
            groups: Sequence[str] = GROUPS) -> Dict[str, Any]:
    per: Dict[str, List[Dict[str, Any]]] = {g: [] for g in groups}
    for sd in seeds:
        for g in groups:
            log = run_relay(g, trials, fail_at, mode, sd, recovered_at=recovered_at,
                            triggers=triggers, params=params)
            per[g].append(log.final)
    out: Dict[str, Any] = {"seeds": list(seeds), "mode": mode, "trials": trials,
                           "fail_at": fail_at, "groups": {}, "per_seed": per,
                           "triggers": (triggers or Triggers(noise_floor=0.05, repair_threshold=0.25,
                                                             unrepaired_integral=1.0)).to_dict()}
    cap = trials - fail_at
    for g in groups:
        recs = [f["recovery"] if f["recovery"] is not None else cap for f in per[g]]
        out["groups"][g] = {
            "recovered_rate": float(np.mean([f["recovered"] for f in per[g]])),
            "recovered_within_c1": float(np.mean([f["recovery"] is not None and f["recovery"] <= c1_trials
                                                  for f in per[g]])),
            "median_recovery": float(np.median(recs)),
            "q1": float(np.percentile(recs, 25)), "q3": float(np.percentile(recs, 75)),
            "false_reroutes_per_10k": 1e4 * sum(f["false_reroutes"] for f in per[g]) / (len(per[g]) * trials),
            "median_final_acc": float(np.median([f["final_acc"] for f in per[g]])),
        }
    o, s, n = out["groups"].get("organism"), out["groups"].get("supervisor"), out["groups"].get("none")
    v: Dict[str, Any] = {}
    if o is not None:
        v["C1_capability"] = o["recovered_within_c1"] >= c1_rate
        v["C3_false_alarms"] = o["false_reroutes_per_10k"] <= c3_per_10k
    if o is not None and s is not None:
        v["C2_latency_vs_supervisor"] = o["median_recovery"] <= c2_ratio * s["median_recovery"]
        v["ratio_vs_supervisor"] = (o["median_recovery"] / s["median_recovery"]
                                    if s["median_recovery"] else None)
    if n is not None:
        v["sanity_none_stuck"] = n["median_final_acc"] <= 0.75
    v["pass"] = all(bool(v.get(k, False)) for k in
                    ("C1_capability", "C2_latency_vs_supervisor", "C3_false_alarms", "sanity_none_stuck"))
    out["verdict"] = v
    return out


__all__ = ["run_relay", "compare", "RelayLog", "RelayRow", "GROUPS", "SENDER_DNA"]
