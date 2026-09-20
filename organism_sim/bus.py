"""
organism_sim.bus — Routed message bus
=====================================

``Payload`` carries two semantic fields — ``content`` and ``pressure`` (≥ 0,
task difficulty / error rate) — plus routing metadata. ``RoutedBus`` is a
tick-ordered queue with directed routes between named agents; ``deliver()``
hands each payload to its recipient(s), which must expose ``receive(payload)``.
"""

from __future__ import annotations

import itertools
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence, Set, Union

import numpy as np

Content = Union[str, Dict[str, Any], List[Any]]
_ids = itertools.count(1)

KINDS = ("signal", "result", "credit", "state")


@dataclass
class Payload:
    content: Content
    pressure: float
    sender: str
    recipient: Optional[str] = None     # None = every node routed from sender
    kind: str = "signal"
    tick: int = 0
    id: int = field(default_factory=lambda: next(_ids))
    reply_to: Optional[int] = None

    def __post_init__(self):
        if self.pressure < 0:
            raise ValueError("pressure must be >= 0")
        if self.kind not in KINDS:
            raise ValueError(f"unknown payload kind {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Receiver(Protocol):
    name: str
    routes: Set[str]

    def receive(self, payload: Payload) -> None: ...


class Relay:
    """Dumb forwarding node. ``signal`` payloads are re-sent to every route; ``credit``
    payloads are forwarded to the upstream sender of the last signal. Modes:
    ``ok`` (forward), ``dead`` (drop everything), ``poisoned`` (forward a random
    symbol at pressure 1.0)."""

    def __init__(self, name: str, bus: "RoutedBus", symbols: Sequence[str] = (),
                 seed: int = 0):
        self.name = name
        self.bus = bus
        self.routes: Set[str] = set()
        self.symbols = list(symbols)
        self.rng = np.random.default_rng(seed)
        self.mode = "ok"
        self.upstream: Optional[str] = None
        self.forwarded = 0
        self.dropped = 0
        self.credits = 0

    def receive(self, payload: Payload) -> None:
        if payload.kind == "credit":
            self.credits += 1
            if self.upstream:
                self.bus.send(Payload(content=payload.content, pressure=payload.pressure,
                                      sender=self.name, recipient=self.upstream, kind="credit"))
            return
        if payload.kind != "signal":
            return
        self.upstream = payload.sender
        if self.mode == "dead":
            self.dropped += 1
            return
        content, pressure = payload.content, payload.pressure
        if self.mode == "poisoned" and self.symbols:
            content = self.symbols[int(self.rng.integers(len(self.symbols)))]
            pressure = 1.0
        for t in sorted(self.routes):
            self.bus.send(Payload(content=content, pressure=pressure, sender=self.name,
                                  recipient=t, kind="signal"))
            self.forwarded += 1


class RoutedBus:
    def __init__(self):
        self.nodes: Dict[str, Receiver] = {}
        self.queue: Deque[Payload] = deque()
        self.tick = 0
        self.delivered = 0
        self.dropped = 0
        self.log: List[Dict[str, Any]] = []

    def add(self, node: Receiver) -> Receiver:
        if node.name in self.nodes:
            raise ValueError(f"duplicate node {node.name!r}")
        self.nodes[node.name] = node
        return node

    def connect(self, a: str, b: str, bidirectional: bool = True) -> None:
        self.nodes[a].routes.add(b)
        if bidirectional:
            self.nodes[b].routes.add(a)

    def sever(self, a: str, b: str) -> bool:
        if b in self.nodes[a].routes:
            self.nodes[a].routes.discard(b)
            self.log.append({"tick": self.tick, "event": "sever", "from": a, "to": b})
            return True
        return False

    def edges(self) -> List[Dict[str, str]]:
        return [{"from": n, "to": t} for n, nd in self.nodes.items() for t in sorted(nd.routes)]

    def send(self, payload: Payload) -> Payload:
        payload.tick = self.tick
        self.queue.append(payload)
        return payload

    def inject(self, content: Content, pressure: float = 0.0, target: Optional[str] = None,
               kind: str = "signal") -> Payload:
        return self.send(Payload(content=content, pressure=pressure, sender="env",
                                 recipient=target, kind=kind))

    def targets(self, p: Payload) -> List[str]:
        if p.recipient is not None:
            return [p.recipient] if p.recipient in self.nodes else []
        if p.sender == "env":
            return list(self.nodes)
        src = self.nodes.get(p.sender)
        return sorted(src.routes) if src else []

    def deliver(self) -> int:
        n = 0
        while self.queue:
            p = self.queue.popleft()
            ts = self.targets(p)
            if not ts:
                self.dropped += 1
                continue
            for t in ts:
                self.nodes[t].receive(p)
                n += 1
        self.delivered += n
        return n

    def next_tick(self) -> int:
        self.tick += 1
        return self.tick

    def describe(self) -> Dict[str, Any]:
        return {"tick": self.tick, "nodes": list(self.nodes), "edges": self.edges(),
                "queued": len(self.queue), "delivered": self.delivered, "dropped": self.dropped}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.describe(), indent=indent)


__all__ = ["Payload", "RoutedBus", "Relay", "Receiver", "KINDS"]
