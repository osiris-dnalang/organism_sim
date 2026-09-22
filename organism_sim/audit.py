"""
organism_sim.audit — Hash-chained telemetry log
===============================================

Each ``TelemetryRecord`` is hashed as ``SHA-256(prev_hash ‖ payload)``. The
chain can be verified end-to-end and optionally HMAC-sealed with a secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Iterable, List, Optional

from .spec import TelemetryRecord

GENESIS_HASH = "0" * 64


class AuditChain:
    """``max_records`` bounds memory: only the newest records are retained, but the head hash
    is carried forward so every append still chains onto the true head and ``verify``
    checks the retained window plus head continuity. ``len`` is the number ever appended."""

    def __init__(self, secret: Optional[bytes] = None, max_records: Optional[int] = None):
        self.records: List[TelemetryRecord] = []
        self._secret = secret
        self.max_records = max_records
        self._head = GENESIS_HASH
        self._count = 0
        self.dropped = 0

    @property
    def head(self) -> str:
        return self._head

    @staticmethod
    def digest(prev_hash: str, payload: str) -> str:
        return hashlib.sha256((prev_hash + payload).encode()).hexdigest()

    def append(self, record: TelemetryRecord) -> TelemetryRecord:
        record.prev_hash = self._head
        record.hash = self.digest(record.prev_hash, record.payload())
        self.records.append(record)
        self._head = record.hash
        self._count += 1
        if self.max_records is not None and len(self.records) > self.max_records:
            excess = len(self.records) - self.max_records
            del self.records[:excess]
            self.dropped += excess
        return record

    def verify(self) -> bool:
        if not self.records:
            return self._head == GENESIS_HASH or self.dropped == self._count
        prev = self.records[0].prev_hash if self.dropped else GENESIS_HASH
        for rec in self.records:
            if rec.prev_hash != prev or self.digest(rec.prev_hash, rec.payload()) != rec.hash:
                return False
            prev = rec.hash
        return prev == self._head

    def seal(self) -> Optional[str]:
        if self._secret is None:
            return None
        return hmac.new(self._secret, self.head.encode(), hashlib.sha256).hexdigest()

    def check_seal(self, seal: str) -> bool:
        expected = self.seal()
        return expected is not None and hmac.compare_digest(expected, seal)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.to_dict(), sort_keys=True) for r in self.records)

    @classmethod
    def from_records(cls, records: Iterable[TelemetryRecord],
                     secret: Optional[bytes] = None) -> "AuditChain":
        chain = cls(secret)
        chain.records = list(records)
        chain._count = len(chain.records)
        chain._head = chain.records[-1].hash if chain.records else GENESIS_HASH
        return chain

    def __len__(self) -> int:
        return self._count


__all__ = ["AuditChain", "GENESIS_HASH"]
