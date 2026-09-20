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
    def __init__(self, secret: Optional[bytes] = None):
        self.records: List[TelemetryRecord] = []
        self._secret = secret

    @property
    def head(self) -> str:
        return self.records[-1].hash if self.records else GENESIS_HASH

    @staticmethod
    def digest(prev_hash: str, payload: str) -> str:
        return hashlib.sha256((prev_hash + payload).encode()).hexdigest()

    def append(self, record: TelemetryRecord) -> TelemetryRecord:
        record.prev_hash = self.head
        record.hash = self.digest(record.prev_hash, record.payload())
        self.records.append(record)
        return record

    def verify(self) -> bool:
        prev = GENESIS_HASH
        for rec in self.records:
            if rec.prev_hash != prev or self.digest(rec.prev_hash, rec.payload()) != rec.hash:
                return False
            prev = rec.hash
        return True

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
        return chain

    def __len__(self) -> int:
        return len(self.records)


__all__ = ["AuditChain", "GENESIS_HASH"]
