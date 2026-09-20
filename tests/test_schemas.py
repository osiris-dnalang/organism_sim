"""The telemetry-row and payload contracts: every row the code emits validates, and the
schemas reject malformed rows."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
jsonschema = pytest.importorskip("jsonschema")

from organism_sim.alpha import build_alpha  # noqa: E402
from organism_sim.bus import Payload  # noqa: E402

ROW = json.loads((ROOT / "schemas" / "telemetry_row.schema.json").read_text())
PAY = json.loads((ROOT / "schemas" / "payload.schema.json").read_text())


def test_emitted_rows_validate():
    org = build_alpha(seed=0)
    org.run(30)
    v = jsonschema.Draft7Validator(ROW)
    for line in org.chain.to_jsonl().splitlines():
        v.validate(json.loads(line))


def test_schema_rejects_bad_rows():
    org = build_alpha(seed=0)
    org.step()
    row = org.chain.records[0].to_dict()
    v = jsonschema.Draft7Validator(ROW)
    v.validate(row)
    for mutate in (lambda d: d.update(hash="nothex"), lambda d: d.pop("prev_hash"),
                   lambda d: d.update(extra=1)):
        bad = dict(row)
        mutate(bad)
        with pytest.raises(jsonschema.ValidationError):
            v.validate(bad)


def test_payloads_validate():
    v = jsonschema.Draft7Validator(PAY)
    v.validate(Payload(content="s1", pressure=0.0, sender="A", recipient="B").to_dict())
    v.validate(Payload(content={"reward": 1.0, "explore": False}, pressure=0.0, sender="B",
                       recipient="A", kind="credit").to_dict())
    with pytest.raises(jsonschema.ValidationError):
        v.validate({**Payload(content="x", pressure=0.0, sender="A").to_dict(), "kind": "spam"})


def test_checked_in_jsonl_results_validate():
    files = list((ROOT / "results").glob("**/*.jsonl"))
    v = jsonschema.Draft7Validator(ROW)
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                v.validate(json.loads(line))
