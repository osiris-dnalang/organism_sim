#!/usr/bin/env python3
"""
scripts/validate_release.py — release pre-flight for organism_sim.

Checks, in order, and reports every failure rather than stopping at the first:

  1. ``zenodo.json`` is valid JSON, carries the required keys, and its ``license`` and
     ``version`` agree with ``pyproject.toml`` / ``CITATION.cff`` — a deposit that
     mislabels its licence is worse than no deposit.
  2. ``docs/RESEARCH_PAPER.md`` and ``docs/ZENODO_UPDATE.md`` exist and are non-empty.
  3. Every result file the release notes cite actually exists under ``results/``.
  4. The scorecard embedded in ``README.md`` and ``docs/PROGRAM.md`` still equals the one
     generated from ``results/`` (documents may not drift from data).
  5. ``pytest`` passes (skip with ``--no-tests`` when a suite has just been run).

Exit code 0 = releasable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ZENODO_KEYS = ("title", "description", "upload_type", "publication_date", "version",
                        "creators", "access_right", "license", "keywords")
REQUIRED_DOCS = ("docs/RESEARCH_PAPER.md", "docs/ZENODO_UPDATE.md")
MIN_DOC_BYTES = 500


def _fail(problems: List[str], msg: str) -> None:
    problems.append(msg)
    print(f"  FAIL  {msg}")


def check_zenodo(problems: List[str]) -> None:
    path = ROOT / "zenodo.json"
    if not path.exists():
        return _fail(problems, "zenodo.json is missing")
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return _fail(problems, f"zenodo.json is not valid JSON: {e}")
    missing = [k for k in REQUIRED_ZENODO_KEYS if k not in meta or meta[k] in ("", [], {})]
    if missing:
        _fail(problems, f"zenodo.json missing/empty keys: {missing}")
    if not isinstance(meta.get("creators"), list) or not meta["creators"]:
        _fail(problems, "zenodo.json: creators must be a non-empty list")
    pyproject = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'license\s*=\s*\{\s*text\s*=\s*"([^"]+)"', pyproject)
    if m and meta.get("license") != m.group(1):
        _fail(problems, f"zenodo.json license {meta.get('license')!r} != pyproject {m.group(1)!r}")
    cff = ROOT / "CITATION.cff"
    if cff.exists():
        c = re.search(r"^license:\s*(\S+)", cff.read_text(), re.M)
        if c and meta.get("license") != c.group(1):
            _fail(problems, f"zenodo.json license {meta.get('license')!r} != CITATION.cff {c.group(1)!r}")
    if not problems:
        print(f"  ok    zenodo.json — {meta['version']}, {meta['license']}, {len(meta['keywords'])} keywords")


def check_docs(problems: List[str]) -> None:
    for rel in REQUIRED_DOCS:
        p = ROOT / rel
        if not p.exists():
            _fail(problems, f"{rel} is missing")
        elif len(p.read_bytes()) < MIN_DOC_BYTES:
            _fail(problems, f"{rel} is present but nearly empty (< {MIN_DOC_BYTES} bytes)")
        else:
            print(f"  ok    {rel} — {len(p.read_bytes()):,} bytes")


def check_cited_results(problems: List[str]) -> None:
    """Every results/… path named in the release notes must exist."""
    notes = ROOT / "docs" / "ZENODO_UPDATE.md"
    if not notes.exists():
        return
    cited = set()
    for raw in re.findall(r"`(results/[^`]+)`", notes.read_text()):
        # expand brace groups: results/a_{b,c}.json -> two paths
        m = re.match(r"([^{]*)\{([^}]*)\}(.*)", raw)
        cited.update(f"{m.group(1)}{part}{m.group(3)}" for part in m.group(2).split(",")) if m else cited.add(raw)
    for rel in sorted(cited):
        if not (ROOT / rel).exists():
            _fail(problems, f"release notes cite {rel}, which does not exist")
    print(f"  ok    {len(cited)} result file(s) cited by the release notes exist")


def check_scorecard(problems: List[str]) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from organism_sim.terminal.state import State
        table = State().scorecard_markdown()
    except Exception as e:                                   # noqa: BLE001 - report, don't crash
        return _fail(problems, f"could not generate the scorecard: {type(e).__name__}: {e}")
    for rel in ("README.md", "docs/PROGRAM.md"):
        text = (ROOT / rel).read_text()
        m = re.search(r"<!-- scorecard:start -->\n(.*?)\n<!-- scorecard:end -->", text, re.S)
        if not m:
            _fail(problems, f"{rel} has no scorecard block")
        elif m.group(1) != table:
            _fail(problems, f"{rel} scorecard has drifted from results/ — regenerate it")
        else:
            print(f"  ok    {rel} scorecard matches results/ ({table.count(chr(10)) - 1} rows)")


def check_tests(problems: List[str]) -> None:
    print("  ...   running pytest (this takes a few minutes)")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
    if r.returncode != 0:
        _fail(problems, f"pytest failed: {tail[0]}")
    else:
        print(f"  ok    pytest — {tail[0]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-tests", action="store_true", help="skip pytest (use when it has just run)")
    args = ap.parse_args(argv)
    print(f"release pre-flight: {ROOT}")
    problems: List[str] = []
    check_zenodo(problems)
    check_docs(problems)
    check_cited_results(problems)
    check_scorecard(problems)
    if not args.no_tests:
        check_tests(problems)
    if problems:
        print(f"\nNOT RELEASABLE — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nRELEASABLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
