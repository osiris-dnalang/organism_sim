"""The release pre-flight must actually fail when an artifact is wrong."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release as vr  # noqa: E402


def test_passes_on_the_current_tree():
    assert vr.main(["--no-tests"]) == 0


def test_zenodo_metadata_is_consistent_with_the_packaging():
    meta = json.loads((ROOT / "zenodo.json").read_text())
    assert meta["license"] == "Apache-2.0"                       # not MIT
    assert all(k in meta for k in vr.REQUIRED_ZENODO_KEYS)
    assert meta["upload_type"] == "software" and meta["access_right"] == "open"
    assert any(r["identifier"] == "10.5281/zenodo.22862566" for r in meta["related_identifiers"])
    assert "organism_sim" in meta["title"]


def test_detects_bad_json_missing_docs_and_scorecard_drift(tmp_path, monkeypatch):
    fake = tmp_path / "repo"
    shutil.copytree(ROOT, fake, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "osiris_env", ".pytest_cache"))
    monkeypatch.setattr(vr, "ROOT", fake)

    problems = []
    (fake / "zenodo.json").write_text("{not json")
    vr.check_zenodo(problems)
    assert any("not valid JSON" in p for p in problems)

    problems = []
    meta = json.loads((ROOT / "zenodo.json").read_text())
    meta["license"] = "MIT"
    (fake / "zenodo.json").write_text(json.dumps(meta))
    vr.check_zenodo(problems)
    assert any("license" in p for p in problems)

    problems = []
    (fake / "docs" / "RESEARCH_PAPER.md").unlink()
    vr.check_docs(problems)
    assert any("RESEARCH_PAPER.md is missing" in p for p in problems)

    problems = []
    readme = fake / "README.md"
    readme.write_text(readme.read_text().replace("| PASS |", "| TOTALLY PASSED |", 1))
    vr.check_scorecard(problems)
    assert any("drifted" in p for p in problems)

    problems = []
    notes = fake / "docs" / "ZENODO_UPDATE.md"
    notes.write_text(notes.read_text() + "\ncites `results/does_not_exist.json`\n")
    vr.check_cited_results(problems)
    assert any("does not exist" in p for p in problems)


def test_runs_as_a_script():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_release.py"), "--no-tests"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "RELEASABLE" in r.stdout
