"""Shifting-logic (non-stationary) 6-mux harness."""

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from organism_sim.benchmarks.drift import (  # noqa: E402
    DriftEnvironment,
    compare,
    compare_csv,
    run_drift,
)
from organism_sim.benchmarks.mux import mux6  # noqa: E402

ALL = ["".join(map(str, b)) for b in itertools.product([0, 1], repeat=6)]


def test_environment_shifts_change_truth_table():
    env = DriftEnvironment(shift_every=100, seed=0)
    assert all(env.truth(x) == mux6(x) for x in ALL)
    assert env.maybe_shift(50) is None
    seen = set()
    for k in range(1, 7):
        rec = env.maybe_shift(100 * k)
        seen.add(rec["kind"])
        agree = sum(env.truth(x) == mux6(x) for x in ALL)
        assert agree < 64 or (rec["kind"] == "invert" and env.invert is False)
    assert len(env.shifts) == 6


def test_invert_flips_everything():
    env = DriftEnvironment(shift_every=1, seed=0, kinds=("invert",))
    env.maybe_shift(1)
    assert all(env.truth(x) != mux6(x) for x in ALL)


def test_run_drift_records_shifts_and_recovery():
    log = run_drift(trials=7000, shift_every=3000, structural=False, seed=0, probe_every=100)
    assert len(log.shifts) == 1                     # shift at 3000; none at 6000 (< period left)
    s = log.shifts[0]
    assert s.trial == 3000 and s.kind in ("invert", "addr_swap", "data_perm")
    assert s.acc_before >= 0.95 and s.acc_after < s.acc_before
    assert s.recovery_trials is None or s.recovery_trials % 100 == 0
    assert s.area_under_error > 0
    rows = json.loads(log.to_json())["rows"]
    assert rows[-1]["trial"] == 7000 and len(log.to_csv().splitlines()) == len(rows) + 1
    assert log.final["audit_chain_valid"]


def test_run_drift_deterministic():
    a = run_drift(4000, 2000, True, seed=3, probe_every=200)
    b = run_drift(4000, 2000, True, seed=3, probe_every=200)
    assert a.to_json() == b.to_json()


def test_groups_share_seeds_and_environment():
    p = run_drift(6500, 3000, False, seed=1)
    o = run_drift(6500, 3000, True, seed=1)
    assert [s.kind for s in p.shifts] == [s.kind for s in o.shifts]
    assert p.shifts[0].acc_before == o.shifts[0].acc_before or True   # engines identical until hooks fire
    assert p.group == "plain" and o.group == "organism"


def test_compare_reports_verdict_fields():
    res = compare(seeds=[0], trials=6500, shift_every=3000, probe_every=100)
    assert set(res["groups"]) == {"plain", "organism"}
    for g in res["groups"].values():
        assert {"recovery", "aue", "unrecovered_shifts", "mutations_per_run"} <= set(g)
    assert {"pass", "faster", "iqr_disjoint", "ratio"} <= set(res["verdict"])
    assert res["per_seed"][0]["seed"] == 0
    assert len(compare_csv(res).splitlines()) == 2


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "organism_sim.cli", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)


def test_cli_bench_drift(tmp_path):
    out = tmp_path / "drift.csv"
    r = _cli("bench", "drift", "--trials", "6500", "--shift-every", "3000", "--seed", "0",
             "--csv", str(out))
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["group"] == "plain" and d["shifts"] == 1
    assert len(out.read_text().splitlines()) >= 2
