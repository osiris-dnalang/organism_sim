"""Typed queries over the repository's own state: git commit, checked-in results, schemas."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
SIBLINGS = {"organism_sim": ROOT, "bridge": ROOT.parent / "bridge",
            "dnalang-core": ROOT.parent / "dnalang-core"}


def git_head(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class State:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.results = self.root / "results"
        self.schemas = self.root / "schemas"

    # ── raw access ───────────────────────────────────────────────────────────
    def load(self, name: str) -> Any:
        return json.loads((self.results / name).read_text())

    def has(self, name: str) -> bool:
        return (self.results / name).exists()

    def commits(self) -> Dict[str, Optional[str]]:
        return {k: git_head(p) for k, p in SIBLINGS.items()}

    def schema(self, which: str = "telemetry") -> str:
        f = {"telemetry": "telemetry_row.schema.json", "payload": "payload.schema.json"}[which]
        return (self.schemas / f).read_text()

    # ── queries ──────────────────────────────────────────────────────────────
    def relay(self, mode: str = "dead", trigger: str = "threshold") -> Dict[str, Any]:
        fname = ("relay_cusum_eval30_seeds0-29.json" if trigger == "cusum"
                 else "relay_eval30_seeds0-29.json")
        d = self.load(fname)[mode]
        arm = "organism-cusum" if trigger == "cusum" else "organism"
        g = d["groups"]
        return {"file": fname, "mode": mode, "trigger": trigger, "seeds": len(d["seeds"]),
                "organism": g[arm], "supervisor": g["supervisor"], "none": g["none"],
                "verdict": d["verdict"]}

    def false_alarms(self, mode: str = "dead", trigger: str = "threshold") -> Dict[str, Any]:
        r = self.relay(mode, trigger)
        return {"mode": mode, "trigger": trigger,
                "organism_false_per_10k": r["organism"]["false_reroutes_per_10k"],
                "supervisor_false_per_10k": r["supervisor"]["false_reroutes_per_10k"],
                "recovered_within_c1": r["organism"]["recovered_within_c1"],
                "median_recovery": r["organism"]["median_recovery"], "pass": r["verdict"]["pass"]}

    def signalling(self) -> Dict[str, Any]:
        rows = self.load("signalling_sweep_50seeds.json")
        acc = [r["acc_15k"] for r in rows]
        return {"seeds": len(rows), "full": sum(a >= 0.95 for a in acc),
                "partial": sum(0.75 <= a < 0.95 for a in acc), "pooling": sum(a < 0.75 for a in acc),
                "median_acc": sorted(acc)[len(acc) // 2]}

    def drift(self) -> Dict[str, Any]:
        d = self.load("drift_eval30_seeds0-29.json")
        S = d["shifts"]

        def med(g):
            v = sorted(s["recovery"] for s in S if s["group"] == g)
            return v[len(v) // 2]
        return {"plain_median": med("plain"), "organism_median": med("organism"),
                "ratio": round(med("organism") / med("plain"), 3), "verdict": "FAIL"}

    def bridge(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        p = SIBLINGS["bridge"] / "results" / "step3_hard" / "compare.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["step3"] = {"organism_median": d["organism_median_verify"],
                            "ga_median": d["ga_median_verify"], "best_baseline": d["best_baseline"],
                            "verdict": d["verdict"]}
        p = SIBLINGS["bridge"] / "results" / "tier4" / "eval" / "drift_compare.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["tier4"] = {"median_evals_to_target": d["median_evals_to_target"],
                            "verdict": d["verdict"]}
        return out

    def scorecard(self) -> List[Dict[str, Any]]:
        rows = []
        try:
            s = self.signalling()
            rows.append({"experiment": "signalling (50 seeds)", "verdict": "quantified",
                         "number": f"{s['full']}/50 full, {s['partial']}/50 partial, {s['pooling']}/50 pooling"})
        except FileNotFoundError:
            pass
        try:
            d = self.drift()
            rows.append({"experiment": "concept drift 6-mux", "verdict": d["verdict"],
                         "number": f"ratio {d['ratio']}"})
        except FileNotFoundError:
            pass
        for mode in ("dead", "poisoned"):
            for trig in ("threshold", "cusum"):
                try:
                    r = self.relay(mode, trig)
                    o = r["organism"]
                    rows.append({"experiment": f"relay {mode} ({trig})",
                                 "verdict": "PASS" if r["verdict"]["pass"] else "FAIL",
                                 "number": f"median {o['median_recovery']:.0f}, C1 {o['recovered_within_c1']:.2f}, false/10k {o['false_reroutes_per_10k']:.2f}"})
                except FileNotFoundError:
                    pass
        b = self.bridge()
        if "step3" in b:
            rows.append({"experiment": "bridge step 3", "verdict": "PASS (tie)" if b["step3"]["verdict"]["pass"] else "FAIL",
                         "number": f"organism {b['step3']['organism_median']:.4f} / GA {b['step3']['ga_median']:.4f} / baseline {b['step3']['best_baseline']:.4f}"})
        if "tier4" in b:
            m = b["tier4"]["median_evals_to_target"]
            rows.append({"experiment": "bridge tier 4 shock", "verdict": "PASS" if b["tier4"]["verdict"]["pass"] else "FAIL",
                         "number": ", ".join(f"{k} {v:.0f}" for k, v in m.items())})
        return rows

    def status(self) -> Dict[str, Any]:
        return {"commits": self.commits(),
                "results": sorted(p.name for p in self.results.glob("*.json")) if self.results.exists() else [],
                "schemas": sorted(p.name for p in self.schemas.glob("*.json")) if self.schemas.exists() else [],
                "scorecard": self.scorecard()}


__all__ = ["State", "git_head", "ROOT", "SIBLINGS"]
