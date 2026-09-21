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
        for fname, label, arm, base in (("expzero_eval_seeds0-4.json", "exp zero: informed priors", "informed", "control"),
                                        ("expone_eval_seeds0-4.json", "exp one: cusum injection", "cusum", "plain"),
                                        ("m1_eval_seeds0-4.json", "m1: regulatory genes", "grn-evolved", "cusum"),
                                        ("m2_eval_seeds0-4.json", "m2: priors, 16-bit family", "informed", "shape")):
            try:
                d = self.load(fname)
                v = d["verdict"]
                key = "pooled_median" if "pooled_median" in d else ("pooled" if "pooled" in d else "pooled_median95")
                pooled = d[key]

                def med(a, pooled=pooled):
                    return pooled[a]["median95"] if isinstance(pooled[a], dict) else pooled[a]
                rows.append({"experiment": label, "verdict": "PASS" if v.get("pass") else "FAIL",
                             "number": f"{arm} {med(arm):.0f} vs {base} {med(base):.0f}"})
            except (FileNotFoundError, KeyError):
                pass
        try:
            d = self.load("m2b_eval_seeds10-14.json")
            f = d["families"]["16bit"]
            pm = f["pooled_median"]
            rows.append({"experiment": "m2b: specificity prior, 16-bit", "verdict": "PASS" if f["verdict"]["pass"] else "FAIL",
                         "number": f"shape {pm['shape']:.0f} vs covering {pm['covering']:.0f} vs periodic {pm['periodic']:.0f}"})
        except (FileNotFoundError, KeyError):
            pass
        try:
            d = self.load("m2c_eval_seeds30-34.json")
            f = d["families"]["16bit"]
            pm, v = f["pooled_median"], f["verdict"]
            t = v["C3_tests"]
            rows.append({"experiment": "m2c: specificity prior replication, 16-bit",
                         "verdict": "PASS" if v["pass"] else "FAIL",
                         "number": f"shape {pm['shape']:.0f} vs covering {pm['covering']:.0f} vs periodic {pm['periodic']:.0f}; "
                                   f"C1 {v['C1_shape_vs_covering']}/5 C2 {v['C2_shape_vs_periodic']}/5 "
                                   f"p={t['shape_vs_covering']['p_one_sided']:.3f}/{t['shape_vs_periodic']['p_one_sided']:.3f}"})
        except (FileNotFoundError, KeyError):
            pass
        try:
            d = self.load("m3_eval_seeds20-24.json")
            rows.append({"experiment": "m3: open-ended tasks, 6-10 bit", "verdict": "PASS" if d["verdict"]["pass"] else "FAIL",
                         "number": f"poet {d['pooled_annecs']['poet']:.0f} vs random {d['pooled_annecs']['random']:.0f} ANNECS"})
        except (FileNotFoundError, KeyError):
            pass
        try:
            d = self.load("substrate_opt1_eval_seeds40-44.json")
            w, df = d["winner"], d["default"]
            rows.append({"experiment": "substrate-opt-1: XCS grid, 16-bit", "verdict": d["verdict"],
                         "number": f"winner {w['config']} {w['pooled_median']:.0f} vs default {df['pooled_median']:.0f} "
                                   f"(winner<default {d['winner_beats_default']})"})
        except (FileNotFoundError, KeyError):
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
