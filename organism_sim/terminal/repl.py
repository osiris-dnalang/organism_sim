"""Executor: an Intent becomes either a query answer, a benchmark argv for the existing CLI
(shown before it runs; ``dry`` = show only), a schema dump, or a scaffold file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .intent import HELP_TEXT, Intent, parse
from .scaffold import scaffold
from .state import State


def argv_for(intent: Intent) -> Optional[List[str]]:
    """Map RUN_* intents to the organism_sim.cli argument vector (or None)."""
    s = intent.slots
    if intent.name == "RUN_GUARD":
        return ["guard"]
    if intent.name != "RUN_EVAL":
        return None
    bench = s.get("benchmark", "relay")
    if bench in ("relay",):
        av = ["bench", "relay", "--mode", s.get("mode", "dead"), "--group", s.get("group", "organism")]
        if "seeds" in s:
            av += ["--compare", str(s["seeds"])]
        if s.get("trigger") == "cusum":
            av[av.index("--group") + 1] = "organism-cusum"
        return av
    if bench == "drift":
        av = ["bench", "drift"]
        if "seeds" in s:
            av += ["--compare", str(s["seeds"])]
        return av
    if bench == "mux":
        av = ["bench", "mux", "--phase", s.get("phase", "A")]
        if s.get("isolated"):
            av.append("--isolated")
        if "trials" in s:
            av += ["--trials", str(s["trials"])]
        return av
    if bench == "swarm":
        av = ["swarm"]
        if "trials" in s:
            av += ["--ticks", str(s["trials"])]
        return av
    return None


def answer(intent: Intent, state: State) -> Dict[str, Any]:
    s = intent.slots
    if intent.name == "QUERY_RESULTS":
        t = intent.text.lower()
        try:
            if "false" in t or "alarm" in t:
                return state.false_alarms(s.get("mode", "dead"), s.get("trigger", "threshold"))
            if "signal" in t or "pooling" in t:
                return state.signalling()
            if "drift" in t:
                return state.drift()
            if "bridge" in t or "step 3" in t or "tier" in t:
                return state.bridge()
            if "relay" in t or "recover" in t:
                return state.relay(s.get("mode", "dead"), s.get("trigger", "threshold"))
            return {"scorecard": state.scorecard()}
        except FileNotFoundError as exc:
            return {"error": f"no such result on disk: {exc}"}
    if intent.name == "STATUS":
        return state.status()
    if intent.name == "SHOW_SCHEMA":
        which = "payload" if "payload" in intent.text.lower() else "telemetry"
        return {"schema": which, "text": state.schema(which)}
    return {"help": HELP_TEXT}


def handle(text: str, state: Optional[State] = None, dry: bool = False,
           scaffold_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Route one sentence. Returns a dict describing what was done (never raises on input)."""
    st = state or State()
    it = parse(text)
    out: Dict[str, Any] = {"intent": it.to_dict()}
    if it.name == "SCAFFOLD":
        req = text.split(":", 1)[1].strip() if ":" in text else text
        path = (scaffold_dir or Path.cwd()) / "scaffold.md"
        scaffold(req, st, out=path, copy=not dry)
        out["scaffold"] = str(path)
        return out
    av = argv_for(it)
    if av is not None:
        out["argv"] = ["python", "-m", "organism_sim.cli", *av]
        if not dry:
            from ..cli import main as cli_main
            if av == ["guard"]:
                from ..benchmarks.ci_guard import main as guard_main
                out["exit"] = guard_main([])
            else:
                out["exit"] = cli_main(av)
        return out
    out["answer"] = answer(it, st)
    return out


def render(res: Dict[str, Any]) -> str:
    it = res["intent"]
    head = f"[{it['intent']}] slots={json.dumps(it['slots'])}"
    if "scaffold" in res:
        return f"{head}\nscaffold written: {res['scaffold']}"
    if "argv" in res:
        line = f"{head}\n$ {' '.join(res['argv'])}"
        return line + (f"\nexit {res['exit']}" if "exit" in res else "  (dry run)")
    a = res["answer"]
    if "help" in a:
        return f"{head}\n{a['help']}"
    if "text" in a and "schema" in a:
        return f"{head}\n{a['text']}"
    if "scorecard" in a and isinstance(a["scorecard"], list):
        rows = a["scorecard"]
        w = max(len(r["experiment"]) for r in rows) if rows else 10
        body = "\n".join(f"  {r['experiment']:<{w}}  {r['verdict']:<11} {r['number']}" for r in rows)
        rest = {k: v for k, v in a.items() if k != "scorecard"}
        return f"{head}\n{body}" + (f"\n{json.dumps(rest, indent=1)}" if rest else "")
    return f"{head}\n{json.dumps(a, indent=1)}"


def repl(dry: bool = False) -> int:
    st = State()
    print("organism_sim terminal — deterministic routing, no model in the loop. 'help' for intents, 'quit' to exit.")
    while True:
        try:
            line = input("osim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            return 0
        print(render(handle(line, st, dry=dry)))


__all__ = ["argv_for", "answer", "handle", "render", "repl"]
