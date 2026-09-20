"""
organism_sim.cli
================

    python -m organism_sim.cli run       [--ticks 100] [--seed 0] [--spec x.json|x.dna]
                                         [--repair-threshold 0.45] [--noise-floor 0.40]
                                         [--entropy-floor 3.0] [--window 5] [--integral 1.0]
                                         [--json] [--telemetry out.jsonl] [-v]
    python -m organism_sim.cli swarm     [--n 8] [--ticks 100] [--seed 0] [--ramp 0.0]
                                         [--topology all|ring] [--phase-gain 0.15]
                                         [--no-coupling] [--csv out.csv] [--json-log out.json]
    python -m organism_sim.cli dump-spec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .alpha import build_alpha_spec, format_summary, format_tick
from .noise import Environment
from .organism import Organism
from .spec import OrganismSpec, Triggers, parse_dna
from .swarm import Coupling, Population


def load_spec(path: Optional[str]) -> OrganismSpec:
    if not path:
        return build_alpha_spec()
    p = Path(path)
    text = p.read_text()
    return parse_dna(text) if p.suffix.lower() in (".dna", ".dnalang") else OrganismSpec.from_json(text)


def _add_trigger_flags(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("triggers (override the spec)")
    g.add_argument("--repair-threshold", type=float, default=None)
    g.add_argument("--noise-floor", type=float, default=None)
    g.add_argument("--entropy-floor", type=float, default=None, dest="entropy_floor_bits")
    g.add_argument("--window", type=int, default=None)
    g.add_argument("--integral", type=float, default=None, dest="unrepaired_integral")
    g.add_argument("--repair-gain", type=float, default=None)


def apply_trigger_overrides(base: Triggers, args) -> Triggers:
    kw = base.to_dict()
    for k in ("repair_threshold", "noise_floor", "entropy_floor_bits", "window",
              "unrepaired_integral", "repair_gain"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v
    kw["window"] = int(kw["window"])
    return Triggers(**kw)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="organism-sim",
                                 description="ALife state-machine agent simulation")
    sub = ap.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run one organism")
    run.add_argument("--spec")
    run.add_argument("--ticks", type=int, default=100)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--burst-prob", type=float, default=0.05)
    run.add_argument("--json", action="store_true")
    run.add_argument("--telemetry", help="write per-tick telemetry JSONL")
    run.add_argument("-v", "--verbose", action="store_true")
    _add_trigger_flags(run)

    sw = sub.add_parser("swarm", help="run a coupled population")
    sw.add_argument("--n", type=int, default=8)
    sw.add_argument("--ticks", type=int, default=100)
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--burst-prob", type=float, default=0.05)
    sw.add_argument("--ramp", type=float, default=0.0, help="noise-floor increase per tick")
    sw.add_argument("--topology", choices=("all", "ring"), default="all")
    sw.add_argument("--phase-gain", type=float, default=0.15)
    sw.add_argument("--transfer-rate", type=float, default=0.10)
    sw.add_argument("--no-coupling", action="store_true")
    sw.add_argument("--csv", help="write per-tick population metrics CSV")
    sw.add_argument("--json-log", help="write per-tick population metrics JSON")
    _add_trigger_flags(sw)

    sub.add_parser("dump-spec", help="print the reference spec as JSON")

    be = sub.add_parser("bench", help="run a benchmark")
    be.add_argument("name", choices=("mux",))
    be.add_argument("--phase", choices=("A", "B"), default="A")
    be.add_argument("--sender", choices=("protocol", "learn"), default="protocol")
    be.add_argument("--isolated", action="store_true", help="phase B: sever the A→B route")
    be.add_argument("--trials", type=int, default=5000)
    be.add_argument("--seed", type=int, default=0)
    be.add_argument("--log-every", type=int, default=250)
    be.add_argument("--p-explore", type=float, default=None)
    be.add_argument("--csv")
    be.add_argument("--json-log")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "dump-spec":
        print(build_alpha_spec().to_json())
        return 0
    if args.cmd is None:
        ap.print_help()
        return 2
    if args.cmd == "bench":
        from .benchmarks.mux import run_single, run_split
        from .lcs import Params
        params = Params(p_explore=args.p_explore) if args.p_explore is not None else None
        if args.phase == "A":
            log = run_single(args.trials, seed=args.seed, log_every=args.log_every, params=params)
        else:
            log = run_split(args.trials, seed=args.seed, coupled=not args.isolated,
                            log_every=args.log_every, params=params, sender=args.sender)
        if args.csv:
            Path(args.csv).write_text(log.to_csv())
        if args.json_log:
            Path(args.json_log).write_text(log.to_json())
        print(json.dumps(log.final, indent=2, default=str))
        return 0
    try:
        if args.cmd == "run":
            spec = load_spec(args.spec)
            spec.triggers = apply_trigger_overrides(spec.triggers, args)
            env = Environment(n_genes=len(spec.genome), seed=args.seed,
                              noise_floor=spec.triggers.noise_floor, burst_prob=args.burst_prob)
            org = Organism(spec, env=env, seed=args.seed)
            for _ in range(args.ticks):
                rec = org.step()
                if args.verbose and (rec.events or rec.tick % 10 == 0):
                    print(format_tick(rec), file=sys.stderr)
            r = org.summary()
            if args.telemetry:
                Path(args.telemetry).write_text(org.chain.to_jsonl() + "\n")
            print(json.dumps(r.to_dict(), indent=2) if args.json else format_summary(r))
            return 0 if r.audit_chain_valid else 1

        triggers = apply_trigger_overrides(Triggers(), args)
        coupling = Coupling(phase_gain=0.0 if args.no_coupling else args.phase_gain,
                            transfer_rate=0.0 if args.no_coupling else args.transfer_rate,
                            topology=args.topology)
        pop = Population(n=args.n, seed=args.seed, triggers=triggers, coupling=coupling,
                         burst_prob=args.burst_prob, ramp=args.ramp)
        pop.run(args.ticks)
        if args.csv:
            Path(args.csv).write_text(pop.log.to_csv())
        if args.json_log:
            Path(args.json_log).write_text(pop.log.to_json())
        print(json.dumps(pop.summary(), indent=2))
        return 0 if pop.summary()["audit_chains_valid"] else 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
