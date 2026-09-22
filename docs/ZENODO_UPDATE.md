# Zenodo release notes — v0.4.0 (2026-09-22)

Previous deposit: **10.5281/zenodo.22862567** (2026-09-20 snapshot of `organism_sim`, `dnalang-core`
and `bridge`; concept DOI 10.5281/zenodo.22862566). This release is a new version of that concept
record and covers `organism_sim` only.

## What changed since 10.5281/zenodo.22862567

| | result | data |
|---|---|---|
| **Substrate-Opt-1** | BOUNDED — a 54-point grid over θ_GA, action-set subsumption, μ and P# does not move 16-bit recovery; the shipped default ranks 3/55 | `results/substrate_opt1_{sweep_seeds110-114,eval_seeds40-44}.json` |
| **Substrate-Opt-2** | **PASS** — N 4000 + tournament selection: 6,000 vs 8,750 trials/shift on fresh seeds 50–54, 5/5. Frozen as `lcs.PARAMS16` | `results/substrate_opt2_{sweep_seeds120-124,eval_seeds50-54}.json` |
| **M2c** | FAIL — second replication of the specificity prior (1/5, p = 0.86); three samples read +13 %, +3 %, −13 %; direction closed | `results/m2c_eval_seeds30-34.{json,csv}` |
| **M3b** | **PASS** — open-ended generation at 12–16 bits with `PARAMS16`: pooled ANNECS 35 vs 25, ratio 1.40, 4/5; still rising 5/5 | `results/m3b_eval_seeds60-64.json` |
| **M3c** | pre-registered, **running at deposit time** — separates curriculum from inheritance on fresh seeds 70–74 | pending |
| neuro-symbolic bridge | `telemetry.capture_frame` + `chat_bridge.py`: a model may author dna::}{::lang genes, but a parse → check → bind → atomic-merge gate stands between it and the runtime | `schemas/telemetry_frame.schema.json`, `prompts/bridge_system.md` |
| `AuditChain` window | bounded retention with head-hash continuity — worker residency 1.2 GB → 150 MB in long open-ended runs | `organism_sim/audit.py` |

Two caveats are part of the M3b record and are stated in the paper, the README and `PROGRAM.md`:
its third criterion is confounded by input-width coverage (withdrawn as evidence, reported
width-matched), and its winning arm bundles curriculum with competence inheritance (which is exactly
what M3c is running to separate). **No claim in this release depends on M3c's outcome.**

## Reproduction

Python 3.9+, NumPy only for the core; `jsonschema` for the contract tests. Each driver fixes its own
seeds — pass a worker count as the sole argument.

```bash
git clone https://github.com/osiris-dnalang/organism_sim && cd organism_sim
pip install -e ".[dev]"
pytest                                            # 186 tests, ~2 min

python experiments/substrate_opt.py  7            # Substrate-Opt-1  tune 110-114, judge 40-44   (~100 min)
python experiments/substrate_opt2.py 7            # Substrate-Opt-2  tune 120-124, judge 50-54   (~80 min)
python experiments/m3_wide.py        5            # M3b              fresh seeds 60-64           (~2.7 h)
python experiments/m3_ablate.py      5            # M3c              fresh seeds 70-74           (~4 h)
python -m organism_sim.benchmarks.m2c             # M2c              fresh seeds 30-34           (~26 min)
```

Wall times are for 8 threads on a laptop-class CPU (WSL2). Every run writes JSON, CSV and a log to
`results/`; the scorecard in `README.md` and `docs/PROGRAM.md` is generated from those files by
`State().scorecard_markdown()` and a test fails if the documents drift from the data.

## Integrity

Telemetry rows form a SHA-256 hash chain (`organism_sim/audit.py`); `AuditChain.verify()` checks the
chain end to end, and a bounded window keeps the head hash continuous so the property survives long
runs. Rows validate against `schemas/telemetry_row.schema.json`, bus payloads against
`schemas/payload.schema.json`, and bridge frames against `schemas/telemetry_frame.schema.json` — all
three are asserted over every checked-in result file by `tests/test_schemas.py`.

Provenance for this release is the git history: each experiment appears as a **pre-registration
commit** (criterion, before the run) followed by a **result commit** (data, verdict, scorecard row).
Reviewers should check that ordering rather than take any number here on trust.

## System requirements

- **Core:** Python 3.9+, NumPy ≥ 1.24, `dnalang` ≥ 0.2.0. Single-core runs are supported; the drivers
  parallelise over seeds with `multiprocessing`.
- **Memory:** ≈ 150 MB per worker for the open-ended runs with the default `audit_window=2000`.
  Unbounded audit chains at N = 4000 exceed 8 GB with 16 live agents — do not remove the window.
- **Optional:** `jsonschema` (contract tests), a local Ollama for the chat bridge (never required by
  the substrate; all tests are offline).
