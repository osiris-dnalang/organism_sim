# Open-ended task generation works where tasks are hard: a pre-registered study on drifting hidden multiplexers

**Author:** Devin Phillip Davis · `osiris.dnalang@gmail.com`
**Repositories:** [`organism_sim`](https://github.com/osiris-dnalang/organism_sim) · [`dnalang-core`](https://github.com/osiris-dnalang/dnalang-core) · [`bridge`](https://github.com/osiris-dnalang/bridge)
**Version:** v0.4.0 (draft, 2026-09-22) · **License:** Apache-2.0
**Status:** working paper. Every number below is read from a file under `results/`; every experiment
was pre-registered (criterion committed to git before the run) and is reported whether it passed or
failed. One experiment (M3c) is **running as this draft is written** and its result is not yet known.

---

## Abstract

We ask whether an open-ended task generator — a population of environments held at the edge of an
agent's competence, in the style of POET (Wang et al. 2019) — produces more competence than a random
task stream, and under what conditions. On a small task space (6–10-bit hidden multiplexers) it does
not: a pre-registered run found the random stream *out-generating* the POET loop (ANNECS 46 vs 52,
ratio 0.885, 1/5 seeds), because every task there falls within a few iterations and the minimal
criterion is pure overhead. Scaling the task space required a faster base learner. A 55-point grid
over four XCS parameters failed to move a 16-bit recovery ceiling of ≈ 10,000 trials per drift
(the shipped default ranked 3rd of 55), but a second pre-registered sweep over population size,
tournament selection (Butz, Sastry & Goldberg 2003) and the specify operator (Lanzi 1997) reduced it
to **6,000 vs 8,750 trials on 5/5 held-out seeds**. With that learner, the same POET machinery at
12–16 bits reversed the earlier result: **pooled ANNECS 35 vs 25, ratio 1.40, 4/5 seeds**, with the
ANNECS curve still rising at the budget limit on 5/5. We report two confounds in that result — one
measurement artifact and one bundled mechanism — and the ablation that separates them is in flight.
We also report, as part of the same record, six negative results that closed three research
directions: offline and LLM-authored rule priors, specificity-matched priors (three independent
samples), and evolved regulatory genes.

---

## 1. Introduction

The claim under test is narrow and falsifiable: *does generating tasks at the edge of competence
produce competence that a fixed or random curriculum does not?* The literature (POET; minimal-
criterion coevolution, Brant & Stanley 2017) answers yes on rich continuous domains. We test it on a
domain small enough to run thousands of trials per seed on a laptop, and simple enough that the
confounds are visible: Boolean hidden multiplexers whose logic drifts.

The contribution is not a new mechanism. Every mechanism here is named prior art. What is new is the
**measurement discipline** applied uniformly: each rung of the research ladder states its criterion
in a git commit before the run, uses seeds disjoint from any tuning, and is published as PASS or
FAIL. Of the eleven pre-registered experiments in this record, **seven failed**. The two positive
results reported here are only interpretable because the failures were kept.

### 1.1 Task family

A hidden generalised multiplexer of width *w* with *k* address bits: *k* positions (unknown to the
agent) select one of 2^*k* data positions among the remaining bits; the rest are irrelevant. Variants
add output inversion and a *parity twist* (output XOR an irrelevant bit). Drift replaces the hidden
instance; the agent is not told. The metric throughout is **trials to recovery**: trials from a drift
until a held-out probe of 256 inputs reaches ≥ 0.95 accuracy.

### 1.2 Base learner

An XCS-lineage Learning Classifier System (Wilson 1995): ternary-condition rules (`0/1/#`) whose
action bodies are programs in a closed s-expression DSL, accuracy-based fitness, a niche GA with
subsumption, and a bucket brigade. The DSL is closed by construction — `validate` rejects any operator
outside a fixed set — so every evolved program is enumerable and inspectable. Implementation is pure
NumPy, seeded and deterministic: every number in this paper reproduces from its seed.

---

## 2. System

### 2.1 Layers

| layer | what it is | where |
|---|---|---|
| L0 substrate | `Organism`: five-phase tick (ingest → update → decide → maintain → log), noise EMA, silent repair, mutation on structural collapse, hash-chained telemetry | `organism.py`, `audit.py` |
| L1 computation | `RuleEngine`: XCS over a fixed-length register; `LCSAgent` binds one engine to one organism and a routed `Payload(content, pressure)` bus with cross-agent credit | `lcs.py`, `agent.py`, `bus.py` |
| L2 regulation | `GRN`: genes with triggers (`when <metric> <op> <v>`, `after G`, `on_signal s`), dependencies, outputs; a signal board; control programs that act on the learner | `grn.py` |
| language | dna::}{::lang v0.2 supplies the grammar, the closed action DSL and the trigger grammar; the language imports nothing from the runtime | `dnalang-core` |

### 2.2 Tamper-evident telemetry under a memory bound

Every tick appends a record to an `AuditChain`: `hash = SHA-256(prev_hash ‖ canonical JSON)`, verified
end-to-end, optionally HMAC-sealed. Long open-ended runs hold many agents alive at once, and unbounded
chains are a memory leak in disguise — 16 live agents at N = 4000 exceeded 8 GB and forced a restart
during this study. `AuditChain(max_records=w)` retains a window of records while **carrying the head
hash forward**, so appends still chain onto the true head and `verify()` checks the retained window
plus head continuity. With `audit_window=2000`, worker residency fell from ~1.2 GB to ~150 MB
(measured), and the integrity property is unchanged for the retained window.

### 2.3 Neuro-symbolic gate

A language model may propose genes in dna::}{::lang, but nothing it writes reaches a running organism
until it passes a gate: `parse → static check against the *merged* genome (live + proposal) → runtime
binding (trigger metrics ∈ runtime metrics, `adjust` parameters ∈ runtime parameters, rule-condition
width = register width) → atomic insertion`. A rejection returns the exact diagnostics to the model
for self-correction and leaves the runtime byte-for-byte unchanged (asserted against a full state
snapshot in tests). The model is strictly outside the execution loop; the substrate stays
deterministic. Verified end-to-end against a local 7B model, which read live telemetry and authored
two valid regulator genes that were injected into a running organism.

---

## 3. Experiments

All runs: pre-registered criterion in a git commit before execution; tuning seeds 100–124, evaluation
seeds disjoint and never reused; results under `results/`.

### 3.1 Base learner ceiling — Substrate-Opt-1 (BOUNDED)

Four parameters × 5 tuning seeds (54 configurations plus the shipped default), 16-bit family
(k = 3, w = 16, N = 1000), no injection. Winner judged against the default on **fresh seeds 40–44**.

| | sweep pooled median (seeds 110–114) |
|---|---|
| grid minimum | 10,000 (θ_GA 25, AS on, μ 0.05, P# 0.75) |
| shipped default | 10,500 — **rank 3 of 55** |
| grid median / max | 21,750 / 30,000 (never recovers) |

Judged on fresh seeds: winner 11,500 vs default 10,000, winner better on 2/5 → **BOUNDED**. Marginals:
θ_GA and μ are monotonically harmful away from the default; action-set subsumption and P# are flat.
193 of 825 sweep shifts never recovered; the fastest single recovery anywhere was 5,000 trials.

### 3.2 Base learner breakthrough — Substrate-Opt-2 (PASS)

Three mechanisms the multiplexer literature names, absent from the Opt-1 grid: tournament selection
(Butz et al. 2003), the specify operator (Lanzi 1997), population size N (Wilson 1995). 12
configurations; sweep on tuning seeds 120–124; winner and default judged on **fresh seeds 50–54**.
Criterion committed first: winner < 7,000 **and** < default on ≥ 4/5.

| arm | per-seed median trials to recovery (seeds 50–54) | pooled |
|---|---|---|
| default (N 1000, roulette, no specify) | 9500, 11000, 6250, 8750, 8750 | 8,750 |
| **N 4000 · tournament · specify** | 6000, 7250, 5750, 5500, 6500 | **6,000** |

Winner < default on **5/5** → **PASS**. Sweep marginals isolate the mechanism: N 1000/2000/4000 →
9,750 / 7,375 / 6,375; tournament −375; specify +375 (noise). **Population size does the work**,
tournament selection adds a consistent small gain, and specify is not established — it is retained only
because it is the configuration that was judged. Cost: N = 4000 is ≈ 4.5× wall time per trial
(≈ 2.7× compute per recovered shift); the criterion was trials, and the record says so. The judged
configuration is frozen as `lcs.PARAMS16`.

### 3.3 Open-ended generation at 6–10 bits — M3 (FAIL)

POET-style agent–task pairs, children admitted only under a minimal criterion (some current agent
scores in [0.6, 0.95) *before* training), capped task population, agent transfer; control is a random
task stream at the same creation rate and training budget. Metric **ANNECS**: tasks unsolved at
creation and solved later. Criterion: poet > random on ≥ 4/5 and pooled ratio ≥ 1.25 (seeds 20–24).

Result: poet 46, random 52, ratio 0.885, C1 1/5 → **FAIL**. At this scale every task is solvable in a
few iterations, so ANNECS counts task *creation* and uniform sampling covers the space faster than
mutate-and-filter. The minimal criterion and transfer are overhead when nothing is hard.

### 3.4 Open-ended generation at 12–16 bits — M3b (PASS)

The same harness, generalised to a task `Space` (k address bits, width range, learner) whose default
reproduces M3 bit-for-bit (verified: seed 20 poet — ANNECS 48, archive 69, transfers 10, identical
curve). k = 3, widths 12–16, learner `PARAMS16`; 30 iterations × 3,000 trials per task; **fresh seeds
60–64**.

| seed | poet ANNECS / created | random ANNECS / created |
|---|---|---|
| 60 | 33 / 38 | 25 / 80 |
| 61 | 41 / 49 | 25 / 80 |
| 62 | 26 / 34 | **30** / 80 |
| 63 | 35 / 41 | 24 / 80 |
| 64 | 38 / 53 | 23 / 80 |

Pooled **35 vs 25, ratio 1.40**; poet > random on **4/5**; ANNECS still rising over the final five
iterations on **5/5** → **PASS**. Poet solved 75–85 % of what it created; the random stream ≈ 30 % of
the 80 tasks it created per seed. This is the reverse of §3.3 and the reversal is explained by the
same mechanism: when tasks are hard, admitting only the just-unsolvable ones is what produces
solved-later tasks; when they are not, it is overhead.

**Confound 1 (measurement).** A third criterion scored each arm's agents on the poet arm's final task
population. An agent of the wrong input width scores 0 by construction, so on one seed the random arm
scored 0.000 for a reason that is not competence. That criterion is withdrawn as evidence and is
reported width-matched from here on.

**Confound 2 (mechanism).** The poet arm bundles a **curriculum** (the minimal criterion) with
**inheritance** (an admitted child starts from a copy of its parent's agent; a better agent is
periodically transferred onto a task). M3b cannot say which carried the result.

### 3.5 Curriculum or inheritance — M3c (pre-registered, **running**)

Three arms on **fresh seeds 70–74**: `poet` (as M3b), `poet-fresh` (minimal criterion kept, **both**
transfer paths off — the curriculum alone), `random`. Criteria committed before the run: **A** —
poet-fresh > random on ≥ 4/5 and pooled ratio ≥ 1.25 (the curriculum alone produces the effect);
**B** — poet > poet-fresh on ≥ 4/5 (inheritance adds to it). The four outcomes (A∧B, A∧¬B, ¬A∧B,
¬A∧¬B) are all informative, and `¬A ∧ B` would mean M3b's PASS was competence transfer rather than
task generation. This section will be completed with the result; **it is not yet known, and no claim
in this paper depends on it.**

### 3.6 Negative results retained

| experiment | question | result |
|---|---|---|
| Experiment Zero | do offline structural priors beat random init? | FAIL |
| Experiment One | does CUSUM-triggered injection speed recovery at 6 bits? | **PASS** (950 vs 1,250 trials, 0.27 false/10k) |
| M1 | do evolved regulatory genes beat a single detector? | FAIL — *specification gaming*: fitness priced recovery only, so evolution disabled detection and maximised injection (284 injections/15k, 169 false/10k) |
| M2 | do family-shaped priors carry signal at 16 bits? | FAIL (0/5 vs a shape-matched random control) |
| M2b / M2c | does specificity-matched injection help at 16 bits? | FAIL twice on fresh seeds; three samples read +13 %, +3 %, −13 % — the effect is noise; direction closed |
| concept drift (6-mux) | does the organism layer help a drifting XCS? | FAIL (layer redundant with XCS) |
| relay failure | does local detection reroute faster than a global supervisor? | **PASS** for `dead` (median 100 trials, 0 false/10k, 2× faster); FAIL for `poisoned` (20/30) |

M1 produced a design rule now applied to every rung: **an evolved learner's fitness must price its
interventions**, or it will buy the metric with side effects.

---

## 4. Discussion

**What is established.** (i) In this task family, mechanisms *above* the learner — priors of any
content or specificity, blind injection, evolved regulatory genes — do not change wide-space sample
efficiency; six pre-registered runs agree. (ii) The learner's own capacity and selection pressure do:
N = 4000 with tournament selection cuts 16-bit recovery by 31 % on held-out seeds, at ≈ 2.7× the
compute. (iii) Given that learner, open-ended generation beats a random stream at 12–16 bits by 1.40×
ANNECS, having lost at 6–10 bits — the crossover is the result, not either endpoint alone.

**What is not.** Whether the M3b advantage is curriculum or competence transfer (M3c, in flight).
Whether any of this holds outside drifting Boolean multiplexers — nothing here tests that. Whether the
gain survives at widths beyond 16, where the learner would again become the bottleneck. And the
efficiency result is bounded by what was tested: Opt-1 bounded *its four knobs*, not XCS, which Opt-2
then demonstrated by moving the ceiling with three different ones. A bound over a grid is not a
theorem.

**Honest cost accounting.** The headline efficiency gain is bought with compute: 4.5× wall time per
trial. Reported as such throughout, because a per-trial metric that ignores per-trial cost is how a
result gets overstated.

## 5. Reproduction

```bash
cd organism_sim && pip install -e ".[dev]" && pytest          # 186 tests
python experiments/substrate_opt.py 7                          # Substrate-Opt-1 (tune 110-114, judge 40-44)
python experiments/substrate_opt2.py 7                         # Substrate-Opt-2 (tune 120-124, judge 50-54)
python experiments/m3_wide.py 5                                # M3b  (fresh seeds 60-64)
python experiments/m3_ablate.py 5                              # M3c  (fresh seeds 70-74)
python -m organism_sim.benchmarks.m2c                          # M2c  (fresh seeds 30-34)
```

Seeds are fixed in each driver; every run writes its JSON, CSV and log to `results/`, and the
scorecard in `README.md` and `docs/PROGRAM.md` is generated from those files and guarded by a test.

## References

Brant & Stanley (2017), *Minimal criterion coevolution*. — Butz, Sastry & Goldberg (2003),
*Tournament selection in XCS*. — Lanzi (1997), *A study of the generalization capabilities of XCS*. —
Wang et al. (2019), *POET: Paired Open-Ended Trailblazer*. — Wilson (1995), *Classifier fitness based
on accuracy*.
