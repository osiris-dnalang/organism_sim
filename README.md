# organism_sim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22862567.svg)](https://doi.org/10.5281/zenodo.22862567)  
Snapshot 2026-09-20 of the three repositories with their pre-registered results: [10.5281/zenodo.22862567](https://doi.org/10.5281/zenodo.22862567) (concept DOI 10.5281/zenodo.22862566).

Artificial-life simulation of autonomous **rule-evolving** agents. The organism is
the computation: a Learning Classifier System (XCS lineage) whose rules are ternary
conditions plus action bodies in a closed DSL, governed by a noise-driven state
machine and connected to other agents over a routed message bus. Pure NumPy, seeded
and deterministic, no neural networks, no hardware, no physical constants.

```bash
pip install -e ".[dev]"
pytest                                   # 74 tests, ~1 min

python -m organism_sim.cli bench mux --phase A --trials 5000 --csv muxA.csv
python -m organism_sim.cli bench mux --phase B --sender protocol --trials 6000
python -m organism_sim.cli bench mux --phase B --sender protocol --isolated
python -m organism_sim.cli bench mux --phase B --sender learn --seed 2 --trials 10000
python -m organism_sim.cli run   --ticks 100                      # substrate only
python -m organism_sim.cli swarm --n 8 --ticks 200 --ramp 0.005 --csv pop.csv
```

## Language

`organism_sim` consumes **dna::}{::lang** (`pip install dnalang`, repo `dnalang-core`): the
closed action DSL (`organism_sim.rules` re-exports `dnalang.action_dsl`), the trigger grammar
(`dnalang.regulation.Trigger`), and the parser (`spec.parse_dna` goes through `dnalang.parse`
for canonical and 2025 key-value genomes; a tolerant regex reader remains only for old files
with prose blocks). Rule genes lower to `RuleSet` (the LCS population), regulator genes to
`RegulatoryGraph` (the `grn` runtime). The language imports nothing from here.

## Program

`docs/PROGRAM.md` — the research program: an operational target (open-ended, auditable
competence growth), what the original dnalang genomes were actually specifying (gene
regulation and development — the unbuilt half), a five-layer architecture, and a seven-rung
milestone ladder with a kill criterion on every rung. Next build: M1, regulatory genes.

## Terminal (local routing, no model in the loop)

```bash
python -m organism_sim.cli chat                       # REPL
python -m organism_sim.cli ask "what were the false alarm rates on the last dead relay run"
python -m organism_sim.cli ask --dry "run the poisoned relay eval on 30 seeds with cusum"
python -m organism_sim.cli ask "scaffold: replace XCS subsumption with a Q-learning router"
```

Natural language is a *routing* layer: a deterministic keyword-and-slot matcher
(`terminal/intent.py`) maps a sentence to one of seven intents — run a benchmark (shown as
the exact `organism_sim.cli` argv before it runs), run the CI guard, query the checked-in
results (`terminal/state.py`: false-alarm rates, recovery medians, signalling rate, the full
scorecard, commits), print a contract, or **scaffold**: package the commits, the scorecard,
the hard constraints and the telemetry-row schema into `scaffold.md` (and the clipboard) as
a prompt for an external reasoner. The reasoner stays outside the execution loop.

## Contracts and guards

* `schemas/telemetry_row.schema.json`, `schemas/payload.schema.json` — the telemetry-row and
  bus-payload contracts (JSON Schema, draft-07). Tests validate every emitted row and every
  checked-in `results/**/*.jsonl` against them.
* `python -m organism_sim.benchmarks.ci_guard` — run on every push: steady-state latency of a
  full LCS trial at a 400-rule population must stay under budget (167 µs measured after the
  condition-matrix cache; 600 µs budget for CI runners), and the dead-relay organism arm must
  recover on every seed with median ≤ 150 trials and 0 false reroutes.

## Layers

| Module | Role |
|---|---|
| `rules.py` | Ternary conditions (`0/1/#`), s-expression DSL, step-budgeted tree-walk `Interpreter`. Ops: `emit set send sever route adjust if seq`; exprs: `reg var last not eq and or add`. |
| `lcs.py` | `RuleEngine`: match → cover → prediction array → select (explore/exploit) → act → credit (Widrow–Hoff prediction/error, accuracy-based fitness, action-set + GA subsumption, roulette deletion), GA (two-point crossover, bit↔`#` mutation), bucket brigade for deferred rewards. `compact()` (repair), `gp_mutate()` (structural), `erode()`. `mode="reinforce"` = cumulative Roth–Erev strength. |
| `agent.py` | `LCSAgent`: engine + `Organism` (as its `Processor`) + bus adapter. Register = input bits ⊕ last received symbol. Exploit-trial error → organism noise; incoming `pressure` → noise; reward forwarded upstream as a `credit` payload (cross-agent bucket brigade). |
| `bus.py` | `Payload(content, pressure, …)`, `RoutedBus` (directed routes, tick-ordered delivery, sever). |
| `organism.py` `noise.py` `swarm.py` `audit.py` | Substrate: state machine (noise EMA, silent repair, mutation on entropy collapse / sustained unrepaired noise), environment, population coupling, hash-chained telemetry. |
| `benchmarks/mux.py` | 6-multiplexer: Phase A (single agent), Phase B (split input over the bus). `protocol_sender.dna` shows the `.dna` → rules pipeline. |

## 6-multiplexer results (deterministic per seed)

`mux6(b) = b[2 + 2·b0 + b1]`. Eval = 256 fixed inputs, exploit only, no learning.

**Phase A — one agent, 6 input bits.** ≥ 0.99 eval accuracy by 1,000 trials (seeds 0–1);
population converges on the 8 maximally general correct rules (`11###1→1`, `01#1##→1`, …).

**Phase B — A sees address bits, B sees data bits; A can only `send` a symbol to B.**

| tier | isolated (A→B severed) | coupled |
|---|---|---|
| `protocol` sender (A's 4 rules fixed from `protocol_sender.dna`, B learns), 6k trials, seeds 0–1 | 0.68–0.71 | **1.0** |
| `learn` (both from scratch; A = cumulative reinforcement, B = XCS), 10–15k trials | 0.70 | seed 2: **1.0** · seed 0: 0.88 · seeds 1, 3: 0.71–0.77 |

Isolated B is capped by the majority-of-data-bits ceiling (11/16 = 0.6875). With a
fixed protocol the bus carries the address and B learns the full multiplexer. With both
agents learning, a signalling system emerges on some seeds and a **partial pooling
equilibrium** (one symbol reused for two addresses) on others — the known outcome for
signalling games with > 2 states. Accuracy-based fitness alone never breaks the pooling
symmetry; the sender needs state-specific cumulative strength (`mode="reinforce"`).

**Signalling reliability (50 seeds, 15k trials, `results/signalling_sweep_50seeds.json`):**
9/50 reach a full injective protocol (≥ 0.95), 36/50 partial (3 symbols, ≈ 0.87),
5/50 pool; median eval accuracy 0.867.

**Measured, not assumed:** the organism-level structural hooks (fitness erosion under
unrepaired noise, compaction on repair, GP mutation on collapse) *hurt* an XCS receiver —
B plateaus at 0.93 with them on vs 1.0 with them off (3 seeds). They exist, are tested,
and default to **off** for `LCSAgent` (`structural=True` to enable).

## Shifting-logic (concept drift) test — pre-registered, FAILED

`benchmarks/drift.py`: the 6-mux truth table is perturbed every 3,000 trials
(`invert`, `addr_swap`, `data_perm`, unannounced). Two groups on identical seeds and
engines; only the organism's structural hooks differ (`LCSAgent(structural=…)`). The
organism response on sustained error: rule-set compaction, GP variants, and a *shock*
(reset experience of high-error rules, exploration boost).

Pre-registered criterion: organism median recovery ≤ 0.75 × plain **and** disjoint IQRs.
Trigger profile tuned on held-out seeds 100–104 (`results/drift_tuning_seeds100-104.json`),
then evaluated once on seeds 0–29 (`results/drift_eval30_seeds0-29.json`):

| 150 shifts / group | plain XCS | organism |
|---|---|---|
| median recovery (trials to ≥ 0.95) | 500 | 400 |
| Q1–Q3 | 200–700 | 200–700 |
| mean area-under-error | 123.0 | 117.4 |

Ratio 0.80, IQRs identical → **FAIL** on both halves. Per kind (exploratory): `invert`
200 → 100, `addr_swap` 700 → 600, `data_perm` 700 → 700. Conclusion: on a task the base
learner can relearn by itself, the organism layer is redundant with XCS's own GA,
deletion and subsumption. It stays available and tested but is telemetry-only by default.
Two bugs found on the way and fixed: the drift probe set was one repeated input, and
repair was scaling a *measured* error rate as if compaction had reduced it (it cannot).

`python -m organism_sim.cli bench drift --compare 30 --repair-threshold-drift 0.25` reruns it.

## Relay-failure recovery — pre-registered, split result

`benchmarks/relay.py`: split 6-mux over A → relay → B with a spare relay unrouted. At trial
3,000 the relay A is using fails (`dead`: drops signals; `poisoned`: random symbols at
pressure 1.0). Transient 20-trial reward-corruption bursts occur elsewhere and must not
trigger reroutes. Groups on identical seeds: `none` (no rerouting), `supervisor`
(centralized monitor with global probe access, reroutes after two probes < 0.8),
`organism` (A's local trigger only: credit pressure → noise → sustained unrepaired excess
→ sever + route to an unrouted peer). Criteria stated first: C1 recover ≥ 0.95 within
1,500 trials on ≥ 90 % of seeds; C2 median ≤ 1.25 × supervisor; C3 ≤ 1 false reroute per
10k trials. Profile tuned on seeds 100–104 (`results/relay_tuning_seeds100-104.json`:
`repair_threshold 0.25, ∫excess 10, window 100`), evaluated on 0–29
(`results/relay_eval30_seeds0-29.json`).

| 30 seeds | `dead` | `poisoned` |
|---|---|---|
| none | stuck 0.70 | stuck 0.68 |
| supervisor | median 200, 30/30 | median 200, 30/30 |
| organism | **median 100, 30/30, 0 false** | median 200, **20/30**, 0 false |
| verdict | **PASS** (ratio 0.5 vs supervisor) | **FAIL** (C1 = 0.67) |

Silent failure is the first positive result for the organism layer: a job the base learner
cannot do at all, done twice as fast as a monitor with strictly more information, with no
false alarms. Poisoning fails because B learns to ignore random symbols and settles at
≈ 0.31 error, marginal against the 0.25 threshold — fires on some seeds, not others; the
5-seed tuning set did not expose it. Not retuned on evaluation seeds.

**Follow-up (pre-registered): CUSUM change-point trigger.** `LCSAgent(cusum=(μ₀, k, h))`
replaces the level threshold with a one-sided CUSUM on per-trial upstream pressure —
its false-alarm rate is a designed quantity. `(0.05, 0.10, 8)` tuned on seeds 100–109
(`results/relay_cusum_tuning_seeds100-109.json`), evaluated once on 0–29
(`results/relay_cusum_eval30_seeds0-29.json`): `dead` median 200, 30/30, 0.25 false/10k →
PASS; `poisoned` median 200, **24/30**, 0.08 false/10k → **FAIL** (C1 0.80 < 0.90).
Change-point detection lifts poisoning from 20/30 to 24/30 inside the false-alarm budget
but doubles silent-failure latency (h must exceed what a burst can accumulate). A ~0.17
mean-pressure shift against 20-trial bursts is near the detectability limit for any local
statistic on this scalar; passing needs a richer downstream signal, not a better detector.

Design decisions that the harness forced (each was a bug or an unfairness found by
running it): a frozen genome has no entropy trigger; the relay A is *using* is the one that
fails; false alarms are counted only after B has converged; reroutes have a cooldown;
upstream ``credit`` pressure carries exploit-trial error only (exploration misses are the
downstream agent's own choice, not evidence about the link).

## Experiment Zero — offline structural priors vs random init, pre-registered, FAIL

`benchmarks/expzero.py`: a hidden *generalised* 6-input multiplexer per seed (unknown address
positions, data order and inversion), switching to a fresh instance every 3,000 trials. Arms
on identical seeds: **A** covering only; **B** + 20 family-shaped rules written once from the
family definition (`expzero_seed_rules.dna`, sha `819ac399…`), injected at start and at each
shift as zero-experience hypotheses; **C** + 20 *random* well-formed rules, same schedule.
Metric: trials to ≥ 0.95 for the initial window and each of 4 recoveries. Criterion stated
first: B beats both A and C on the per-seed median on ≥ 4/5 seeds (seeds 0–4, nothing tuned).

| arm | pooled median |
|---|---|
| A random | 1000 |
| B informed | 1000 |
| C control (random seeding) | 900 |

Informed wins 2/5 → **FAIL** (`results/expzero_eval_seeds0-4.{json,csv}`). B beats A on 4/5
seeds, but C does at least as well: injecting fresh hypotheses at each shift helps
re-convergence; the *information* in the priors adds nothing measurable. Without the control
arm this would have read as a win.

## Experiment One — detector-triggered random injection under drift, pre-registered, PASS

`benchmarks/expone.py`: same hidden drifting family as Experiment Zero, but the agent is
not told when shifts happen. Arms: `plain` XCS; `cusum` — one-sided CUSUM on the agent's own
exploit error, arming only after first convergence, on detection inject N random
zero-experience rules (cooldown 300); `periodic` — N random rules every 500 trials, no
detector; `oracle` — inject at the true shift. Criterion stated first: cusum median
recovery (≥ 0.95) < plain on ≥ 4/5 seeds **and** ≤ 1 false injection per 10k trials.
Profile `(μ₀ 0.05, k 0.10, h 8, N 40)` from seeds 100–104
(`results/expone_tuning_seeds100-104.json`), evaluated once on 0–4
(`results/expone_eval_seeds0-4.{json,csv}`):

| arm | median to ≥ 0.95 | to ≥ 0.99 | injections / 15k | false / 10k |
|---|---|---|---|---|
| plain | 1250 | 1700 | 0 | — |
| cusum | **950** | 1500 | 4.6 | 0.27 |
| oracle | 900 | 1450 | 4.0 | 0 |
| periodic | 650 | 1050 | 30 | — |

C1 4/5, C2 met → **PASS**. Exploratory: blind periodic injection is faster still (4/5 seeds)
at 6× the injections. So the claim is narrow: **detection buys parsimony, not speed** — most
of the diversity benefit with ~5 injections instead of 30, which matters only where an
injection costs something (e.g. hardware shots). The mechanism is *random immigrants*
(Grefenstette 1992; Cobb & Grefenstette 1993) with a change detector; the contribution here
is the pre-registered measurement with blind-periodic and oracle controls, not the mechanism.

## M1 — regulatory genes (dnalang v0.2), pre-registered, FAIL — and specification gaming

`grn.py`: genes with `trigger` (`on_genesis` / `continuous` / `on_error` / `after G` /
`on_signal s` / `when <metric> <op> <v>`), `dependencies`, `outputs`; a signal board;
control genes in the closed DSL (`(adjust inject|explore|cusum_reset|compact …)`), rule
genes that inject their rule on expression; `.dna` round-trip; mutation/crossover over
triggers, wiring and parameters. `benchmarks/m1.py`: hand genome (`m1_hand.dna`) and a
genome meta-evolved on seeds 100–102 (`m1_evolved.dna`, fitness = recovery speed), judged on
0–4 against Experiment One's arms. Criterion committed first: evolved beats `cusum` on ≥ 4/5
**and** ≤ 1 false injection / 10k.

| arm | median recovery | injections / 15k | false / 10k |
|---|---|---|---|
| plain | 1250 | 0 | — |
| cusum | 950 | 4.6 | 0.27 |
| periodic | 650 | 30 | — |
| grn-hand | 750 | 16 | 1.1 |
| grn-evolved | **400** | **284** | **169** |

C1 5/5, C2 failed → **FAIL** (`results/m1_eval_seeds0-4.json`, `results/m1_evolution_seeds100-102.json`).
Evolution disabled the detector (`when error > 11.29`), made the injection gene fire almost
always (`when error < 0.35`), and ran at the one-per-50-ticks floor — the fitness priced
recovery speed and not interventions, so that was the optimum. Specification gaming, caught
by the pre-registered false-alarm criterion. Two honest readings: on this family injection
is monotonically better up to the floor and detection is unnecessary when injection is
free; and the hand-composed cascade beat the single mechanism on 4/5 seeds (exploratory).
Design rule recorded for the ladder: **any evolved learner's fitness must price its
interventions.**

## M2 — do priors matter when injection stops being free? Pre-registered, FAIL

`benchmarks/m2.py`: 16-bit hidden family (3 address positions selecting one of 8 data
positions, 5 irrelevant, optional inversion), N = 1000, fresh instance every 30k trials, 3
scored shifts. Arms: `none`; `informed` (40 family-shaped rules, action = guessed data bit,
`m2_seed_rules.dna` sha `033cbaa2…`); `shape` (same specificity, random content — isolates
shape from content); `random` (p# 0.5); `periodic` (blind, every 500). Criteria first:
informed < shape on ≥ 4/5 **and** informed < random on ≥ 4/5.

| arm | pooled median recovery |
|---|---|
| none | 9500 |
| informed | 9500 |
| shape | 8250 |
| random | 9000 |
| periodic | 10500 |

C1 0/5, C2 2/5 → **FAIL** (`results/m2_eval_seeds0-4.{json,csv}`). The priors carry no
signal; the action coupling is wrong for every inverted instance (half of them), so the
shape-matched control with random actions beats it 5/5 — a prior true less often than chance
is worse than none. Exploratory: blind injection now *hurts* (periodic < none 0/5: the
Experiment One mechanism is a small-space effect); specificity alone helps (shape > none
5/5, ≈ 13 %) — a candidate for its own pre-registration, about the search, not the layer.

## Substrate reference numbers

Single organism, default triggers (`repair_threshold` 0.45 over a 0.40 noise floor):
~16 silent repairs, 0 mutations, 96 % stable over 100 ticks. Population of 8, 200 ticks:
mean `sync_r` 0.5 → 0.75 → 0.83 for `phase_gain` 0 → 0.15 → 0.3. Noise ramp
0.005/tick: mutations ≈ 0 until the floor crosses `repair_threshold`, then rise.
