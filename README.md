# organism_sim

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

## Substrate reference numbers

Single organism, default triggers (`repair_threshold` 0.45 over a 0.40 noise floor):
~16 silent repairs, 0 mutations, 96 % stable over 100 ticks. Population of 8, 200 ticks:
mean `sync_r` 0.5 → 0.75 → 0.83 for `phase_gain` 0 → 0.15 → 0.3. Noise ramp
0.005/tick: mutations ≈ 0 until the floor crosses `repair_threshold`, then rise.
