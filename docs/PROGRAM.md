# The dnalang program: engineering open-ended, auditable competence growth

*Working document, 2026-09-21. Every claim here is either measured (cited to a results file
or record) or labelled as a hypothesis with the experiment that would kill it.*

## 0. The word, and the definition we will use instead

"Superintelligence" has no operational definition, so it cannot be engineered toward; it can
only be named after the fact. The measurable target this program pursues is:

> **Open-ended, auditable competence growth (OACG):** a system whose measured competence
> over a task distribution *it did not receive from a human* keeps increasing for as long as
> compute is supplied, faster than any fixed learner we can build for the same distribution,
> with every improvement replayable from seed and attributable to a logged change.

Three parts, each falsifiable: *open-ended* (the task distribution grows with the system:
POET's ANNECS-style count of newly solved, previously unsolved tasks keeps rising);
*faster than fixed* (a pre-registered comparison against the strongest static baseline,
as in every experiment in this repository); *auditable* (hash-chained provenance and JSON
Schema contracts, as shipped). Nobody has OACG. LLMs do not have it: their competence
is fixed at training time and their improvements are not attributable. That is the gap,
and it is the only sense in which "beyond current AI" is honest here.

## 1. What dnalang was actually saying

Reading the 2025 genomes (`organism/*.dna`, 72–128 genes) with the vocabulary removed,
five structural commitments survive. They are the design intent.

| commitment | in the 2025 genomes | in the rebuilt stack today | literature |
|---|---|---|---|
| **Programs are genomes** — modular genes with expression levels, mutated and crossed | `GENE {expression, mutation}` | `dnalang-core` (gene = circuit macro), `organism_sim` (gene = LCS rule) | GP, LCS |
| **Regulation** — genes fire on internal events and other genes' outputs | `trigger: on_genesis / after_G2 / when_gamma>0.3 / on_error`, `dependencies`, `outputs` | **absent** (rules match a register; no gene-to-gene signalling) | artificial gene regulatory networks (Banzhaf 2003; Bongard 2002) |
| **Development** — the genotype builds a phenotype through a process, not a lookup | clusters: structural → dynamic → autopoietic; agents own gene subsets | **absent** (genotype→phenotype is a direct compile) | evolvable developmental encodings; Stanley & Miikkulainen |
| **Autopoiesis** — the organism repairs and rewrites itself under internal pressure | `HEALING {}`, `when_performance_below_threshold` | measured: helps only where the base learner has no move (routing, diversity); redundant elsewhere | random immigrants; concept-drift ensembles |
| **Environment as ground truth** — fitness from the world, including physics | `fitness survival_plus` on IBM hardware | `dnalang-core` + `bridge` (Aer/IBM), joined provenance | hardware-in-the-loop optimisation |

The missing two — regulation and development — are exactly what open-endedness needs.
A rule that matches a register cannot build anything; a gene that activates other genes
in response to what they produced can build a controller, a protocol, a circuit family.
That is the unbuilt half of the language.

## 2. Architecture

```
            ┌───────────────────────────────────────────────────────────────┐
  L4  META  │  evolution of the learner itself (operators, credit rules,    │  hypothesis
            │  trigger profiles) — pre-registered, seeds disjoint            │
            ├───────────────────────────────────────────────────────────────┤
  L3  OPEN  │  task generator co-evolves with the population (POET-style);  │  hypothesis
            │  library of abstracted genes grows (DreamCoder-style)         │
            ├───────────────────────────────────────────────────────────────┤
  L2  GRN   │  dnalang v0.2: genes with triggers / dependencies / outputs   │  built (M1 FAIL)
            │  = regulatory graph; development = staged expression;          │
            │  compiled to L1 rule sets, DSL programs, circuits             │
            ├───────────────────────────────────────────────────────────────┤
  L1  MICRO │  organism_sim: LCS engine, closed DSL, routed bus, CUSUM     │  measured
            │  triggers, random-immigrant injection, reroute (167 µs/trial) │
            ├───────────────────────────────────────────────────────────────┤
  L0  TRUTH │  dnalang-core + bridge: Aer physics env, IBM hardware,        │  measured
            │  hash-chained ledgers, joined provenance, JSON Schemas, CI    │
            └───────────────────────────────────────────────────────────────┘
  Reasoner (offline): scaffold → external model → text → reviewed → committed. Never in a loop.
```

Rules that hold at every layer: the language (`dnalang`) imports nothing from any runtime;
`organism_sim` and `bridge` consume its targets (`RuleSet`, `RegulatoryGraph`, `Circuit`)
and never the reverse; every hardware or simulated evaluation writes a ledger row before it runs; every
structural change writes a telemetry row; a claim exists only after a pre-registered
criterion has been met on evaluation seeds never used for tuning; negative results are
kept and cited.

## 3. The milestone ladder

Each rung is an experiment with a stated kill criterion. Passing a rung does not prove
OACG; failing one ends that branch and is published.

| # | milestone | mechanism (prior art) | metric | kill criterion |
|---|---|---|---|---|
| M1 | **Regulatory genes** — dnalang v0.2 interpreter: genes with `trigger`, `dependencies`, `outputs`; staged expression; compiles to an L1 rule set | artificial GRNs (Banzhaf; Bongard) | on the drifting hidden-mux family, a GRN-encoded organism vs a flat LCS at equal evaluations | **FAIL (specification gaming; hand cascade 4/5 exploratory).** Language built (`dnalang` 0.2) |
| M2 | **Content must matter** — priors vs random injection on a 12–16-bit family where random rules almost never match | Experiment Zero, widened | B (informed) vs C (random) vs periodic, as before | **FAIL (0/5 vs shape-matched control).** LLM-prior branch closed. Specificity prior: M2b FAIL (3/5), M2c FAIL (1/5) — closed |
| M3 | **Open-ended task generation** — the environment is a population too: instances mutate, are kept if "just solvable" by some agent | POET (Wang et al. 2019); minimal-criterion coevolution (Brant & Stanley 2017) | ANNECS: count of tasks solved by later agents that no earlier agent solved, vs a fixed random task stream | **FAIL at 6–10 bits (ratio 0.885, 1/5)** — no hard tasks to reach; **PASS at 12–16 bits with `PARAMS16` (M3b: ratio 1.40, 4/5)** — with hard tasks the minimal criterion is what produces solved-later tasks. M3c (curriculum vs inheritance) pre-registered |
| M4 | **Library learning** — evolved DSL programs are abstracted into new primitives when they recur; the DSL grows | DreamCoder (Ellis et al. 2021); ADFs (Koza) | held-out task solve rate and description length before/after abstraction | no gain in solve rate on held-out families → abstraction adds nothing |
| M5 | **Division of labour** — populations on the bus with the signalling result (18 % full protocols, 50 seeds) as baseline; tasks that require ≥ 2 agents | emergent communication; Lewis signalling | fraction of communication-dependent tasks solved; protocol injectivity rate | no rise over the 18 % baseline with structure (M1) present → communication is not helped by regulation |
| M6 | **Physics as the open-ended environment** — hardware calibration drift as the task generator (the Flywheel Aim 3) | hardware-in-the-loop | re-convergence after `calibration_hash` changes, four arms | as pre-registered in `flywheel-2026/PREREGISTRATION.md` |
| M7 | **Meta-evolution** — the learner's own operators and trigger profiles are genomes; selection on *learning speed* across task families with seeds disjoint from evaluation | AutoML-Zero (Real et al. 2020); evolved learning rules | learning-speed improvement of generation *g+1* over *g* on held-out families | no monotone improvement over 5 generations → the "self-improving learner" branch closes |

**M1 result (2026-09-21): FAIL on the false-alarm criterion.** Meta-evolution of the
regulatory genome disabled detection and maximised injection (284 injections / 15k, 169
false / 10k) because fitness priced recovery speed only — specification gaming. Rule
adopted for every rung from here: *fitness must price interventions* (injections,
reroutes, evaluations). **M2 result (2026-09-21): FAIL.** On the 16-bit family, family-shaped priors carry no
signal (0/5 vs a shape-matched random control; the action coupling is wrong for inverted
instances), and blind injection *hurts* (0/5 vs none) — Experiment One's diversity mechanism
is a small-space effect. The only mover was specificity (shape-matched random rules beat
none 5/5, exploratory). Vector 1 (LLM/offline priors) is closed on this substrate. **M2b (2026-09-21): FAIL** — the specificity effect did not replicate on fresh seeds; at
16 bits neither injection (any content, any specificity) nor the covering wildcard rate
changes recovery (~9,500 trials/shift). What the substrate has measurably contributed:
topological self-repair (relay dead) and small-space diversity injection. **Search
efficiency in wide spaces is not a layer property**; if M3 needs it, it is a base-learner
pre-registration (tournament selection, specify operator, N, θ_GA — Butz et al.), or M3 runs
at 6–10 bits and says so. **M3 (2026-09-21): FAIL** at 6–10 bits — the random stream out-generates the POET-style
loop because nothing there is hard; at 16 bits the learner is too slow to generate for.
**M2c (2026-09-21): FAIL — autopsy.** A second replication of the specificity prior on
fresh seeds 30–34 (M2b harness unchanged, plus a permutation rank-sum criterion) was run as
the prerequisite for scaling M3 to 16 bits: shape-matched injection vs covering 1/5 (shape
*slower* on four seeds), vs blind periodic 3/5, p = 0.86 / 0.063. Three samples of the same
effect read +13 %, +3 %, −13 %; the M2 finding was a seed draw. Closed. Nothing that
injects rules — informed, shape-matched, or blind — changes 16-bit recovery, which sits at
≈ 9,500–12,000 trials/shift for every arm. M3 at 16 bits therefore was **not** scaffolded:
a POET-style generator cannot sit at the edge of competence when the learner takes ~10k
trials per instance, and at 6–10 bits M3 already showed it is overhead. **Next rung is
base-learner efficiency** (a pre-registered XCS-engineering comparison at 16
bits: tournament selection, specify operator, N, θ_GA), because both M3 and M7 are blocked
on it. M3 and M7 remain where OACG lives or dies. M6 is funded by the
proposal if it is accepted, and is the only rung with a physical environment.

### Substrate-Opt-1 — the learner's own knobs at 16 bits (pre-registered 2026-09-21)

Every rung above the learner left 16-bit recovery at ≈ 9,500–12,000 trials/shift. This is
the first experiment *on* the learner: `experiments/substrate_opt.py`, a 54-point grid over
XCS parameters — GA period θ_GA ∈ {25, 100, 500}, action-set subsumption ∈ {on, off} (new
`Params.as_subsumption`, default on = today's behaviour), mutation μ ∈ {0.01, 0.05, 0.10},
covering wildcard P# ∈ {0.33, 0.50, 0.75} — plus the current default as a 55th point, on the
M2b/M2c 16-bit family with its budgets, no injection of any kind.

Two stages, disjoint seeds, because a grid search is tuning: **sweep** every configuration
on tuning seeds 110–114 (score = pooled median of per-seed median recovery, cap 30,000);
**judge** the single best configuration and the default on fresh seeds 40–44. Decision,
stated before the run, on the judge seeds only:

| verdict | condition | consequence |
|---|---|---|
| PASS | winner pooled median < 3,000 | adopt as the mandatory 16-bit default |
| PARTIAL | 3,000 ≤ winner < 8,000 and winner < default on ≥ 4/5 judge seeds | real improvement, target missed; record, do not adopt as mandatory |
| BOUNDED | winner ≥ 8,000, or no 4/5 win over default | the XCS substrate's sample-efficiency ceiling at 16 bits is empirically bounded on this family: none of θ_GA, action-set subsumption, μ, P# moves it |

The sweep minimum is reported next to the judged number so a lucky tuning-seed draw cannot
be read as the result. Nothing in the grid, metric, family or budgets changes after the
first run.

**Result (2026-09-21): BOUNDED.** Sweep minimum 10,000 (θ_GA 25, AS on, μ 0.05, P# 0.75)
against the default's 10,500 (rank 3 of 55); on the fresh judge seeds the winner scored
11,500 vs the default's 10,000 and beat it on 2/5. θ_GA and μ are monotonically harmful
away from the default, subsumption and P# are flat, 193/825 sweep shifts never recovered,
and the fastest recovery observed anywhere was 5,000 trials. The learner's exposed
parameters do not move the 16-bit ceiling. With M2/M2b/M2c/M3 this closes the question
from both sides: neither the layers above the learner nor the learner's own knobs change
wide-space sample efficiency on this family. What would: a different learner (e.g. a
gradient or tree model over the register), or a task family with exploitable structure —
both are new programs, not rungs of this one.

### Substrate-Opt-2 — the literature's XCS mechanisms at 16 bits (pre-registered 2026-09-21)

Substrate-Opt-1 bounded the four knobs it tested. Three mechanisms it did not test are the
ones the multiplexer literature names as what lets XCS scale: **tournament selection** in the
GA (Butz, Sastry & Goldberg 2003), Lanzi's **specify** operator (1997), and **population
size N** (Wilson 1995: 20-mux at N = 2000; the 16-bit family here ran at 1000). All three are
now `Params` switches (`selection`, `specify`, `N`), defaults unchanged so no earlier result
moves. `experiments/substrate_opt2.py`: 2 × 2 × 3 = 12 configurations including the default;
sweep on tuning seeds 120–124, winner and default judged on fresh seeds 50–54; family,
budgets, metric and threshold as Opt-1.

| verdict | condition (judge seeds) | consequence |
|---|---|---|
| PASS | winner < 7,000 and winner < default on ≥ 4/5 | winner becomes `Params16`, the pre-registered learner for any further 16-bit rung; M3 at width is unblocked for that learner |
| PARTIAL | 7,000 ≤ winner < 8,000 and ≥ 4/5 | improvement recorded, not adopted |
| BOUNDED | otherwise | the literature's mechanisms do not move this family; the next step is a different learner |

Opt-1's 3,000 line is reported, not required. Cost (N = 4000 ≈ 4.5× wall time) is reported;
the criterion is trials. One tuning-seed observation motivated the run and is not evidence:
tournament + specify + N 4000 on seed 120 recovered in 5,750 / 5,750 / 7,500.

**Result (2026-09-21): PASS.** Judge seeds 50–54: winner (N 4000, tournament, specify) 6,000
[6000, 7250, 5750, 5500, 6500] vs default 8,750 [9500, 11000, 6250, 8750, 8750], 5/5. Sweep
marginals: N 1000 / 2000 / 4000 → 9,750 / 7,375 / 6,375; tournament −375; specify +375
(noise). Population size is the mechanism; tournament selection is a consistent small gain;
specify is not established. `lcs.PARAMS16` is the pre-registered 16-bit learner from here.
What this changes in the boundary statement: the ceiling Opt-1 bounded was over its four
knobs, not over XCS — the substrate's sample efficiency at 16 bits is ≈ 6,000 trials/shift
at N 4000, at 2.7× the compute per shift. What it does not change: nothing above the
learner moved it; the gain came from the learner. **Next rung: M3 at 16 bits with
`PARAMS16`**, pre-registered below.

### M3b — open-ended task generation at 12–16 bits with `PARAMS16` (pre-registered 2026-09-21)

The rung Substrate-Opt-2 unblocked. `benchmarks/m3.py` is generalised to a `Space`
(k address bits, width range, learner) whose default reproduces M3 bit-for-bit (verified:
seed 20 poet — ANNECS 48, archive 69, transfers 10, identical curve). `experiments/m3_wide.py`:
k = 3, widths 12–16, learner `PARAMS16`; 30 iterations × 3,000 training trials per task, 4
initial pairs, ≤ 8 tasks, every other harness parameter M3's; fresh seeds 60–64.

| criterion | statement |
|---|---|
| C1 | ANNECS(poet) > ANNECS(random) on ≥ 4/5 seeds |
| C2 | pooled ANNECS ratio ≥ 1.25 |
| C3 | on the poet arm's final task population (the self-generated distribution), the best accuracy reachable by poet's agents exceeds that reachable by the random arm's agents on ≥ 4/5 — competence over the self-generated distribution beats a fixed-curriculum learner of equal budget on that same distribution |

PASS ⇔ C1 ∧ C2 ∧ C3. Reported, not judged: ANNECS still rising over the last 5 iterations
(continuous growth), max width, transfers. If PASS: the first positive wide-space result for
open-ended generation; next rungs M4 / M7 on this learner. If FAIL: with the learner no
longer the excuse, the POET loop itself is what does not deliver on this family, and M3
closes.

**Result (2026-09-22): PASS.** Pooled ANNECS 35 vs 25 (ratio 1.40); C1 4/5 (seed 62: 26 vs
30), C3 5/5, still rising 5/5. Poet solved 75–85 % of the tasks it admitted, the random
stream ≈ 30 % of the 80 it created per seed. Caveats recorded with the result: C3 on seed
61 is 0.798 vs 0.000 because no random-arm agent had the width of poet's final tasks (C3
partly measures width coverage); and the poet arm bundles curriculum (minimal criterion)
with inheritance (a child starts from a copy of its parent's agent). What this changes in
the ladder: M3 is no longer FAIL-and-blocked; at a width where tasks are hard, the
POET-style loop out-generates a random stream. What it does not yet say: which of the two
bundled mechanisms carries it.

### M3c — curriculum or inheritance? (pre-registered 2026-09-22, not yet run)

Three arms, same harness, budgets and fresh seeds 70–74: `poet` (as M3b), `poet-fresh`
(minimal criterion kept, every admitted child gets a fresh agent — curriculum only), and
`random` (as M3b). Criteria stated first: **A** poet-fresh > random on ANNECS on ≥ 4/5 with
pooled ratio ≥ 1.25 → the minimal criterion alone produces the effect; **B** poet >
poet-fresh on ≥ 4/5 → inheritance adds to it. Four outcomes, all informative: A ∧ B (both
mechanisms), A ∧ ¬B (curriculum is the mechanism), ¬A ∧ B (inheritance is the mechanism —
M3b's PASS was competence transfer, not task generation), ¬A ∧ ¬B (M3b does not replicate).
C3 is dropped as a judged criterion (width-coverage confound) and reported width-matched
instead. Cost ≈ 4 h on 5 workers. Launch is the next step; nothing is tuned before it.

## 4. What this is not

- Not a language model, and not competitive with one on language. The DSL is closed by
  design; the reasoner stays outside the loop. Anything that needs open-domain text is out.
- Not "sub-millisecond NISQ control." 167 µs is a simulated trial on a laptop.
- Not a physics claim. No constants; the environment is either a documented simulation or a
  ledgered hardware job.
- Not new mechanisms. Every rung names its prior art. What is new, if anything, is the
  measurement discipline applied across all rungs and the joined provenance from a
  genome to a hardware job.

## 5. Immediate build: M1 — dnalang v0.2 regulatory genes

**Language:** extend the `.dna` genome with
`trigger:` (`on_genesis` | `after <gene>` | `when <metric> <op> <value>` | `on_error` |
`continuous`), `dependencies:` (gene ids whose outputs are inputs), `outputs:` (named
signals). A gene's body is either an LCS rule (`condition`/`action`) or a DSL program.
**Interpreter:** each tick, evaluate triggers against the organism's state and the signal
board; express active genes in dependency order; their outputs update the board.
**Compilation:** an expressed gene set *is* the L1 rule population for that tick — so
regulation changes *which rules exist* rather than only which fire.
**Evolution:** mutate triggers, wiring and bodies; crossover on clusters.
**Experiment (pre-registered before the interpreter is written):** GRN organism vs flat LCS
on the drifting hidden-mux family (Experiment One's harness), equal evaluations, seeds
100–104 to tune, 0–4 to judge, kill criterion in the table above.

**Cost:** three to four days. **Data it produces:** the first measurement of whether
dnalang's central unbuilt idea — regulation — does anything at all.

## 6. Scorecard (generated from `results/`)

<!-- scorecard:start -->
| experiment | verdict | number |
|---|---|---|
| signalling (50 seeds) | quantified | 9/50 full, 36/50 partial, 5/50 pooling |
| concept drift 6-mux | FAIL | ratio 0.8 |
| relay dead (threshold) | PASS | median 100, C1 1.00, false/10k 0.00 |
| relay dead (cusum) | PASS | median 200, C1 1.00, false/10k 0.25 |
| relay poisoned (threshold) | FAIL | median 200, C1 0.67, false/10k 0.00 |
| relay poisoned (cusum) | FAIL | median 200, C1 0.80, false/10k 0.08 |
| exp zero: informed priors | FAIL | informed 1000 vs control 900 |
| exp one: cusum injection | PASS | cusum 950 vs plain 1250 |
| m1: regulatory genes | FAIL | grn-evolved 400 vs cusum 950 |
| m2: priors, 16-bit family | FAIL | informed 9500 vs shape 8250 |
| m2b: specificity prior, 16-bit | FAIL | shape 9500 vs covering 9750 vs periodic 10500 |
| m2c: specificity prior replication, 16-bit | FAIL | shape 10750 vs covering 9500 vs periodic 11750; C1 1/5 C2 3/5 p=0.856/0.063 |
| m3: open-ended tasks, 6-10 bit | FAIL | poet 46 vs random 52 ANNECS |
| substrate-opt-1: XCS grid, 16-bit | BOUNDED | winner tga25_ason_mu0.05_pw0.75 11500 vs default 10000 (winner<default 2/5) |
| substrate-opt-2: tournament/specify/N, 16-bit | PASS | winner N4000_selectiontournament_specifyon 6000 vs default 8750 (winner<default 5/5) |
| m3b: open-ended tasks, 12-16 bit, PARAMS16 | PASS | poet 35 vs random 25 ANNECS, ratio 1.40, C1 4/5 C3 5/5 |
| bridge step 3 | PASS (tie) | organism 0.9763 / GA 0.9758 / baseline 0.9705 |
| bridge tier 4 shock | FAIL | organism-structural 14, organism-plain 11, ga-continued 30, ga-restarted 38 |
<!-- scorecard:end -->

The program's empirical boundary, in one line: **the substrate delivers topological
self-repair and small-space diversity injection; nothing above the learner, and none of the
learner's exposed parameters, changes wide-space sample efficiency.** The remaining rungs
that depend on wide-space learning (M3 at width, M7) are blocked until a different base
learner is pre-registered; M4, M5 and M6 do not depend on it.
