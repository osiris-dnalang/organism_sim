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
| M3 | **Open-ended task generation** — the environment is a population too: instances mutate, are kept if "just solvable" by some agent | POET (Wang et al. 2019); minimal-criterion coevolution (Brant & Stanley 2017) | ANNECS: count of tasks solved by later agents that no earlier agent solved, vs a fixed random task stream | **FAIL at 6–10 bits (ratio 0.885, 1/5):** no hard tasks to reach, so the generator is overhead. Blocked on learner sample efficiency at wide inputs |
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
