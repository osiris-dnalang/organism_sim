You are improving the learning operators of an XCS learning classifier system. A fixed
program will run your version and measure it. Your job is to propose one concrete change at a
time that should make the learner adapt faster.

## The learner

A population of at most N rules. Each rule has a ternary `condition` over the input bits
('0', '1' or '#' = don't care), an `action` (a string, one of `engine.actions`), and learned
statistics: `prediction` (expected reward, 0..1), `error` (0..1), `fitness` (relative
accuracy, 0..1), `experience`, `numerosity` (copies), `as_size`, `time_stamp`, `id`.

Each trial: the rules matching the input form the match set; an action is chosen; the rules
advocating it form the action set `aset`; reward 1 (correct) or 0 updates their statistics.
Fixed parts you cannot change: matching, action choice, the statistics update, subsumption,
the GA period, N.

## The task

Inputs are 16 random bits. A hidden instance picks 3 address positions and 8 data positions
(the other 5 bits are irrelevant); the address bits select which data bit is the answer,
possibly inverted. Every 30,000 trials the hidden instance changes. The score is how many
trials the learner needs, after each change, to reach 95% accuracy on a fixed probe set.
Lower is better. The operators never see the right answer; only rule statistics carry it.

## Current parameter values (engine.p; fixed, you cannot change them)

{{PARAMS}}

Branches that depend on these values and are not taken (for example the roulette branch
while selection is 'tournament') have no effect; change code that actually runs.

## The six operators you may change (signatures are fixed)

{{SIGNATURES}}

- `cover_condition`: a condition for a new rule that MUST match `register` (every
  position is the register's bit or '#').
- `select_parents`: two rules taken from `aset`.
- `crossover`: returns (condition1, condition2, action1, action2); conditions are strings of
  the same length as the register over '0', '1', '#'; actions from `engine.actions`.
- `mutate`: returns (condition, action) for one offspring.
- `child_estimates`: returns (prediction, error, fitness), each in [0, 1].
- `deletion_votes`: one non-negative number per rule in `engine.rules`, same order, positive
  sum; when the population exceeds N one rule copy is removed with probability proportional
  to its vote.

## Rules (a checker enforces them; a violating change is rejected)

- No import statements. `np` and `math` are provided with only these members:
  np: {{NUMPY}}
  math: {{MATH}}
- Randomness only through `engine.rng` (random, integers, choice, permutation, normal,
  uniform, ...). Never use any other source of randomness.
- No names or attributes that start with an underscore. Do not assign to attributes and do
  not modify `engine.rules`, `aset` or any rule: return values instead.
- Attributes, by object (nothing else exists on these objects):
{{GROUPS}}
  Other allowed method names (on lists, strings, numpy arrays): {{OTHER_METHODS}}
- Only these builtins exist: {{BUILTINS}}
- Keep it fast: the learner runs about 120,000 trials; a version that uses more than twice
  the current CPU time is rejected. `deletion_votes` runs over the whole population often.
- Top level of the program: only function definitions. Keep the `# EVOLVE-BLOCK-START` and
  `# EVOLVE-BLOCK-END` comment lines.

## Output format

First, the idea in one or two sentences. Then the change, in ONE of these two forms:

1. SEARCH/REPLACE blocks; the SEARCH text must match the current program exactly:

<<<<<<< SEARCH
exact lines from the current program
=======
the replacement lines
>>>>>>> REPLACE

2. The complete new definition of ONLY the function(s) you change (usually one), in a single
   ```python block. Do not repeat functions you do not change; they stay as they are.
