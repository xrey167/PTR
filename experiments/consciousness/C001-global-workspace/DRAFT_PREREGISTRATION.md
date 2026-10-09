# C001 global workspace: preregistration (draft v2; renamed to PREREGISTRATION.md at the freeze)

Status: draft. Version 1 of this experiment (six position-specific modules, held-out position pairs, online data,
five declared seeds 17/29/43/71/101, a lower-bound-only decision) was withdrawn before any freeze after review: its
outcome was fixed by a ceiling (every comparator was at 95-97% accuracy, the margin equalled the whole headroom) and
the position-pair split was not out of distribution. The one dry run of version 1 (seed 17, scratchpad
`c001-dry17.json`) belongs to that superseded design and is disclosed here; its seed is no longer declared. Nothing
in this version has been run on a declared seed (3001-3010). The pinned values are `[pinned_thresholds]` in
`config.toml`; `criteria.toml` holds the rules; this file states the reasons and the limits.

## 1. Scope and non-claims

C001 is a functional-mechanism test on a CPU toy task of whether a **learned-attention** global workspace (the
mechanism of Goyal et al., ICLR 2022: a few slots, competitive write, gated slot update, broadcast read) has an advantage
as an inductive bias. **It is not a test of consciousness and no result is evidence about consciousness.** No
self-report is produced, trained or measured.

**Relation to `ptr-pods::workspace`.** The workspace implemented in `crates/ptr-pods/src/workspace/` (package A) is a
deterministic contract: bounded slots, proposals that pass the verifier gate compete by bid and then by a SHA-256
tie-break, snapshots are broadcast to bounded mailboxes, findings are broadcast as refusals. It learns nothing. C001 tests a
learned analogue (attention write, GRU slot update, attention read). A supportive C001 does not validate that contract and a
refuting C001 does not touch it; the contract's evidence is its own tests and C006 on Pods. C001 informs one design
question only: whether a *learned* write and read are worth building on top of the contract (GW-L1/GW-L2, doc 38).

## 2. Hypothesis and the claim being tested

Goyal et al. claim an inductive-bias advantage (sample efficiency, generalisation, linear cost in modules), not an
expressivity advantage: an all-to-all attention can express everything the workspace can. So the test is at an
operating point where inductive bias can matter: a finite training set and more modules at test time than at training.

H1. With a finite training set of eight-module instances, the workspace arm is more accurate than each of the
all-to-all, the bandwidth-matched all-to-all and the no-broadcast arms on instances of **sixteen** modules (more
distractors than ever seen in training), by at least the pinned margin (50 per mille), shown by the lower one-sided 95% t
bound over ten seeds.

## 3. Task, arms, training

* N_train = 8 sensory modules **sharing one encoder and one update network** (plus a reporter module with no input);
  there is no position information, so the number of modules is free at test time. Each module sees a noisy observation
  (type one-hot over 4 types, value one-hot over 3 values, cue flag, query one-hot, Gaussian noise 0.3). One module is the
  cue: it carries the query type q. Exactly two other modules have type q, the others have another type. The answer is the
  sum of the two values of type q (5 classes; the majority class has probability 1/3). A single module's value does not
  determine it, and which modules matter is known only after the cue's content has reached the others.
* Evaluation sets (2048 fresh instances each): in distribution (8 modules, a secondary check), **primary: 16 modules**, secondary:
  24 modules (same rule, more distractors; a real change of the input size, unlike a held-out pair of positions).
* Data: a finite training set of 4096 eight-module examples generated once per seed, cycled; every arm trains on the same
  set in the same order (paired). 3000 Adam steps, batch 64, gradient clip 1; the same step count for every arm; no early stopping or
  checkpoint selection.
* Arms (parameter counts within 2% of the workspace arm by choosing the update-network width; communication-cost proxies
  and message width are reported):
  * workspace: 3 slots, per-slot top-3 competitive write, GRU slot update, broadcast read by every module;
  * all_to_all: single-head pairwise attention among all modules;
  * **bw_all_to_all**: pairwise attention whose value width is reduced so that the values stored (N x d_value) equal the
    workspace's slot content (3 x 32), isolating "shared store" from "pairwise" at matched bandwidth;
  * no_broadcast: the same workspace, but only the reporter reads it back;
  * no_comm: no communication (a floor);
  * **competition_off**: the workspace with soft write over all modules (no top-k); **capacity_off**: as many slots as
    training modules (8); both are ablations of the claimed ingredients, reported with bounds, not judged;
  * **crippled_all_to_all**: all-to-all with one round instead of three (the cue cannot reach the selection in time): the
    **sensitivity control**, a comparator crippled in a known direction.
* Module dropout, ignition-style curves, learning curves (area under them = sample efficiency) and the communication
  cost per module count are descriptive.

## 4. Decision rule (frozen in criteria.toml) and what each outcome means

Per seed: **structural** coverage rules and ten hard counters (parameter match, no communication leak in the isolated arms,
write attention with at most k sources, dropped modules cannot influence an output, no out-of-distribution module count in
any training batch actually drawn, no evaluation instance identical to a training instance, determinism, finite losses, a
no-communication arm that does not beat the majority class). A failing rule is a defect of the product arm, the task or the harness and
FALSIFIES (kind structural) -- a broken workspace falsifies, it does not turn into a harness error: the negative controls
run on doctored inputs or on the all-to-all net, never on the arm under test. Six controls must trip or the run is exit 2.

**Validity guards** (the aggregator, not the harness): the better of the all-to-all and the bandwidth-matched arm on the
primary metric must lie in [300, 900] per mille on every seed, else the run is saturated or floored and the experiment is
**INVALID**, never FALSIFIED; the workspace arm must have trained (ID accuracy at least 100 per mille above the
no-communication arm) on at least 8 of 10 seeds; and the sensitivity control must be resolved: the lower bound of
(all_to_all minus crippled_all_to_all) must reach 50 per mille, else INVALID (insensitive).

Across the ten seeds, with the one-sided 95% t quantile for 9 degrees of freedom (1.833113), bounds of the mean paired
difference (workspace minus comparator) on the primary metric, against all_to_all, bw_all_to_all and no_broadcast:

| Outcome | Condition | Consequence |
|---|---|---|
| COMPLETED (supported) | guards hold, every lower bound >= 50 per mille | Build the learned L2 prototype; run C006 on Pods |
| FALSIFIED (refuted) | guards hold and some UPPER bound < 50 per mille | Record a negative result for this setting; downgrade the learned-workspace requirements GW-L1 and GW-L2 to exploratory |
| NOT_SHOWN | guards hold, bounds straddle the margin | Nothing is concluded; status stays planned; a larger successor may follow; no downgrade |
| INVALID | saturated, floored, insensitive, workspace untrained on more than 2 seeds, seeds or mutation evidence missing | Nothing is concluded; repair and register a successor |

Exit status of `aggregate.py`: 0, 1, 3, 2 respectively (2 also for any refusal to aggregate).

**Patterns, stated in advance.** *Supports:* the guards hold, the workspace beats all three comparators by the margin on 16
modules; (reported, not judged) the advantage is at least as large on 24 modules as on 16, competition_off and capacity_off are
worse than the workspace, the slot count behaves like a bottleneck. *Refutes (for this setting):* the guards hold, the sensitivity
control resolves, and the workspace is not better than a pairwise comparator by the margin with power; competition_off and
capacity_off are not worse (so neither bottleneck nor competition adds anything). *Partial (NOT_SHOWN, reported):* the
workspace beats no_broadcast but not the pairwise arms: it supports "communication/broadcast matters" (GWT-3), not "the
bottleneck helps" (GWT-2). *Uninformative (INVALID):* saturation, floor, an insensitive control, an untrained workspace.

## 5. Mutation evidence

`tests/mutations.toml` plants 17 defects (model, harness data handling, and checks blinded), each naming the counter or control
it must trip; every hard counter and every negative control is expected by at least one; defects of the workspace arm must
reach a counter (exit 1), not a control (exit 2). They are planted in a throwaway copy by `research/consciousness/gw/run_mutations.py`
(`scripts/mutation_check.py` cannot plant a defect in a Python entrypoint; `--check` verifies the anchors). The statistics and the
four outcomes are exercised by `scripts/tests/test_c001_aggregate.py`, which pins the bounds to hand-computed values.
`aggregate.py` refuses unless `results/mutations.json` counts as many mutations as the plan lists, all killed, and
`research/consciousness` and this directory are identical to the mutation commit.

## 6. Pilot disclosure and tuning (seeds 1001, 1002, 1003 only; none is declared)

Equal tuning budget: every arm got the same two-point learning-rate grid {0.001, 0.003} on pilot seeds 1001 and 1002, the
rate with the higher mean in-distribution accuracy was pinned for that arm (not out-of-distribution accuracy, the primary
endpoint). Nothing else was tuned per arm. The workspace's structural choices (3 slots, top-3 write, 3 rounds, d=32) were
fixed from earlier pilots of version 1 (below) and not re-tuned; the comparators' structure (single head, d=32, 3 rounds) is
the same module bank. The training-set size and the task difficulty were chosen on the pilots: the smallest of {1024, 2048,
4096, 8192} at which the strongest comparator's accuracy on the primary metric lies between 400 and 800 per mille.

History, so that the reader sees how the operating point was found:
1. Version 1 (position-specific modules, online data, six modules): nothing was learned in some variants; after two task
   reductions every comparator was at 95-97% and the workspace at 96%; the outcome was fixed by the ceiling. Withdrawn.
2. Version 2, first operating point (6 types, noise 0.5, shared weights, finite data): at 512, 1024, 2048, 8192 and 32768
   training examples neither the workspace nor the all-to-all arm generalised beyond about 30-39% on the pilot seed (the
   task was too hard for 3000 steps); on 6 types the workspace learned more slowly. Task reduced to 4 types and noise 0.3.
3. Pinned operating point (4 types, noise 0.3, 4096 examples), pilot seed 1001, lr 0.003 unless stated, accuracy per mille
   in distribution / 16 modules / 24 modules: workspace 880/845/824, all_to_all 867/744/638 (lr 0.001: 860/798/705),
   bw_all_to_all 865/814/718, no_broadcast 876/835/810, competition_off 891/808/736, no_comm 358/331/349, crippled
   all-to-all 371/317/308. At 1024 and 2048 examples the workspace reached 467/412/395 and 792/734/683.
   **The workspace was found better than the all-to-all arms out of distribution in this pilot, and the no-broadcast arm
   about as good: the operating point was chosen to show headroom, with the workspace's pilot behaviour visible, which
   favours H1; the pilot seeds are not evidence for either direction and are not part of the result.**
4. Tuning table and the dry run (pilot seed 1003): TBD-PILOT

## 7. Limits

See `aggregate.py` LIMITATIONS: one task family and size; ten seeds and a t bound (normality unverifiable); the ablations are
definitions; the learned analogue is not the deterministic contract; sample efficiency, dropout, 24-module accuracy and ignition
are unjudged; the mutation tooling gap; and the standing sentence that this experiment says nothing about whether any system
is conscious. An equivalence reading ("same accuracy at lower communication cost at 8 modules and large data") was considered
and is not part of this preregistration; it would need arms at a larger training set.

## 8. Entry point, environment and freeze steps (for the head)

The entrypoint is `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads
--project research/consciousness python -B research/consciousness/gw/harness.py --benchmark c001 --seed <seed>`. Reasons, each found by
rehearsing the path through `scripts/run_experiment.py` in a scratch clone (`c001-report.md`):

* The runner passes only PATH, HOME, TMPDIR/TEMP/TMP, locale and the cargo/rustup homes (plus PYTHONNOUSERSITE=1 and PYTHONHASHSEED=0): no
  proxy, certificate or `UV_*` variable. So `--offline` (nothing is downloaded at run time) and `--no-python-downloads`.
* The runner **refuses to run while the repository holds git-ignored files** (`.venv/`, `__pycache__/`). A project-local `.venv` would
  therefore stop every run, so the environment lives outside the repository (`UV_PROJECT_ENVIRONMENT`, resolved against
  `research/consciousness`: the directory `c001-venv` next to the repository checkout), bytecode is not written (`-B`,
  `PYTHONDONTWRITEBYTECODE`), and the freeze must be run from a tree without `__pycache__` directories (the freeze scripts of F003 do
  `find . -name __pycache__ -prune -exec rm -rf {} +`).
* The locked environment holds **torch only** (no numpy); the harness is written for it (a numpy call hid in the development environment
  and was found by the rehearsal).

Before the first run, with the network variables set: `UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project
research/consciousness` as the same user and HOME, so that the environment and the uv cache are complete; then the smoke check
`env -i PATH=... HOME=... UV_PROJECT_ENVIRONMENT=../../../c001-venv uv run --locked --offline --no-python-downloads --project
research/consciousness python -c "import torch"` must pass. The harness checks its environment (python 3.11, torch 2.7.1, the config file)
before training and exits 2 with a message otherwise, so an incomplete environment cannot be mistaken for a falsification (a Python crash
would exit 1). `uv` and CPython 3.11 must be on PATH (absolute entries). A listed seed runs once.

1. Rename this file to `PREREGISTRATION.md`; pin the two sha256 placeholders of the `[preregistration]` table of `config.toml`;
   set the manifest digests, status `prepared` (manifest and registry); take C001 out of `LISTED_UNFROZEN` in
   `scripts/tests/test_research_gates.py` as F003 was.
2. Run `python research/consciousness/gw/run_mutations.py` from the clean committed tree (writes `results/mutations.json`; tens of minutes
   on a loaded machine), commit it, then `prepare` and `run` seeds 3001-3010 once each, then `aggregate.py`.
