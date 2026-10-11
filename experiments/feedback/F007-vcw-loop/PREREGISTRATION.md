# F007 vcw-loop: preregistration

Design: `docs/architecture/39-vcw-moe.md` (sections 3.3, 6.1). Status of this file: final text for the owner to sign;
the `must-be-signed-...` values of `config.toml [preregistration]` are the owner's to replace by the numbers stated
in section 6 (or by others, which must then be derived as section 6 says). Nothing here has been run on a
confirmatory seed.

## 1. What is tested

Whether the **verifier-closed loop with finding-routed repair** (`ptr_runtime::vcw::VcwLoop` with the finding
router, the learned router promoted through the off-policy gate, the workspace, the self-model and attention
schema, and L3 through the ledger) verifies more requests, at a **matched verifier-call budget**, than (2) a trained
MoE-style softmax gate over the same Pods, (3) best-of-N with the same verifier, on seeded simulated expert Pods.
It is a **mechanism check**: the Pods are simulations with declared skill tables, a gain is a consequence of those
tables, and nothing is claimed about any model.

The wording of a COMPLETED verdict is fixed here: *a verifier-closed loop with finding-routed repair beats best-of-N
at equal verifier budget, and beats the task-feature gate and the shuffled finding map, in a world whose Pod skills
are declared.* It is **not** "finding routing beats best-of-N": best-of-N has no repair Pod at all, so the contrast
bundles "a repair Pod exists" with "repair is routed" (in the null world, where nothing is specialised, the full loop
still beats best-of-N by about ten points; the aggregator reports that row). **The routing claim is the contrast to
the gate, to the untrained generic pool (arm 7 against arm 5) and to the shuffled map.**

## 2. The assumption under test

`world.rs` declares, per (Pod, finding code), the probability that a repair call clears the code. In the primary
world `hetero`: nine repair Pods, two specialists for each of the roles Format, Value, Schema, Code (the roles of the
real `ptr-verifier` code table) and one generalist; a specialist clears the codes of its role with probability 0.80
+- 0.08 (a keyed offset per Pod and code), the generalist every code with 0.60, a specialist of another role 0.30;
nobody can repair `code.timeout` at the start (0.00) and no Pod is registered for it (the capability gap L3 can
fill; a Birth is trained at skill 0.80); three proposers, each with 0.6x the defect rates on its own task kind and
1.3x on the others. **Under this assumption a loop that routes by the verifier's finding code has information a
task-feature gate cannot have.** That is the assumption, not a result. Three more worlds show how much of any gain
is the assumption:

| world | role | what differs |
|---|---|---|
| `hetero` | **primary, gates the verdict** | the table above |
| `homog` | null world, gates validity | every Pod 0.60 at every code, identical proposers, no capability gap: finding routing has nothing to exploit and must **tie** |
| `mild` | descriptive | on-role 0.72 +- 0.04, generalist 0.62, off-role 0.40 |
| `ctx` | declared mechanism world of the workspace secondary | as `hetero`, and a Pod that reads the workspace broadcast fixes half of its failed repairs |

The difficulty (defect rates arithmetic 200/450/100 permille, JSON 200/350/350/250, code 250/450/250; a budget of 3
verifier calls) was set so that no arm sits near the ceiling or floor in the planning model `power.py`.

## 3. Arms, at the same requests and the same verifier-call budget (3 calls, 3 rounds)

All are the same `VcwLoop`, configured differently (`arms.rs`); the same draws are used for the same (request, Pod,
code, occurrence), so arms are paired.

| # | arm | registry | scorer | self-model | broadcast | L3 |
|---|---|---|---|---|---|---|
| 1 | full | true role map | learned router, promoted through `PromotionGate` | hints steer (bounded boost) | yes | live |
| 2 | gate | none: every Pod for every code | softmax gate on (step, Pod), no finding code | no | yes | no |
| 3 | best-of-N | no repair Pod: a failure is regenerated | the proposers' own tuned gate | no | yes | no |
| 4 | no workspace | as 1 | as 1 | as 1 | **withheld from Pods** | live |
| 5 | no finding routing | every Pod for every code | uniform | no | yes | no (cannot trigger) |
| 6 | no self-model | as 1 | as 1 | **no hints, no schema** | yes | live |
| 7 | no L3 | as 1 | as 1 | as 1 | yes | **off** |
| 8 | shuffled map (sensitivity control) | each code to the Pods of the wrong role | as 1 (not retrained) | as 1 | yes | live |
| 9 | oracle (upper reference, descriptive) | each code to the one Pod that is best at it, and the good proposer only | uniform | no | yes | no |

**Round 0 is shared.** The first draft of a request is proposed by the kind's specialised proposer, whatever the arm,
and its defects are keyed by (seed, request id, Pod, code) inside the simulated Pod. Neither an arm's scorer nor the
router's nonce sequence (which drifts once L3 changes the number of draws) can reach it, so every arm starts every
request from the same candidate (`first_candidates_differ` is exactly 0, control `control_first_draft`). The price: the
experiment no longer tests whether a learned router picks the right *proposer* for round 0 (earlier pilots mixed that
into every arm's learned part); it tests finding routing of repairs, and the proposer choice of later regenerations. The
training streams route round 0 freely (the learned parts are fitted on traffic in which the proposer choice shows) and
the evaluation streams share it. This makes the untrained and tuned baselines no weaker than before and the full loop
no stronger; the planning model, the derived thresholds and the pilots were re-done on it (section 6 chronology, round 4).

Choices: the workspace ablation removes the broadcast to Pods (the loop always runs its workspace as gate-keeper); the
self-model acts through advisory hints applied to the attention schema (`ControlArm::HintOnly`, boost at most 2x);
removing finding routing (5) also removes the unroutable-code statistic L3 triggers on, so the clean contrast for
routing alone is **7 against 5**; the baselines have no L3 (the gap is a capability only the full loop can add; the
robustness line reports arm 7 against the baselines); best-of-N is the same loop with no repair Pod; the gate is a
softmax over all Pods conditioned on the kind (one gate per lane) and the step, never on the finding code. **L3 live**
is Birth only (no world gives the other operations a defined benefit), through `StructureDriver` ->
`admit_expert_proposal`. **Non-goal: a model-backed (Qwen) arm.** In package VE the CPU Qwen Pods repaired 0 of 40 in
stage (a) and ran at 2 to 6 tokens per second: such an arm would be a floor, not a comparison; the extension point is
`RolePods`.

## 4. Tasks and verifiers

Arithmetic (`ArithmeticVerifier`), a JSON object under a schema (`JsonSchemaVerifier`), a small program with 4 tests
(`CodeTestVerifier`), mixed 40/35/25 per mille; ten finding codes. A candidate is the clean answer with planted
defects rendered to text; the **real** verifier decides what it reports. The oracle is independent: the planted set and
the clean text, and a fresh verifier instance. Task kinds are separate lanes (learners and L3 per kind).

## 5. Metric and pairing

Primary: **verified success per request** (the loop returned a candidate that the oracle and a fresh verifier both
pass), paired over requests within a seed, 4000 requests per world and seed, seeds 17, 29, 43, 71, 101. Also reported:
verifier calls, Pod calls, latency per request per arm. The unit of replication is the **seed**: per-seed paired
difference, mean over seeds, one-sided 95% t bound (df 4). The learned arms are retrained for each run seed on
pilot-domain streams (seeds >= 1001 derived from the run seed, never a declared seed), so the interval includes
training variance.

## 6. Decision rule, margins and their derivation

**Primary margin m = 3 points** against **each** baseline. Rationale (planning model only): the architecture adds a
second learned component, a workspace, a self-model and a ledger-backed structure path; a gain smaller than what
half an extra verifier call buys best-of-N (about 6 points per call in the planning model) is not worth that; at a
baseline of 0.87, 3 points is a 23% cut in unverified requests. It is a judgement; the owner may move it.

* COMPLETED: lower bound of (full - baseline) above m for **every** baseline, with every control valid.
* FALSIFIED: upper bound below m for **any** baseline; or an integrity falsifier tripped (a product defect).
* NOT_SHOWN: otherwise.
* INVALID: a control or coverage rule fails, a seed is missing or unreadable, a result lacks a pair table the analysis
  reads, or the mutation check does not kill every mutation. **INVALID takes precedence over FALSIFIED**: a product
  defect seen in a run that cannot be trusted is a note, not a verdict.

**Chronology, disclosed exactly.** Nothing is committed, so file times prove nothing about the order (files were
rewritten in later rounds) and are not relied on.
1. *Before the first pilots (round 1):* the primary margin (3 points), the null band (1.5), the shuffled bar (5), the
   difficulty (defect rates, budget 3) and the planning model were written; the rationale of the margin refers to the
   planning model only. Round-1 pilots (seeds 1007 to 1009, gate temperature grid {50..400}) were then run and **seen by the
   reviewer and the owner**: full - gate +7.75, the gate-over-pool gap +24, best-of-N over single shot +36, workspace in
   `ctx` +2.3, L3 +0.15, `mild` full - best-of-N +3.7 (close to the margin).
2. *After seeing them (round 2):* everything else in this section was written: the controls against crippled baselines
   (gate over the untrained pool 20 -> 120 permille in round 2, 43 in round 4; best-of-N over single shot, the oracle reference), the
   secondary hypotheses, the agreement rule and the saturation band, the wider tuning grids and the edge rule. The
   **arithmetic** of each derived threshold is pilot-free (`power.py derive` reads `config.toml` and the model only), but
   **the choice of "half of the modelled gain" as the rule, and of adding these controls, was a judgement made after
   seeing the +24 and +36 gaps**. The L3 power figure uses a between-seed sd of 7.5 points **measured in a pilot**.
   The primary margin, the band and the shuffled bar were not changed by it.
3. *Round 3:* the edge rule was replaced and the grids widened again (section 8), the source digest, the first-candidate
   check and the mechanism control for the broadcast were added, and the pilots were re-run on the resulting code.
4. *Round 4:* the round-3 pilots showed the first candidate of the full and no-L3 arms differing for about 15% of the
   L3 population (router nonce drift); a drift tolerance of 30% had been set after seeing that 15%. **That tolerance is
   removed**: the first draft is now shared by construction (section 3) and its identity across arms is an exact
   control. Because the planning model had to follow (round 0 is the specialised proposer in every arm), every derived
   threshold was **re-derived by the same pilot-free rule before the new pilots were run**: gate over the untrained pool
   120 -> 43 permille, best-of-N over single shot 181 -> 176, workspace secondary 13 -> 9, L3 secondary 31 -> 54 (the
   untrained pool gets the good first draft, which shrinks the modelled gate gap). The grid-edge rule gained a
   sensitivity check and a logged paired SE (section 8), `Cargo.toml` and `Cargo.lock` joined the source digest, and the
   pilots were re-run with the release profile of the entrypoint.
The owner has seen all pilots before signing. The owner states at signing whether the margin is kept or moved and
why; any margin moved must be derived from `power.py` without pilot numbers. The tamper-evident record is the commit of
the freeze followed by a re-run of pilot 1007 on the committed tree (section 12).

**Thresholds derived from the planning model alone** (`python3 experiments/feedback/F007-vcw-loop/power.py derive`,
n = 20000, seed 11; each is half the modelled gain; a test pins them to `criteria.toml`):

| threshold | modelled gain | threshold |
|---|---|---|
| gate beats the untrained pool (control) | 8.5 points | 4.3 points (43 permille) |
| best-of-N beats its own single shot (control) | 35.1 points | 17.6 points (176 permille) |
| workspace secondary: full - no workspace, world `ctx` | 1.9 points | 0.9 points (9 permille) |
| L3 secondary: full - no L3 on requests whose first candidate carried the unrepairable code | 10.8 points | 5.4 points (54 permille) |

Other controls (**judgements, not derived**: the band, the oracle tolerance 2 points, the shuffled bar 5, the
agreement 2 points, the ceiling 0.985 and the floor 0.50; their reasons are in `criteria.toml`): null-world equivalence band +-1.5 points (half the margin; both one-sided 95% bounds of each of the
no-L3 loop and the full loop minus the gate, the generic pool and the shuffled map inside it); the full loop beats the
shuffled map by at least 5 points (margin plus 2; the only one of these not derived from the model); the oracle
reference is not below the full loop by more than 2 points (lower bound; the oracle has no L3); **the harness agrees
with the planning model within 2 points** on full, gate and best-of-N in `hetero` (predictions 0.962, 0.883, 0.891)
**or the run is INVALID**; no compared arm outside 0.50 to 0.985. Per-seed harness restatements at looser point
thresholds: shuffled 5, null 3, baseline 4.3, best-of-N 17.6, oracle -2 points.

**Secondary hypotheses** (reported with their own SUPPORTED / NOT_SHOWN / REFUTED line, never gating the verdict):
(1) *workspace*: full - no workspace in `ctx`, margin 0.9 points; this tests a **harness-defined reader** of the
broadcast (an implementation check of the broadcast path), not the value of the workspace, which needs Pods that read
it (F004, C006). (2) *L3*: full - no L3 restricted to the requests whose first candidate carried `code.timeout` as a hard
finding (the population L3 acts on: the first draft is shared by every arm, so it is fixed before any arm acts and is read from the no-L3 arm; its first-draft identity across arms is a control, section 7), margin 5.4 points;
about 120 requests per seed, so its power is low (a NOT_SHOWN there is not evidence against L3). **Self-model:
untested in F007** (no discriminating condition in this design); the arm runs. In the primary world the workspace,
self-model and L3 contrasts are labelled **non-discriminating** and never "no effect": the workspace tie is by
construction, the self-model and L3 cannot reach a component margin there (the L3 ceiling is about 0.5 to 1 point of
all requests).

## 7. Integrity falsifiers (any > 0 on a valid seed: FALSIFIED), from observed events

`false_accepts`, `unverified_returns`, `double_verifications`, `budget_violations`, `routing_outside_admissible`,
`verdict_mismatches`, `verdict_overrides` (a learned score cannot override a verifier verdict),
`verifier_call_mismatches`, `calls_after_acceptance`, `l3_unadmitted`, `loop_errors` (also in the training streams),
`product_panics` (caught per request), `exhausted_as_success`. Per-seed controls (INVALID): `control_shuffled`,
`control_null`, `control_baseline`, `control_workspace`, `control_ctx` (a mechanism check of the broadcast path that does not read any success rate, so that it is not the quantity of the workspace secondary: in a seeded probe a Pod reads the broadcast when it is delivered and never when it is withheld, delivering it changes the candidates produced, and in the `ctx` world the full arm reads it and the no-workspace arm never does), `control_training` (any learned part of
the primary world not promoted: no silent fall-back to uniform routing; `hetero` only), `control_grid_edge` (an edge-chosen tuned value flagged by the paired test, or the test blind to a materially worse policy; `hetero` only), `control_oracle`,
`control_bon`, `control_l3` (the L3 ablation is wired), `control_first_draft` (the first draft is the same candidate in every arm on every request: the count of (arm, request) pairs whose first candidate carries other planted defects than the full arm's is exactly 0 in every world; **the condition that invalidates the run**: any count above 0 means arm state reached round 0, a harness defect that touches every paired contrast and the L3 population alike. It is never merely an annotation of the secondary, because the primary pairing relies on the same property). Coverage (INVALID), each with its own counter: every kind at
least 20% of requests, every code observed 10 times, 200 repairs, 20 exhausted runs of the full arm, one Birth. Exit
codes: product defects and failed controls exit 1, harness faults exit 2.

## 8. Tuning

Both learned arms and best-of-N's proposers see identical budgets: three pilot-domain training streams per lane under
uniform routing (A 1500 statistics, B 1000 tuning, C 1500 promotion), **nine** candidates each (Beta prior in
{1,2,4,8,16,32,64,128,256}; softmax temperature in {1,2,5,10,20,50,100,200,400} permille, for the gate and, separately, for
best-of-N's proposers), chosen on stream B by the doubly robust estimate, promoted only if the lower 95% bound of the
DR difference to uniform exceeds 2 points on stream C (VCW: the real `PromotionGate`; the others: `compare_policies`).
**Grid-edge rule.** A tuned parameter chosen at an edge of its grid is a grid that was too narrow when it beats its
inner neighbour on stream B with a lower bound, at 3 standard errors of their **paired** difference (both are
evaluated on the same stream B, so the pair's own noise is what matters, and it shrinks as the two become the same
policy), above zero: `control_grid_edge`, INVALID. The multiplier **z = 3 is a judgement, fixed after the round-2
episode** (a one-sided 0.13% level, so that the nine tests per seed rarely flag noise); the rule has no tolerance on the
value itself, but it does have this constant. **What can escape it:** a real edge gain smaller than 3 paired standard
errors per grid step, or a slow monotone trend of sub-3-SE steps; the largest raw edge-over-inner gap in the round-4
pilots is 0.08 points (766 micro) against a margin of 3, so the bound that matters for the verdict is that the baselines are
not under-tuned by more than a fraction of a point. The paired standard error of every edge decision is logged in the
result (`*_edge_se_micro`) so that the bound is auditable. **The test is checked for blindness**: the tuned gate (and
best-of-N's proposers) are also compared with the worst candidate of the grid, which the test must flag
(`*_edge_test_sees`); a blind test counts in `control_grid_edge` exactly like a flagged edge, and two mutations (the
lower bound required above 10^12; the gain required to exceed 100 paired standard errors) are killed through it. The rule and the
sensitivity check run in the primary world `hetero` only; in `homog` the tuned parts are not edge-checked, which can only
make the null tie harder to pass (conservative). Grids: Beta prior {1,2,4,8,16,32,64,128,256},
softmax temperature {1,2,5,10,20,50,100,200,400} permille, nine candidates each, for the learned router, the gate and
best-of-N's proposers. **History:** round 1 tuned the gate at the edge of {50..400}; round 2 widened to
{5..400} and used a fixed tolerance of 0.001 on the DR value, which fired on pilot 1008 (prior 64, tau 5 drifting by
0.002); I then raised it to 0.005 with an argument that read the standard error off a pilot, a **post-hoc judgement
that loosened a control on the baseline's tuning**; round 3 replaced it by the paired-noise rule above and widened the
grids instead of loosening the rule. Round 4 kept z = 3, labelled it a judgement, logged the paired SE, and added the blindness check. The bounded consequence of the grid for the verdict: widening the gate grid
from {50..400} to {5..400} moved full - gate by 0.3 points against a margin of 3. Each arm trains on traffic routed inside its own registry; no confirmatory seed runs
before the freeze; the training seeds are `1001 + 1000 * (seed + 1) + {0,1,2}`.

## 9. Power

`power.py` is an independent Python model of the arms (planning, not evidence). In `hetero` it predicts full 0.962,
gate 0.883, best-of-N 0.891 (gains of 7.8 and 7.1 points). Power of "lower one-sided 95% bound above the margin" (`power.py power`, t interval over 5 seeds, between-seed
sd and within-seed sd as measured in the pilots, max 0.9 points):

| contrast | margin | modelled true gain | P(lower bound > margin) at the modelled gain | gain at which power is 0.8 |
|---|---|---|---|---|
| primary: full - gate, full - best-of-N | 3.0 | 7.8 / 7.1 | 1.00 | about 4.3 (margin + 1.3) |
| workspace secondary (world ctx) | 0.9 | 1.9 | 0.99 | about 1.7 |
| L3 secondary (about 120 requests per seed; between-seed sd 7.5 **measured in a pilot**) | 5.4 | 10.8 | **0.31** | not reachable below 14 |

A true primary gain between 3 and 4 points is more likely NOT_SHOWN than COMPLETED. The L3 secondary is
underpowered by construction (a rare code); its NOT_SHOWN is not evidence against L3. One seed takes 3 to 5 minutes
of CPU (release), far under the one-hour budget.

## 10. What would refute the idea

FALSIFIED against a baseline means: under tables where specialisation exists and the loop is given every advantage
the design names, it still does not beat that baseline by 3 points. COMPLETED with a passing null world means only that
the loop exploits specialisation when it exists. A model-backed experiment (F004, F005) is where the idea is tested.

## 11. Pilots (seeds >= 1001), not evidence

Three pinned-scale pilots on the final code, seeds 1007, 1008, 1009 (`pilots/`, summary in `pilots/summary.md`), run with
the release profile of the entrypoint (`cargo run --release`, no override): all integrity counters 0, all controls pass,
coverage met, `first_candidates_differ` 0 in all four worlds, one source digest in all three results. Primary world, mean
over seeds: full 0.958, gate 0.883, best-of-N 0.892, untrained pool 0.799, shuffled map 0.802, no L3 0.957, oracle about 0.96.
**full - gate +7.50 points [6.70, 8.30], full - best-of-N +6.60 [5.60, 7.60]**; between-seed sd 0.5 and 0.6 points.
The harness agrees with the planning model (0.958 / 0.883 / 0.892 against 0.962 / 0.883 / 0.891). Null world: the no-L3 loop
minus gate, generic pool and shuffled map is -0.2, -0.2, +0.6 points. Secondary: workspace in `ctx` +2.3 points
[1.7, 2.9] against 0.9 (SUPPORTED); L3 on the gap population +3.0 [-5.7, 11.6] against 5.4 (NOT_SHOWN, low power);
self-model untested (+0.00, non-discriminating). `mild`: full - best-of-N +3.7 [3.6, 3.8], at the margin. The grid
edge rule flagged none of the 108 tunings and its sensitivity check passed in all of them; the largest edge-over-inner gap
is 0.08 points (766 micro).
Disclosed history of the pilots: round-1 pilots (grid {50..400}) had full - gate +7.75; round-3 pilots (nine-point grids,
shared first draft not yet in place) +7.4; the round-4 numbers above change little although the planning model, the
derived thresholds (gate over the untrained pool 120 -> 43 permille) and the first draft changed: the untrained pool
gained 16 points (0.635 -> 0.799) from the shared good first draft, the gate gained 0.3, the full loop 0.4.
Seeds 1007 to 1009 are never confirmatory. Every result carries `procedure.source_digest` (a hash of the harness
sources, the harness crate manifest, `Cargo.lock` and the pinned configuration as compiled). The claim that the
pilots ran on the final code is **conditional on freeze step 5** (an outcome-digest match on the committed tree); a
mismatch means the pilots are re-run, never that a digest is edited. **The pilots' source digest can never equal the
frozen one** (freeze steps 1 and 3 write signed values into `config.toml`, which is part of it): the aggregator compares
source digests across the five confirmatory seeds only, and the guard for the pilots is the outcome digest.

## 12. Freeze procedure (run by the owner; not done)

1. Sign: replace the `must-be-signed-...` values of `config.toml [preregistration]` by integers equal to
   `criteria.toml [decision]` (30, 15, 50, 950; `aggregate.py` and a test refuse a disagreement).
2. Merge `claude/dev-integration` (runner support for the manifest `preflight` command and `command_environment`); the
   manifest already declares `preflight` in the format that support defines (a command string of the entrypoint's program,
   no placeholder). The preflight itself runs after step 5.
3. Digests: `preregistration_text_sha256` and `criteria_sha256` into `config.toml`; `preregistration_sha256` and
   `preregistration_rules_sha256` into `experiment.toml` (`experiment_records.preregistration_digest`); status
   `prepared` in the manifest and the registry; `update_component_docs.py --write`; `check_research_gates.py`. The harness
   compiles `config.toml` and `experiment.toml`, so this precedes the mutation run. **Commit.**
4. On the committed tree: the full `python3 scripts/mutation_check.py F007` (no `--only`) and
   `python3 experiments/feedback/F007-vcw-loop/tests/mutate_aggregator.py`; **commit** `results/mutations.json` and
   `tests/aggregator-mutations.json` in a results-only commit (check that `experiment_records.mutation_evidence`
   accepts it). No mutation count is cited in this text before that run.
5. Re-run `ptr-bench vcw-loop 1007` on the committed tree with the entrypoint's release profile and require its
   `outcome_digest` values to equal those of `pilots/pilot-1007.json`. The outcome digest covers every arm's success bits in all
   worlds, so it also guards what the source digest does not hash (the product crates `ptr-runtime`, `ptr-verifier`,
   `ptr-router`, `ptr-ledger`, and the toolchain), but only on seed 1007. A mismatch means the pilots are re-run, never that a
   digest is edited; **first re-run the pilot with the same build profile as the entrypoint (the pilots here already used
   it) to separate profile from code**. A different `source_digest` among the five seeds' results is INVALID (source digests
   are not compared with the pilots').
6. `run_experiment.py preflight F007` (it reserves nothing), then `run_experiment.py prepare F007` and **commit** its record
   (the runner refuses a tree HEAD does not hold).
7. Only then `run_experiment.py run F007 --seed <s>` for each of the five seeds, **committing each run record before the
   next seed**, and `aggregate.py`. Mutation evidence is the record written in step 4; the earlier logs under `tests/`
   (`mutation-run-*.log`) are development history, not evidence (run 5 still shows the pre-fix crash of
   `edge-test-uses-a-huge-z`, which was re-expressed as a 100-SE requirement and killed in a later run).

## 13. Limitations

See `aggregate.py` (LIMITATIONS): simulated Pods; a task-feature gate handicapped by the structure of the tables; no
Pod reads the broadcast outside `ctx`; the self-model acts through bounded hints only and is untested; L3 is Birth on one
code; no online learning during evaluation; five seeds make a wide interval; success is verified success.
