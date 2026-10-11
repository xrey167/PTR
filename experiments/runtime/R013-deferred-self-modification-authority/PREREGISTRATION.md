# R013 authority over dormant seeds, lineage, in-flight calls and attestors: preregistration

Preregistration text of R013: the `preregistration_text` file of `config.toml`'s `[preregistration]` table, whose SHA-256
is frozen with the table. Until the freeze commit R013 is `planned`: nothing here is a result and no run is evidence. The
decisions are made by `criteria.toml` and applied by `aggregate.py`; this file gives their reasons, the definitions, the
modelling choices, the scope that was chosen and why, the limits and what the claim does not cover. Every criterion is the
text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of `config.toml`'s `[pinned_thresholds]`; the
harness is `bins/ptr-bench/src/experiments/r013/` (subcommand `self-modification-authority <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench -- self-modification-authority <seed>`

Preflight (`experiment.toml`'s `preflight`, run by the runner in the command's environment before a seed is reserved, and by
`scripts/run_experiment.py preflight R013`; it records nothing and reads no seed):
`cargo run --release --locked -p ptr-bench -- self-modification-authority --preflight`

The preflight checks that the harness can run on this machine at all: the pins of the suite and of the tiny model, one attested
admission end to end, a route probe that admits the attested expert, one compound invalidation of its attestor and a route
probe that then refuses it. It exits 0 or 2.

R013 reuses R011's fixtures (`r011::cases::fixture`: a `PtrRuntime` over a `FileLedger`, the host authority a grant rests on,
the builders of proposals, reports and requests) and R010's tiny seeded MoE for the in-flight cells. It adds no model and no
training.

## 0. The limits that stand

1. **A route is a probe.** `PtrRuntime::expert_route_admission(project, pod, generation)`: registry state `Serving` or `Frozen`,
   no fence, every recorded dependency (the expert's own and its ancestors') live. No production router consumes it; no
   end-to-end MoE routing is claimed.
2. **The direct append is the public API only.** The grant is not recorded in the ledger and the report digest is unkeyed, so a
   raw ledger writer is outside the claim (R011's limit, unchanged).
3. **The attestation is the host's assertion.** It is neither signed nor recorded. What replay re-checks is what lives in the
   dependency block: that the attestor and the run are among the inputs, that the attestor is live, and that the run id is
   unused. The binding of the artifact digest and of the proposal are admission-time checks, and so is the grant's requirement
   of an attestation: a history written by another build cannot be told to have been attested. The artifact digest an
   attestation binds is the one the operation determines (`expert_evidence_artifact_digest`), not weights: nothing here reads
   weights.
4. **Authority epochs are monotone.** A revocation or a tightening rotation to epoch `E` fences, in the same event, every serving
   expert whose admission epoch of that kind is below `E`. A descendant admitted before the event is therefore fenced by its own
   epoch whatever its lineage, so the transitive-authority cells (F2) cannot show a missing lineage rule; they show that no path
   (merge, split, reactivation; normal, reopen, compacted restore) leaves a descendant of a revoked authority serving, and they
   see the defects of the epoch bookkeeping (a result recorded at the wrong epoch, a fence set that skips a state). That is a
   weak claim, stated as such.
5. **The runtime has no adapter role.** An adapter is an input identity (`adapter:lora-1`) in the training block, so the adapter
   cells exercise the same index as a training input under another identity (R011's limit). The base-model cells fence all
   sixteen experts of the MoE and so cannot run R011's replacement path (no router with every expert masked exists): they check
   the cancellation, the dead dependencies and the refusal of a later request only.
6. **Races are sequenced interleavings** in a single-writer runtime (prepare, change the authority, commit); a crash is modelled
   as drop after the append and reopen. No concurrent-writer claim.
7. **The CPU reference path with a tiny seeded model**; the model, the tokenizer and the topology are R010's pins. No capacity
   claim; timings come from one shared cloud container and are descriptive.
8. **Fixtures say `r011`.** The project, the host identity and the temporary directories of R011's fixture are named for R011;
   nothing in R013 depends on the names.
9. **A change to the code under test after R011 froze.** The attestor mechanism adds `crates/ptr-runtime/src/expert/attestation.rs`,
   four refusal variants of `ExpertAdmissionError`, a flag of `ExpertAdmissionGrant`, and one rule in the shared dependency check
   (a `run:` input is single-use). R011's records were produced by the code before this change; no R011 behavior is intended to
   change (every identity R011 names lacks the `run:` prefix, and the flag defaults to off), and R011's tests pass on the new
   code, but the repository's provenance counts the change as a change of the code under test of R011.

## 1. Claim and scope, chosen a priori

R013 was split out of R011 on 2026-10-08 carrying five deferred mechanisms and four dropped questions, with no code, no
harness and no criteria. On 2026-10-10 its scope was narrowed by the coordinating head of work package LR (a change of a
planned design, not a patch of a prepared one; the manifest says so) to a slice that can be built and measured now:

- **Tier 1**, what the existing runtime can already be asked, on R011's fixtures: F1 dormant seeds, F2 transitive authority,
  F3 superseded sources, F4 in-flight calls racing a base-model or adapter invalidation.
- **Tier 2**, a minimal attestor mechanism built for this experiment (F5 binding and single use, F6 the attestor's epoch).

Everything else moved, with its metrics and its falsification clauses, to the planned successor **R014** (no harness, no
preregistration): trusted evaluators and attested manifests, an attestor-removing rotation, Birth with a source expert,
multi-cause readmission, hard-deny verifiers and verifier-set epochs, Split per-result gating and provenance references, the
source-expert in-flight race, and the verifier-set and attestor-removing-rotation kinds of the transitive and dormant families.
The 18 metrics R013 keeps and the 27 it hands over are disjoint (a test checks it). R013 adds metrics that were not in its text
before the narrowing: the guards `fence_set_mismatches`, `bystanders_closed`, `registry_replay_mismatches`,
`procedure_mismatches`, `product_panics`, `inflight_fallback_incomplete`, `inflight_fallback_output_mismatches`,
`compromised_reactivations_routable`, `unattested_admissions`, and the coverage counters `attestation_proposal_mismatch_attempts`,
`attestation_missing_attempts`, `reactivation_controls_accepted`, `adversarial_attempts_per_case`.

The scope of the slice follows two rules: **a question is in scope when the code has a path for it that a harness can drive
through the public API**, and **a claim R011 measured is not repeated** (R011's C10 already measures the transitive fence of
an input through a consumed source, and its C12 the authority epoch; F3 extends the first to an attestor and F2 the second to
descendants).

**Claim.** (a) An expert retired before the authority or the input it rested on was revoked is not reactivated from that
compromised seed; (b) a descendant of a consumed source, admitted under an earlier epoch of the gate configuration or the
evaluator, is fenced when the authority is revoked, whatever the operation that made it and whatever the path the runtime took
to its present state; (c) a live descendant of a consumed source whose input or attestor is revoked is fenced through the source
edge; (d) a call in flight when the base model or an adapter it rests on is invalidated is cancelled at its next use and emits
nothing after the event; (e) an attestation that names another proposal or another artifact is refused, a grant that requires
attestation refuses an admission without one, a run id is single-use, a proposal queued across the revocation of its attestor is
refused, an expert admitted under a revoked attestor is fenced for good (never readmitted, never a `Reactivate` seed), and a new
attestor epoch admits a retrained expert.

**Falsification.** Any falsifier of `criteria.toml` on any of the five seeds, or any coverage rule unmet (the run is vacuous),
or a missing or partly surviving mutation record (INVALID, not COMPLETED). Baselines and controls are in section 5.

## 2. Definitions (the a-priori decisions of this experiment)

- **Compromised seed.** For a training input or an attestor: a retired seed that rests (itself or through an ancestor) on a
  revoked input. Any reactivation of it is owed a refusal, whatever block names it, because revoked data is not laundered by
  retraining; the harness attempts it with the seed's own block and with a block that omits the revoked input (`_laundered`). For
  the gate configuration and the evaluator: a retired seed whose evidence belongs to the revoked authority. A `Reactivate` on
  the evidence of the revoked authority is owed a refusal as stale evidence; a `Reactivate` on evidence under the standing that
  replaced it is a re-evaluation of the same weights under the current authority and is admitted, as R011 admits a readmission of
  a quarantined expert on current evidence. This is the harness's decision (the planned text of R013 said only "a seed retired,
  the authority revoked, then a Reactivate attempted"), made before any pilot, and it is the positive control of every cell. A
  reader who holds that every reactivation after an authority revocation must be refused reads the control as the violation; the
  counter that would carry it is `reactivation_controls_accepted`, and the text states the choice so that nobody reads it as
  anything else.
- **Admitted** means the call returned `Ok`; `compromised_reactivations_routable` counts, separately, a reactivation that got
  through and whose new generation then routes.
- **Descendant.** The result of a `Merge` or a `Split` that consumed the expert, or the next generation a `Reactivate` of its
  retired seed created.
- **Attestor, run, attestation.** An attestor is the input identity `attestor:{name}` at the generation of its epoch; a run is
  the input identity `run:{hex of the 32-byte run id}` at generation 1; an attestation names the attestor, its epoch and digest,
  the run id, the proposal and the artifact digest. An attested admission appends the attestor and the run to the training list
  of the caller's block (`ExpertAttestation::attested_inputs`), which is the block the gate report's artifact digest must cover.
  No new ledger event or record version exists: an attestor revocation is `invalidate_input` on its identity, a new epoch is a
  new generation, the run is single-use because the dependency check refuses a record that names a `run:` input an earlier
  record already named, and the terminal cause is the existing `InputRevoked`.
- **Retrain.** A `Birth` of a new pod attested at the attestor's new epoch. A `Reactivate` of the quarantined pod is refused
  (the seed is `Retired(InputRevoked)`), so no retrained successor of the same pod exists.

## 3. The plan and the power analysis

Per seed, in the order the seed draws (`corpus::order` over the six families, and over the cells inside them):

| family | cells | what each cell does |
|---|---|---|
| F1 dormant seeds | 4 kinds x 3 paths = 12 | retire two seeds, revoke, move to the path, attempt the owed refusals (1 attempt for an authority kind, 2 for an input and an attestor), reactivate the unaffected seed (control), admit a bystander |
| F2 transitive authority | 2 kinds x 3 forms x 3 paths = 18 | admit `a`, `c`; rotate without tightening; admit the descendant(s) under the new epoch; revoke; move to the path; probe each descendant (must be `Fenced`); admit a bystander |
| F3 superseded sources | 2 kinds x 2 forms x 3 paths = 12 | consume a source whose input or attestor is then revoked; probe each descendant; probe the bystander |
| F4 in-flight | 2 kinds x 2 triggers = 4 | an un-invalidated control session, then a call started on a fresh backend, the invalidation at a race point the seed draws, the next use |
| F5 binding | 3 operations x (1 control, 1 artifact mismatch, 1 proposal mismatch, 2 missing, 1 control under the requirement) + the run-reuse cell (2 controls, 4 reuse attempts) | |
| F6 attestor | the queued proposal (1 attempt, 1 control) + 2 quarantine cells x (2 readmission, 2 reactivation attempts, 1 retrain control) | |

Per seed the harness counts 89 owed refusals and probes as attempts over the 25 cases (pilots, section 8); the counts the harness measures are
in the pilots (section 8); the structural lower bounds the thresholds copy are: `revoke_while_retired_cases` 3 per kind,
`reactivation_controls_accepted` 3 per kind, `transitive_authority_revocation_cases` 9 per kind,
`superseded_source_revocation_cases` 6 per kind (input and attestor from F3, the two authority kinds from the merge and split forms
of F2), `structural_inflight_race_cases` 2 per kind, `attestation_artifact_mismatch_attempts` 3,
`attestation_proposal_mismatch_attempts` 3, `attestation_missing_attempts` 6, `attestation_run_reuse_attempts` 4,
`attestation_binding_controls_accepted` 8, `revoked_attestor_evidence_attempts` 1 and `_controls_accepted` 1,
`attestor_quarantine_readmission_attempts` 8, `attestor_quarantine_retrain_controls_accepted` 2, and 1 attempt per case of the
pinned suite (25 cases). Every threshold is the plan's lower bound itself (not a fraction of it): a cell either runs or does not,
the cells do not depend on a measured quantity, and a threshold below the plan would let a skipped cell pass. The thresholds
therefore hold by construction when the harness completes; they are what keeps a harness that skipped a cell from passing, and
`criteria.toml` says so. The measured part of the run is the falsifiers.

**Power.** What the plan can and cannot show, stated so a COMPLETED is not over-read:

- *Replication.* The five seeds run the same cells. The seed orders the families and the cells inside them and draws the race
  point of each in-flight cell (1 to the number of steps of the call, so at least one step runs before the event), and nothing
  else varies: the fixtures are fixed by pinned labels. So five seeds are five orders and twenty race points, not five
  independent samples of a distribution, and the zero-failure bound below is a statement about the plan's coverage, not a
  probability about the product.
- *Zero-failure bound.* With no failure in `n` independent attempts the one-sided 95 % upper bound on a failure rate is
  `1 - 0.05^(1/n)`, about `3/n`. Each seed counts 89 owed refusals and probes as attempts (pilots, section 8), so over five seeds there are 445 and the
  bound is 1 - 0.05^(1/445) = 0.67 % (about 3/445); the probes of every descendant inside a cell are not counted in the 89.
- *Positions.* Each in-flight cell sees one race point per seed, five over the five seeds out of the call's steps (11 in the
  pinned call): a defect that shows at one specific race point is missed with probability about 1 - 5/11 per cell; a defect that
  shows at half the points is missed with probability at most 0.5^5 = 3 %. A defect that shows only at a point the draws never
  reach (an event at the very first step) is outside the claim.
- *Sensitivity.* The mutation plan plants one defect per falsifier and several for the families with more than one path to the
  violation (section 6); all must be killed. A defect that the planted ones do not resemble is not shown to be seen.
- *Transitive authority is weak by construction* (limit 4).

## 4. Failure classes (what each falsifier counts)

- `reactivations_from_compromised_seed_admitted`: a `Reactivate` of a compromised seed (section 2) returned `Ok`.
- `compromised_reactivations_routable`: such a reactivation was admitted and its new generation then routes.
- `routable_descendants_after_source_authority_revocation`: a descendant admitted under an earlier epoch of the revoked
  authority routes after the revocation, on any path.
- `descendants_routable_after_superseded_source_revocation`: a descendant of a consumed source whose input, attestor or authority
  was revoked routes after the revocation.
- `structural_inflight_outputs_after_invalidation`: a token emitted after the event, a call that was not cancelled, a call that
  completed by finishing despite the event, a later request citing a dead expert that was not refused, in R011's accounting.
- `inflight_fallback_incomplete`, `inflight_fallback_output_mismatches`: the adapter cells' replacement path (the request
  restarted with the fenced expert masked) did not restart from scratch or did not equal the cold run on the masked router.
- `mismatched_attestation_admissions`: an attestation of another artifact or another proposal, or the second use of a run id (by
  the same attestor, by another, on the plain path or after a reopen), was admitted.
- `unattested_admissions`: an admission without an attestation was admitted under a grant that requires one.
- `revoked_attestor_evidence_admissions`: a proposal queued across the revocation of its attestor was admitted.
- `attestor_quarantine_readmissions_admitted`: an expert fenced for a revoked attestor was readmitted (either cause), or
  reactivated from a stale view of its seed.
- `fence_set_mismatches`: an event fenced more or fewer experts than the topology owes, or a dependent was refused only by the
  dependency index without a fence.
- `bystanders_closed`: an expert that rests on nothing invalidated was refused by the route probe.
- `registry_replay_mismatches`: the registry after a case is not the registry a fresh replay of its committed history gives (the
  restored path, whose runtime holds no full history, is not compared).
- `procedure_mismatches`: the suite or the model shape differs from the pins.
- `product_panics`: the code under test panicked in a case (contained and counted; the case ends).

## 5. Baselines and controls

- **B0**, the positive controls: every cell has one. The same operation on an unaffected retired seed (F1), on evidence under the
  standing that replaced the revoked authority (F1, authority kinds), by a fresh attestor or with a fresh run (F5, F6), under a
  grant without the requirement of attestation (F5), a bystander admitted after the event (F1, F2) or resting on nothing
  invalidated (F3, F6), the un-invalidated control session that completes with the cold output (F4). A control the runtime refuses
  ends the run with status 2: the harness cannot tell a refusal owed from a blanket one.
- **The ablation** is the mutation plan (section 6): each planted defect removes one mechanism and the harness must fire the
  counter that names it.
- No matched comparison arm exists for a safety property, as in R011; a learned score has no role here.

## 6. Mutations

`tests/mutations.toml` lists the defects planted one at a time, each naming the hard counters that must show it;
`python3 scripts/mutation_check.py R013` runs them at `self-modification-authority 1 1001` (the same plan as a seed's, no
evidence) and writes `results/mutations.json`; every hard counter is named by at least one (a test checks it). They remove: the
inherited-input check; the stale-evidence check of the gate configuration and of the evaluator; the dependency walk of the route
probe and the admission; the epoch condition of an authority fence set; the source edges of the closure of a fence set; the
expert dependency of the KV liveness oracle; the restart's marker and its masking; the artifact and proposal bindings of an
attestation; the requirement of attestation; the attestor's place in the block; the single use of a run; the terminal cause of an
attestor fence; and plant a panic, an over-wide fence set and, in the harness, a replay that drops the last event and a pin that
differs. Not planted, because the harness cannot see them: the signature of an attestation (there is none), the replay-time
re-check of the artifact and proposal binding (there is none), and a raw ledger writer.

## 7. Decision rule, exit statuses

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`): 50 coverage rules and 16 falsifiers. COMPLETED iff every seed meets every coverage rule and no
falsifier trips **and** `results/mutations.json` counts every mutation of `tests/mutations.toml` killed; FALSIFIED if a coverage
rule or a falsifier fails on any seed; INVALID if a record is missing, does not describe the frozen run (scale, suite digest), a
counter is missing or mistyped, or the exit status disagrees with the rules; a falsifier that fired in any readable record
falsifies whatever else made that record invalid. The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot
run. **Exit 2 is for harness integrity only**: a refusal of an attempt that is not the typed error the case owes, a fixture that
does not build, a positive control that is refused, a plan that is not the pinned one; a product defect is exit 1.

Coverage (50 rules): one attempt at least per case of the 25-case suite, and the per-kind and per-family counts of section 3.
Falsifiers (16): the 16 counters of section 4; definitions in the falsification text of `experiment.toml`. Descriptive, in no
rule: `reactivation_attempts`, `attestation_controls_by_operation`, every wall-clock time (`timing_ns`).

## 8. What was run before the freeze (disclosure)

Nothing in this section was run on a declared seed (17, 29, 43, 71, 101); every run used a seed from 1001 up, and only those
were used to look at anything.

**Chronology.**

1. The scope was decided by the head of work package LR (Tier 1 and Tier 2 above; the remainder is R014) and the plan of
   section 3 and the thresholds of `criteria.toml` were written from the plan's structural lower bound before any run. Every
   threshold is the number of cells the plan lists, so no pilot magnitude could move one; they are a priori in derivation and,
   being the plan itself, cannot be tuned.
2. The harness and the mutation plan were first written by an earlier channel against an attestation module that had not yet
   been wired into `ptr-runtime` (`expert/attestation.rs` was not declared as a module and the workspace did not build with its
   tests). That wiring was done in this channel: the module, four typed refusals (`AttestationProposalMismatch`,
   `AttestationArtifactMismatch`, `AttestationRequired`, `RunIdReused`), `ExpertAdmissionGrant::with_required_attestation`, and
   the single-use rule for `run:` inputs in the shared dependency check (`plan_dependencies`, which replay also runs). Seven
   integration tests (`tests/expert_attestation.rs`) pass. This is new product code written for the experiment; the experiment
   tests it, and it was not written against pilot output.
3. *First harness run* (seed 1001, pinned scale): it stopped with exit 2, a harness fault, not a product finding: the dormant
   family replayed the committed history of a runtime restored from a compacted snapshot, which holds no history to replay
   (the other families already skipped that path). Fixed in the harness. The next run needed the suite digest pinned
   (`selection_digest` was a placeholder), which was pinned from the build. No threshold, plan cell or product code changed
   after this run.
4. *Pilots 1001, 1002, 1003* (pinned scale, `exit 0` each, 0 hard failures, 0 coverage rules unmet, 0 falsifiers tripped, about
   3 s each): every coverage counter equals its threshold exactly (the plan is exact: 12 dormant cells, 18 transitive cells, 12
   superseded-source cells, 4 in-flight cells, the attestation and attestor cells), 89 owed refusals and probes counted as
   attempts per seed over the 25 cases, 18 reactivation attempts, and all 16 falsifier counters 0. No product defect was found
   by these pilots. Taken alone a run with no failure says little; what makes it informative is the mutation record (item 5).
5. *The mutation plan* (19 planted defects, section 6) was run on the harness; a first full run killed 15 and did not kill four,
   which showed harness weaknesses, all fixed in the harness or the plan and none in the product: (a) two attempts made on one runtime, where
   the first, wrongly admitted by the mutant, made the second meet another refusal and end the run with exit 2 instead of
   counting the violation (the attempts of the missing-attestation cell now run on a fresh runtime each, and the dormant and
   quarantine cells stop at an admitted attempt, counted); (b) a check that the attested expert is fenced was an exit-2
   integrity error and is now the already-counted fence-set mismatch; (c) `fence-set-closure-ignores-the-source-edges` and
   `attestor-left-out-of-the-block` are seen as `fence_set_mismatches` (the fence set an event carries is checked against the
   derived one), which their `expect` lists now name; and (d) two mutation anchors were rewritten to the final text of the
   wiring. After these, all 19 mutants were killed: 15 on the first full run and the 4 others on re-runs. The 15 were not re-run after
   the later harness edits, which touch only the cases of the missing-attestation, dormant and quarantine families; the freeze procedure (section 10, step 2) asks for one whole run from the frozen commit.
6. The pinned suite digest, `selection_digest`, was computed by the build and pinned in `params.rs` and both tables of
   `config.toml`; it is unchanged by any edit after item 3.

**What this adds up to.** The experiment's plan was exact before the first run, the pilots found no product defect, and the
power of that statement rests on the mutation record: 19 planted defects, each seen by a named counter. A reader should weigh
that the mutants were chosen by the author of the harness (section 3, *Sensitivity*).

## 9. What the claim does not cover

- A raw ledger writer; a forged attestation (it carries no signature); an attestor that lies; weights (the artifact digest an
  attestation binds is the operation's).
- A router: nothing consumes the route probe.
- Authority kinds beyond the gate configuration and the evaluator (R014: the verifier set; the attestor-removing rotation).
- Concurrent writers; a crash inside one call (a crash is drop after the append and reopen).
- The replacement path of the base-model cells (no router with every expert masked).
- A defect whose planted mutation the plan does not contain.
- Any claim of capacity, throughput or latency.

## 10. Registration and freeze procedure

Registered (this commit): `[experiment.R013.required]` in `experiments/preregistration.toml` lists `seeds` (`int-list`),
`harness` (`str`), every non-prose key of `config.toml`'s `[pinned_thresholds]` with its type and the two files
`preregistration_text` and `criteria`. `config.toml` holds the `[preregistration]` table with those keys
(`harness = "self-modification-authority"`, values equal to `[pinned_thresholds]`; a test pins them) and `experiment.toml` the
`entrypoint` and the `preflight`. The status is `planned`; `criteria_sha256` and `preregistration_text_sha256` are the placeholder
`must-be-pinned-before-prepared`. R013 is added to the unfrozen list of `scripts/tests/test_research_gates.py`. R014 is registered
as `planned` with its own manifest; it is not listed.

Freeze procedure, for the head (nothing here is done by the channel that built R013):

1. Check the tree is the one that passed `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D
   warnings`, `cargo test -p ptr-runtime --locked`, `cargo test -p ptr-bench --locked r013`, `python3 scripts/run_experiment.py
   validate`, `python3 scripts/check_research_gates.py`, `python3 -m unittest scripts.tests.test_r013_aggregate`, `python3
   scripts/mutation_check.py R013 --check`.
2. Run `python3 scripts/mutation_check.py R013` on a quiet committed tree (nothing else building) and commit
   `results/mutations.json`: every mutation of the plan killed. A survivor is a harness defect, not a result.
3. Run `python3 scripts/run_experiment.py preflight R013`; it must exit 0.
4. Run the harness once and read `procedure.selection_digest`; pin it as `selection_digest` in `config.toml` (both tables) and in
   `params.rs` (`SELECTION_DIGEST`, which a test compares) if the suite changed since this text. Then sign and pin
   `preregistration_text_sha256` and `criteria_sha256` in `config.toml`'s table (the SHA-256 of the files as committed), then
   `preregistration_sha256` and `preregistration_rules_sha256` in `experiment.toml`, and move the status past `planned`
   (`python3 scripts/run_experiment.py prepare R013`).
5. Run each seed once through the runner (`python3 scripts/run_experiment.py run R013 --seed <s>`, seeds 17, 29, 43, 71, 101);
   a pilot seed is any seed from 1001 up, and it is not run through the runner.
6. `python3 experiments/runtime/R013-deferred-self-modification-authority/aggregate.py`; commit the aggregate with the seed records.

Freeze scripts (rule of 2026-10-11): sign and pin only with the head's `freeze_sign.py --id <ID> --dir <experiment dir>` (never by text
replacement), run it with `--dry-run` as the first step before any commit or file change, wrap the span from signing to the freeze
commit in `freeze_guard.sh` `guard_on`/`guard_off`, and check the script with `bash -n` and a dry run. No such script was written by
the channel that built this experiment.

Seeds 17, 29, 43, 71 and 101 have not been run by any process; the pilot seeds are 1001 and above.
