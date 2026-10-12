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
   restore_recovery_snapshot and migrate_legacy_log replay with an empty attestor registry (no _with_attestors variant), so a
   ledger holding attested records cannot be restored through them; this fails closed and is outside the claim.
3. **The attestation is a signature by a host-configured key, and the registry is host configuration.** The attestation is an
   Ed25519 signature (strict verification) by an attestor key in the registry the host builds and hands to the runtime's
   constructor (`new_with_attestors`, `open_durable_with_attestors`, `replay_with_attestors`,
   `restore_compacted_with_attestors`); **the trusted set is fixed at construction and no call changes it**: an earlier
   `register_expert_attestor` under a gate grant was removed after a review showed that a caller holding only an admission
   grant could assemble a gate fence grant from `ExpertAdmissionGrant::binding()` and make its own key trusted (case
   `attestation_forged_gate_grant`, counter `self_trusted_attestor_admissions`, crate test
   `a_caller_holding_an_admission_grant_cannot_make_its_own_attestor_key_trusted`). The attestation binds claims (proposal,
   artifact digest, base model, results, run id), not weights: the artifact digest it binds is the one the operation determines
   (`expert_evidence_artifact_digest`), and nothing here reads what the attestor read. It does not bind the other training or
   evaluation inputs of the dependency block or the policy marker (only the gate report's digest covers the block, and that is
   the proposer's word); doc 37 lists the manifest digest as signed and that part is not implemented (doc 37's implementation
   status says so). Stated limits of the registry: it is not a ledger event (no audit trail of who trusted a key; a host that
   supplies a looser registry can trust any key for a new name or epoch, a smaller one fails closed); the same key may be
   registered under two names and revoking one name does not revoke the key; a global-revocation authority can burn a run id
   nobody has used (`invalidate_input` Superseded on an unseen `run:X`); the host's construction, not Gate authority, decides
   which keys are trusted. The attestation requirement becomes durable with the first attested record admitted under a requiring
   grant (the marker is recorded with it): a restart before that record, or a ledger whose earlier artifact-producing records were
   unattested, admits an unattested record under a looser grant, and `open_durable` does not refuse such a ledger (stated, not
   fixed: the host's configuration of the requirement is not a ledger fact). The attestor is not an evidence-authority kind of
   the gate-policy epoch.
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
   typed refusal variants of `ExpertAdmissionError`, a flag of `ExpertAdmissionGrant`, an attestor registry on `PtrRuntime`
   fixed at construction (`new_with_attestors`, `replay_with_attestors`, `open_durable_with_attestors`,
   `open_durable_at_with_attestors`, `restore_compacted_with_attestors`), and two rules in the shared dependency check (a `run:` input is single-use in any role at
   any generation; the reserved identities and the recorded requirement are checked on admission and replay). R011's records
   were produced by the code before this change; no R011 behavior is intended to change (every identity R011 names lacks the
   reserved prefixes, the flag defaults to off and the registry to empty), and R011's tests pass on the new code, but the
   repository's provenance counts the change as a change of the code under test of R011.

## 1. Claim and scope, chosen a priori

R013 was split out of R011 on 2026-10-08 carrying five deferred mechanisms and four dropped questions, with no code, no
harness and no criteria. On 2026-10-10 its scope was narrowed by the coordinating head of work package LR (a change of a
planned design, not a patch of a prepared one; the manifest says so) to a slice that can be built and measured now:

- **Tier 1**, what the existing runtime can already be asked, on R011's fixtures: F1 dormant seeds, F2 transitive authority,
  F3 superseded sources, F4 in-flight calls racing a base-model or adapter invalidation.
- **Tier 2**, a signed attestor mechanism built for this experiment (F5 trust, binding, single use and the recorded requirement; F6 the attestor's epoch).

Everything else moved, with its metrics and its falsification clauses, to the planned successor **R014** (no harness, no
preregistration): trusted evaluators and attested manifests, an attestor-removing rotation, Birth with a source expert,
multi-cause readmission, hard-deny verifiers and verifier-set epochs, Split per-result gating and provenance references, the
source-expert in-flight race, and the verifier-set and attestor-removing-rotation kinds of the transitive and dormant families.
The 18 metrics R013 keeps and the 27 it hands over are disjoint (a test checks it). R013 adds metrics that were not in its text
before the narrowing: the guards `fence_set_mismatches`, `bystanders_closed`, `registry_replay_mismatches`,
`procedure_mismatches`, `product_panics`, `inflight_fallback_incomplete`, `inflight_fallback_output_mismatches`,
`compromised_reactivations_routable`, `unattested_admissions`, `untrusted_attestor_admissions`, `self_trusted_attestor_admissions`, `wrong_typed_refusals`,
`skipped_cells`, and the coverage counters `attestation_proposal_mismatch_attempts`, `attestation_missing_attempts`,
`attestation_base_model_mismatch_attempts`, `attestation_untrusted_attestor_attempts`, `forged_gate_grant_attempts`,
`attestation_requirement_after_reopen_attempts`, `reactivation_controls_accepted`, `adversarial_attempts_per_case`.

The scope of the slice follows two rules: **a question is in scope when the code has a path for it that a harness can drive
through the public API**, and **a claim R011 measured is not repeated** (R011's C10 already measures the transitive fence of
an input through a consumed source, and its C12 the authority epoch; F3 extends the first to an attestor and F2 the second to
descendants).

**Claim.** (a) An expert retired before the authority or the input it rested on was revoked is not reactivated from that
compromised seed; (b) a descendant of a consumed source, admitted under an earlier epoch of the gate configuration or the
evaluator, is fenced when the authority is revoked, whatever the operation that made it and whatever the path the runtime took
to its present state; (c) a live descendant of a consumed source whose input or attestor is revoked is fenced through the source
edge; (d) a call in flight when the base model or an adapter it rests on is invalidated is cancelled at its next use and emits
nothing after the event; (e) an attestation signed by a key the host never registered (including one a caller tried to make trusted with a gate fence grant it assembled from its admission grant), or whose signature does not verify, or that names another
proposal, artifact, base model or result set than the operation, is refused with the typed error the case owes; a grant that
requires attestation refuses an artifact-producing admission without one, and the requirement is recorded, so a restart with a
looser grant cannot admit an unattested artifact-producing record and a replay refuses one; a run id is single-use in every
role and at every generation; a proposal queued across the revocation of its attestor is refused, an expert admitted under a
revoked attestor is fenced for good (never readmitted, never a `Reactivate` seed), and a new attestor epoch admits a retrained
expert.

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
- **Attestor, run, attestation.** An attestor is the input identity `attestor:{name}` at the generation of its epoch, with the
  id of its Ed25519 key as digest; a run is the input identity `run:{hex of the 32-byte run id}` at generation 1; an attestation
  is a claim (attestor, epoch, run id, proposal id, artifact digest, base model, ordered results) and the signature of the
  attestor key over its domain-separated payload. An attested admission appends the attestor, the run, the signature
  (`attestation:{hex}`) and, under a grant that requires attestation, the marker `policy:attestation-required` to the training
  list of the caller's block (`ExpertAttestation::attested_inputs`), which is the block the gate report's artifact digest must
  cover. The identities are reserved: only an attested admission sets them. No new ledger event or record version exists: an
  attestor revocation is `invalidate_input` on its identity, a new epoch is a new generation, the run is single-use because the
  dependency check refuses a record that names a `run:` input an earlier record already named, and the terminal cause is the
  existing `InputRevoked`. Replay rebuilds the payload from the record and re-verifies the signature against the registry's key.
- **Recorded requirement.** The marker input registers `policy:attestation-required` in the dependency index; from the first
  record that carries it, every later artifact-producing record (Birth, Merge, Split, a Reactivate that creates a generation)
  must carry a complete attestation, whatever grant admits it. Freeze, Retire, Prune and the unfreeze are not attested.
- **Wrong typed refusal.** A refusal of an attempt with a typed error other than the one the case owes is a violation of the
  product's contract and counts as `wrong_typed_refusals` (exit 1). Only an error the harness itself made is a harness error.
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
| F5 attestation | 3 operations x (1 control, 1 artifact mismatch, 1 proposal mismatch, 1 base-model mismatch, 1 self-invented attestor, 1 forged signature, 1 forged gate grant (the caller assembles a gate fence grant from its admission grant's binding, tries to trust its own key, signs and admits), 2 missing, 1 control under the requirement) = 24 attempts and 6 controls; the run-reuse cell (2 controls, 6 reuse attempts: same attestor, another attestor, plain training input, evaluation input, generation 2, after a reopen); the recorded-requirement cell (2 controls, 3 attempts: two unattested admissions after a restart with a looser grant, one unattested record in a replay; a Freeze under the requirement is the second control); the replayed-binding cell (1 control, 1 attempt: the attestation of one proposal carried by the record of another, replayed) | the owed typed refusal of each attempt; the control the runtime must accept |
| F6 attestor | the queued proposal (1 attempt, 1 control) + 2 quarantine cells x (2 readmission, 2 reactivation attempts, 1 retrain control) | revoke the attestor, probe the fence set, attempt the owed refusals, admit the retrained expert at the new epoch |

Per seed the harness counts 107 owed refusals and probes as attempts over the 34 cases (pilots, section 8; 104 over 33 cases before the
forged-gate-grant case of fix round 2, 89 over 25 before the signed attestation, section 8 items 2 and 5); the structural lower bounds the thresholds copy are: `revoke_while_retired_cases` 3 per kind,
`reactivation_controls_accepted` 3 per kind, `transitive_authority_revocation_cases` 9 per kind,
`superseded_source_revocation_cases` 6 per kind (input and attestor from F3, the two authority kinds from the merge and split forms
of F2), `structural_inflight_race_cases` 2 per kind, `attestation_artifact_mismatch_attempts` 4 (3 operations plus the replayed-binding cell, 3 + 1),
`attestation_proposal_mismatch_attempts` 3, `attestation_missing_attempts` 6, `attestation_run_reuse_attempts` 6,
`attestation_base_model_mismatch_attempts` 3, `attestation_untrusted_attestor_attempts` 6 (3 operations x self-invented and forged signature), `forged_gate_grant_attempts` 3 (one per operation),
`attestation_requirement_after_reopen_attempts` 3, `attestation_binding_controls_accepted` 11, `revoked_attestor_evidence_attempts` 1 and `_controls_accepted` 1,
`attestor_quarantine_readmission_attempts` 8, `attestor_quarantine_retrain_controls_accepted` 2, and 1 attempt per case of the
pinned suite (34 cases). Every threshold is the plan's lower bound itself (not a fraction of it): a cell either runs or does not,
the cells do not depend on a measured quantity, and a threshold below the plan would let a skipped cell pass. The thresholds
therefore hold by construction when the harness completes; they are what keeps a harness that skipped a cell from passing, and
`criteria.toml` says so. The measured part of the run is the falsifiers.

**Power.** What the plan can and cannot show, stated so a COMPLETED is not over-read:

- *Replication.* The five seeds run the same cells. The seed orders the families and the cells inside them and draws the race
  point of each in-flight cell (1 to the number of steps of the call, so at least one step runs before the event), and nothing
  else varies: the fixtures are fixed by pinned labels. So five seeds are five orders and twenty race points, not five
  independent samples of a distribution, and the zero-failure bound below is a statement about the plan's coverage, not a
  probability about the product.
- *Zero-failure bound, two readings.* With no failure in `n` independent attempts the one-sided 95 % upper bound on a failure
  rate is `1 - 0.05^(1/n)`, about `3/n`. Each seed counts 107 owed refusals and probes as attempts (section 8). Counted as
  independent, the five seeds give 535 attempts and a bound of 1 - 0.05^(1/535) = 0.56 %. But the five seeds run the same
  cells in another order, so the distinct attempts are the 107 of the plan, and the bound on those is 1 - 0.05^(1/107) =
  2.8 %; that is the figure that does not depend on treating the seeds as samples. (Before the forged-gate-grant case the same two readings were 520 attempts, 0.57 %, and 104 distinct attempts,
  2.8 %; on the tree before the signed attestation 445 attempts, 0.67 %, and 89 distinct attempts, 3.3 %.) Neither is a probability about the
  product: the plan is fixed and the bound is a statement about its coverage. The probes of every descendant inside a cell
  are not counted in the 104.
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
- `mismatched_attestation_admissions`: an attestation of another artifact, proposal or base model, or the second use of a run id
  (by the same attestor, by another, on the plain path as a training input, as an evaluation input or at generation 2, or after
  a reopen), was admitted, or a recorded attestation made for another proposal and artifact replayed.
- `unattested_admissions`: an artifact-producing admission without an attestation was admitted under a grant that requires one,
  or after a restart with a looser grant once the ledger recorded the requirement, or an unattested record replayed under it.
- `untrusted_attestor_admissions`: an attestation by an attestor the host never registered, or a registered attestor's id
  carrying another key's signature, was admitted.
- `self_trusted_attestor_admissions`: a key a caller tried to make trusted through a gate fence grant it assembled from its
  admission grant's binding is in the runtime's registry afterwards, or its self-signed attestation was admitted.
- `wrong_typed_refusals`: the product refused an attempt with a typed error other than the one the case owes (a contract
  violation of the product; a regression that still refuses but with another error is a falsification, not a harness error).
- `skipped_cells`: the number of coverage prerequisites a run left unmet (a cell the plan lists did not run). The run is vacuous
  then; the counter exists so that the mutation plan can name a skipped cell like any other defect.
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
evidence) and writes `results/mutations.json`; every hard counter is named by at least one (a test checks it). They remove:
the inherited-input check; the stale-evidence check of the gate configuration and of the evaluator; the dependency walk of the
route probe and the admission; the epoch condition of an authority fence set; the source edges of the closure of a fence set;
the expert dependency of the KV liveness oracle; the restart's marker and its masking; the proposal, artifact and base-model
bindings of an attestation at admission (killed by `wrong_typed_refusals`: the replay-time check still refuses the attempt,
with another error than the one owed); the artifact binding at admission and in the recorded signature together (`artifact-binding-not-checked-at-admission-or-replay`; it plants the artifact binding only, which an earlier name overstated); the replay-time
re-verification of a recorded attestation; the signature check; the registry check (a self-invented attestor trusted); the live registration of an attestor key under a forged gate grant
(the pre-review product, put back by a plant in the product and in the harness's `try_to_trust` stub); the
single use of a run id, at its own generation only and in the training role only; the requirement of attestation, its
recording and its enforcement from the record; the liveness check of an attestor; the terminal cause of an attestor fence;
and plant a panic, an over-wide fence set and, in the harness, a replay that drops the last event, a pin that differs and a
case family that skips a cell (the Merge cells of the binding case, and the recorded-requirement cells of the attestor family: `skipped_cells`). **The per-kind and per-case coverage rules are not each targeted by a mutation**: only `skipped_cells` has harness skip-plants (two), so that the generic mechanism (an unmet rule is counted) is shown; that a particular rule would have fallen for a particular skipped cell follows from the rule being the plan's lower bound and the test that sets each counter to a violating value and expects exactly that rule to fail (`every_rule_fails_when_the_emitted_counter_is_set_to_a_violation`), not from a mutation each. Not planted, because the harness cannot
or need not see them: a raw ledger writer; the scope of the requirement to artifact-producing operations (a positive control
that is refused ends the run with status 2, which is no kill by definition); the refusal of an unconsumed reserved `run:` input
on the plain path (held by the crate tests in `crates/ptr-runtime/tests/expert_attestation.rs`).

## 7. Decision rule, exit statuses

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`): 63 coverage rules and 20 falsifiers. COMPLETED iff every seed meets every coverage rule and no
falsifier trips **and** `results/mutations.json` counts every mutation of `tests/mutations.toml` killed; FALSIFIED if a coverage
rule or a falsifier fails on any seed; INVALID if a record is missing, does not describe the frozen run (scale, suite digest), a
counter is missing or mistyped, or the exit status disagrees with the rules; a falsifier that fired in any readable record
falsifies whatever else made that record invalid. The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot
run. **Exit 2 is for harness integrity only**: a request the fixtures cannot build, a fixture that does not build, a positive
control that is refused, a plan that is not the pinned one; a product defect is exit 1, and so is a refusal with a typed error
other than the one the case owes (`wrong_typed_refusals`; in an earlier version of this plan that was exit 2 and a regression
that still refused would have made the seed INVALID instead of FALSIFIED).

Coverage (63 rules): one attempt at least per case of the 34-case suite, and the per-kind and per-family counts of section 3.
Falsifiers (20): the 20 counters of section 4; definitions in the falsification text of `experiment.toml`. Descriptive, in no
rule: `reactivation_attempts`, `attestation_controls_by_operation`, every wall-clock time (`timing_ns`).

## 8. What was run before the freeze (disclosure)

Nothing in this section was run on a declared seed (17, 29, 43, 71, 101); every run used a seed from 1001 up, and only those
were used to look at anything.

**Chronology.**

1. The scope was decided by the head of work package LR (Tier 1 and Tier 2 above; the remainder is R014) and the plan of
   section 3 and the thresholds of `criteria.toml` were written from the plan's structural lower bound before any run. Every
   threshold is the number of cells the plan lists, so no pilot magnitude could move one; they are a priori in derivation and,
   being the plan itself, cannot be tuned. (The rules that item 4 added for the signed tree were derived from the plan before
   any run on that tree; the pilots of the unsigned tree of item 2 had been seen by then. The rules for the forged gate grant of
   fix round 2, item 8, were derived from the plan too, after the pilots of item 5 had been seen.)
2. *Earlier tree.* The harness and a first mutation plan were written against an unsigned attestation: a plain record naming an
   attestor string, a single-use run id and the proposal and artifact digest, with no signature and no registry. The pilots
   1001, 1002 and 1003 (pinned scale, `exit 0`, no hard failure, no coverage rule unmet, no falsifier tripped, about 3 s each) and
   the 19 mutations of that round (15 killed on the first full run, 4 more on re-runs after harness fixes) all ran on that
   earlier tree. **None of them is evidence about the tree this text describes.** A review of that tree found that the
   "attestation" was a record the caller wrote, that the requirement was not in the ledger, that the run id was single-use in
   the training role at generation 1 only, and that the attestor name was unchecked; the head ordered the product change below.
3. *The product change (this round).* In `crates/ptr-runtime`: an attestation is an Ed25519 signature (strict verification, the
   scheme `ptr-security` and `ptr-net` already use, no new dependency) by a key of the host's registry with its epoch (in this
   round registered under a gate grant; item 8 fixes the trusted set at construction); the payload is domain separated and covers attestor, epoch, key id, run id, proposal id, the artifact digest the
   operation determines, the base model (id, digest, generation) and the result experts; the attestor, the run, the signature
   and the requirement marker are inputs of the version 3 dependency block, so the requirement is in the ledger and replay
   re-verifies the signature and the bindings against a registry the host supplies; the run id is single-use in every role
   and generation; the `attestor:`, `run:` and `attestation:` identities and the marker are reserved against the plain path;
   attestor names are validated; attestation and the run id apply to artifact-producing operations only. 16 integration tests
   (`tests/expert_attestation.rs`; 17 after item 8) cover each negative case (self-invented attestor, forged signature, unattested record after a
   reopen or in a replay, a looser grant after a restart, run reuse as an evaluation input and at generation 2, a wrong base
   model, non-producing operations under a required grant, a descendant of an attested expert after the attestor's revocation).
   This is new product code written for the experiment; it was not written against pilot output.
4. *The harness and the plan were rewritten to match* before any run on the new tree: the self-invented attestor, the unattested
   record after a reopen and in a replay, run reuse as an evaluation input and at generation 2, the wrong base model and the
   replayed binding are cases with a falsifier counter and a coverage rule each; a typed refusal other than the one owed is a
   falsifier (`wrong_typed_refusals`, exit 1), not a harness error (exit 2); a `skipped_cells` counter makes a skipped cell a
   falsifier. The mutation plan grew from 19 to 30 defects, among them the self-invented attestor trusted, the signature not
   verified, the recorded requirement not enforced, the run id checked at its own generation or in the training role only, and a
   harness that skips the Merge cells of the binding case.
5. *Pilots on the new tree, before fix round 2 (item 8)* (release build, the pinned entrypoint, seeds 1001, 1002, 1003, none declared): `exit 0` each, no hard
   failure, no coverage rule unmet, all 19 falsifier counters 0, 104 owed refusals and probes counted as attempts over the 33
   cases per seed, 18 reactivation attempts, suite digest unchanged. The outcome (everything but wall-clock time and the
   environment) of seed 1001 is identical to an earlier run on an intermediate tree of this round. No product defect was found
   by these pilots. Taken alone a run with no failure says little; what makes it informative is the mutation record.
6. *The mutation plan, whole run on the tree before fix round 2 (item 8) (uncommitted: `mutation_check.py` records only from a committed tree, so the run
   used `--only` with every name; nothing was recorded).* 27 of the 29 then planned defects were killed by a counter they name. Two
   were not: `attestation-bindings-not-checked-anywhere` and `replay-does-not-verify-the-recorded-attestation` ended
   `failed-elsewhere` (only `wrong_typed_refusals` fired, 4 and 1 times). The cause is a plan defect, not a product one: the replay
   check has two layers, the signature and the comparison of the recorded payload digest, and a plant that removed only the
   signature layer was still refused by the digest layer, with `AttestationMalformed` instead of `AttestationSignatureInvalid`. The
   plan was changed (no product change): both plants now remove both layers and still expect `mismatched_attestation_admissions`,
   and a new mutation, `replay-checks-the-payload-digest-only`, names the single-layer case with `wrong_typed_refusals`. These
   three were re-run alone after the change (all three killed); the other 27 ran on the tree before
   this edit of `tests/mutations.toml`, whose other entries are unchanged. Earlier, partial runs (a subset of 28 names, interrupted
   at the hand-over between two working sessions, the mutated file restored from its backup) are not claimed. The whole plan is
   re-run **on the final committed tree** at the freeze (section 10, step 6).
7. The pinned suite digest, `selection_digest`, is computed by the build and pinned in `params.rs` and both tables of
   `config.toml`; the suite has 34 cases since fix round 2 (item 8) and its digest was re-pinned then (it had been unchanged for the 33-case suite since the first pin).

8. *Fix round 2* (after a re-review of the tree of items 3 to 6; nothing here was run on a declared seed). Product: the review
   showed that `register_expert_attestor` under a gate grant did not keep the trust root away from the caller of an admission
   (`ExpertFenceGrant::gate` and `ExpertAdmissionGrant::binding()` are public and the binding names no kind, so a holder of an
   admission grant could assemble a gate fence grant and register its own key). The call was removed: the trusted set is a
   registry given to the constructor (`new_with_attestors`, `open_durable_with_attestors`, `open_durable_at_with_attestors`,
   `replay_with_attestors`, `restore_compacted_with_attestors`) and nothing changes it later. A crate test
   (`a_caller_holding_an_admission_grant_cannot_make_its_own_attestor_key_trusted`) builds the forged gate grant and shows the
   self-signed attestation refused `AttestorUnknown`, and the registry's own tests moved to the registry (17 tests in
   `tests/expert_attestation.rs`). Stated as limits instead of fixed (section 0, item 3): the requirement becomes durable with
   the first attested record, the signature does not cover the other inputs of the block (doc 37's implementation status was
   corrected), the registry is not a ledger event, one key under two names, a burnt run id, Gate authority and the host. Harness
   and plan: the case `attestation_forged_gate_grant` (3 attempts per seed, one per operation), the counter
   `forged_gate_grant_attempts` (coverage, 3) and the falsifier `self_trusted_attestor_admissions`; the suite grew to 34 cases
   (new digest, pinned in `params.rs` and both tables of `config.toml`), 63 coverage rules and 20 falsifiers; the mutation
   plan to 32: the live registration put back (`live-attestor-registration-under-a-forged-gate-grant`, a plant in the product
   and one in the harness's `try_to_trust` stub), a second skip plant on an attestor cell
   (`harness-skips-the-recorded-requirement-cells`), and the rename of `attestation-bindings-not-checked-anywhere` to
   `artifact-binding-not-checked-at-admission-or-replay`. The rules for the new case were written from the plan (3 operations)
   after the pilots of item 5 had been seen. **Pilots** on the final tree (release, the pinned entrypoint, seeds 1001, 1002, 1003,
   none declared): `exit 0` each, `hard_failures` 0, no coverage rule unmet, all 20 falsifier counters 0, 107 owed refusals and
   probes counted as attempts over the 34 cases per seed, 18 reactivation attempts, suite digest equal to its pin; their stdout
   is in `pilots/` (pilots, not evidence). The pilots and the mutation record of items 5 and 6 are of the tree before this round
   and are superseded by these. **Whole mutation set** on the final tree (uncommitted, so `--only` with all 32 names and no
   record): 32 of 32 killed, each by a counter it names (the log of the run is in the channel report). The live registration is
   killed by `self_trusted_attestor_admissions` (6: the key trusted and the self-signed attestation admitted, for each of the
   three operations) and by no other counter. The whole plan is re-run on the final committed tree at the freeze (section 10,
   step 6).

**What this adds up to.** The experiment's plan was exact before the first run on this tree, the pilots found no product defect,
and the power of that statement rests on the mutation record: 32 planted defects, each seen by a named counter, to be shown on
the frozen tree. A reader should weigh that the mutants were chosen by the author of the harness (section 3, *Sensitivity*).

## 9. What the claim does not cover

- A raw ledger writer; an attestor that signs a false claim (the signature binds claims, not what was trained); the compromise of
  a registered attestor key before it is revoked; the host's choice of registry (the registry is host configuration, fixed at construction and not a ledger fact); weights (the
  artifact digest an attestation binds is the operation's).
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

Freeze procedure, for the head (nothing here is done by the channel that built R013; the draft script is `freeze_r013.sh`). The
order matters and each step must succeed before the next:

1. `freeze_sign.py --id R013 --dir <experiment dir> --dry-run`, the first step, before any commit or file change. R013 has no
   threshold to sign (every threshold is the plan itself); the script pins two hashes.
2. Commit the work of the channel as one commit, explicit paths only (`experiments`, `crates`, `bins`, `docs`, `scripts/tests`,
   `STATUS.md`), which includes `pilots/` with `pilot-1001.json`, the outcome reference of step 7.
3. The pre-sign checks, on the committed tree and **before** signing, exactly the ones the head's script runs, in its order:
   a clean tree after the work commit; `python3 scripts/check_research_gates.py`; `python3 scripts/check_repo.py`; `python3
   scripts/update_component_docs.py --check`; `python3 scripts/mutation_check.py R013 --check`; `python3
   scripts/run_experiment.py validate`; `python3 -m unittest scripts.tests.test_r013_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests`; `cargo fmt --all -- --check`;
   `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test -p ptr-runtime --locked`; `cargo test -p
   ptr-bench --locked r013`; then `target` is removed and the tree must be clean again.
4. Sign with `freeze_sign.py` under `freeze_guard.sh` `guard_on`: it pins `preregistration_text_sha256` and `criteria_sha256`
   in `config.toml` and moves the status past `planned`; never by text replacement.
5. The freeze commit and the tag; `guard_off`.
6. **The whole** `python3 scripts/mutation_check.py R013` on the committed tree, all 32 mutations killed, then commit
   `results/mutations.json`. **No edit to a mutation-bound file** (`crates/ptr-runtime/src/**`, `bins/ptr-bench/src/experiments/r013/**`,
   `tests/mutations.toml`, `config.toml`) after this run. A survivor is a harness defect, not a result.
7. Rerun pilot 1001 on the frozen commit (release, the pinned entrypoint) and compare its outcome (everything but wall-clock time
   and the environment) with the committed reference `pilots/pilot-1001.json` (pilots, not evidence: `pilots/README.md`); a difference stops the procedure.
8. `python3 scripts/run_experiment.py preflight R013` (must exit 0), then `prepare R013`.
9. Each seed once through the runner (`run R013 --seed <s>`, seeds 17, 29, 43, 71, 101); a pilot seed is any seed from 1001 up and
   is not run through the runner. Then `aggregate.py`; commit the aggregate with the seed records.

Freeze scripts (rule of 2026-10-11): sign and pin only with the head's `freeze_sign.py` (never by text replacement), run it with
`--dry-run` as the first step, wrap the span from signing to the freeze commit in `guard_on`/`guard_off`, and check the script with
`bash -n` and a dry run. The draft was checked with `bash -n`; it was not run.

Seeds 17, 29, 43, 71 and 101 have not been run by any process; the pilot seeds are 1001 and above.
