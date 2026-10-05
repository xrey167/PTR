# 37 — Pod-MoE: a mixture of experts whose experts are pods

Status: **design**. Nothing in this document is implemented as a Pod-MoE. The pieces
it builds on exist as prototypes (`ptr-lineage`, `ptr-branch`, `ptr-router`,
`ptr-pods`, `ptr-verifier`, see [35](35-agentic-substrate.md)); the parts marked
*new* below do not exist yet. The base model is OLMoE-1B-7B. Training runs in Python
(TRL/PEFT, later verl) with PTR governance reached through the Rust binding
`bindings/ptr-py` (PyO3, merged in #123; §6). The first increment produces only contracts and a CPU
smoke pipeline: **no quality claim is made**, and every claim beyond the cited code is an
experiment (§5) with a falsification rule, per invariant 15.

**Dependencies on other PRs.** This document describes three APIs by PR: `aimer_score`
(#114, merged), `ExpertProposal` in `crates/ptr-pods/src/expert.rs` (#115) and the
`ptr-router` routing API (#116: `Router::policy` returning a `RouteDistribution`, the
trusted `route_checked` / `CheckedRouter` that samples with a secret `SamplingKey` and
a fresh `DecisionNonce` and records the sampled pod's quantized probability as a
`Propensity` (exact width/2^53), and `RoutedError` carrying the request id and
digest). Statements about #115 and #116 hold **once those PRs are merged**; the merge
order is #116, #115, then this document. They are not in this branch's tree today.

## 1. Principle

Every expert is a pod: it has a manifest, a generation, a lineage record, a
lifecycle state and it is revocable. Where it executes is a separate decision made
by latency: an expert consulted once per token per layer cannot be a network
service, one consulted once per request can. Placement changes how an expert is
called, never what it is, so the same lifecycle, gates and revocation rules apply
at every level.

## 2. Three levels

| Level | Unit | Where it runs | Router | Evaluation / decision |
|---|---|---|---|---|
| L1 token | Frozen OLMoE base experts plus routed LoRA experts (PERFT-R) per layer | **In-process**, per token per layer; network latency is impossible here | Learned top-k router with auxiliary-loss-free bias | Training metrics (§5) and the lifecycle gates (§4) |
| L2 segment / request | Whole models or adapter sets as networked pods over PodWire ([33](33-pod-wire.md)) | Separate pods, one routing decision per call | Learned router in `ptr-router` (Router-R1 / RouteLLM style) | Offline evaluation needs a *new* contextual-routing log and categorical estimator (see below) |
| L3 meta | Structural operators Birth, Merge, Split, Freeze, Reactivate, Prune, Retire | Policy / agent pods | None; it proposes | Proposals are decided by the verifier and the gates, never by the policy |

`evaluate_off_policy` (`crates/ptr-branch/src/arbiter.rs:1027`) cannot evaluate L2:
it is specific to the fixed `TriageDecision` action space (`AutoPropose` / Escalate /
Discard, `arbiter.rs:13`) and cannot score a categorical choice among pods. L2 needs
a **new** contextual-routing log (request features, candidate set, chosen pod, and
the logging propensity as `ptr-router`'s trusted `route_checked` records it from the
`RouteDistribution` of `Router::policy`, a validated `Propensity` of the chosen pod in (0, 1]) and a categorical
IPS / SNIPS / doubly-robust estimator. Every L2 record logs the **complete
per-candidate distribution** (the `RouteDistribution`'s quantized probabilities for
every candidate) plus the logging-policy version or digest, not only the chosen
propensity. The estimator refuses a target policy that puts mass on an action whose
logging probability is 0 (positivity), as the arbiter does with
`ArbiterError::PositivityViolation` (`crates/ptr-branch/src/arbiter.rs:1174-1178`). The
arbiter's estimators are a template only.

**Actions and candidates are generation-qualified.** The log records, for every
candidate and for the chosen action, `(ProjectId, PodId, generation, execution-manifest
digest)`. The project is part of the identity because `PodRegistry` is keyed by
`(ProjectId, PodId)` (`crates/ptr-pods/src/lib.rs:174-192`), so two projects may each
register an `echo` Pod. The Pod
manifest alone does not carry a generation: `PodManifest` has `id: PodId` and a
`digest()` over the invocation contract (`crates/ptr-pods/src/lib.rs:96` and `:107`),
while the generation and the `manifest_digest` live in the `ExecutionManifest`
(`crates/ptr-pods/src/manifest.rs:14` and `:26`) that binds the served artifacts. The
log therefore takes the execution-manifest digest (or the artifact digest) together
with its generation, so rewards and propensities from different artifacts never
collapse onto one action label. Off-policy estimation groups actions by this
generation-qualified identity, and a record whose chosen generation was revoked or
replaced before verification is evaluated against **that** generation, not the new
one.

**The log must also record the observed outcome.** Context, action and propensity are
not enough: IPS, SNIPS and doubly-robust estimators weight an observed reward, and
the template reads it from `LoggedTriage.reward`
(`crates/ptr-branch/src/arbiter.rs:924`). Each L2 record therefore carries an
outcome field bound to `(request_id, chosen action)`: the verifier's
`VerificationStatus` and the derived reward, using the four-status mapping of §5
(`Pass` yields the report's `score`; `Fail`, `Disputed` and `Unknown` yield 0, and
an output whose **weakest** per-verifier `VerificationLevel` is below the task's
minimum also yields 0; see §5).
It is written **after** verification, as an update of the record the routing decision
created, never by the router at decision time. Outcomes must not be censored by the action
taken, so a failure is an outcome and is **not** excluded: a timeout, crash,
unreachable pod or verification failure is assigned a preregistered status
(`Fail` or `Unknown`) with reward 0, recorded at a fixed deadline. Only records still
pending before their deadline are not yet evaluable and are left out until it
passes. Every estimate reports the counts per outcome type. Propensity weighting does
**not** correct outcome censoring: it reweights by the chance of the action, not by
the chance that the outcome was observed, so dropping failures would bias the
estimate toward pods that fail silently.

Token-level experts must be co-located, because every MoE layer finishes routing
before the next layer starts. Request-level experts can be networked services,
because there is one routing decision per call.

## 3. Self-observation and self-modification without self-promotion

**Self-observation.** Routing traces, expert load, churn, interference and AIMER
scores are written as observations into SemDB as derived keys with provenance. The
model can read them as context. They are derived and non-authoritative
(invariant 5): reading them informs a proposal, it does not decide anything.

**Self-modification flow.**

1. The L3 policy emits an `ExpertProposal`.
2. The proposal runs as a speculative overlay in `ptr-branch`.
3. Sandbox training produces a candidate.
4. The candidate must pass the `ForgettingGate`, `measure_interference` against the
   calibrated null threshold (a preregistered margin, or a high quantile of overlaps
   between independently trained random adapters of the same shape; not the mean
   `chance_overlap`), the verifier and the public suites.
5. Only then is it committed as a ledger event (`ExpertPromoted` / `ExpertRetired`).

**Gate evidence must be bound to the evaluated artifact.** This is a design
requirement, not existing behaviour. `ForgettingGate::evaluate` takes an `AdapterId`
plus caller-supplied matrices and scores (`crates/ptr-lineage/src/forgetting.rs:235`),
and `GateReport` binds the verdict to that id and to nothing else: the gate cannot
tell which adapter the accuracy matrix and public-suite scores were measured on, so
matching the measurements to the adapter is the caller's obligation
(`forgetting.rs:172-176`); in particular "the report names no lineage or base model" (`forgetting.rs:175`), so a pass on one base does not transfer to another. A promotion that rests on such a report would only be as
trustworthy as its caller. Therefore the `ExpertPromoted` event must bind **trusted
evidence**: the artifact digest and generation of the evaluated expert, the **exact evaluated
base model identity, generation and digest** (or the complete evaluated
execution-manifest digest, which folds them in), the digest of
the evaluation-data manifest (its **identity and generations**, not only a digest), the digest of the gate configuration (thresholds), and
the evaluator identity, and the **attested training manifest**: the training sandbox
is untrusted, so the full consumed-input manifest (every input identity, generation
and digest the trainer actually read) is recorded by the trusted orchestrator that
feeds the data, never self-reported by the trainer. `affected_by` uses this attested
manifest, **and the evaluation-manifest bindings as well as the training inputs**, and an expert whose attested manifest is missing or mismatched cannot be
promoted (`SECURITY.md:17-21`: revoked generations cannot become usable). The
evaluation must run in a **trusted evaluator**, not in
the proposer's sandbox, which produced the candidate and could hand the gate
measurements of a different artifact. The Python binding (§6) only calls the gate; it
does not make the result trustworthy.

**One shared evidence-admission function** admits every gate-driven structural event, and the list is exactly: Birth or promotion (`ExpertPromoted`), Merge, Split, Prune, Freeze, Reactivate (including unfreeze), and proposal-driven Retire (reasons `Superseded` and `GateFailed`). `ExpertFenced` and the revocation-driven retirement are **excluded**: they run under the runtime revocation grant. R011 runs the evidence-binding negative cases for each event type on that list.

**Every gate-driven structural ledger event** (`ExpertRetired`, and likewise the
results of Merge, Split, Prune, Freeze and Reactivate, not only `ExpertPromoted`)
carries the target identity, generation, artifact digest and the trusted gate
evidence above. Commit is **compare-and-set**: the event is rejected if the target's
current generation or digest differs from the evaluated one, so a delayed event can
never retire a newer, unevaluated generation. At commit the compare-and-set **also
checks that every attested training input (identity, generation, digest) is live at
the commit anchor**: an input revoked after the gate passed and before the commit
makes the commit fail.

**Evaluation inputs are live inputs too.** The dependency index (`affected_by`), the use-time admission at serving, and R011 all include the evaluation-manifest bindings, not only the training inputs. The commit is rejected if any input named by the evaluation-data manifest is revoked or superseded at the commit anchor, and a **post-promotion revocation of an evaluation input** fences the experts whose promotion evidence depends on it, through the same compound `InputRevoked` path below.

**Revocation after promotion.** When a training-input generation is revoked, every
expert that `affected_by` returns over the **attested consumed-input manifest** is
**fenced** (not routable) and then retired. The ledger appends **one `LedgerEvent` per
commit** (`crates/ptr-ledger/src/lib.rs:350-358`, `Ledger::append`), so a fence appended
as a second event would leave a crash window between the revocation and the fence.
Revocation is therefore **one compound ledger event**,
`InputRevoked { input, generation, fence_set_commitment }`. It carries **no per-expert
list**, only a commitment to the fence set (the count and a digest), so the record stays
bounded: a dependency fan-out whose explicit list would exceed `MAX_RECORD_BYTES`
(8 MiB, `crates/ptr-ledger/src/integrity.rs:16`) still fits. **Applying the event
atomically derives the fenced experts** from the dependency index as of the event's
commit index, which is deterministic and replay-identical, so there is no state in which
the input is revoked and a dependent expert is still routable. A projection that cannot
derive the set **refuses to serve the dependents**, and replay refuses any mismatch with
the committed count and digest. The fence part of the event is not gate-driven and needs no gate
pass, it has its own runtime grant (`ExpertFenced` is the name of that fence record),
and it can never be issued by an L3 policy. The **revocation-driven retirement** that follows the fence is issued under the same runtime grant as `ExpertFenced`; it is not gate-driven, and it is explicitly excepted from R011's case that an `ExpertRetired` without the gate grant must fail. **In-flight expert calls are fenced too:** when an expert is fenced by a training-input or evaluation-input revocation, every in-flight L1 or L2 invocation of that expert is cancelled or dropped by completion-time dependency admission (no output, no candidate, no effect). In addition, serving admission for L1 and L2 routing revalidates
each expert's attested training dependencies against live generations **at use and
after restart or replay**, so revocation-processing or index lag cannot leave a
dependent expert routable (`SECURITY.md:17-21`: revoked generations cannot become
usable after restart/cache/index lag). Current code does neither:
`Lineage::serving()` selects only on `AdapterStatus::Serving`
(`crates/ptr-lineage/src/lineage.rs:292-296`) and `affected_by()` returns adapter ids
only, from unstructured manifest strings (`:264-270`).

`ExpertPromoted` and `ExpertRetired` are **not implemented yet**. The `ptr-lineage`
README lists the `AdapterPromoted` ledger event as a next milestone
(`crates/ptr-lineage/README.md:40`), and
[35-agentic-substrate.md](35-agentic-substrate.md) states at line 1313 that until
adapter promotion is a committed ledger event admitted through checkpoint binding,
no registry row decides which adapter serves. The same holds for experts.

Invariants preserved:

- **#4 authority:** a candidate is uncommitted until the ledger event exists.
- **#5 derived search:** observations in SemDB cannot self-promote.
- **#11 verifier precedence:** a learned score cannot override a deterministic
  contradiction. For merges this is enforced by `PtrRuntime::merge_branch`
  (`crates/ptr-runtime/src/merge.rs:849`, ADR-0020); an expert promotion
  must be routed through the same grant-controlled path once it exists.
- **#15 falsifiability:** R011 (§5) states the failure criterion.
- **#17 revocable derivation:** the intent is that an expert names the input
  generations it depends on, so revoking an input identifies the experts it influenced
  (`affected_by`, §4). **Existing behaviour falls short of this:** `affected_by`
  intersects unstructured `data_manifest: BTreeSet<String>` entries
  (`crates/ptr-lineage/src/lineage.rs:109`, `:264-268`), with no generation and no
  digest, so a revocation of one generation of an input cannot be told from another
  and an id that was never bound to content matches by name only. Revocable
  derivation for experts therefore **requires** validated
  `(input identity, generation, digest)` bindings in the expert lineage record. This
  is a required extension, not existing behaviour, and a prerequisite of M013 and
  R011 (§5).

Hard invariants are excluded from learnable optimisation
([09-training.md](09-training.md)). The L3 policy can never alter gates, thresholds
or verifiers, and this is enforced by types: `ExpertProposal` carries no gate,
threshold or verifier parameter, and `ExpertProposal::parts` exposes its complete
field list (`crates/ptr-pods/src/expert.rs`, #115). That list is exactly the
operation, the **project** the experts belong to, the targets, the level of a newborn
expert and a digest of the proposer's rationale. The project is identity, not a
policy knob: an expert is identified by `(project, pod, generation)`, matching the
`(ProjectId, PodId)` registry key, and `ExpertTarget` and `ExpertPodSpec` each carry
a `project` and the `base_model` digest of the base model the expert adapts.
`ExpertProposal::new` takes the project and rejects a blank project
(`BlankProject`), a target of another project (`MixedProjects`) and a Merge whose
targets were adapted from different base models (`MixedBaseModels`). A **Birth**
proposal carries the base-model digest of the expert it creates, and Merge sources are
deduplicated by **logical expert** `(project, pod)`, so two generations of the same
expert can never both be Merge sources (these two rules come with the #115 update). The
`base_model` digest of a target is the value that the trusted evidence of this
section must match when it binds the evaluated base model.

## 4. Lifecycle mapped to existing code

| Operator | Mechanism | Existing code / status |
|---|---|---|
| Birth | New expert registered as Candidate; DynMoE-style initialisation from unrouted tokens | `Lineage::register` sets `AdapterStatus::Candidate` (`crates/ptr-lineage/src/lineage.rs:151`, status set at `:183`); initialisation is *new* |
| Merge | `subspace_overlap` above a **preregistered margin or a calibrated null threshold**, or `ConsolidationPolicy::consolidation_due`, then `ties_merge`, new Candidate, then gate. `chance_overlap` is the *expected* overlap of two random subspaces (`max(rank_a, rank_b) / dim`, `crates/ptr-lineage/src/subspace.rs:452-456`), a mean and not an upper confidence bound, so a merge must never trigger merely because the overlap exceeds it: about half of independent random pairs do. The threshold is a high quantile of overlaps between independently trained random adapters of the same shape, fixed before the runs | `subspace_overlap` `crates/ptr-lineage/src/subspace.rs:444`, `chance_overlap` `:462`; `consolidation_due` `crates/ptr-lineage/src/lineage.rs:341`; `ties_merge` `crates/ptr-lineage/src/merge.rs:32` |
| Split | High load plus bimodal routing clusters, then duplicate (expert upcycling), then fine-tune | *new* |
| Freeze | Stop gate updates after convergence (continual-learning theory, 2406.16437) | *new* flag |
| Prune | AIMER score plus usage; `ForgettingGate` on a public suite; then `retire` | `ForgettingGate::evaluate` `crates/ptr-lineage/src/forgetting.rs:235` (type at `:132`); `retire` `crates/ptr-lineage/src/lineage.rs:220`; `aimer_score` and `rank_experts_for_pruning` (`crates/ptr-lineage/src/aimer.rs`) |
| Reactivate | `Retired` is final today: `retire` refuses a retired adapter (`lineage.rs:225`) and no transition leaves it | Open research question with its own claim; no literature (§7) |
| Revoke | `affected_by` finds the adapters that depend on a revoked input, by comparing unstructured `data_manifest` string ids; the per-layer KV chain key invalidates layers after the affected layer j. Reliable revocation needs validated `(input identity, generation, digest)` bindings in the expert lineage: a **required extension, not existing behaviour**, and a prerequisite of M013 and R011 | `affected_by` `crates/ptr-lineage/src/lineage.rs:264` (manifest type at `:109`); per-layer KV chain key in [36](36-kv-reuse.md), section D |

## 5. Training programme

All runs happen in Python; the verifier contract is the reward source. No result
below exists yet.

- **SFT of PERFT-R LoRA experts on OLMoE.** Router balancing compared across
  auxiliary-loss-free bias, auxiliary loss and φ-balancing. Ablation: prescribed
  specialisation (code / math / tool) versus emergent specialisation.
- **RLVR.** Reward is derived from the `ptr-verifier` `VerificationReport` (a status
  plus `score` and findings; `crates/ptr-verifier/src/lib.rs:19`) and is defined for
  **all four** `VerificationStatus` values (`lib.rs:4-10`): `Pass` yields the
  report's `score`; `Fail`, `Disputed` and `Unknown` yield 0. A non-Pass report never
  receives a passing reward and its independent `score` is never used, so a high
  score on a disputed or unknown report is worth nothing. The reward is also **gated on the weakest verification level**:
the minimum acceptable `VerificationLevel`
(`Unverified`, `LatentAgreement`, `SampleVerified`, `FullSemantic`, `Deterministic`,
`crates/ptr-types/src/lib.rs:425-431`) is preregistered per task. The gate must
**not** read the combined `VerificationReport.level` of a `VerifierFabric`: the fabric
reports as its level "the strongest level any report reached" and documents that this
"is not an admission rule" (`crates/ptr-verifier/src/lib.rs:34-41`, computed with
`max_by_key` at `:81-85`). The reward is therefore gated on the **weakest** level
among all individual verifier reports for that output, or on an admission verdict
that records the weakest level (as the runtime's semantic grant does). An output
whose weakest level is below the minimum yields reward 0 regardless of status or
score. A hard finding
  (`Finding::hard`) forces `Fail` whatever the reported status or score. Arms: GRPO (clip-higher), GRPO with
  routing replay, GSPO, SAPO, REINFORCE++. Factors: mini-steps 1 versus 4 (at 1
  mini-step GRPO, GSPO and SAPO coincide, which is the null control among those
  three; REINFORCE++ uses a different advantage estimator, global batch
  normalisation, and is excluded from it) and same versus separate
  rollout engine (R3). At least 3 seeds. Collapse is defined before the runs.
- **Distillation control** with pass@k up to 256. HSD later (no code).
- **Routing alignment** (RA-MoE adapted to tasks) as an ablation. MTP deferred.
- **Prune plus recovery.** AIMER at 25% and 50%, short SFT, then GSPO/SAPO; quality
  against VRAM.
- **Routing metrics in every run:** per-layer expert churn (old versus new policy;
  rollout versus training), routing entropy, load coefficient of variation, dead
  experts, train-inference KL, clip fraction / soft-gate mass, crash step.

| Experiment | Question |
|---|---|
| M010 `moe-rl-stability` | Which RL arm keeps routing stable, by factor |
| M011 `routed-peft` | Does PERFT-R help OLMoE under our verifiers |
| M012 `lora-expert-balancing` | Aux-free bias vs aux loss vs φ-balancing |
| M013 `expert-lifecycle` | Does the gated lifecycle forget less than a static expert set at equal compute, and does reactivation beat a new expert. **Claim 11:** gated lifecycle versus a static expert set at equal training compute (FLOPs, where training_flops includes discarded candidates, the L3 policy and gate evaluation) and equal active parameters (per seed the lifecycle arm's final active parameters equal the static arm's within active_params_relative_tolerance = 0.01, pinned; active_param_mismatch_seeds above 0 invalidates the run); metrics: average forgetting (the `ForgettingGate` definition), task quality on the public suite, and final per-domain and average accuracy. **Claim 12:** reactivation versus training a new expert on the returning domain, in a **paired trial** per seed (a retired expert reactivated versus a freshly trained new expert on the same returning domain); metric: steps (or tokens) to reach a pinned recovery target (for example 95% of the expert's pre-retirement domain score). The net quality metric and the domain sequence are pinned and held out from the gate suite. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** fewer than 3 seeds; paired_reactivation_trials below 1 in any seed (Reactivate is in the claim 12 coverage; missing makes claim 12 vacuous); min_proposals_accepted_per_arm below 10 (preregistered) for any **proposal-generating arm** (full-context, context-ablated, heuristic); for the **static arm**, which generates no proposals, the matched-observation requirement instead: the same domain-shift schedule, the same evaluation stages, and training_flops within flops_relative_tolerance of the other arms. **The claim 11 run is also vacuous** if any of the operators Birth, Merge, Split, Freeze and Prune has fewer than 2 accepted and gate-processed proposals in any lifecycle arm (accepted_proposals_by_operator); the claim is not narrowed to the exercised operators (preregistered: vacuous). **Falsified if any of, in any seed (every seed must meet the criteria):** forgetting above the `ForgettingGate` bound; forgetting of the lifecycle arm not strictly below the static set's; quality more than a pinned margin (for example 1 point) below the static set; final per-domain or average accuracy of the lifecycle arm not non-inferior to the static arm within 1 point (preregistered); reactivation not reaching the recovery target in fewer steps than a new expert. It also carries the **context-ablation arm** below |
| M014 `aimer-prune-recovery` | Quality against VRAM after pruning and recovery |
| M015 `rlvr-vs-distill-passk` | RLVR vs distillation at large k |
| R011 `self-modification-safety` | The L3 policy cannot reach gates, thresholds or verifiers, **and cannot bypass the gate**. **The run is invalid (vacuous) and counts as falsified if any coverage prerequisite is unmet:** fewer than 1 attempt on any of these paths: appending `ExpertPromoted` / `ExpertRetired` directly to the ledger (a gate-driven `ExpertRetired`; the revocation-driven retirement under the `ExpertFenced` grant is excepted), mutating the serving registry or state, self-promoting; a stale-generation retirement; revoking an input after gate pass and before commit; revoking a training input of a serving expert and then attempting to route to it; the fence without a gate pass; the fault-injection case that crashes **between append and apply** of the compound `InputRevoked` event; the restart or replay arm; each evidence-binding case below; runtime_fence_with_grant_succeeded below 1 (a fence under the dedicated runtime revocation grant must succeed); stale_eval_input_commit_attempts below 1; inflight_expert_call_race_cases below 1 (an expert call that starts before the revocation and completes after it); post_promotion_eval_revocation_cases below 1 (an evaluation input revoked after promotion); evidence_binding_negative_cases_per_event_type below 1 for any event type on the list (Birth or promotion, Merge, Split, Prune, Freeze, Reactivate including unfreeze, proposal-driven Retire; each with a positive control); large_fanout_revocation_cases below 1 (a dependency fan-out whose explicit expert list would exceed `MAX_RECORD_BYTES`); `control_accepts` below 1 on any path, including the revocation-driven retirement under the runtime revocation grant (a well-formed event with the trusted grant and a current generation must be accepted on each path, so a pass is not vacuous). **Falsified if any of:** any attempt on the paths above succeeds without the trusted gate grant (the revocation fence and the revocation-driven retirement are **excluded** from this blanket gate-grant check and require instead the dedicated runtime revocation grant); l3_fence_attempts_admitted above 0; fence_without_runtime_grant_admitted above 0; stale_eval_input_commits_admitted above 0 (a commit while an evaluation input was revoked or superseded); routable_after_eval_revocation above 0; inflight_expert_outputs_after_revocation above 0; evidence_binding_admissions above 0 (any event type on the list); experts_serving_after_revocation above 0 in the large fan-out case; fence_set_digest_mismatches above 0; a stale-generation retirement is committed (the compare-and-set must reject it); a commit succeeds after an attested training input was revoked before commit; any route to a fenced expert at any time, including between append and apply and after a restart or replay (replay re-derives the fence set and refuses a mismatch); a fence needs a gate pass or can be issued by an L3 policy; any evidence-binding case is accepted: promotion_with_evidence_for_other_artifact, promotion_with_evidence_for_other_base_model, `promotion_with_untrusted_evaluator`, `promotion_missing_attested_manifest`, promotion_trainer_self_reported_manifest, promotion_with_eval_manifest_of_other_evaluation, promotion_with_gate_config_of_other_evaluation (each has a matching positive control) |

**M013 context-ablation arm** (claim 13, an arm of `experiments/model/M013`). L3 proposals are
generated with full self-observation context (routing traces, load, churn,
interference, AIMER) versus the same policy with that context removed, versus a
fixed heuristic baseline (threshold rules on load and overlap). Metrics: the
fraction of proposals that pass the gates, and the net quality change per accepted
proposal. The test is **joint**, because a conservative policy can pass the gates
more often while gaining nothing. Falsified if the full-context policy does not beat
both the context-ablated policy and the heuristic baseline on gate pass rate by a
pre-registered margin, **or** if its mean net quality change per accepted proposal is
less than or equal to 0, **or** is not at least the baseline plus the pinned margin for both baselines, in any
seed (at least 3 seeds are run; every seed must meet the criteria). Failure of **any one** of these conditions falsifies
the claim. R011 is a different question (can the
policy reach gates at all) and does not test this claim.

**Preregistered pins common to M010-M015.** Every seed must meet the criteria. Each
arm gets an equal tuning budget (preregistered), or its hyperparameters are pinned:
learning rate, batch size, clip epsilons, SAPO tau_pos / tau_neg, GSPO range, aux
coefficient, bias update rate, LoRA rank and alpha. M010:
policy_ratio_variance_per_token is the variance over all tokens of a rollout batch,
averaged over steps; `max_rl_steps` counts optimizer steps. Rewards are in [0, 1] by
construction (verifiable reward). **Collapse** means the smoothed reward is at most
0.5 x the smoothed running maximum while the running maximum is above 0, or a
non-finite loss; while the running maximum is 0 only a non-finite loss counts. The
smoothing window is 20 steps (preregistered). M011: emergent specialisation is
non-inferior within 1 point, and mean task accuracy is the macro average over
domains. M012: `load_cv` is computed per layer (population standard deviation),
averaged over layers, at the final step. The pinned balance criteria, per arm and per layer on a pinned eval set at the final step, are: final `load_cv` at most 0.5; the maximum expert token fraction at most 2/E (E is the number of experts in that layer); `dead_experts == 0` (zero routed tokens). Failing any one is an arm failure. The hypothesis's ranking claim between arms is falsified if it is violated by more than 0.05 `load_cv`. `max_violation` is **reported only**. M014: the claim includes
AIMER against baselines; two recovery sequences are pinned, **SFT then GSPO** and
**SFT then SAPO**; the run requires the full method x fraction x optimizer x seed
matrix, and **a missing cell falsifies**; pruned experts are physically removed, and
the quality metric and suite are pinned; an unpruned control with an identical recovery budget is added, `quality_retention` is measured against it, and both SFT-then-GSPO and SFT-then-SAPO must meet all criteria. M015: the teacher is pinned as a Hugging Face
model id plus revision sha, and the problem set as dataset id, revision sha and split
plus the selection rule (the full GSM8K test split, 1319 problems, at a pinned sha, or a pinned set); the criteria use paired bootstrap 95% confidence intervals over problems (10000 resamples, preregistered); temperature, top-p, the
distillation data and held-out status are pinned; compute matching is on training
FLOPs with `compute_relative_tolerance = 0.01` (preregistered), training_flops is
declared per arm, and a mismatch makes the run invalid.

## 6. Rust binding

`bindings/ptr-py` is **merged** (#123): a PyO3/maturin crate `ptr_py` in its own
workspace, excluded from the root workspace like `model/burn-a0` (root `Cargo.toml`
`exclude` list at `Cargo.toml:7`: `bindings/ptr-py`, `fuzz`, `model/burn-a0`,
`vendor/*`). Its API table is in `bindings/ptr-py/README.md:23`. It exposes
`aimer_score`, `rank_experts_for_pruning`, `subspace_overlap`, `chance_overlap`,
`measure_interference`, `ties_merge` and `forgetting_gate_evaluate` (the
`ForgettingGate` evaluation). #124 calls `aimer_score`, `rank_experts_for_pruning`, `subspace_overlap` and `chance_overlap` for reporting only; no code calls `forgetting_gate_evaluate`, and no lifecycle decision depends on a gate result. Python
is to call it at every lifecycle gate; the import is to be guarded so the CPU smoke
pipeline still runs without the extension, reporting the gate as not evaluated rather
than passed. The binding
only calls the gate: it does not make the gate's result trustworthy. A promotion
(and every other structural event) must rest on the complete trusted evidence contract of §3: artifact digest and
generation of the evaluated expert, the **evaluated base-model identity, generation
and digest** (or the complete evaluated execution-manifest digest), the
evaluation-data manifest digest, the gate configuration digest, the evaluator
identity, and the **orchestrator-attested consumed-input (training) manifest**,
produced by a trusted evaluator and not by the proposer's sandbox, because the gate
binds its verdict only to the `AdapterId` it is given.

## 7. Verified literature

Sources were checked; corrections to common readings are in the notes.

### Reinforcement learning

| Work | arXiv | Note |
|---|---|---|
| GRPO (DeepSeekMath) | 2402.03300 | The original already found RL improves Maj@K, not Pass@K |
| GSPO (Qwen) | 2507.18071 | Sequence-level ratio and clipping; about 10% of activated experts change per update on Qwen3-30B-A3B. "Removes the need for routing replay" is overstated: in R3 (2510.11370) GSPO without replay crashed. Curves only, no seeds |
| SAPO (Qwen) | 2511.20347 | Sigmoid soft gate; τ_neg > τ_pos required; baseline was GRPO with routing replay; all methods eventually unstable, SAPO later |
| R3 Rollout Routing Replay (Xiaomi) | 2510.11370 | Replays inference-engine expert choices in training |
| REINFORCE++ | 2501.03262 | Critic-free, global normalisation; "more stable than GRPO" only in small or extreme setups, about equal in Yue et al. |
| Yue et al., RLVR vs distillation | 2504.13837 | RLVR raises pass@1, not large-k pass@k; the distillation claim rests on one comparison |
| ProRL | 2505.24864 | Counter-evidence: boundary expansion where the base is weak |
| HSD (GWU) | 2606.15576 | Single preprint, no code, no seeds, traces up to 4k, not tested on MoE |
| MTP: Gloeckle et al. | 2404.19737 | Hurts at 0.3B / 0.6B |
| MTP: DeepSeek-V3 | 2412.19437 | |
| MTP curriculum | 2505.22757 | Tested at 1.3B / 3B; stays below plain NTP on NTP benchmarks |

### Mixture of experts

| Work | arXiv | Note |
|---|---|---|
| PERFT-R, "Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules" | 2508.02587 | Predecessor PERFT 2411.08212; up to +17.2% on OLMoE commonsense |
| MixLoRA | 2404.15159 | Makes a dense model an MoE; about 2x latency |
| Mixture of Routers | 2503.23362 | +0.4 to 1.2 points; not tested on real pretrained MoEs |
| RA-MoE | 2605.28306 | About **multilingual** adaptation |
| Aux-loss-free balancing (DeepSeek) | 2408.15664 | Bias only for top-k selection; u = 1e-3 sign update |
| Aux-free theory | 2512.03915 | First posted December 2025, not 2026; regret bound only with a diminishing step; its own experiments show slightly **worse** balance than aux loss |
| φ-Balancing | 2605.15403 | |
| CL theory | 2406.16437 | ICLR 2025; per-task routing, terminate gate updates, more experts only delay convergence |
| DynMoE | 2405.14297 | Lifecycle piece |
| Lifelong-MoE | 2305.12281 | Lifecycle piece |
| MC-SMoE | 2310.01334 | Lifecycle piece |
| Sparse upcycling | 2212.05055 | Lifecycle piece |
| Expert upcycling | 2604.19835 | Lifecycle piece |
| BTX | 2403.07816 | Lifecycle piece |
| EMO | 2605.13247 | Lifecycle piece |
| AIMER | 2603.18492 | Weights-only score ‖w‖₁/(√N‖w‖₂); higher score is pruned first; seconds instead of hours |
| OLMoE-1B-7B | 2409.02060 | 7B total / 1B active; 12.9 GB in bf16; the only fully open small MoE: weights, data, code, logs |

No paper implements the full lifecycle, and "reactivate" has no literature.

### Self-referential and model-as-expert

| Work | arXiv | Note |
|---|---|---|
| Self-MoE | 2406.12034 | |
| Mixture-of-Agents | 2406.04692 | |
| Router-R1 | 2506.09033 | |
| RouteLLM | 2406.18665 | |
| BTM | 2208.03306 | |
| Darwin Gödel Machine | 2505.22954 | Modifies code / scaffolding, not MoE weights |
| Gödel Agent | 2410.04444 | Modifies code / scaffolding, not MoE weights |
| AlphaEvolve | 2506.13131 | Modifies code / scaffolding, not MoE weights |

## 8. Diagram

```mermaid
flowchart TB
  subgraph L1["L1 token, in-process"]
    BASE["Frozen OLMoE experts"] --- LORA["Routed LoRA experts (PERFT-R)"]
    LORA --- R1["top-k router + aux-free bias"]
  end
  subgraph L2["L2 request, PodWire"]
    R2["ptr-router (learned)"] --> PODS["Model / adapter pods"]
  end
  subgraph L3["L3 meta"]
    POL["Policy pod: Birth, Merge, Split, Freeze, Reactivate, Prune, Retire"]
  end
  L1 --> OBS["SemDB observations: traces, load, churn, AIMER (derived)"]
  L2 --> OBS
  OBS -->|context only| POL
  POL -->|ExpertProposal| BR["ptr-branch speculative overlay"]
  BR --> TR["Sandbox training: candidate"]
  TR --> GATES["ForgettingGate, interference vs calibrated null threshold, verifier, public suites"]
  GATES -->|pass| LEDGER["Ledger event: ExpertPromoted / ExpertRetired (not yet implemented)"]
  GATES -.->|fail: held| BR
  LEDGER --> L1
  LEDGER --> L2
```

The policy has no edge into `GATES`: it only proposes.

## 9. Risks

- The first increment runs only a CPU smoke pipeline on a tiny random OLMoE, so
  there are no quality claims until GPU runs exist.
- TRL API version drift: the training scripts must pin versions and the smoke test
  must fail loudly on a signature change.
- `make py-bindings` builds `bindings/ptr-py` locally, but no workflow under
  `.github/workflows` does, so a break in the binding is invisible to CI until a job
  is added.
- The L3 policy must never gain write access to gates, thresholds or the verifier.
  This is the claim R011 tries to falsify, and it holds only as long as
  `ExpertProposal` stays free of such parameters.
