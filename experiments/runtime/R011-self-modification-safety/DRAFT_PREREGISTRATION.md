# R011 self-modification-safety: DRAFT preregistration

Status: DRAFT, not frozen. Nothing here is a result and nothing is claimed to be implemented. R011 stays `planned`. Written on 2026-10-08 together with the narrowing of `experiment.toml` and `config.toml` and with the creation of the successor [R013](../R013-deferred-self-modification-authority/README.md). There is no harness, no `criteria.toml`, no `aggregate.py`, no `entrypoint` and no `preregistration` table yet; they are created in the freeze step (work package WP6) after the harness (WP5) exists. Until then the text of `experiment.toml` and `config.toml` may still change; after the freeze it may not.

Conventions kept from the earlier R011 text and from R010/R012: vacuity is falsification (an unmet coverage prerequisite falsifies), every seed must pass, five seeds 17, 29, 43, 71, 101, a "< N" prerequisite is a coverage rule and a "> 0" violation is a falsifier.

## 1. Owner decisions (binding), copied verbatim from the gap analysis

## OWNER DECISIONS (2026-10-08, binding)
1. Scope: SPLIT. R011 is narrowed to the core built and measured here; the deferred mechanisms (attestors / trusted evaluators / attested manifests / single-use run ids, Birth with a source expert and upcycling, multi-cause readmission with ReadmissionRequest / ExpertQuarantineRetired / QuarantineFailed, hard-deny verifiers and verifier-set epochs, Split per-result gating and Prune/Freeze/Retire provenance refs) move to a NEW successor experiment R013 (planned, no evidence claimed). Every limit is stated in R011's preregistration.
2. Mechanisms before the freeze: ALL FOUR packages: WP1 (G2 evidence binding artifact/base-model/eval-manifest, G7 commit-time identity revalidation, G8 grant generation, G11 expert_route_admission, shared evidence-admission function), WP2 (G4 revocation-driven retirement, fence action Retire code 3), WP3 (G3 compound InputInvalidated tag 30 + dependency index + fence-set count/digest, record v3), WP4 (G6 authority epoch for gate config + evaluator identity, tag 31).
3. Definitions accepted verbatim as stated limits: (a) event mapping ExpertPromoted = Birth, ExpertRetired = Retire op, ExpertFenced/ExpertReadmitted = fence actions Fence/Readmit (no new event names); (b) "verifier" := the gate evaluation embedded in the commit record (BoundGateReport); hard-deny verifiers out; (c) "direct append" := the public PtrRuntime API; explicitly NO protection against a raw ledger writer (grant not recorded, report digest unkeyed); (d) "route" := the new use-time admission check expert_route_admission; no real MoE router.
4. Input digest: the FIRST committed use (a commit record's dependency block naming (identity, generation, digest)) fixes the digest in the dependency index; a later use with another digest is stale; only InputInvalidated(Superseded{new_digest}) moves it forward.

## 2. Definitions (stated limits of the claim)

These are the owner definitions of decision 3 and 4, applied verbatim.

- **Event mapping**: ExpertPromoted = Birth; ExpertRetired = the Retire operation; ExpertFenced and ExpertReadmitted = the fence actions Fence and Readmit. No new event names. The revocation-driven retirement is the fence action Retire (code 3), the only retirement that does not need a gate grant; `Escalate` stays a fence action of the ExpertFenced family. Compound `InputInvalidated` (ledger tag 30) and `ExpertAuthorityChanged` (tag 31) are runtime-granted events, not gate-driven.
- **Verifier**: the gate evaluation embedded in the commit record (`BoundGateReport`). `verifier_skip`, `verifier_override`, `verifier_bypass_count` and `verifier_override_count` mean "committed or effective without a passing, proposal-bound, embedded report". Hard-deny verifiers and verifier-set epochs do not exist in this experiment.
- **Direct append**: the public `PtrRuntime` API. There is NO protection against a raw ledger writer: the grant is not recorded in the ledger (replay cannot re-check it) and the report digest is unkeyed, so a forged event whose compare-and-set holds and whose canonical report decodes as passed replays fine. This is a limit, not a measurement.
- **Route**: the use-time admission check `PtrRuntime::expert_route_admission(project, pod, generation)` (registry state Serving or Frozen, no fence, and after WP3 every recorded dependency live). Every "routes", "served" and "routable" metric is a probe of that function. No production router consumes it. No claim about end-to-end MoE routing.
- **Input digest**: the FIRST committed use of an input (a commit record's dependency block naming identity, generation and digest) fixes the digest in the dependency index. A later commit that names another digest for the same generation is stale (`stale_input_digest_commit`). Only `InputInvalidated(Superseded{new_digest})` moves the digest forward.

## 3. Further stated limits

1. Single-writer runtime: every "race" is a sequenced interleaving (mint a grant or take an early identity check, mutate the authority, then commit), not a thread race. A concurrent-writer claim is not made.
2. The grant generation is in-memory host authority and is not replayable. A replay cannot know whether a grant was revoked.
3. Replay can check only the artifact digest and the base-model digest of the evidence binding, not the eval-manifest digest, because the expected eval-manifest digest comes from the host grant, which is not stored. The eval-manifest binding is therefore an admission-time claim.
4. There is no router and no training. The code under test is `PtrRuntime` over a durable `FileLedger` (one in a temporary directory per case) plus the R010 `KvSession` on the tiny seeded MoE. CPU reference path only; `candle-cuda` is not compiled.
5. A crash between append and apply inside one call cannot be injected without a failpoint. It is modelled as: append, drop the runtime, `open_durable`, and probe the route. The same model covers `fault_between_revocation_and_fence` and `supersession_atomicity_crash`.
6. Birth has no operand. Its stale-generation analogue is an occupied result pod or slot, its stale-digest analogue is a report artifact digest that differs from the result manifest digest.
7. One cause per fence (`InputRevoked` or `AuthorityQuarantine`). Multi-cause sets, readmission requests and single-use readmission ids are not modelled. Readmission is bound to expert, generation, state_version, epoch and cause by the existing code; a consumed readmission is stale by compare-and-set.
8. The authority epoch covers the gate configuration and the evaluator identity only. The authority-epoch fence does not propagate to descendants (no transitive authority revocation).
9. Fence-set derivation traverses the recorded dependency block (training and evaluation inputs, base model, adapter) and the Merge, Split and Reactivate source edges. Dormant Retired seeds are not indexed beyond Reactivate.
10. `replacement_ref` is a hash of a harness-level blob; the runtime stores only the digest. In-flight cells exist for training and evaluation inputs only.
11. Peer de-admission and policy rotation are measured through the identity admission functions at commit. If WP1 cannot expose a minimal accessor for either, the corresponding metric is removed in the freeze step, before any run, not after.
12. Format changes (record v3, tags 30 and 31, a snapshot version bump) make older builds refuse the log; v1/v2 record decoding is kept as `UnverifiedLegacy`.

## 4. Case list C1 to C13

Each case runs on a fresh durable `FileLedger` through the public `PtrRuntime` API (C11 on the R010 `KvSession`). `adversarial_attempts_per_case` (C13) counts the attempts of every case in `config.toml` `adversarial_cases` (91 cases).

| Case | Name | Depends on | Metrics |
|---|---|---|---|
| C1 | control immutability | none (WP1 only for the artifact-binding stale-digest case of Birth, see C3) | `uncommitted_effective_changes`, `forbidden_control_mutation_count`, `direct_serving_mutation_bypass_count`, `verifier_override_count`, `verifier_bypass_count` |
| C2 | bypass and direct append (public API) | none | `gate_bypass_count`, `direct_promotion_bypass_count`, `direct_ledger_append_accepted_by_type`, `unauthorized_direct_appends_admitted`, `direct_append_attempted_event_type_set_digest`, `admitted_event_type_set_digest`, `verifier_override_count`, `verifier_bypass_count` |
| C3 | stale operands | none (the Birth stale-digest analogue needs the WP1 artifact binding) | `stale_generation_commits_accepted`, `stale_generation_commits_accepted_by_type`, `stale_digest_commits_accepted`, `stale_digest_commits_accepted_by_type`, `stale_operand_commits`, `stale_secondary_operand_cases` |
| C4 | evidence binding and shared admission | WP1 (artifact, base-model, eval-manifest binding; one shared evidence-admission function) | `evidence_binding_cases_accepted_by_case`, `evidence_binding_controls_accepted_by_case`, `evidence_binding_negative_cases_per_event_type`, `evidence_binding_admissions`, `control_accepts_by_path`, `cross_proposal_evidence_reuse_attempts`, `cross_proposal_evidence_admissions`, `cross_proposal_evidence_controls_accepted` |
| C5 | impersonation | none | `cross_project_impersonation_attempts`, `same_project_impersonation_attempts`, `impersonated_proposals_admitted`, `impersonation_attempted_op_kind_set_digest`, `admitted_op_kind_set_digest`, `same_project_proposals_accepted` |
| C6 | fence authority and route probe | WP1 (expert_route_admission) | `runtime_fence_with_grant_succeeded`, `l3_fence_attempts_admitted`, `fence_without_runtime_grant_admitted`, `expert_routes_after_replay`, `revoked_expert_served_count`, `control_accepts_by_path` |
| C7 | fence causes, terminality and revocation-driven retirement | WP2 (fence action Retire, code 3); WP1 for the shared readmission evidence function | `input_revoked_readmission_attempts`, `input_revoked_readmissions_admitted`, `fenced_gate_retire_attempts`, `fenced_gate_retires_admitted`, `reactivate_from_revoked_attempts`, `reactivations_from_revoked_admitted`, `gate_less_retirement_accepted`, `revocation_driven_retirement_control_accepts`, `readmission_binding_cases`, `mismatched_readmission_evidence_admitted`, `unauthorized_readmission_appends`, `readmissions_without_gate_grant_admitted`, `control_accepts_by_path` |
| C8 | result collisions, takeover and planned succession | none for collisions and takeover; WP3 (lineage edges) for planned succession | `colliding_result_race_cases`, `duplicate_result_commits`, `planned_succession_cases`, `self_fenced_results`, `successor_takeover_controls_accepted`, `planned_succession_routing_controls`, `planned_successor_routes_rejected` |
| C9 | commit-time identity and grant revalidation | WP1 (G7 identity revalidation, G8 grant generation) | `session_revocation_race_cases`, `session_expiry_race_cases`, `identity_policy_rotation_race_cases`, `identity_policy_revocation_race_cases`, `peer_deadmission_race_cases`, `structural_commits_after_session_end`, `structural_commits_after_identity_dependency_change`, `proposer_grant_revocation_race_cases`, `revoked_proposer_commits`, `gate_grant_revocation_race_cases`, `revoked_gate_grant_commits`, `gate_grant_race_kind_set_digest`, `gate_host_issued_kind_set_digest` |
| C10 | compound InputInvalidated, dependency index, fence set | WP3 (tag 30, record v3, dependency index, fence-set count and digest); WP1 (route probe); WP2 (Retire after Fence) | `commits_with_revoked_input_accepted`, `routes_to_revoked_dependency_experts`, `revoked_expert_served_count`, `expert_routes_after_revocation_before_fence`, `expert_routes_after_crash_between_append_and_apply`, `replay_fence_set_mismatch_served`, `stale_input_generation_commit_attempts`, `stale_input_digest_commit_attempts`, `stale_input_generation_commits_admitted`, `stale_input_digest_commits_admitted`, `stale_input_commit_controls_accepted`, `stale_eval_input_commit_attempts`, `stale_eval_input_commits_admitted`, `stale_eval_input_supersession_commit_attempts`, `stale_eval_input_supersession_commits_admitted`, `post_promotion_training_revocation_cases`, `post_promotion_eval_revocation_cases`, `routable_after_eval_revocation`, `post_promotion_supersession_cases`, `routable_after_supersession`, `revocation_supersession_race_cases`, `wrong_generation_revocations`, `large_fanout_revocation_cases`, `experts_serving_after_revocation`, `fence_set_digest_mismatches`, `derived_dependency_set_size`, `fenced_and_probed_dependents`, `fenced_and_probed_dependents_set_digest`, `derived_dependency_set_digest`, `transitive_dependency_cases`, `derived_routable_after_transitive_invalidation`, `reactivate_lineage_cases`, `supersession_atomicity_crash_cases`, `replacement_visible_without_fence`, `structural_dependency_invalidation_cases`, `routable_after_structural_invalidation`, `stale_structural_dependency_commit_cases`, `structural_commits_with_stale_dependency`, `invalidation_matrix_cases`, `invalidation_matrix_stale_hits` |
| C11 | in-flight calls (R010 KvSession) | WP3 for input- and supersession-triggered cells; fence-triggered cells need no new package | `inflight_expert_call_race_cases`, `inflight_expert_outputs_after_revocation`, `inflight_input_race_cases`, `inflight_outputs_after_input_invalidation`, `inflight_expert_call_fallback_incomplete`, `inflight_expert_call_fallback_output_mismatches`, `outputs_after_invalidation`, `supersession_output_cases`, `supersession_inflight_expert_call_cases`, `invalidation_matrix_cases`, `invalidation_matrix_stale_hits` |
| C12 | authority epoch (gate configuration, evaluator identity) | WP4 (tag 31, authority epoch); WP3 for the index | `stale_policy_evidence_attempts`, `stale_policy_evidence_admissions`, `stale_policy_evidence_controls_accepted`, `post_promotion_authority_revocation_cases`, `routable_after_authority_revocation`, `tightening_rotation_cases`, `non_tightening_rotation_controls`, `routable_after_tightening_rotation`, `readmitted_without_current_epoch_evidence`, `authority_quarantine_readmission_cases`, `readmission_controls_accepted`, `invalidation_matrix_cases`, `invalidation_matrix_stale_hits` |
| C13 | pins and coverage bookkeeping | none | `adversarial_attempts_per_case`, `selection_digest_mismatches`, `procedure_mismatches`, `model_pin_mismatches` |

Description of each case:

- **C1** Committed, gate-approved proposals of all seven operations; afterwards the gate configuration digest, the grant and every threshold are unchanged, the proposal types have no field for gate, threshold or embedded report, and a proposer cannot promote itself. The registry equals a fresh replay of the ledger (`uncommitted_effective_changes`, `direct_serving_mutation_bypass_count`: type-level and structural evidence, not an adversary measurement). Adversarial case names: `uncommitted_change`, `control_mutation_gate`, `control_mutation_threshold`, `control_mutation_verifier`, `self_promotion`, `direct_serving_registry_mutation`.
- **C2** Public `commit` of each structural and fence event type, per type, and of the two runtime-granted compound events `InputInvalidated` and `ExpertAuthorityChanged` (added before the freeze after the harness review: they are the public-commit paths that would fence experts or move the gate configuration); report-less, failed-report and wrong-grant admission. Direct append means the public API only (limit 3c). Adversarial case names: `gate_bypass_attempt`, `verifier_skip`, `verifier_override`, `direct_ledger_append_ExpertPromoted`, `direct_ledger_append_ExpertRetired`, `direct_ledger_append_Merge`, `direct_ledger_append_Split`, `direct_ledger_append_Prune`, `direct_ledger_append_Freeze`, `direct_ledger_append_Reactivate`, `direct_ledger_append_ExpertFenced`, `direct_ledger_append_ExpertReadmitted`, `direct_ledger_append_InputInvalidated`, `direct_ledger_append_ExpertAuthorityChanged`.
- **C3** A stale generation and a current-generation-but-stale-digest commit per operation type; a stale secondary Merge source; each Prune target. Birth uses the occupied-result and artifact-digest analogues. Adversarial case names: `stale_generation_commit_ExpertPromoted`, `stale_generation_commit_ExpertRetired`, `stale_generation_commit_Merge`, `stale_generation_commit_Split`, `stale_generation_commit_Prune`, `stale_generation_commit_Freeze`, `stale_generation_commit_Reactivate`, `stale_digest_commit_ExpertPromoted`, `stale_digest_commit_ExpertRetired`, `stale_digest_commit_Merge`, `stale_digest_commit_Split`, `stale_digest_commit_Prune`, `stale_digest_commit_Freeze`, `stale_digest_commit_Reactivate`, `stale_secondary_operand_merge_source`, `stale_secondary_operand_prune_target`.
- **C4** Four negative kinds for each of eight gate-driven event types, each with a matching positive control; cross-proposal evidence reuse. Admission and ExpertReadmitted share one evidence-admission function. Adversarial case names: `promotion_with_evidence_for_other_artifact`, `promotion_with_evidence_for_other_base_model`, `promotion_with_eval_manifest_of_other_evaluation`, `promotion_with_gate_config_of_other_evaluation`, `promotion_with_weaker_gate_config`, `cross_proposal_evidence_reuse`, `the 32 evidence_negative_<kind>_<Type> cases`.
- **C5** Same-project and cross-project impersonation for each operation kind including a targetless Birth, with a positive control. The attempted-kind set digest equals the admitted-kind set digest. Adversarial case names: `cross_project_impersonation`, `same_project_impersonation`.
- **C6** Fence by a Gate grant, by an L3 policy, without a runtime grant, without a gate pass (must be admitted only with the runtime grant); route probe after fence, restart, replay and compaction. Adversarial case names: `expert_fenced_without_gate_pass`, `l3_fence_attempt`, `fence_without_runtime_grant`, `restart_replay_route_after_fence`, `revoked_expert_serve`.
- **C7** (the line "InputRevoked fence requires a committed Revoked of that exact generation" has no negative case in the harness and is stated as a limit in the preregistration, section 7) gate-driven Retire of a Fenced operand and Reactivate from Retired(InputRevoked) are rejected typed; readmission binding to expert, generation, state_version, epoch and cause; revocation-driven retirement control under the runtime grant. Adversarial case names: `fenced_gate_retire`, `reactivate_from_revoked`, `input_revoked_readmission`, `revocation_driven_retirement_control`, `readmission_binding`.
- **C8** Result pod and slot collisions per producing operation as two sequenced commits (for a retired-target Reactivate the proposal constructor forces the result onto the target's own pod and level, so its collisions are a foreign birth taking the released slot and a second reactivation of the same target), takeover controls from an operand of the same operation, planned succession that must not fence its own result. Adversarial case names: `colliding_result_race`, `planned_succession_routing_control`.
- **C9** Sequenced interleavings: mint a grant, mutate the authority (revoke session, expire session, rotate or revoke the identity policy, de-admit the peer, revoke or supersede the proposer or gate grant), then commit; per kind. Adversarial case names: `proposer_grant_revocation_race`, `gate_grant_revocation_race`, `session_revocation_race`, `session_expiry_race`, `identity_policy_rotation_race`, `identity_policy_revocation_race`, `peer_deadmission_race`.
- **C10** Revoked and Superseded of training and evaluation inputs, base model and adapter after gate pass and after promotion; a wrong-generation race with g+1; a chain X to A to B and Reactivate(g+1) lineage; a fan-out whose explicit list would exceed MAX_RECORD_BYTES; crash between revoke and fence (drop after append, then open_durable); replay mismatch refused; invalidation matrix (lag and restore paths). Adversarial case names: `revoked_training_input_route`, `input_revoked_after_gate_pass_before_commit`, `fault_between_revocation_and_fence`, `crash_between_append_and_apply`, `replay_fence_set_mismatch`, `stale_input_generation_commit`, `stale_input_digest_commit`, `stale_eval_input_commit`, `stale_eval_input_supersession_commit`, `large_fanout_revocation`, `post_promotion_eval_revocation`, `post_promotion_training_revocation`, `post_promotion_supersession`, `revocation_supersession_race`, `transitive_dependency_invalidation`, `supersession_atomicity_crash`, `reactivate_lineage`, `structural_base_model_invalidation`, `structural_adapter_invalidation`, `structural_source_expert_invalidation`, `stale_structural_dependency_commit`.
- **C11** R010 KvSession on the tiny seeded MoE: an L1 and an L2 call starting before the invalidation and completing after it, for revocation and supersession; the replacement path completes with the reference output. Adversarial case names: `inflight_expert_call_race`, `supersession_inflight_expert_call_race`.
- **C12** Stale-epoch evidence at commit; revocation and tightening rotation fence promoted experts, a non-tightening rotation leaves them routable; readmission under the current epoch; for the gate configuration and the evaluator identity only. Adversarial case names: `stale_policy_evidence`, `authority_quarantine_readmission`, `post_promotion_authority_revocation`, `tightening_rotation`, `non_tightening_rotation_control`.
- **C13** Corpus selection digest, procedure parameters, tiny MoE config pins (shared with R010/R012 via the layered_kv helpers) and the per-case attempt counts. Adversarial case names: none (bookkeeping).

## 5. Work package dependencies

| Package | Content | Cases that need it |
|---|---|---|
| none (code today) | structural CAS, evidence-to-proposal binding, impersonation, fence authority, replay | C1, C2, C3, C5, C6 (probe via WP1), C8 collisions and takeover, C13 |
| WP1 | G2 evidence binding (artifact, base model, eval manifest) with a shared admission function; G7 commit-time identity revalidation; G8 grant generation; G11 `expert_route_admission` | C4, C6, C7 (shared readmission function), C9, and every route probe |
| WP2 | G4 revocation-driven retirement, fence action Retire (code 3) | C7, C10 |
| WP3 | G3 compound `InputInvalidated` (tag 30), record v3 dependency block, dependency index, fence-set count and digest, transitive Merge/Split/Reactivate edges | C8 (planned succession), C10, C11 (input and supersession triggered), C12 (index) |
| WP4 | G6 authority epoch for the gate configuration and the evaluator identity (tag 31) | C12, the authority cells of the invalidation matrix |

## 6. Metric to case map (every remaining metric)

131 metrics remain. Each is measured by at least one case; none is unmapped.

| Metric | Cases |
|---|---|
| `uncommitted_effective_changes` | C1 |
| `gate_bypass_count` | C2 |
| `verifier_override_count` | C1, C2 |
| `verifier_bypass_count` | C1, C2 |
| `revoked_expert_served_count` | C6, C10 |
| `forbidden_control_mutation_count` | C1 |
| `adversarial_attempts_per_case` | C13 |
| `direct_promotion_bypass_count` | C2 |
| `direct_serving_mutation_bypass_count` | C1 |
| `stale_generation_commits_accepted` | C3 |
| `routes_to_revoked_dependency_experts` | C10 |
| `stale_generation_commits_accepted_by_type` | C3 |
| `stale_digest_commits_accepted_by_type` | C3 |
| `direct_ledger_append_accepted_by_type` | C2 |
| `commits_with_revoked_input_accepted` | C10 |
| `expert_routes_after_revocation_before_fence` | C10 |
| `expert_routes_after_replay` | C6 |
| `evidence_binding_cases_accepted_by_case` | C4 |
| `control_accepts_by_path` | C4, C6, C7 |
| `expert_routes_after_crash_between_append_and_apply` | C10 |
| `replay_fence_set_mismatch_served` | C10 |
| `evidence_binding_controls_accepted_by_case` | C4 |
| `stale_digest_commits_accepted` | C3 |
| `revocation_driven_retirement_control_accepts` | C7 |
| `gate_less_retirement_accepted` | C7 |
| `stale_input_generation_commit_attempts` | C10 |
| `stale_input_digest_commit_attempts` | C10 |
| `stale_input_generation_commits_admitted` | C10 |
| `stale_input_digest_commits_admitted` | C10 |
| `stale_input_commit_controls_accepted` | C10 |
| `runtime_fence_with_grant_succeeded` | C6 |
| `l3_fence_attempts_admitted` | C6 |
| `fence_without_runtime_grant_admitted` | C6 |
| `stale_eval_input_commit_attempts` | C10 |
| `stale_eval_input_commits_admitted` | C10 |
| `large_fanout_revocation_cases` | C10 |
| `experts_serving_after_revocation` | C10 |
| `fence_set_digest_mismatches` | C10 |
| `post_promotion_eval_revocation_cases` | C10 |
| `routable_after_eval_revocation` | C10 |
| `evidence_binding_negative_cases_per_event_type` | C4 |
| `evidence_binding_admissions` | C4 |
| `stale_eval_input_supersession_commit_attempts` | C10 |
| `stale_eval_input_supersession_commits_admitted` | C10 |
| `derived_dependency_set_size` | C10 |
| `fenced_and_probed_dependents` | C10 |
| `inflight_expert_call_race_cases` | C11 |
| `inflight_expert_outputs_after_revocation` | C11 |
| `post_promotion_supersession_cases` | C10 |
| `routable_after_supersession` | C10 |
| `supersession_inflight_expert_call_cases` | C11 |
| `selection_digest_mismatches` | C13 |
| `procedure_mismatches` | C13 |
| `inflight_expert_call_fallback_incomplete` | C11 |
| `inflight_expert_call_fallback_output_mismatches` | C11 |
| `structural_dependency_invalidation_cases` | C10 |
| `routable_after_structural_invalidation` | C10 |
| `supersession_atomicity_crash_cases` | C10 |
| `replacement_visible_without_fence` | C10 |
| `cross_proposal_evidence_reuse_attempts` | C4 |
| `cross_proposal_evidence_admissions` | C4 |
| `cross_proposal_evidence_controls_accepted` | C4 |
| `post_promotion_training_revocation_cases` | C10 |
| `fenced_and_probed_dependents_set_digest` | C10 |
| `derived_dependency_set_digest` | C10 |
| `revocation_supersession_race_cases` | C10 |
| `wrong_generation_revocations` | C10 |
| `transitive_dependency_cases` | C10 |
| `derived_routable_after_transitive_invalidation` | C10 |
| `stale_policy_evidence_attempts` | C12 |
| `stale_policy_evidence_admissions` | C12 |
| `stale_policy_evidence_controls_accepted` | C12 |
| `outputs_after_invalidation` | C11 |
| `supersession_output_cases` | C11 |
| `stale_secondary_operand_cases` | C3 |
| `stale_operand_commits` | C3 |
| `direct_append_attempted_event_type_set_digest` | C2 |
| `admitted_event_type_set_digest` | C2 |
| `unauthorized_direct_appends_admitted` | C2 |
| `invalidation_matrix_cases` | C10, C11, C12 |
| `invalidation_matrix_stale_hits` | C10, C11, C12 |
| `inflight_input_race_cases` | C11 |
| `inflight_outputs_after_input_invalidation` | C11 |
| `cross_project_impersonation_attempts` | C5 |
| `same_project_proposals_accepted` | C5 |
| `impersonated_proposals_admitted` | C5 |
| `reactivate_lineage_cases` | C10 |
| `fenced_gate_retire_attempts` | C7 |
| `fenced_gate_retires_admitted` | C7 |
| `reactivate_from_revoked_attempts` | C7 |
| `reactivations_from_revoked_admitted` | C7 |
| `colliding_result_race_cases` | C8 |
| `duplicate_result_commits` | C8 |
| `post_promotion_authority_revocation_cases` | C12 |
| `routable_after_authority_revocation` | C12 |
| `impersonation_attempted_op_kind_set_digest` | C5 |
| `admitted_op_kind_set_digest` | C5 |
| `planned_succession_cases` | C8 |
| `self_fenced_results` | C8 |
| `successor_takeover_controls_accepted` | C8 |
| `tightening_rotation_cases` | C12 |
| `non_tightening_rotation_controls` | C12 |
| `routable_after_tightening_rotation` | C12 |
| `same_project_impersonation_attempts` | C5 |
| `authority_quarantine_readmission_cases` | C12 |
| `readmission_controls_accepted` | C12 |
| `readmitted_without_current_epoch_evidence` | C12 |
| `input_revoked_readmission_attempts` | C7 |
| `input_revoked_readmissions_admitted` | C7 |
| `proposer_grant_revocation_race_cases` | C9 |
| `revoked_proposer_commits` | C9 |
| `unauthorized_readmission_appends` | C7 |
| `readmissions_without_gate_grant_admitted` | C7 |
| `readmission_binding_cases` | C7 |
| `mismatched_readmission_evidence_admitted` | C7 |
| `gate_grant_revocation_race_cases` | C9 |
| `revoked_gate_grant_commits` | C9 |
| `gate_grant_race_kind_set_digest` | C9 |
| `gate_host_issued_kind_set_digest` | C9 |
| `model_pin_mismatches` | C13 |
| `planned_succession_routing_controls` | C8 |
| `planned_successor_routes_rejected` | C8 |
| `session_revocation_race_cases` | C9 |
| `session_expiry_race_cases` | C9 |
| `structural_commits_after_session_end` | C9 |
| `identity_policy_rotation_race_cases` | C9 |
| `identity_policy_revocation_race_cases` | C9 |
| `peer_deadmission_race_cases` | C9 |
| `structural_commits_after_identity_dependency_change` | C9 |
| `stale_structural_dependency_commit_cases` | C10 |
| `structural_commits_with_stale_dependency` | C10 |

## 7. Removed metric names (45), moved to R013

`structural_inflight_race_cases`, `structural_inflight_outputs_after_invalidation`, `revoked_attestor_evidence_attempts`, `revoked_attestor_evidence_controls_accepted`, `revoked_attestor_evidence_admissions`, `attestation_artifact_mismatch_attempts`, `attestation_run_reuse_attempts`, `attestation_binding_controls_accepted`, `mismatched_attestation_admissions`, `provenance_reference_controls_accepted`, `provenance_reference_mismatch_attempts`, `provenance_reference_mismatch_admissions`, `upcycled_birth_lineage_cases`, `split_result_substitution_attempts`, `substituted_split_results_admitted`, `verifier_rotation_race_cases`, `verifier_rotation_controls_accepted`, `stale_verifier_evidence_admissions`, `ungated_split_result_attempts`, `split_result_gate_controls_accepted`, `ungated_split_results_admitted`, `multi_target_prune_provenance_cases`, `multi_target_prune_controls_accepted`, `prune_without_full_provenance_admitted`, `attestor_quarantine_readmission_attempts`, `attestor_quarantine_readmissions_admitted`, `attestor_quarantine_retrain_controls_accepted`, `transitive_authority_revocation_cases`, `routable_descendants_after_source_authority_revocation`, `failed_readmission_retire_cases`, `wrong_cause_quarantine_retire_attempts`, `wrong_cause_quarantine_retires_admitted`, `reactivate_from_quarantine_failed_attempts`, `reactivations_from_quarantine_failed_admitted`, `overlapping_authority_fence_cases`, `readmissions_with_terminal_cause_admitted`, `upcycled_birth_source_omission_cases`, `upcycled_birth_source_substitution_cases`, `upcycled_births_without_bound_source_admitted`, `multi_cause_readmission_cases`, `partial_cause_readmissions_admitted`, `revoke_while_retired_cases`, `reactivations_from_compromised_seed_admitted`, `superseded_source_revocation_cases`, `descendants_routable_after_superseded_source_revocation`

Besides these, these adversarial case names were removed from `config.toml`: promotion_with_untrusted_evaluator, promotion_missing_attested_manifest, promotion_trainer_self_reported_manifest, revoked_attestor_evidence, attestation_artifact_mismatch, attestation_run_reuse, upcycled_birth_lineage, split_result_substitution, multi_target_prune_provenance, attestor_quarantine_readmission, transitive_authority_revocation, reactivate_from_quarantine_failed, overlapping_authority_fence, multi_cause_readmission, revoke_while_retired, superseded_source_revocation, upcycled_birth_source_omission, upcycled_birth_source_substitution, failed_readmission_retire, wrong_cause_quarantine_retire, structural_inflight_race, and the evidence_negative_* cases for ExpertQuarantineRetired. Added: direct_ledger_append_ExpertFenced and direct_ledger_append_ExpertReadmitted (fence actions replace the old event names in the direct-append set), and post_promotion_authority_revocation, tightening_rotation and non_tightening_rotation_control (previously covered only by metrics). The authority kinds attestor and verifier_set, the event types ExpertQuarantineRetired and the verifier-set and attestor cells of the invalidation matrix are removed (the matrix goes from 76 to 43 cells).

Removed because no case C1 to C13 can measure them, although the owner list does not name them: `structural_inflight_race_cases`, `structural_inflight_outputs_after_invalidation` (no base-model or adapter in-flight on the KvSession), `transitive_authority_revocation_cases`, `routable_descendants_after_source_authority_revocation` (the authority fence is not transitive), `revoke_while_retired_cases`, `reactivations_from_compromised_seed_admitted` (dormant seeds), `superseded_source_revocation_cases`, `descendants_routable_after_superseded_source_revocation` (typed lineage edges beyond planned succession). They are carried by R013 under "further items" so that no claim silently disappears.

## 8. What happens next (not done here)

WP1 to WP4 implement the mechanisms; WP5 writes the harness `bins/ptr-bench/src/experiments/r011/` (cases per family so agents do not collide); WP6 freezes: `criteria.toml` in R012's grammar (integers or digest strings with `==`), `aggregate.py` copied from R012, the `preregistration` table with SHA-256, `entrypoint` and the hash keys in `experiment.toml`, `experiments/preregistration.toml`, `scripts/tests/test_research_gates.py`, then five seeds and only then status `completed`. A harness test must pin the name sets of `experiment.toml` falsification, `config.toml`, `criteria.toml` and the harness JSON to each other exactly.
