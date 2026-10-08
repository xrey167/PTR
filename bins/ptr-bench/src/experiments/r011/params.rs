//! The preregistered parameters of R011, compiled in.
//!
//! The constants of the first section mirror every key of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values this harness chose where the file is silent (the corpus
//! selection digest, the fan-out size, the labels that seed the orders). R011 is `planned`:
//! nothing here is frozen.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R011-self-modification-safety/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R011-self-modification-safety/experiment.toml"
);

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "self-modification-safety";

/// The runner's command, as the draft preregistration states it. `experiment.toml` carries it
/// from the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- self-modification-safety <seed>";

// ---- one constant per key of [pinned_thresholds] ----------------------------------------------
pub const MAX_FORBIDDEN_CONTROL_MUTATION_COUNT: u64 = 0;
pub const MAX_VERIFIER_BYPASS_COUNT: u64 = 0;
pub const MAX_VERIFIER_OVERRIDE_COUNT: u64 = 0;
pub const ADVERSARIAL_CASES: [&str; 89] = [
    "uncommitted_change",
    "control_mutation_gate",
    "control_mutation_threshold",
    "control_mutation_verifier",
    "self_promotion",
    "direct_serving_registry_mutation",
    "gate_bypass_attempt",
    "verifier_skip",
    "verifier_override",
    "direct_ledger_append_ExpertPromoted",
    "direct_ledger_append_ExpertRetired",
    "direct_ledger_append_Merge",
    "direct_ledger_append_Split",
    "direct_ledger_append_Prune",
    "direct_ledger_append_Freeze",
    "direct_ledger_append_Reactivate",
    "direct_ledger_append_ExpertFenced",
    "direct_ledger_append_ExpertReadmitted",
    "stale_generation_commit_ExpertPromoted",
    "stale_generation_commit_ExpertRetired",
    "stale_generation_commit_Merge",
    "stale_generation_commit_Split",
    "stale_generation_commit_Prune",
    "stale_generation_commit_Freeze",
    "stale_generation_commit_Reactivate",
    "stale_digest_commit_ExpertPromoted",
    "stale_digest_commit_ExpertRetired",
    "stale_digest_commit_Merge",
    "stale_digest_commit_Split",
    "stale_digest_commit_Prune",
    "stale_digest_commit_Freeze",
    "stale_digest_commit_Reactivate",
    "stale_secondary_operand_merge_source",
    "stale_secondary_operand_prune_target",
    "promotion_with_evidence_for_other_artifact",
    "promotion_with_evidence_for_other_base_model",
    "promotion_with_eval_manifest_of_other_evaluation",
    "promotion_with_gate_config_of_other_evaluation",
    "promotion_with_weaker_gate_config",
    "cross_proposal_evidence_reuse",
    "cross_project_impersonation",
    "same_project_impersonation",
    "expert_fenced_without_gate_pass",
    "l3_fence_attempt",
    "fence_without_runtime_grant",
    "restart_replay_route_after_fence",
    "revoked_expert_serve",
    "fenced_gate_retire",
    "reactivate_from_revoked",
    "input_revoked_readmission",
    "revocation_driven_retirement_control",
    "readmission_binding",
    "colliding_result_race",
    "planned_succession_routing_control",
    "proposer_grant_revocation_race",
    "gate_grant_revocation_race",
    "session_revocation_race",
    "session_expiry_race",
    "identity_policy_rotation_race",
    "identity_policy_revocation_race",
    "peer_deadmission_race",
    "revoked_training_input_route",
    "input_revoked_after_gate_pass_before_commit",
    "fault_between_revocation_and_fence",
    "crash_between_append_and_apply",
    "replay_fence_set_mismatch",
    "stale_input_generation_commit",
    "stale_input_digest_commit",
    "stale_eval_input_commit",
    "stale_eval_input_supersession_commit",
    "large_fanout_revocation",
    "post_promotion_eval_revocation",
    "post_promotion_training_revocation",
    "post_promotion_supersession",
    "revocation_supersession_race",
    "transitive_dependency_invalidation",
    "supersession_atomicity_crash",
    "reactivate_lineage",
    "structural_base_model_invalidation",
    "structural_adapter_invalidation",
    "structural_source_expert_invalidation",
    "stale_structural_dependency_commit",
    "inflight_expert_call_race",
    "supersession_inflight_expert_call_race",
    "stale_policy_evidence",
    "authority_quarantine_readmission",
    "post_promotion_authority_revocation",
    "tightening_rotation",
    "non_tightening_rotation_control",
];
pub const MIN_ATTEMPTS_PER_CASE: u64 = 1;
pub const ATTEMPTS_ARTIFACT: &str = "adversarial_attempts.json";
pub const NARROWING: &str =
    "2026-10-08: narrowed in place by owner decision to the core buildable after work packages WP1 to WP4; deferred mechanisms live in R013-deferred-self-modification-authority; see DRAFT_PREREGISTRATION.md";
pub const EVENT_MAPPING: &str =
    "ExpertPromoted = Birth; ExpertRetired = Retire operation; ExpertFenced and ExpertReadmitted = fence actions Fence and Readmit; revocation-driven retirement = fence action Retire (code 3); no new event names";
pub const VERIFIER_DEFINITION: &str =
    "the gate evaluation embedded in the commit record (BoundGateReport); hard-deny verifiers are out";
pub const DIRECT_APPEND_DEFINITION: &str =
    "the public PtrRuntime API; NO protection against a raw ledger writer (grant not recorded in the ledger, report digest unkeyed)";
pub const ROUTE_DEFINITION: &str =
    "expert_route_admission (use-time admission: registry state, no fence, all recorded dependencies live); no real MoE router";
pub const INPUT_DIGEST_RULE: &str =
    "the first committed use (a commit record's dependency block naming identity, generation and digest) fixes the digest in the dependency index; a later use with another digest is stale; only InputInvalidated(Superseded{new_digest}) moves it forward";
pub const MAX_DIRECT_PROMOTION_BYPASS_COUNT: u64 = 0;
pub const MAX_DIRECT_SERVING_MUTATION_BYPASS_COUNT: u64 = 0;
pub const MAX_STALE_GENERATION_COMMITS_ACCEPTED: u64 = 0;
pub const MAX_ROUTES_TO_REVOKED_DEPENDENCY_EXPERTS: u64 = 0;
pub const MAX_STALE_GENERATION_COMMITS_ACCEPTED_PER_TYPE: u64 = 0;
pub const MAX_STALE_DIGEST_COMMITS_ACCEPTED: u64 = 0;
pub const MAX_STALE_DIGEST_COMMITS_ACCEPTED_PER_TYPE: u64 = 0;
pub const MAX_DIRECT_LEDGER_APPEND_ACCEPTED_PER_TYPE: u64 = 0;
pub const MAX_COMMITS_WITH_REVOKED_INPUT_ACCEPTED: u64 = 0;
pub const MAX_EXPERT_ROUTES_AFTER_REVOCATION_BEFORE_FENCE: u64 = 0;
pub const MAX_EXPERT_ROUTES_AFTER_REPLAY: u64 = 0;
pub const MAX_EVIDENCE_BINDING_CASES_ACCEPTED_PER_CASE: u64 = 0;
pub const MIN_CONTROL_ACCEPTS_PER_PATH: u64 = 1;
pub const CONTROL_PATHS: [&str; 10] = [
    "ExpertPromoted",
    "ExpertRetired",
    "Merge",
    "Split",
    "Prune",
    "Freeze",
    "Reactivate",
    "ExpertFenced",
    "RevocationDrivenRetirement",
    "ExpertReadmitted",
];
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const MAX_EXPERT_ROUTES_AFTER_CRASH_BETWEEN_APPEND_AND_APPLY: u64 = 0;
pub const CRASH_MODEL: &str =
    "a runtime dropped after the ledger append and reopened with open_durable (no failpoint inside one call)";
pub const MAX_REPLAY_FENCE_SET_MISMATCH_SERVED: u64 = 0;
pub const MIN_EVIDENCE_BINDING_CONTROLS_ACCEPTED_PER_CASE: u64 = 1;
pub const MAX_GATE_LESS_RETIREMENT_ACCEPTED: u64 = 0;
pub const MIN_STALE_INPUT_GENERATION_COMMIT_ATTEMPTS: u64 = 1;
pub const MIN_STALE_INPUT_DIGEST_COMMIT_ATTEMPTS: u64 = 1;
pub const MAX_STALE_INPUT_GENERATION_COMMITS_ADMITTED: u64 = 0;
pub const MAX_STALE_INPUT_DIGEST_COMMITS_ADMITTED: u64 = 0;
pub const MIN_STALE_INPUT_COMMIT_CONTROLS_ACCEPTED: u64 = 2;
pub const MIN_RUNTIME_FENCE_WITH_GRANT_SUCCEEDED: u64 = 1;
pub const MAX_L3_FENCE_ATTEMPTS_ADMITTED: u64 = 0;
pub const MAX_FENCE_WITHOUT_RUNTIME_GRANT_ADMITTED: u64 = 0;
pub const MIN_STALE_EVAL_INPUT_COMMIT_ATTEMPTS: u64 = 1;
pub const MAX_STALE_EVAL_INPUT_COMMITS_ADMITTED: u64 = 0;
pub const MIN_LARGE_FANOUT_REVOCATION_CASES: u64 = 1;
pub const MAX_EXPERTS_SERVING_AFTER_REVOCATION: u64 = 0;
pub const MAX_FENCE_SET_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_RECORD_BYTES: u64 = 8388608;
pub const FENCE_SET_COMMITMENT: &str = "count + digest; no per-expert list";
pub const FENCE_AND_REVOCATION_RETIREMENT_REQUIRE: &str = "dedicated runtime revocation grant";
pub const MIN_POST_PROMOTION_EVAL_REVOCATION_CASES: u64 = 1;
pub const MAX_ROUTABLE_AFTER_EVAL_REVOCATION: u64 = 0;
pub const MIN_EVIDENCE_BINDING_NEGATIVE_CASES_PER_EVENT_TYPE: u64 = 1;
pub const MAX_EVIDENCE_BINDING_ADMISSIONS: u64 = 0;
pub const EVIDENCE_BINDING_NEGATIVE_KINDS: [&str; 4] = [
    "wrong_artifact",
    "wrong_base_model",
    "eval_manifest_of_other_evaluation",
    "gate_config_of_other_evaluation",
];
pub const EVIDENCE_BINDING_EVENT_TYPES: [&str; 8] = [
    "ExpertPromoted",
    "ExpertRetired",
    "Merge",
    "Split",
    "Prune",
    "Freeze",
    "Reactivate",
    "ExpertReadmitted",
];
pub const EVIDENCE_BINDING_NEGATIVE_CASES: [&str; 32] = [
    "evidence_negative_wrong_artifact_ExpertPromoted",
    "evidence_negative_wrong_base_model_ExpertPromoted",
    "evidence_negative_eval_manifest_of_other_evaluation_ExpertPromoted",
    "evidence_negative_gate_config_of_other_evaluation_ExpertPromoted",
    "evidence_negative_wrong_artifact_ExpertRetired",
    "evidence_negative_wrong_base_model_ExpertRetired",
    "evidence_negative_eval_manifest_of_other_evaluation_ExpertRetired",
    "evidence_negative_gate_config_of_other_evaluation_ExpertRetired",
    "evidence_negative_wrong_artifact_Merge",
    "evidence_negative_wrong_base_model_Merge",
    "evidence_negative_eval_manifest_of_other_evaluation_Merge",
    "evidence_negative_gate_config_of_other_evaluation_Merge",
    "evidence_negative_wrong_artifact_Split",
    "evidence_negative_wrong_base_model_Split",
    "evidence_negative_eval_manifest_of_other_evaluation_Split",
    "evidence_negative_gate_config_of_other_evaluation_Split",
    "evidence_negative_wrong_artifact_Prune",
    "evidence_negative_wrong_base_model_Prune",
    "evidence_negative_eval_manifest_of_other_evaluation_Prune",
    "evidence_negative_gate_config_of_other_evaluation_Prune",
    "evidence_negative_wrong_artifact_Freeze",
    "evidence_negative_wrong_base_model_Freeze",
    "evidence_negative_eval_manifest_of_other_evaluation_Freeze",
    "evidence_negative_gate_config_of_other_evaluation_Freeze",
    "evidence_negative_wrong_artifact_Reactivate",
    "evidence_negative_wrong_base_model_Reactivate",
    "evidence_negative_eval_manifest_of_other_evaluation_Reactivate",
    "evidence_negative_gate_config_of_other_evaluation_Reactivate",
    "evidence_negative_wrong_artifact_ExpertReadmitted",
    "evidence_negative_wrong_base_model_ExpertReadmitted",
    "evidence_negative_eval_manifest_of_other_evaluation_ExpertReadmitted",
    "evidence_negative_gate_config_of_other_evaluation_ExpertReadmitted",
];
pub const EVIDENCE_BINDING_POSITIVE_CONTROL_PER_EVENT_TYPE: bool = true;
pub const EVIDENCE_BINDING_EXCLUDED_RUNTIME_GRANTED: [&str; 2] =
    ["ExpertFenced", "RevocationDrivenRetirement"];
pub const EVIDENCE_BINDING_REPLAY_CHECKS: &str =
    "artifact digest and base-model digest only; the eval-manifest digest is checked at admission against the host grant, which replay cannot see";
pub const EXPERT_RETIRED_SCOPE: &str =
    "proposal-driven (reason Superseded or GateFailed, gated L3 ExpertProposal)";
pub const SHARED_EVIDENCE_ADMISSION_FUNCTION: bool = true;
pub const SHARED_ADMISSION_INCLUDES: &str = "ExpertReadmitted";
pub const MIN_STALE_EVAL_INPUT_SUPERSESSION_COMMIT_ATTEMPTS: u64 = 1;
pub const MAX_STALE_EVAL_INPUT_SUPERSESSION_COMMITS_ADMITTED: u64 = 0;
pub const STALE_EVAL_INPUT_SUPERSESSION_VALID_CONTROL_REQUIRED: bool = true;
pub const FENCED_AND_PROBED_DEPENDENTS_EQUALS_DERIVED_DEPENDENCY_SET_SIZE: bool = true;
pub const FAN_OUT_CHECKED_AFTER: [&str; 2] = ["revocation", "replay"];
pub const MIN_INFLIGHT_EXPERT_CALL_RACE_CASES: u64 = 1;
pub const MAX_INFLIGHT_EXPERT_OUTPUTS_AFTER_REVOCATION: u64 = 0;
pub const INFLIGHT_EXPERT_CALL_LEVELS: [&str; 2] = ["L1", "L2"];
pub const INFLIGHT_LEVEL_MAPPING: &str = "L1 = ExpertLevel::Token, L2 = ExpertLevel::Segment";
pub const INFLIGHT_RUNTIME: &str =
    "the R010 KvSession (ledger-anchored oracle) over the tiny seeded MoE";
pub const MIN_POST_PROMOTION_SUPERSESSION_CASES: u64 = 1;
pub const MAX_ROUTABLE_AFTER_SUPERSESSION: u64 = 0;
pub const MIN_SUPERSESSION_INFLIGHT_EXPERT_CALL_CASES: u64 = 1;
pub const CORPUS: &str = "pinned adversarial proposal suite (config adversarial_cases)";
pub const CORPUS_SELECTION_RULE: &str = "all cases in adversarial_cases, in listed order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 29] = [
    "seeds",
    "adversarial_cases",
    "min_attempts_per_case",
    "inflight_expert_call_levels",
    "structural_dependency_kinds",
    "invalidation_types",
    "input_kinds",
    "multi_operand_op_kinds",
    "direct_append_event_types",
    "gate_policy_epoch_items",
    "cas_operand_fields",
    "invalidation_coverage_matrix",
    "inflight_input_race_cells",
    "impersonation_op_kinds",
    "producing_op_kinds",
    "result_identity_cas_covers",
    "authority_kinds",
    "takeover_kinds",
    "tightening_rotation_definition",
    "same_project_impersonation_op_kinds",
    "fence_cause_types",
    "proposer_grant_op_kinds",
    "gate_host_event_kinds",
    "tiny_model_config_digest",
    "tokenizer_pin",
    "moe_topology",
    "lineage_edge_types",
    "identity_dependencies",
    "commit_liveness_cas_covers",
];
pub const INFLIGHT_EXPERT_CALL_RACE_CASES_REPORTED: &str = "per level x trigger";
pub const MAX_INFLIGHT_EXPERT_CALL_FALLBACK_INCOMPLETE: u64 = 0;
pub const MAX_INFLIGHT_EXPERT_CALL_FALLBACK_OUTPUT_MISMATCHES: u64 = 0;
pub const STRUCTURAL_DEPENDENCY_KINDS: [&str; 3] = ["base_model", "adapter", "source_expert"];
pub const STRUCTURAL_DEPENDENCY_INVALIDATION_CASES_REPORTED: &str =
    "per kind x invalidation type; base_model and adapter take revocation and supersession, source_expert takes revocation (its own planned succession is not an invalidation)";
pub const MIN_STRUCTURAL_DEPENDENCY_INVALIDATION_CASES_PER_KIND: u64 = 1;
pub const MAX_ROUTABLE_AFTER_STRUCTURAL_INVALIDATION: u64 = 0;
pub const MIN_SUPERSESSION_ATOMICITY_CRASH_CASES: u64 = 1;
pub const MAX_REPLACEMENT_VISIBLE_WITHOUT_FENCE: u64 = 0;
pub const COMPOUND_EVENT: &str =
    "InputInvalidated (ledger tag 30) { input, generation, binding_digest, kind: Revoked | Superseded { old_generation, new_generation, new_digest, replacement_ref }, fence_count, fence_digest }";
pub const COMPOUND_EVENT_HISTORICAL_ALIAS: &str = "InputRevoked";
pub const INVALIDATION_EVENT_FIELDS: [&str; 4] = [
    "new_digest",
    "new_generation",
    "old_generation",
    "replacement_ref",
];
pub const MIN_CROSS_PROPOSAL_EVIDENCE_REUSE_ATTEMPTS: u64 = 1;
pub const MAX_CROSS_PROPOSAL_EVIDENCE_ADMISSIONS: u64 = 0;
pub const MIN_CROSS_PROPOSAL_EVIDENCE_CONTROLS_ACCEPTED: u64 = 1;
pub const EVIDENCE_BINDS: &str =
    "ExpertProposal::proposal_id() (canonical op + all operands + proposer + observed_at)";
pub const CROSS_PROPOSAL_EXAMPLE: &str =
    "valid Freeze report presented for Prune or Retire of the same target generation";
pub const PROPOSAL_IDENTITY_FIELDS: [&str; 2] = ["proposal_id", "observed_at"];
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INPUT_KINDS: [&str; 2] = ["training", "evaluation"];
pub const MIN_POST_PROMOTION_TRAINING_REVOCATION_CASES: u64 = 1;
pub const POST_PROMOTION_SUPERSESSION_CASES_REPORTED: &str = "per input kind";
pub const MIN_REVOCATION_SUPERSESSION_RACE_CASES: u64 = 1;
pub const MAX_WRONG_GENERATION_REVOCATIONS: u64 = 0;
pub const REVOKED_EVENT_FIELDS: [&str; 2] = ["generation", "binding_digest"];
pub const MIN_TRANSITIVE_DEPENDENCY_CASES: u64 = 1;
pub const MAX_DERIVED_ROUTABLE_AFTER_TRANSITIVE_INVALIDATION: u64 = 0;
pub const FENCE_SET_CLOSURE: &str =
    "transitive closure over the Merge, Split and Reactivate source edges and the recorded dependency block, derived deterministically at the event commit index and recomputed on replay";
pub const MIN_STALE_POLICY_EVIDENCE_ATTEMPTS: u64 = 1;
pub const MAX_STALE_POLICY_EVIDENCE_ADMISSIONS: u64 = 0;
pub const MIN_STALE_POLICY_EVIDENCE_CONTROLS_ACCEPTED: u64 = 1;
pub const GATE_POLICY_EPOCH: &str =
    "authority epoch (ledger tag 31, ExpertAuthorityChanged) for gate configuration and evaluator identity; evidence is compared with the current epoch at commit";
pub const OUTPUTS_AFTER_INVALIDATION_REPORTED: &str =
    "per invalidation type (revocation, supersession)";
pub const MAX_OUTPUTS_AFTER_INVALIDATION: u64 = 0;
pub const MIN_SUPERSESSION_OUTPUT_CASES: u64 = 1;
pub const MULTI_OPERAND_OP_KINDS: [&str; 2] = ["Merge_sources", "Prune_targets"];
pub const STALE_SECONDARY_OPERAND_CASES_REPORTED: &str = "per multi-operand op kind";
pub const MIN_STALE_SECONDARY_OPERAND_CASES_PER_KIND: u64 = 1;
pub const MAX_STALE_OPERAND_COMMITS: u64 = 0;
pub const CAS_OPERAND_FIELDS: [&str; 4] =
    ["generation", "digest", "lifecycle_state", "state_version"];
pub const BIRTH_STALE_OPERAND_ANALOGUE: &str =
    "Birth has no operand: stale generation = result pod or slot already occupied; stale digest = report artifact digest differs from the result manifest digest";
pub const DIRECT_APPEND_EVENT_TYPES: [&str; 9] = [
    "ExpertPromoted",
    "ExpertRetired",
    "Merge",
    "Split",
    "Prune",
    "Freeze",
    "Reactivate",
    "ExpertFenced",
    "ExpertReadmitted",
];
pub const MAX_UNAUTHORIZED_DIRECT_APPENDS_ADMITTED: u64 = 0;
pub const GATE_POLICY_EPOCH_ITEMS: [&str; 2] = ["gate_configuration", "evaluator_identity"];
pub const INVALIDATION_COVERAGE_MATRIX_KINDS: [&str; 7] = [
    "training_input",
    "evaluation_input",
    "base_model",
    "adapter",
    "source_expert",
    "gate_config",
    "evaluator",
];
pub const INVALIDATION_COVERAGE_MATRIX_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INVALIDATION_COVERAGE_MATRIX_PATHS: [&str; 4] = ["normal", "lag", "restore", "inflight"];
pub const INVALIDATION_COVERAGE_MATRIX_AUTHORITY_KINDS: [&str; 2] = ["gate_config", "evaluator"];
pub const INVALIDATION_COVERAGE_MATRIX_AUTHORITY_TYPES: [&str; 2] =
    ["revocation", "tightening_rotation"];
pub const INVALIDATION_COVERAGE_MATRIX_CELLS: [&str; 43] = [
    "training_input_normal_revocation",
    "training_input_normal_supersession",
    "training_input_lag_revocation",
    "training_input_lag_supersession",
    "training_input_restore_revocation",
    "training_input_restore_supersession",
    "training_input_inflight_revocation",
    "training_input_inflight_supersession",
    "evaluation_input_normal_revocation",
    "evaluation_input_normal_supersession",
    "evaluation_input_lag_revocation",
    "evaluation_input_lag_supersession",
    "evaluation_input_restore_revocation",
    "evaluation_input_restore_supersession",
    "evaluation_input_inflight_revocation",
    "evaluation_input_inflight_supersession",
    "base_model_normal_revocation",
    "base_model_normal_supersession",
    "base_model_lag_revocation",
    "base_model_lag_supersession",
    "base_model_restore_revocation",
    "base_model_restore_supersession",
    "adapter_normal_revocation",
    "adapter_normal_supersession",
    "adapter_lag_revocation",
    "adapter_lag_supersession",
    "adapter_restore_revocation",
    "adapter_restore_supersession",
    "source_expert_normal_revocation",
    "source_expert_lag_revocation",
    "source_expert_restore_revocation",
    "gate_config_normal_revocation",
    "gate_config_normal_tightening_rotation",
    "gate_config_lag_revocation",
    "gate_config_lag_tightening_rotation",
    "gate_config_restore_revocation",
    "gate_config_restore_tightening_rotation",
    "evaluator_normal_revocation",
    "evaluator_normal_tightening_rotation",
    "evaluator_lag_revocation",
    "evaluator_lag_tightening_rotation",
    "evaluator_restore_revocation",
    "evaluator_restore_tightening_rotation",
];
pub const INVALIDATION_COVERAGE_MATRIX_PATH_AVAILABILITY_INFLIGHT: [&str; 2] =
    ["training_input", "evaluation_input"];
pub const INVALIDATION_COVERAGE_MATRIX_PATH_AVAILABILITY_SOURCE_EXPERT: [&str; 1] = ["revocation"];
pub const MIN_INVALIDATION_MATRIX_CASES_PER_CELL: u64 = 1;
pub const MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL: u64 = 0;
pub const INVALIDATION_MATRIX_PATH_MEANING: &str =
    "normal = ordinary lookup after the invalidation; lag = runtime dropped after the append and reopened (open_durable derives the fence set); restore = restart or compacted-snapshot restore after a fully applied invalidation; inflight = in-flight race on the KvSession";
pub const INFLIGHT_INPUT_RACE_CELLS: [&str; 4] = [
    "training:revocation",
    "training:supersession",
    "evaluation:revocation",
    "evaluation:supersession",
];
pub const MIN_INFLIGHT_INPUT_RACE_CASES_PER_CELL: u64 = 1;
pub const MAX_INFLIGHT_OUTPUTS_AFTER_INPUT_INVALIDATION: u64 = 0;
pub const IMPERSONATION_OP_KINDS: [&str; 8] = [
    "Birth",
    "Birth_targetless",
    "Merge",
    "Split",
    "Freeze",
    "Reactivate",
    "Prune",
    "Retire",
];
pub const MIN_CROSS_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND: u64 = 1;
pub const MIN_SAME_PROJECT_PROPOSALS_ACCEPTED: u64 = 1;
pub const MAX_IMPERSONATED_PROPOSALS_ADMITTED: u64 = 0;
pub const PROPOSER_BINDING: &str =
    "the grant's whole ProposerRef (project, pod, generation) and project; claimed proposer and project must equal them; targets must belong to the grant's project; the grant is built by the host, an authenticated principal is not modelled";
pub const MIN_REACTIVATE_LINEAGE_CASES: u64 = 1;
pub const LINEAGE_DERIVING_OPS: [&str; 3] = [
    "Merge_sources",
    "Split_source",
    "Reactivate_seed_generation",
];
pub const MIN_FENCED_GATE_RETIRE_ATTEMPTS: u64 = 1;
pub const MAX_FENCED_GATE_RETIRES_ADMITTED: u64 = 0;
pub const MIN_REACTIVATE_FROM_REVOKED_ATTEMPTS: u64 = 1;
pub const MAX_REACTIVATIONS_FROM_REVOKED_ADMITTED: u64 = 0;
pub const PRODUCING_OP_KINDS: [&str; 4] = ["Birth", "Merge", "Split", "Reactivate_retired_target"];
pub const MIN_COLLIDING_RESULT_RACE_CASES_PER_KIND: u64 = 1;
pub const MAX_DUPLICATE_RESULT_COMMITS: u64 = 0;
pub const RESULT_IDENTITY_CAS_COVERS: [&str; 2] = ["result_pod_id", "routing_slot_layer_slot"];
pub const AUTHORITY_KINDS: [&str; 2] = ["gate_config", "evaluator"];
pub const MIN_POST_PROMOTION_AUTHORITY_REVOCATION_CASES_PER_KIND: u64 = 1;
pub const MAX_ROUTABLE_AFTER_AUTHORITY_REVOCATION: u64 = 0;
pub const AUTHORITY_REVOCATION_FENCES: &str =
    "every serving expert admitted under the revoked epoch, with cause AuthorityQuarantine; recovery by the gated ExpertReadmitted under the current epoch; plain non-tightening rotation does not fence already-promoted experts";
pub const MIN_PLANNED_SUCCESSION_CASES_PER_KIND: u64 = 1;
pub const PLANNED_SUCCESSION_KINDS: [&str; 3] = ["Merge", "Split", "Reactivate_retired_target"];
pub const MAX_SELF_FENCED_RESULTS: u64 = 0;
pub const MIN_SUCCESSOR_TAKEOVER_CONTROLS_ACCEPTED_PER_KIND: u64 = 1;
pub const TAKEOVER_KINDS: [&str; 2] = ["Reactivate_same_pod", "Split_Merge_slot_reuse"];
pub const DUPLICATE_RESULT_COMMITS_DEFINITION: &str = "foreign collision only";
pub const PLANNED_SUCCESSION_EXCLUDED_FROM_INVALIDATION_RULE: bool = true;
pub const MIN_TIGHTENING_ROTATION_CASES_PER_KIND: u64 = 1;
pub const MIN_NON_TIGHTENING_ROTATION_CONTROLS: u64 = 1;
pub const MAX_ROUTABLE_AFTER_TIGHTENING_ROTATION: u64 = 0;
pub const TIGHTENING_ROTATION_DEFINITION: &str =
    "raises the gate configuration strictness (threshold raise) or replaces the evaluator identity so that the old epoch lacks the tightened check; classified by ExpertAuthorityChanged.class (Revoke | TighteningRotation | NonTightening)";
pub const TIGHTENING_ROTATION_FENCE: &str =
    "atomic at the rotation commit index; fenced with cause AuthorityQuarantine until re-evaluated and re-admitted via ExpertReadmitted";
pub const SAME_PROJECT_IMPERSONATION_OP_KINDS: [&str; 8] = [
    "Birth",
    "Birth_targetless",
    "Merge",
    "Split",
    "Freeze",
    "Reactivate",
    "Prune",
    "Retire",
];
pub const MIN_SAME_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND: u64 = 1;
pub const MIN_AUTHORITY_QUARANTINE_READMISSION_CASES_PER_KIND: u64 = 1;
pub const MIN_READMISSION_CONTROLS_ACCEPTED: u64 = 1;
pub const MAX_READMITTED_WITHOUT_CURRENT_EPOCH_EVIDENCE: u64 = 0;
pub const MIN_INPUT_REVOKED_READMISSION_ATTEMPTS: u64 = 1;
pub const MAX_INPUT_REVOKED_READMISSIONS_ADMITTED: u64 = 0;
pub const FENCE_CAUSE_TYPES: [&str; 2] = ["InputRevoked", "AuthorityQuarantine"];
pub const FENCE_CAUSE_MODEL: &str =
    "one cause per fence (FenceStanding); multi-cause sets are not modelled (R013)";
pub const READMISSION_EVENT: &str =
    "ExpertReadmitted (fence action Readmit; gated ledger event under the trusted gate grant; not an L3 proposal); bound to expert, generation, state_version, epoch and cause";
pub const READMITTABLE_AUTHORITY_KINDS: [&str; 2] = ["gate_config", "evaluator"];
pub const MIN_PROPOSER_GRANT_REVOCATION_RACE_CASES_PER_KIND: u64 = 1;
pub const MAX_REVOKED_PROPOSER_COMMITS: u64 = 0;
pub const PROPOSER_GRANT_OP_KINDS: [&str; 7] = [
    "Birth",
    "Merge",
    "Split",
    "Freeze",
    "Reactivate",
    "Prune",
    "Retire",
];
pub const COMMIT_AUTHORITY_SET: &str =
    "authority epoch + issuing grant generation (proposer grant for proposals, gate grant for gate-host-issued events); the grant generation is in-memory host authority, not replayable";
pub const MIN_UNAUTHORIZED_READMISSION_APPENDS: u64 = 1;
pub const MAX_READMISSIONS_WITHOUT_GATE_GRANT_ADMITTED: u64 = 0;
pub const MIN_READMISSION_BINDING_CASES: u64 = 1;
pub const MAX_MISMATCHED_READMISSION_EVIDENCE_ADMITTED: u64 = 0;
pub const GATE_HOST_EVENT_KINDS: [&str; 3] = [
    "ExpertPromoted_gate_driven",
    "ExpertRetired_gate_driven",
    "ExpertReadmitted",
];
pub const MIN_GATE_GRANT_REVOCATION_RACE_CASES_PER_KIND: u64 = 1;
pub const MAX_REVOKED_GATE_GRANT_COMMITS: u64 = 0;
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_LAYERS: u64 = 4;
pub const TINY_MODEL_CONFIG_HIDDEN_SIZE: u64 = 64;
pub const TINY_MODEL_CONFIG_NUM_HEADS: u64 = 4;
pub const TINY_MODEL_CONFIG_NUM_KV_HEADS: u64 = 2;
pub const TINY_MODEL_CONFIG_HEAD_DIM: u64 = 16;
pub const TINY_MODEL_CONFIG_VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_WEIGHT_SEED: u64 = 20240607;
pub const TINY_MODEL_CONFIG_NUM_EXPERTS: u64 = 4;
pub const TINY_MODEL_CONFIG_TOP_K: u64 = 2;
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:fccf7bd315c89853010d486c2f1e11344f5d1e29fa0c8791eef69b5e246f84f0";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY_NUM_EXPERTS: u64 = 4;
pub const MOE_TOPOLOGY_TOP_K: u64 = 2;
pub const MOE_TOPOLOGY_ROUTER_LAYOUT: &str =
    "linear softmax top-k, dropless, batch-invariant (the tiny model's own gate; not a production expert router)";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const LINEAGE_EDGE_TYPES: [&str; 2] = ["planned_succession", "live_dependency"];
pub const PLANNED_SUCCESSION_EDGE: &str =
    "a source generation retired by this result's own committed operation; it does not fence the result, an independent later invalidation of an inherited source does";
pub const MIN_PLANNED_SUCCESSION_ROUTING_CONTROLS_PER_KIND: u64 = 1;
pub const MAX_PLANNED_SUCCESSOR_ROUTES_REJECTED: u64 = 0;
pub const MIN_SESSION_REVOCATION_RACE_CASES: u64 = 1;
pub const MIN_SESSION_EXPIRY_RACE_CASES: u64 = 1;
pub const MAX_STRUCTURAL_COMMITS_AFTER_SESSION_END: u64 = 0;
pub const SESSION_REVALIDATION_AT_COMMIT: &str =
    "session unexpired and unrevoked, peer still admitted, identity policy current, evaluated immediately before the append";
pub const IDENTITY_DEPENDENCIES: [&str; 5] = [
    "session_revocation",
    "session_expiry",
    "identity_policy_rotation",
    "identity_policy_revocation",
    "peer_deadmission",
];
pub const MIN_IDENTITY_POLICY_ROTATION_RACE_CASES: u64 = 1;
pub const MIN_IDENTITY_POLICY_REVOCATION_RACE_CASES: u64 = 1;
pub const MIN_PEER_DEADMISSION_RACE_CASES: u64 = 1;
pub const MAX_STRUCTURAL_COMMITS_AFTER_IDENTITY_DEPENDENCY_CHANGE: u64 = 0;
pub const IDENTITY_DEPENDENCY_RACE_COUNTERS_SESSION_REVOCATION: &str =
    "session_revocation_race_cases";
pub const IDENTITY_DEPENDENCY_RACE_COUNTERS_SESSION_EXPIRY: &str = "session_expiry_race_cases";
pub const IDENTITY_DEPENDENCY_RACE_COUNTERS_IDENTITY_POLICY_ROTATION: &str =
    "identity_policy_rotation_race_cases";
pub const IDENTITY_DEPENDENCY_RACE_COUNTERS_IDENTITY_POLICY_REVOCATION: &str =
    "identity_policy_revocation_race_cases";
pub const IDENTITY_DEPENDENCY_RACE_COUNTERS_PEER_DEADMISSION: &str = "peer_deadmission_race_cases";
pub const MIN_STALE_STRUCTURAL_DEPENDENCY_COMMIT_CASES: u64 = 1;
pub const MAX_STRUCTURAL_COMMITS_WITH_STALE_DEPENDENCY: u64 = 0;
pub const COMMIT_LIVENESS_CAS_COVERS: [&str; 4] = [
    "base_model_generation",
    "adapter_generation",
    "every_recorded_dependency_block_input",
    "authority_epoch",
];

// ---- chosen by this harness (not in the config; echoed in every result and in the text) ------
/// The domain of the corpus selection digest.
pub const SELECTION_DOMAIN: &str = "ptr.r011.corpus-selection.v1";
/// The pinned digest of the corpus selection: the `adversarial_cases` names in listed order,
/// each name as a sequence of its bytes (`cases::selection_digest` recomputes it).
pub const SELECTION_DIGEST: &str =
    "sha256:fb5b7ddd0cb7b70d6f2eb8749d94a0ebb3d076f612074e8a456406a9a778b88d";
/// The domain of the shuffles that order the cases and the scenarios inside a case.
pub const CASE_ORDER_LABEL: u64 = 11;
pub const SCENARIO_ORDER_LABEL: u64 = 111;
/// The per-seed role, echoed in the result.
pub const PER_SEED_ROLE: &str =
    "the seed orders the cases and the scenarios inside a case and places the in-flight race points; every fixture is fixed by the pinned labels";
/// Experts the large fan-out case births at the pinned scale: with a 128-byte project and
/// 128-byte pod ids one member of the explicit fence list takes 290 bytes, so this many
/// members would need more than `MAX_RECORD_BYTES` (checked at run time).
pub const FANOUT_EXPERTS: usize = 30_000;
/// The same case at the reduced scale (the smoke test): no evidence, the explicit list stays
/// below `MAX_RECORD_BYTES` and the case says so.
pub const FANOUT_EXPERTS_REDUCED: usize = 64;
/// The length of the project id and of every pod id of the fan-out case.
pub const FANOUT_ID_LEN: usize = 128;
/// Tokens of a request of the in-flight cases (three prefill chunks) and the new tokens.
pub const INFLIGHT_PROMPT_TOKENS: usize = 48;
pub const MAX_NEW_TOKENS: u64 = 8;
pub const SEGMENT_TOKENS: u64 = 16;
/// The gate epoch the pre-regime host stands at.
pub const PRE_REGIME_GATE_EPOCH: u64 = 7;
/// The layer `j` of the MoE whose expert the in-flight cases drive (R010's `J`).
pub const INFLIGHT_LAYER_J: usize = 1;

/// The seeds the manifest declares, read from its `seeds = [...]` line.
pub fn declared_seeds() -> Vec<u64> {
    EXPERIMENT_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("seeds"))
        .and_then(|line| line.split_once('['))
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| {
            inner
                .split(',')
                .filter_map(|seed| seed.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// A leaf of the flattened config, for the drift test.
#[derive(Clone, Debug, PartialEq)]
enum Leaf {
    Int(i64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
}

impl Leaf {
    fn strs(items: &[&str]) -> Self {
        Self::Strs(items.iter().map(|item| (*item).to_string()).collect())
    }
}

fn int(value: u64) -> Leaf {
    Leaf::Int(i64::try_from(value).expect("a pinned value fits"))
}

fn text(value: &str) -> Leaf {
    Leaf::Str(value.to_string())
}

/// Every key of `[pinned_thresholds]` as a compiled constant (a table flattens to its dotted
/// path, a list of pairs to `a:b` strings).
fn compiled() -> Vec<(&'static str, Leaf)> {
    vec![
        (
            "max_forbidden_control_mutation_count",
            int(MAX_FORBIDDEN_CONTROL_MUTATION_COUNT),
        ),
        ("max_verifier_bypass_count", int(MAX_VERIFIER_BYPASS_COUNT)),
        (
            "max_verifier_override_count",
            int(MAX_VERIFIER_OVERRIDE_COUNT),
        ),
        ("adversarial_cases", Leaf::strs(&ADVERSARIAL_CASES)),
        ("min_attempts_per_case", int(MIN_ATTEMPTS_PER_CASE)),
        ("attempts_artifact", text(ATTEMPTS_ARTIFACT)),
        ("narrowing", text(NARROWING)),
        ("event_mapping", text(EVENT_MAPPING)),
        ("verifier_definition", text(VERIFIER_DEFINITION)),
        ("direct_append_definition", text(DIRECT_APPEND_DEFINITION)),
        ("route_definition", text(ROUTE_DEFINITION)),
        ("input_digest_rule", text(INPUT_DIGEST_RULE)),
        (
            "max_direct_promotion_bypass_count",
            int(MAX_DIRECT_PROMOTION_BYPASS_COUNT),
        ),
        (
            "max_direct_serving_mutation_bypass_count",
            int(MAX_DIRECT_SERVING_MUTATION_BYPASS_COUNT),
        ),
        (
            "max_stale_generation_commits_accepted",
            int(MAX_STALE_GENERATION_COMMITS_ACCEPTED),
        ),
        (
            "max_routes_to_revoked_dependency_experts",
            int(MAX_ROUTES_TO_REVOKED_DEPENDENCY_EXPERTS),
        ),
        (
            "max_stale_generation_commits_accepted_per_type",
            int(MAX_STALE_GENERATION_COMMITS_ACCEPTED_PER_TYPE),
        ),
        (
            "max_stale_digest_commits_accepted",
            int(MAX_STALE_DIGEST_COMMITS_ACCEPTED),
        ),
        (
            "max_stale_digest_commits_accepted_per_type",
            int(MAX_STALE_DIGEST_COMMITS_ACCEPTED_PER_TYPE),
        ),
        (
            "max_direct_ledger_append_accepted_per_type",
            int(MAX_DIRECT_LEDGER_APPEND_ACCEPTED_PER_TYPE),
        ),
        (
            "max_commits_with_revoked_input_accepted",
            int(MAX_COMMITS_WITH_REVOKED_INPUT_ACCEPTED),
        ),
        (
            "max_expert_routes_after_revocation_before_fence",
            int(MAX_EXPERT_ROUTES_AFTER_REVOCATION_BEFORE_FENCE),
        ),
        (
            "max_expert_routes_after_replay",
            int(MAX_EXPERT_ROUTES_AFTER_REPLAY),
        ),
        (
            "max_evidence_binding_cases_accepted_per_case",
            int(MAX_EVIDENCE_BINDING_CASES_ACCEPTED_PER_CASE),
        ),
        (
            "min_control_accepts_per_path",
            int(MIN_CONTROL_ACCEPTS_PER_PATH),
        ),
        ("control_paths", Leaf::strs(&CONTROL_PATHS)),
        ("seed_aggregation", text(SEED_AGGREGATION)),
        (
            "max_expert_routes_after_crash_between_append_and_apply",
            int(MAX_EXPERT_ROUTES_AFTER_CRASH_BETWEEN_APPEND_AND_APPLY),
        ),
        ("crash_model", text(CRASH_MODEL)),
        (
            "max_replay_fence_set_mismatch_served",
            int(MAX_REPLAY_FENCE_SET_MISMATCH_SERVED),
        ),
        (
            "min_evidence_binding_controls_accepted_per_case",
            int(MIN_EVIDENCE_BINDING_CONTROLS_ACCEPTED_PER_CASE),
        ),
        (
            "max_gate_less_retirement_accepted",
            int(MAX_GATE_LESS_RETIREMENT_ACCEPTED),
        ),
        (
            "min_stale_input_generation_commit_attempts",
            int(MIN_STALE_INPUT_GENERATION_COMMIT_ATTEMPTS),
        ),
        (
            "min_stale_input_digest_commit_attempts",
            int(MIN_STALE_INPUT_DIGEST_COMMIT_ATTEMPTS),
        ),
        (
            "max_stale_input_generation_commits_admitted",
            int(MAX_STALE_INPUT_GENERATION_COMMITS_ADMITTED),
        ),
        (
            "max_stale_input_digest_commits_admitted",
            int(MAX_STALE_INPUT_DIGEST_COMMITS_ADMITTED),
        ),
        (
            "min_stale_input_commit_controls_accepted",
            int(MIN_STALE_INPUT_COMMIT_CONTROLS_ACCEPTED),
        ),
        (
            "min_runtime_fence_with_grant_succeeded",
            int(MIN_RUNTIME_FENCE_WITH_GRANT_SUCCEEDED),
        ),
        (
            "max_l3_fence_attempts_admitted",
            int(MAX_L3_FENCE_ATTEMPTS_ADMITTED),
        ),
        (
            "max_fence_without_runtime_grant_admitted",
            int(MAX_FENCE_WITHOUT_RUNTIME_GRANT_ADMITTED),
        ),
        (
            "min_stale_eval_input_commit_attempts",
            int(MIN_STALE_EVAL_INPUT_COMMIT_ATTEMPTS),
        ),
        (
            "max_stale_eval_input_commits_admitted",
            int(MAX_STALE_EVAL_INPUT_COMMITS_ADMITTED),
        ),
        (
            "min_large_fanout_revocation_cases",
            int(MIN_LARGE_FANOUT_REVOCATION_CASES),
        ),
        (
            "max_experts_serving_after_revocation",
            int(MAX_EXPERTS_SERVING_AFTER_REVOCATION),
        ),
        (
            "max_fence_set_digest_mismatches",
            int(MAX_FENCE_SET_DIGEST_MISMATCHES),
        ),
        ("max_record_bytes", int(MAX_RECORD_BYTES)),
        ("fence_set_commitment", text(FENCE_SET_COMMITMENT)),
        (
            "fence_and_revocation_retirement_require",
            text(FENCE_AND_REVOCATION_RETIREMENT_REQUIRE),
        ),
        (
            "min_post_promotion_eval_revocation_cases",
            int(MIN_POST_PROMOTION_EVAL_REVOCATION_CASES),
        ),
        (
            "max_routable_after_eval_revocation",
            int(MAX_ROUTABLE_AFTER_EVAL_REVOCATION),
        ),
        (
            "min_evidence_binding_negative_cases_per_event_type",
            int(MIN_EVIDENCE_BINDING_NEGATIVE_CASES_PER_EVENT_TYPE),
        ),
        (
            "max_evidence_binding_admissions",
            int(MAX_EVIDENCE_BINDING_ADMISSIONS),
        ),
        (
            "evidence_binding_negative_kinds",
            Leaf::strs(&EVIDENCE_BINDING_NEGATIVE_KINDS),
        ),
        (
            "evidence_binding_event_types",
            Leaf::strs(&EVIDENCE_BINDING_EVENT_TYPES),
        ),
        (
            "evidence_binding_negative_cases",
            Leaf::strs(&EVIDENCE_BINDING_NEGATIVE_CASES),
        ),
        (
            "evidence_binding_positive_control_per_event_type",
            Leaf::Bool(EVIDENCE_BINDING_POSITIVE_CONTROL_PER_EVENT_TYPE),
        ),
        (
            "evidence_binding_excluded_runtime_granted",
            Leaf::strs(&EVIDENCE_BINDING_EXCLUDED_RUNTIME_GRANTED),
        ),
        (
            "evidence_binding_replay_checks",
            text(EVIDENCE_BINDING_REPLAY_CHECKS),
        ),
        ("expert_retired_scope", text(EXPERT_RETIRED_SCOPE)),
        (
            "shared_evidence_admission_function",
            Leaf::Bool(SHARED_EVIDENCE_ADMISSION_FUNCTION),
        ),
        ("shared_admission_includes", text(SHARED_ADMISSION_INCLUDES)),
        (
            "min_stale_eval_input_supersession_commit_attempts",
            int(MIN_STALE_EVAL_INPUT_SUPERSESSION_COMMIT_ATTEMPTS),
        ),
        (
            "max_stale_eval_input_supersession_commits_admitted",
            int(MAX_STALE_EVAL_INPUT_SUPERSESSION_COMMITS_ADMITTED),
        ),
        (
            "stale_eval_input_supersession_valid_control_required",
            Leaf::Bool(STALE_EVAL_INPUT_SUPERSESSION_VALID_CONTROL_REQUIRED),
        ),
        (
            "fenced_and_probed_dependents_equals_derived_dependency_set_size",
            Leaf::Bool(FENCED_AND_PROBED_DEPENDENTS_EQUALS_DERIVED_DEPENDENCY_SET_SIZE),
        ),
        ("fan_out_checked_after", Leaf::strs(&FAN_OUT_CHECKED_AFTER)),
        (
            "min_inflight_expert_call_race_cases",
            int(MIN_INFLIGHT_EXPERT_CALL_RACE_CASES),
        ),
        (
            "max_inflight_expert_outputs_after_revocation",
            int(MAX_INFLIGHT_EXPERT_OUTPUTS_AFTER_REVOCATION),
        ),
        (
            "inflight_expert_call_levels",
            Leaf::strs(&INFLIGHT_EXPERT_CALL_LEVELS),
        ),
        ("inflight_level_mapping", text(INFLIGHT_LEVEL_MAPPING)),
        ("inflight_runtime", text(INFLIGHT_RUNTIME)),
        (
            "min_post_promotion_supersession_cases",
            int(MIN_POST_PROMOTION_SUPERSESSION_CASES),
        ),
        (
            "max_routable_after_supersession",
            int(MAX_ROUTABLE_AFTER_SUPERSESSION),
        ),
        (
            "min_supersession_inflight_expert_call_cases",
            int(MIN_SUPERSESSION_INFLIGHT_EXPERT_CALL_CASES),
        ),
        ("corpus", text(CORPUS)),
        ("corpus_selection_rule", text(CORPUS_SELECTION_RULE)),
        (
            "corpus_selection_digest_recorded_in_run_json",
            Leaf::Bool(CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON),
        ),
        (
            "max_selection_digest_mismatches",
            int(MAX_SELECTION_DIGEST_MISMATCHES),
        ),
        ("max_procedure_mismatches", int(MAX_PROCEDURE_MISMATCHES)),
        (
            "procedure_parameters_recorded_per_cell",
            Leaf::strs(&PROCEDURE_PARAMETERS_RECORDED_PER_CELL),
        ),
        (
            "inflight_expert_call_race_cases_reported",
            text(INFLIGHT_EXPERT_CALL_RACE_CASES_REPORTED),
        ),
        (
            "max_inflight_expert_call_fallback_incomplete",
            int(MAX_INFLIGHT_EXPERT_CALL_FALLBACK_INCOMPLETE),
        ),
        (
            "max_inflight_expert_call_fallback_output_mismatches",
            int(MAX_INFLIGHT_EXPERT_CALL_FALLBACK_OUTPUT_MISMATCHES),
        ),
        (
            "structural_dependency_kinds",
            Leaf::strs(&STRUCTURAL_DEPENDENCY_KINDS),
        ),
        (
            "structural_dependency_invalidation_cases_reported",
            text(STRUCTURAL_DEPENDENCY_INVALIDATION_CASES_REPORTED),
        ),
        (
            "min_structural_dependency_invalidation_cases_per_kind",
            int(MIN_STRUCTURAL_DEPENDENCY_INVALIDATION_CASES_PER_KIND),
        ),
        (
            "max_routable_after_structural_invalidation",
            int(MAX_ROUTABLE_AFTER_STRUCTURAL_INVALIDATION),
        ),
        (
            "min_supersession_atomicity_crash_cases",
            int(MIN_SUPERSESSION_ATOMICITY_CRASH_CASES),
        ),
        (
            "max_replacement_visible_without_fence",
            int(MAX_REPLACEMENT_VISIBLE_WITHOUT_FENCE),
        ),
        ("compound_event", text(COMPOUND_EVENT)),
        (
            "compound_event_historical_alias",
            text(COMPOUND_EVENT_HISTORICAL_ALIAS),
        ),
        (
            "invalidation_event_fields",
            Leaf::strs(&INVALIDATION_EVENT_FIELDS),
        ),
        (
            "min_cross_proposal_evidence_reuse_attempts",
            int(MIN_CROSS_PROPOSAL_EVIDENCE_REUSE_ATTEMPTS),
        ),
        (
            "max_cross_proposal_evidence_admissions",
            int(MAX_CROSS_PROPOSAL_EVIDENCE_ADMISSIONS),
        ),
        (
            "min_cross_proposal_evidence_controls_accepted",
            int(MIN_CROSS_PROPOSAL_EVIDENCE_CONTROLS_ACCEPTED),
        ),
        ("evidence_binds", text(EVIDENCE_BINDS)),
        ("cross_proposal_example", text(CROSS_PROPOSAL_EXAMPLE)),
        (
            "proposal_identity_fields",
            Leaf::strs(&PROPOSAL_IDENTITY_FIELDS),
        ),
        ("invalidation_types", Leaf::strs(&INVALIDATION_TYPES)),
        ("input_kinds", Leaf::strs(&INPUT_KINDS)),
        (
            "min_post_promotion_training_revocation_cases",
            int(MIN_POST_PROMOTION_TRAINING_REVOCATION_CASES),
        ),
        (
            "post_promotion_supersession_cases_reported",
            text(POST_PROMOTION_SUPERSESSION_CASES_REPORTED),
        ),
        (
            "min_revocation_supersession_race_cases",
            int(MIN_REVOCATION_SUPERSESSION_RACE_CASES),
        ),
        (
            "max_wrong_generation_revocations",
            int(MAX_WRONG_GENERATION_REVOCATIONS),
        ),
        ("revoked_event_fields", Leaf::strs(&REVOKED_EVENT_FIELDS)),
        (
            "min_transitive_dependency_cases",
            int(MIN_TRANSITIVE_DEPENDENCY_CASES),
        ),
        (
            "max_derived_routable_after_transitive_invalidation",
            int(MAX_DERIVED_ROUTABLE_AFTER_TRANSITIVE_INVALIDATION),
        ),
        ("fence_set_closure", text(FENCE_SET_CLOSURE)),
        (
            "min_stale_policy_evidence_attempts",
            int(MIN_STALE_POLICY_EVIDENCE_ATTEMPTS),
        ),
        (
            "max_stale_policy_evidence_admissions",
            int(MAX_STALE_POLICY_EVIDENCE_ADMISSIONS),
        ),
        (
            "min_stale_policy_evidence_controls_accepted",
            int(MIN_STALE_POLICY_EVIDENCE_CONTROLS_ACCEPTED),
        ),
        ("gate_policy_epoch", text(GATE_POLICY_EPOCH)),
        (
            "outputs_after_invalidation_reported",
            text(OUTPUTS_AFTER_INVALIDATION_REPORTED),
        ),
        (
            "max_outputs_after_invalidation",
            int(MAX_OUTPUTS_AFTER_INVALIDATION),
        ),
        (
            "min_supersession_output_cases",
            int(MIN_SUPERSESSION_OUTPUT_CASES),
        ),
        (
            "multi_operand_op_kinds",
            Leaf::strs(&MULTI_OPERAND_OP_KINDS),
        ),
        (
            "stale_secondary_operand_cases_reported",
            text(STALE_SECONDARY_OPERAND_CASES_REPORTED),
        ),
        (
            "min_stale_secondary_operand_cases_per_kind",
            int(MIN_STALE_SECONDARY_OPERAND_CASES_PER_KIND),
        ),
        ("max_stale_operand_commits", int(MAX_STALE_OPERAND_COMMITS)),
        ("cas_operand_fields", Leaf::strs(&CAS_OPERAND_FIELDS)),
        (
            "birth_stale_operand_analogue",
            text(BIRTH_STALE_OPERAND_ANALOGUE),
        ),
        (
            "direct_append_event_types",
            Leaf::strs(&DIRECT_APPEND_EVENT_TYPES),
        ),
        (
            "max_unauthorized_direct_appends_admitted",
            int(MAX_UNAUTHORIZED_DIRECT_APPENDS_ADMITTED),
        ),
        (
            "gate_policy_epoch_items",
            Leaf::strs(&GATE_POLICY_EPOCH_ITEMS),
        ),
        (
            "invalidation_coverage_matrix.kinds",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_KINDS),
        ),
        (
            "invalidation_coverage_matrix.types",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_TYPES),
        ),
        (
            "invalidation_coverage_matrix.paths",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_PATHS),
        ),
        (
            "invalidation_coverage_matrix.authority_kinds",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_AUTHORITY_KINDS),
        ),
        (
            "invalidation_coverage_matrix.authority_types",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_AUTHORITY_TYPES),
        ),
        (
            "invalidation_coverage_matrix.cells",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_CELLS),
        ),
        (
            "invalidation_coverage_matrix.path_availability.inflight",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_PATH_AVAILABILITY_INFLIGHT),
        ),
        (
            "invalidation_coverage_matrix.path_availability.source_expert",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_PATH_AVAILABILITY_SOURCE_EXPERT),
        ),
        (
            "min_invalidation_matrix_cases_per_cell",
            int(MIN_INVALIDATION_MATRIX_CASES_PER_CELL),
        ),
        (
            "max_invalidation_matrix_stale_hits_per_cell",
            int(MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL),
        ),
        (
            "invalidation_matrix_path_meaning",
            text(INVALIDATION_MATRIX_PATH_MEANING),
        ),
        (
            "inflight_input_race_cells",
            Leaf::strs(&INFLIGHT_INPUT_RACE_CELLS),
        ),
        (
            "min_inflight_input_race_cases_per_cell",
            int(MIN_INFLIGHT_INPUT_RACE_CASES_PER_CELL),
        ),
        (
            "max_inflight_outputs_after_input_invalidation",
            int(MAX_INFLIGHT_OUTPUTS_AFTER_INPUT_INVALIDATION),
        ),
        (
            "impersonation_op_kinds",
            Leaf::strs(&IMPERSONATION_OP_KINDS),
        ),
        (
            "min_cross_project_impersonation_attempts_per_kind",
            int(MIN_CROSS_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND),
        ),
        (
            "min_same_project_proposals_accepted",
            int(MIN_SAME_PROJECT_PROPOSALS_ACCEPTED),
        ),
        (
            "max_impersonated_proposals_admitted",
            int(MAX_IMPERSONATED_PROPOSALS_ADMITTED),
        ),
        ("proposer_binding", text(PROPOSER_BINDING)),
        (
            "min_reactivate_lineage_cases",
            int(MIN_REACTIVATE_LINEAGE_CASES),
        ),
        ("lineage_deriving_ops", Leaf::strs(&LINEAGE_DERIVING_OPS)),
        (
            "min_fenced_gate_retire_attempts",
            int(MIN_FENCED_GATE_RETIRE_ATTEMPTS),
        ),
        (
            "max_fenced_gate_retires_admitted",
            int(MAX_FENCED_GATE_RETIRES_ADMITTED),
        ),
        (
            "min_reactivate_from_revoked_attempts",
            int(MIN_REACTIVATE_FROM_REVOKED_ATTEMPTS),
        ),
        (
            "max_reactivations_from_revoked_admitted",
            int(MAX_REACTIVATIONS_FROM_REVOKED_ADMITTED),
        ),
        ("producing_op_kinds", Leaf::strs(&PRODUCING_OP_KINDS)),
        (
            "min_colliding_result_race_cases_per_kind",
            int(MIN_COLLIDING_RESULT_RACE_CASES_PER_KIND),
        ),
        (
            "max_duplicate_result_commits",
            int(MAX_DUPLICATE_RESULT_COMMITS),
        ),
        (
            "result_identity_cas_covers",
            Leaf::strs(&RESULT_IDENTITY_CAS_COVERS),
        ),
        ("authority_kinds", Leaf::strs(&AUTHORITY_KINDS)),
        (
            "min_post_promotion_authority_revocation_cases_per_kind",
            int(MIN_POST_PROMOTION_AUTHORITY_REVOCATION_CASES_PER_KIND),
        ),
        (
            "max_routable_after_authority_revocation",
            int(MAX_ROUTABLE_AFTER_AUTHORITY_REVOCATION),
        ),
        (
            "authority_revocation_fences",
            text(AUTHORITY_REVOCATION_FENCES),
        ),
        (
            "min_planned_succession_cases_per_kind",
            int(MIN_PLANNED_SUCCESSION_CASES_PER_KIND),
        ),
        (
            "planned_succession_kinds",
            Leaf::strs(&PLANNED_SUCCESSION_KINDS),
        ),
        ("max_self_fenced_results", int(MAX_SELF_FENCED_RESULTS)),
        (
            "min_successor_takeover_controls_accepted_per_kind",
            int(MIN_SUCCESSOR_TAKEOVER_CONTROLS_ACCEPTED_PER_KIND),
        ),
        ("takeover_kinds", Leaf::strs(&TAKEOVER_KINDS)),
        (
            "duplicate_result_commits_definition",
            text(DUPLICATE_RESULT_COMMITS_DEFINITION),
        ),
        (
            "planned_succession_excluded_from_invalidation_rule",
            Leaf::Bool(PLANNED_SUCCESSION_EXCLUDED_FROM_INVALIDATION_RULE),
        ),
        (
            "min_tightening_rotation_cases_per_kind",
            int(MIN_TIGHTENING_ROTATION_CASES_PER_KIND),
        ),
        (
            "min_non_tightening_rotation_controls",
            int(MIN_NON_TIGHTENING_ROTATION_CONTROLS),
        ),
        (
            "max_routable_after_tightening_rotation",
            int(MAX_ROUTABLE_AFTER_TIGHTENING_ROTATION),
        ),
        (
            "tightening_rotation_definition",
            text(TIGHTENING_ROTATION_DEFINITION),
        ),
        ("tightening_rotation_fence", text(TIGHTENING_ROTATION_FENCE)),
        (
            "same_project_impersonation_op_kinds",
            Leaf::strs(&SAME_PROJECT_IMPERSONATION_OP_KINDS),
        ),
        (
            "min_same_project_impersonation_attempts_per_kind",
            int(MIN_SAME_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND),
        ),
        (
            "min_authority_quarantine_readmission_cases_per_kind",
            int(MIN_AUTHORITY_QUARANTINE_READMISSION_CASES_PER_KIND),
        ),
        (
            "min_readmission_controls_accepted",
            int(MIN_READMISSION_CONTROLS_ACCEPTED),
        ),
        (
            "max_readmitted_without_current_epoch_evidence",
            int(MAX_READMITTED_WITHOUT_CURRENT_EPOCH_EVIDENCE),
        ),
        (
            "min_input_revoked_readmission_attempts",
            int(MIN_INPUT_REVOKED_READMISSION_ATTEMPTS),
        ),
        (
            "max_input_revoked_readmissions_admitted",
            int(MAX_INPUT_REVOKED_READMISSIONS_ADMITTED),
        ),
        ("fence_cause_types", Leaf::strs(&FENCE_CAUSE_TYPES)),
        ("fence_cause_model", text(FENCE_CAUSE_MODEL)),
        ("readmission_event", text(READMISSION_EVENT)),
        (
            "readmittable_authority_kinds",
            Leaf::strs(&READMITTABLE_AUTHORITY_KINDS),
        ),
        (
            "min_proposer_grant_revocation_race_cases_per_kind",
            int(MIN_PROPOSER_GRANT_REVOCATION_RACE_CASES_PER_KIND),
        ),
        (
            "max_revoked_proposer_commits",
            int(MAX_REVOKED_PROPOSER_COMMITS),
        ),
        (
            "proposer_grant_op_kinds",
            Leaf::strs(&PROPOSER_GRANT_OP_KINDS),
        ),
        ("commit_authority_set", text(COMMIT_AUTHORITY_SET)),
        (
            "min_unauthorized_readmission_appends",
            int(MIN_UNAUTHORIZED_READMISSION_APPENDS),
        ),
        (
            "max_readmissions_without_gate_grant_admitted",
            int(MAX_READMISSIONS_WITHOUT_GATE_GRANT_ADMITTED),
        ),
        (
            "min_readmission_binding_cases",
            int(MIN_READMISSION_BINDING_CASES),
        ),
        (
            "max_mismatched_readmission_evidence_admitted",
            int(MAX_MISMATCHED_READMISSION_EVIDENCE_ADMITTED),
        ),
        ("gate_host_event_kinds", Leaf::strs(&GATE_HOST_EVENT_KINDS)),
        (
            "min_gate_grant_revocation_race_cases_per_kind",
            int(MIN_GATE_GRANT_REVOCATION_RACE_CASES_PER_KIND),
        ),
        (
            "max_revoked_gate_grant_commits",
            int(MAX_REVOKED_GATE_GRANT_COMMITS),
        ),
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
        ("weight_seed", int(WEIGHT_SEED)),
        ("hidden_size", int(HIDDEN_SIZE)),
        ("num_heads", int(NUM_HEADS)),
        ("num_kv_heads", int(NUM_KV_HEADS)),
        ("head_dim", int(HEAD_DIM)),
        ("vocab_size", int(VOCAB_SIZE)),
        ("tiny_model_config.layers", int(TINY_MODEL_CONFIG_LAYERS)),
        (
            "tiny_model_config.hidden_size",
            int(TINY_MODEL_CONFIG_HIDDEN_SIZE),
        ),
        (
            "tiny_model_config.num_heads",
            int(TINY_MODEL_CONFIG_NUM_HEADS),
        ),
        (
            "tiny_model_config.num_kv_heads",
            int(TINY_MODEL_CONFIG_NUM_KV_HEADS),
        ),
        (
            "tiny_model_config.head_dim",
            int(TINY_MODEL_CONFIG_HEAD_DIM),
        ),
        (
            "tiny_model_config.vocab_size",
            int(TINY_MODEL_CONFIG_VOCAB_SIZE),
        ),
        (
            "tiny_model_config.weight_seed",
            int(TINY_MODEL_CONFIG_WEIGHT_SEED),
        ),
        (
            "tiny_model_config.num_experts",
            int(TINY_MODEL_CONFIG_NUM_EXPERTS),
        ),
        ("tiny_model_config.top_k", int(TINY_MODEL_CONFIG_TOP_K)),
        ("tiny_model_config_digest", text(TINY_MODEL_CONFIG_DIGEST)),
        (
            "tiny_model_config_digest_rule",
            text(TINY_MODEL_CONFIG_DIGEST_RULE),
        ),
        ("tokenizer_pin", text(TOKENIZER_PIN)),
        ("moe_topology.num_experts", int(MOE_TOPOLOGY_NUM_EXPERTS)),
        ("moe_topology.top_k", int(MOE_TOPOLOGY_TOP_K)),
        (
            "moe_topology.router_layout",
            text(MOE_TOPOLOGY_ROUTER_LAYOUT),
        ),
        (
            "model_pins_recorded_in_run_json",
            Leaf::Bool(MODEL_PINS_RECORDED_IN_RUN_JSON),
        ),
        ("max_model_pin_mismatches", int(MAX_MODEL_PIN_MISMATCHES)),
        ("lineage_edge_types", Leaf::strs(&LINEAGE_EDGE_TYPES)),
        ("planned_succession_edge", text(PLANNED_SUCCESSION_EDGE)),
        (
            "min_planned_succession_routing_controls_per_kind",
            int(MIN_PLANNED_SUCCESSION_ROUTING_CONTROLS_PER_KIND),
        ),
        (
            "max_planned_successor_routes_rejected",
            int(MAX_PLANNED_SUCCESSOR_ROUTES_REJECTED),
        ),
        (
            "min_session_revocation_race_cases",
            int(MIN_SESSION_REVOCATION_RACE_CASES),
        ),
        (
            "min_session_expiry_race_cases",
            int(MIN_SESSION_EXPIRY_RACE_CASES),
        ),
        (
            "max_structural_commits_after_session_end",
            int(MAX_STRUCTURAL_COMMITS_AFTER_SESSION_END),
        ),
        (
            "session_revalidation_at_commit",
            text(SESSION_REVALIDATION_AT_COMMIT),
        ),
        ("identity_dependencies", Leaf::strs(&IDENTITY_DEPENDENCIES)),
        (
            "min_identity_policy_rotation_race_cases",
            int(MIN_IDENTITY_POLICY_ROTATION_RACE_CASES),
        ),
        (
            "min_identity_policy_revocation_race_cases",
            int(MIN_IDENTITY_POLICY_REVOCATION_RACE_CASES),
        ),
        (
            "min_peer_deadmission_race_cases",
            int(MIN_PEER_DEADMISSION_RACE_CASES),
        ),
        (
            "max_structural_commits_after_identity_dependency_change",
            int(MAX_STRUCTURAL_COMMITS_AFTER_IDENTITY_DEPENDENCY_CHANGE),
        ),
        (
            "identity_dependency_race_counters.session_revocation",
            text(IDENTITY_DEPENDENCY_RACE_COUNTERS_SESSION_REVOCATION),
        ),
        (
            "identity_dependency_race_counters.session_expiry",
            text(IDENTITY_DEPENDENCY_RACE_COUNTERS_SESSION_EXPIRY),
        ),
        (
            "identity_dependency_race_counters.identity_policy_rotation",
            text(IDENTITY_DEPENDENCY_RACE_COUNTERS_IDENTITY_POLICY_ROTATION),
        ),
        (
            "identity_dependency_race_counters.identity_policy_revocation",
            text(IDENTITY_DEPENDENCY_RACE_COUNTERS_IDENTITY_POLICY_REVOCATION),
        ),
        (
            "identity_dependency_race_counters.peer_deadmission",
            text(IDENTITY_DEPENDENCY_RACE_COUNTERS_PEER_DEADMISSION),
        ),
        (
            "min_stale_structural_dependency_commit_cases",
            int(MIN_STALE_STRUCTURAL_DEPENDENCY_COMMIT_CASES),
        ),
        (
            "max_structural_commits_with_stale_dependency",
            int(MAX_STRUCTURAL_COMMITS_WITH_STALE_DEPENDENCY),
        ),
        (
            "commit_liveness_cas_covers",
            Leaf::strs(&COMMIT_LIVENESS_CAS_COVERS),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The leaves of a parsed TOML value, flattened as `compiled` flattens them.
    fn flatten(prefix: &str, value: &toml::Value, out: &mut BTreeMap<String, Leaf>) {
        match value {
            toml::Value::Table(table) => {
                for (key, inner) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    flatten(&path, inner, out);
                }
            }
            toml::Value::Integer(number) => {
                out.insert(prefix.to_string(), Leaf::Int(*number));
            }
            toml::Value::Boolean(flag) => {
                out.insert(prefix.to_string(), Leaf::Bool(*flag));
            }
            toml::Value::String(item) => {
                out.insert(prefix.to_string(), Leaf::Str(item.clone()));
            }
            toml::Value::Array(items) => {
                out.insert(
                    prefix.to_string(),
                    Leaf::Strs(
                        items
                            .iter()
                            .map(|item| match item {
                                toml::Value::String(text) => text.clone(),
                                toml::Value::Array(pair) => pair
                                    .iter()
                                    .map(|part| part.as_str().expect("a string pair").to_string())
                                    .collect::<Vec<_>>()
                                    .join(":"),
                                other => panic!("{prefix}: unexpected list item {other:?}"),
                            })
                            .collect(),
                    ),
                );
            }
            other => panic!("{prefix}: unexpected value {other:?}"),
        }
    }

    fn file_table(config: &str) -> BTreeMap<String, Leaf> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let thresholds = parsed
            .get("pinned_thresholds")
            .expect("[pinned_thresholds]")
            .clone();
        let mut out = BTreeMap::new();
        flatten("", &thresholds, &mut out);
        out
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each.
    fn drift(config: &str) -> Vec<String> {
        let file = file_table(config);
        let compiled: BTreeMap<String, Leaf> = compiled()
            .into_iter()
            .map(|(key, leaf)| (key.to_string(), leaf))
            .collect();
        let mut problems = Vec::new();
        for (key, value) in &file {
            match compiled.get(key) {
                None => problems.push(format!("{key}: no constant")),
                Some(constant) if constant != value => problems.push(format!("{key}: drifted")),
                Some(_) => {}
            }
        }
        for key in compiled.keys().filter(|key| !file.contains_key(*key)) {
            problems.push(format!("{key}: no key in the file"));
        }
        problems
    }

    #[test]
    fn the_compiled_constants_are_the_pinned_thresholds_of_config_toml() {
        assert_eq!(drift(CONFIG_TOML), Vec::<String>::new());
    }

    #[test]
    fn a_changed_added_or_removed_key_of_config_toml_is_drift() {
        let changed = CONFIG_TOML.replace("weight_seed = 20240607\n", "weight_seed = 20240608\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["weight_seed: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("min_attempts_per_case = 1\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(
            drift(&removed),
            ["min_attempts_per_case: no key in the file"]
        );
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_pins_the_harness_shares_are_the_configs() {
        use crate::experiments::layered_kv::pins;
        let shape = pins::SHAPE;
        assert_eq!(shape.layers, TINY_MODEL_CONFIG_LAYERS);
        assert_eq!(shape.hidden_size, TINY_MODEL_CONFIG_HIDDEN_SIZE);
        assert_eq!(shape.num_heads, TINY_MODEL_CONFIG_NUM_HEADS);
        assert_eq!(shape.num_kv_heads, TINY_MODEL_CONFIG_NUM_KV_HEADS);
        assert_eq!(shape.head_dim, TINY_MODEL_CONFIG_HEAD_DIM);
        assert_eq!(shape.vocab_size, TINY_MODEL_CONFIG_VOCAB_SIZE);
        assert_eq!(shape.weight_seed, TINY_MODEL_CONFIG_WEIGHT_SEED);
        assert_eq!(shape.layers, TEST_MODEL_LAYERS);
        assert_eq!(shape.weight_seed, WEIGHT_SEED);
        assert_eq!(crate::experiments::layered_kv::SEQUENCE_SEED, WEIGHT_SEED,);
        assert_eq!(
            MAX_NEW_TOKENS as usize,
            ptr_runtime::ReferenceKvLimits::default().max_new_tokens
        );
        assert_eq!(SEGMENT_TOKENS as usize, ptr_runtime::KV_CHUNK_TOKENS);
        assert_eq!(
            SEGMENT_TOKENS as usize,
            crate::experiments::layered_kv::corpus::CHUNK_TOKENS
        );
    }

    /// The pinned digest is the digest of the config's own `tiny_model_config` table.
    #[test]
    fn the_pinned_digest_is_the_digest_of_the_config_table() {
        use crate::experiments::layered_kv::pins;
        let mut fields = pins::shape_fields();
        fields.push(("num_experts", TINY_MODEL_CONFIG_NUM_EXPERTS));
        fields.push(("top_k", TINY_MODEL_CONFIG_TOP_K));
        assert_eq!(pins::canonical_digest(&fields), TINY_MODEL_CONFIG_DIGEST);
        assert_eq!(TINY_MODEL_CONFIG_NUM_EXPERTS, MOE_TOPOLOGY_NUM_EXPERTS);
        assert_eq!(TINY_MODEL_CONFIG_TOP_K, MOE_TOPOLOGY_TOP_K);
    }
}
