//! The metric list of R011 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R011).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`, in the order of
//! that text: every "< N" prerequisite becomes `>= N`, every "> 0" violation becomes `<= 0`,
//! every "!=" of two digests becomes `==`. A family of cells (a by-type, per-case or per-cell
//! object) has one rule per member, so a missing member is an unmet prerequisite and not a
//! silent zero. The harness judges its own output with them, and a test pins `criteria.toml` to
//! exactly these lists.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 131] = [
    "uncommitted_effective_changes",
    "gate_bypass_count",
    "verifier_override_count",
    "verifier_bypass_count",
    "revoked_expert_served_count",
    "forbidden_control_mutation_count",
    "adversarial_attempts_per_case",
    "direct_promotion_bypass_count",
    "direct_serving_mutation_bypass_count",
    "stale_generation_commits_accepted",
    "routes_to_revoked_dependency_experts",
    "stale_generation_commits_accepted_by_type",
    "stale_digest_commits_accepted_by_type",
    "direct_ledger_append_accepted_by_type",
    "commits_with_revoked_input_accepted",
    "expert_routes_after_revocation_before_fence",
    "expert_routes_after_replay",
    "evidence_binding_cases_accepted_by_case",
    "control_accepts_by_path",
    "expert_routes_after_crash_between_append_and_apply",
    "replay_fence_set_mismatch_served",
    "evidence_binding_controls_accepted_by_case",
    "stale_digest_commits_accepted",
    "revocation_driven_retirement_control_accepts",
    "gate_less_retirement_accepted",
    "stale_input_generation_commit_attempts",
    "stale_input_digest_commit_attempts",
    "stale_input_generation_commits_admitted",
    "stale_input_digest_commits_admitted",
    "stale_input_commit_controls_accepted",
    "runtime_fence_with_grant_succeeded",
    "l3_fence_attempts_admitted",
    "fence_without_runtime_grant_admitted",
    "stale_eval_input_commit_attempts",
    "stale_eval_input_commits_admitted",
    "large_fanout_revocation_cases",
    "experts_serving_after_revocation",
    "fence_set_digest_mismatches",
    "post_promotion_eval_revocation_cases",
    "routable_after_eval_revocation",
    "evidence_binding_negative_cases_per_event_type",
    "evidence_binding_admissions",
    "stale_eval_input_supersession_commit_attempts",
    "stale_eval_input_supersession_commits_admitted",
    "derived_dependency_set_size",
    "fenced_and_probed_dependents",
    "inflight_expert_call_race_cases",
    "inflight_expert_outputs_after_revocation",
    "post_promotion_supersession_cases",
    "routable_after_supersession",
    "supersession_inflight_expert_call_cases",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "inflight_expert_call_fallback_incomplete",
    "inflight_expert_call_fallback_output_mismatches",
    "structural_dependency_invalidation_cases",
    "routable_after_structural_invalidation",
    "supersession_atomicity_crash_cases",
    "replacement_visible_without_fence",
    "cross_proposal_evidence_reuse_attempts",
    "cross_proposal_evidence_admissions",
    "cross_proposal_evidence_controls_accepted",
    "post_promotion_training_revocation_cases",
    "fenced_and_probed_dependents_set_digest",
    "derived_dependency_set_digest",
    "revocation_supersession_race_cases",
    "wrong_generation_revocations",
    "transitive_dependency_cases",
    "derived_routable_after_transitive_invalidation",
    "stale_policy_evidence_attempts",
    "stale_policy_evidence_admissions",
    "stale_policy_evidence_controls_accepted",
    "outputs_after_invalidation",
    "supersession_output_cases",
    "stale_secondary_operand_cases",
    "stale_operand_commits",
    "direct_append_attempted_event_type_set_digest",
    "admitted_event_type_set_digest",
    "unauthorized_direct_appends_admitted",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "inflight_input_race_cases",
    "inflight_outputs_after_input_invalidation",
    "cross_project_impersonation_attempts",
    "same_project_proposals_accepted",
    "impersonated_proposals_admitted",
    "reactivate_lineage_cases",
    "fenced_gate_retire_attempts",
    "fenced_gate_retires_admitted",
    "reactivate_from_revoked_attempts",
    "reactivations_from_revoked_admitted",
    "colliding_result_race_cases",
    "duplicate_result_commits",
    "post_promotion_authority_revocation_cases",
    "routable_after_authority_revocation",
    "impersonation_attempted_op_kind_set_digest",
    "admitted_op_kind_set_digest",
    "planned_succession_cases",
    "self_fenced_results",
    "successor_takeover_controls_accepted",
    "tightening_rotation_cases",
    "non_tightening_rotation_controls",
    "routable_after_tightening_rotation",
    "same_project_impersonation_attempts",
    "authority_quarantine_readmission_cases",
    "readmission_controls_accepted",
    "readmitted_without_current_epoch_evidence",
    "input_revoked_readmission_attempts",
    "input_revoked_readmissions_admitted",
    "proposer_grant_revocation_race_cases",
    "revoked_proposer_commits",
    "unauthorized_readmission_appends",
    "readmissions_without_gate_grant_admitted",
    "readmission_binding_cases",
    "mismatched_readmission_evidence_admitted",
    "gate_grant_revocation_race_cases",
    "revoked_gate_grant_commits",
    "gate_grant_race_kind_set_digest",
    "gate_host_issued_kind_set_digest",
    "model_pin_mismatches",
    "planned_succession_routing_controls",
    "planned_successor_routes_rejected",
    "session_revocation_race_cases",
    "session_expiry_race_cases",
    "structural_commits_after_session_end",
    "identity_policy_rotation_race_cases",
    "identity_policy_revocation_race_cases",
    "peer_deadmission_race_cases",
    "structural_commits_after_identity_dependency_change",
    "stale_structural_dependency_commit_cases",
    "structural_commits_with_stale_dependency",
];

/// The metrics printed as an object (a by-type, per-case, per-kind or per-cell family): the
/// members are the keys, `total` is the scalar aggregate. Every other metric is an integer or a
/// digest string.
pub const NESTED: [&str; 28] = [
    "adversarial_attempts_per_case",
    "stale_generation_commits_accepted_by_type",
    "stale_digest_commits_accepted_by_type",
    "direct_ledger_append_accepted_by_type",
    "evidence_binding_cases_accepted_by_case",
    "control_accepts_by_path",
    "evidence_binding_controls_accepted_by_case",
    "evidence_binding_negative_cases_per_event_type",
    "inflight_expert_call_race_cases",
    "post_promotion_supersession_cases",
    "structural_dependency_invalidation_cases",
    "stale_secondary_operand_cases",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "inflight_input_race_cases",
    "inflight_outputs_after_input_invalidation",
    "outputs_after_invalidation",
    "cross_project_impersonation_attempts",
    "same_project_impersonation_attempts",
    "colliding_result_race_cases",
    "post_promotion_authority_revocation_cases",
    "planned_succession_cases",
    "successor_takeover_controls_accepted",
    "tightening_rotation_cases",
    "authority_quarantine_readmission_cases",
    "proposer_grant_revocation_race_cases",
    "gate_grant_revocation_race_cases",
    "planned_succession_routing_controls",
];

/// The metrics that are digest strings (set equalities).
#[cfg_attr(not(test), allow(dead_code))]
pub const DIGESTS: [&str; 8] = [
    "direct_append_attempted_event_type_set_digest",
    "admitted_event_type_set_digest",
    "impersonation_attempted_op_kind_set_digest",
    "admitted_op_kind_set_digest",
    "gate_grant_race_kind_set_digest",
    "gate_host_issued_kind_set_digest",
    "fenced_and_probed_dependents_set_digest",
    "derived_dependency_set_digest",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The members of the object `name` that a run must fill, in a stable order: the pinned
/// cells of the family. A two-level family lists `outer.inner`. Used to seed the zero-based
/// families and to word the rules.
pub fn cells(name: &str) -> Vec<String> {
    let owned = |items: &[&str]| {
        items
            .iter()
            .map(|item| (*item).to_string())
            .collect::<Vec<_>>()
    };
    let cross = |outer: &[&str], inner: &[&str]| {
        outer
            .iter()
            .flat_map(|o| inner.iter().map(move |i| format!("{o}.{i}")))
            .collect::<Vec<_>>()
    };
    match name {
        "adversarial_attempts_per_case" => owned(&params::ADVERSARIAL_CASES),
        "stale_generation_commits_accepted_by_type" | "stale_digest_commits_accepted_by_type" => {
            owned(&STRUCTURAL_TYPES)
        }
        "direct_ledger_append_accepted_by_type" => owned(&params::DIRECT_APPEND_EVENT_TYPES),
        "evidence_binding_cases_accepted_by_case"
        | "evidence_binding_controls_accepted_by_case" => {
            owned(&params::EVIDENCE_BINDING_NEGATIVE_CASES)
        }
        "control_accepts_by_path" => owned(&params::CONTROL_PATHS),
        "evidence_binding_negative_cases_per_event_type" => cross(
            &params::EVIDENCE_BINDING_EVENT_TYPES,
            &params::EVIDENCE_BINDING_NEGATIVE_KINDS,
        ),
        "inflight_expert_call_race_cases" => cross(
            &params::INFLIGHT_EXPERT_CALL_LEVELS,
            &params::INVALIDATION_TYPES,
        ),
        "post_promotion_supersession_cases" => owned(&params::INPUT_KINDS),
        "structural_dependency_invalidation_cases" => owned(&STRUCTURAL_CELLS),
        "stale_secondary_operand_cases" => owned(&params::MULTI_OPERAND_OP_KINDS),
        "invalidation_matrix_cases" | "invalidation_matrix_stale_hits" => {
            owned(&params::INVALIDATION_COVERAGE_MATRIX_CELLS)
        }
        "inflight_input_race_cases" | "inflight_outputs_after_input_invalidation" => {
            params::INFLIGHT_INPUT_RACE_CELLS
                .iter()
                .map(|cell| cell.replace(':', "."))
                .collect()
        }
        "outputs_after_invalidation" => owned(&params::INVALIDATION_TYPES),
        "cross_project_impersonation_attempts" => owned(&params::IMPERSONATION_OP_KINDS),
        "same_project_impersonation_attempts" => owned(&params::IMPERSONATION_OP_KINDS),
        "colliding_result_race_cases" => owned(&params::PRODUCING_OP_KINDS),
        "post_promotion_authority_revocation_cases"
        | "tightening_rotation_cases"
        | "authority_quarantine_readmission_cases" => owned(&params::AUTHORITY_KINDS),
        "planned_succession_cases" | "planned_succession_routing_controls" => {
            owned(&params::PLANNED_SUCCESSION_KINDS)
        }
        "successor_takeover_controls_accepted" => owned(&params::TAKEOVER_KINDS),
        "proposer_grant_revocation_race_cases" => owned(&params::PROPOSER_GRANT_OP_KINDS),
        "gate_grant_revocation_race_cases" => owned(&params::GATE_HOST_EVENT_KINDS),
        _ => Vec::new(),
    }
}

/// The seven operation types, by the event name the draft gives them.
pub const STRUCTURAL_TYPES: [&str; 7] = [
    "ExpertPromoted",
    "ExpertRetired",
    "Merge",
    "Split",
    "Prune",
    "Freeze",
    "Reactivate",
];

/// `structural_dependency_invalidation_cases` cells: kind x type, as the config reports them
/// (base_model and adapter take revocation and supersession, source_expert revocation).
pub const STRUCTURAL_CELLS: [&str; 5] = [
    "base_model.revocation",
    "base_model.supersession",
    "adapter.revocation",
    "adapter.supersession",
    "source_expert.revocation",
];

fn at_least(path: &str, bound: u64) -> String {
    format!("{path} >= {bound}")
}

fn at_most(path: &str, bound: u64) -> String {
    format!("{path} <= {bound}")
}

/// One rule per member of the family `name`: `name.member >= bound`.
fn each_at_least(name: &str, bound: u64, out: &mut Vec<String>) {
    out.extend(
        cells(name)
            .iter()
            .map(|cell| at_least(&format!("{name}.{cell}"), bound)),
    );
}

fn each_at_most(name: &str, bound: u64, out: &mut Vec<String>) {
    out.extend(
        cells(name)
            .iter()
            .map(|cell| at_most(&format!("{name}.{cell}"), bound)),
    );
}

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous, in the
/// order of the falsification text.
pub fn coverage() -> Vec<String> {
    let mut rules = Vec::new();
    // Every case of the pinned adversarial suite, including the direct appends and the stale
    // generation and digest commits per operation type, was attempted.
    each_at_least(
        "adversarial_attempts_per_case",
        params::MIN_ATTEMPTS_PER_CASE,
        &mut rules,
    );
    each_at_least(
        "stale_secondary_operand_cases",
        params::MIN_STALE_SECONDARY_OPERAND_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "evidence_binding_negative_cases_per_event_type",
        params::MIN_EVIDENCE_BINDING_NEGATIVE_CASES_PER_EVENT_TYPE,
        &mut rules,
    );
    each_at_least(
        "evidence_binding_controls_accepted_by_case",
        params::MIN_EVIDENCE_BINDING_CONTROLS_ACCEPTED_PER_CASE,
        &mut rules,
    );
    each_at_least(
        "control_accepts_by_path",
        params::MIN_CONTROL_ACCEPTS_PER_PATH,
        &mut rules,
    );
    rules.push(at_least(
        "cross_proposal_evidence_reuse_attempts",
        params::MIN_CROSS_PROPOSAL_EVIDENCE_REUSE_ATTEMPTS,
    ));
    rules.push(at_least(
        "cross_proposal_evidence_controls_accepted",
        params::MIN_CROSS_PROPOSAL_EVIDENCE_CONTROLS_ACCEPTED,
    ));
    each_at_least(
        "cross_project_impersonation_attempts",
        params::MIN_CROSS_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "same_project_impersonation_attempts",
        params::MIN_SAME_PROJECT_IMPERSONATION_ATTEMPTS_PER_KIND,
        &mut rules,
    );
    rules.push(
        "impersonation_attempted_op_kind_set_digest == admitted_op_kind_set_digest".to_string(),
    );
    rules.push(at_least(
        "same_project_proposals_accepted",
        params::MIN_SAME_PROJECT_PROPOSALS_ACCEPTED,
    ));
    rules.push(at_least(
        "runtime_fence_with_grant_succeeded",
        params::MIN_RUNTIME_FENCE_WITH_GRANT_SUCCEEDED,
    ));
    for (name, bound) in [
        (
            "fenced_gate_retire_attempts",
            params::MIN_FENCED_GATE_RETIRE_ATTEMPTS,
        ),
        (
            "reactivate_from_revoked_attempts",
            params::MIN_REACTIVATE_FROM_REVOKED_ATTEMPTS,
        ),
        (
            "input_revoked_readmission_attempts",
            params::MIN_INPUT_REVOKED_READMISSION_ATTEMPTS,
        ),
        (
            "readmission_binding_cases",
            params::MIN_READMISSION_BINDING_CASES,
        ),
        (
            "unauthorized_readmission_appends",
            params::MIN_UNAUTHORIZED_READMISSION_APPENDS,
        ),
        // The revocation-driven retirement control is a control of the path as well as a
        // counter of its own; the manifest names it with the readmission prerequisites.
        (
            "revocation_driven_retirement_control_accepts",
            params::MIN_CONTROL_ACCEPTS_PER_PATH,
        ),
    ] {
        rules.push(at_least(name, bound));
    }
    each_at_least(
        "colliding_result_race_cases",
        params::MIN_COLLIDING_RESULT_RACE_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "planned_succession_cases",
        params::MIN_PLANNED_SUCCESSION_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "planned_succession_routing_controls",
        params::MIN_PLANNED_SUCCESSION_ROUTING_CONTROLS_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "successor_takeover_controls_accepted",
        params::MIN_SUCCESSOR_TAKEOVER_CONTROLS_ACCEPTED_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "proposer_grant_revocation_race_cases",
        params::MIN_PROPOSER_GRANT_REVOCATION_RACE_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "gate_grant_revocation_race_cases",
        params::MIN_GATE_GRANT_REVOCATION_RACE_CASES_PER_KIND,
        &mut rules,
    );
    rules.push("gate_grant_race_kind_set_digest == gate_host_issued_kind_set_digest".to_string());
    for (name, bound) in [
        (
            "session_revocation_race_cases",
            params::MIN_SESSION_REVOCATION_RACE_CASES,
        ),
        (
            "session_expiry_race_cases",
            params::MIN_SESSION_EXPIRY_RACE_CASES,
        ),
        (
            "identity_policy_rotation_race_cases",
            params::MIN_IDENTITY_POLICY_ROTATION_RACE_CASES,
        ),
        (
            "identity_policy_revocation_race_cases",
            params::MIN_IDENTITY_POLICY_REVOCATION_RACE_CASES,
        ),
        (
            "peer_deadmission_race_cases",
            params::MIN_PEER_DEADMISSION_RACE_CASES,
        ),
        (
            "stale_input_generation_commit_attempts",
            params::MIN_STALE_INPUT_GENERATION_COMMIT_ATTEMPTS,
        ),
        (
            "stale_input_digest_commit_attempts",
            params::MIN_STALE_INPUT_DIGEST_COMMIT_ATTEMPTS,
        ),
        (
            "stale_input_commit_controls_accepted",
            params::MIN_STALE_INPUT_COMMIT_CONTROLS_ACCEPTED,
        ),
        (
            "stale_eval_input_commit_attempts",
            params::MIN_STALE_EVAL_INPUT_COMMIT_ATTEMPTS,
        ),
        (
            "stale_eval_input_supersession_commit_attempts",
            params::MIN_STALE_EVAL_INPUT_SUPERSESSION_COMMIT_ATTEMPTS,
        ),
        (
            "stale_structural_dependency_commit_cases",
            params::MIN_STALE_STRUCTURAL_DEPENDENCY_COMMIT_CASES,
        ),
        (
            "post_promotion_training_revocation_cases",
            params::MIN_POST_PROMOTION_TRAINING_REVOCATION_CASES,
        ),
        (
            "post_promotion_eval_revocation_cases",
            params::MIN_POST_PROMOTION_EVAL_REVOCATION_CASES,
        ),
    ] {
        rules.push(at_least(name, bound));
    }
    each_at_least(
        "post_promotion_supersession_cases",
        params::MIN_POST_PROMOTION_SUPERSESSION_CASES,
        &mut rules,
    );
    for (name, bound) in [
        (
            "revocation_supersession_race_cases",
            params::MIN_REVOCATION_SUPERSESSION_RACE_CASES,
        ),
        (
            "large_fanout_revocation_cases",
            params::MIN_LARGE_FANOUT_REVOCATION_CASES,
        ),
        (
            "transitive_dependency_cases",
            params::MIN_TRANSITIVE_DEPENDENCY_CASES,
        ),
        (
            "reactivate_lineage_cases",
            params::MIN_REACTIVATE_LINEAGE_CASES,
        ),
    ] {
        rules.push(at_least(name, bound));
    }
    each_at_least(
        "structural_dependency_invalidation_cases",
        params::MIN_STRUCTURAL_DEPENDENCY_INVALIDATION_CASES_PER_KIND,
        &mut rules,
    );
    rules.push(at_least(
        "supersession_atomicity_crash_cases",
        params::MIN_SUPERSESSION_ATOMICITY_CRASH_CASES,
    ));
    each_at_least(
        "invalidation_matrix_cases",
        params::MIN_INVALIDATION_MATRIX_CASES_PER_CELL,
        &mut rules,
    );
    each_at_least(
        "inflight_expert_call_race_cases",
        params::MIN_INFLIGHT_EXPERT_CALL_RACE_CASES,
        &mut rules,
    );
    rules.push(at_least(
        "supersession_inflight_expert_call_cases",
        params::MIN_SUPERSESSION_INFLIGHT_EXPERT_CALL_CASES,
    ));
    rules.push(at_least(
        "supersession_output_cases",
        params::MIN_SUPERSESSION_OUTPUT_CASES,
    ));
    each_at_least(
        "inflight_input_race_cases",
        params::MIN_INFLIGHT_INPUT_RACE_CASES_PER_CELL,
        &mut rules,
    );
    rules.push(at_least(
        "stale_policy_evidence_attempts",
        params::MIN_STALE_POLICY_EVIDENCE_ATTEMPTS,
    ));
    rules.push(at_least(
        "stale_policy_evidence_controls_accepted",
        params::MIN_STALE_POLICY_EVIDENCE_CONTROLS_ACCEPTED,
    ));
    each_at_least(
        "post_promotion_authority_revocation_cases",
        params::MIN_POST_PROMOTION_AUTHORITY_REVOCATION_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "tightening_rotation_cases",
        params::MIN_TIGHTENING_ROTATION_CASES_PER_KIND,
        &mut rules,
    );
    each_at_least(
        "authority_quarantine_readmission_cases",
        params::MIN_AUTHORITY_QUARANTINE_READMISSION_CASES_PER_KIND,
        &mut rules,
    );
    rules.push(at_least(
        "non_tightening_rotation_controls",
        params::MIN_NON_TIGHTENING_ROTATION_CONTROLS,
    ));
    rules.push(at_least(
        "readmission_controls_accepted",
        params::MIN_READMISSION_CONTROLS_ACCEPTED,
    ));
    for (name, bound) in [
        (
            "selection_digest_mismatches",
            params::MAX_SELECTION_DIGEST_MISMATCHES,
        ),
        ("procedure_mismatches", params::MAX_PROCEDURE_MISMATCHES),
        ("model_pin_mismatches", params::MAX_MODEL_PIN_MISMATCHES),
        (
            "inflight_expert_call_fallback_incomplete",
            params::MAX_INFLIGHT_EXPERT_CALL_FALLBACK_INCOMPLETE,
        ),
    ] {
        rules.push(at_most(name, bound));
    }
    rules
}

/// The falsifiers of the manifest: each tripped one falsifies R011 on this seed, in the order
/// of the falsification text.
pub fn falsifiers() -> Vec<String> {
    let mut rules = Vec::new();
    for (name, bound) in [
        ("uncommitted_effective_changes", 0),
        (
            "forbidden_control_mutation_count",
            params::MAX_FORBIDDEN_CONTROL_MUTATION_COUNT,
        ),
        ("gate_bypass_count", 0),
        (
            "verifier_override_count",
            params::MAX_VERIFIER_OVERRIDE_COUNT,
        ),
        ("verifier_bypass_count", params::MAX_VERIFIER_BYPASS_COUNT),
        (
            "direct_promotion_bypass_count",
            params::MAX_DIRECT_PROMOTION_BYPASS_COUNT,
        ),
        (
            "direct_serving_mutation_bypass_count",
            params::MAX_DIRECT_SERVING_MUTATION_BYPASS_COUNT,
        ),
    ] {
        rules.push(at_most(name, bound));
    }
    each_at_most(
        "direct_ledger_append_accepted_by_type",
        params::MAX_DIRECT_LEDGER_APPEND_ACCEPTED_PER_TYPE,
        &mut rules,
    );
    rules.push(at_most(
        "unauthorized_direct_appends_admitted",
        params::MAX_UNAUTHORIZED_DIRECT_APPENDS_ADMITTED,
    ));
    rules.push(
        "direct_append_attempted_event_type_set_digest == admitted_event_type_set_digest"
            .to_string(),
    );
    rules.push(at_most(
        "stale_generation_commits_accepted",
        params::MAX_STALE_GENERATION_COMMITS_ACCEPTED,
    ));
    rules.push(at_most(
        "stale_digest_commits_accepted",
        params::MAX_STALE_DIGEST_COMMITS_ACCEPTED,
    ));
    each_at_most(
        "stale_generation_commits_accepted_by_type",
        params::MAX_STALE_GENERATION_COMMITS_ACCEPTED_PER_TYPE,
        &mut rules,
    );
    each_at_most(
        "stale_digest_commits_accepted_by_type",
        params::MAX_STALE_DIGEST_COMMITS_ACCEPTED_PER_TYPE,
        &mut rules,
    );
    rules.push(at_most(
        "stale_operand_commits",
        params::MAX_STALE_OPERAND_COMMITS,
    ));
    each_at_most(
        "evidence_binding_cases_accepted_by_case",
        params::MAX_EVIDENCE_BINDING_CASES_ACCEPTED_PER_CASE,
        &mut rules,
    );
    for (name, bound) in [
        (
            "evidence_binding_admissions",
            params::MAX_EVIDENCE_BINDING_ADMISSIONS,
        ),
        (
            "cross_proposal_evidence_admissions",
            params::MAX_CROSS_PROPOSAL_EVIDENCE_ADMISSIONS,
        ),
        (
            "impersonated_proposals_admitted",
            params::MAX_IMPERSONATED_PROPOSALS_ADMITTED,
        ),
        (
            "l3_fence_attempts_admitted",
            params::MAX_L3_FENCE_ATTEMPTS_ADMITTED,
        ),
        (
            "fence_without_runtime_grant_admitted",
            params::MAX_FENCE_WITHOUT_RUNTIME_GRANT_ADMITTED,
        ),
        ("revoked_expert_served_count", 0),
        (
            "routes_to_revoked_dependency_experts",
            params::MAX_ROUTES_TO_REVOKED_DEPENDENCY_EXPERTS,
        ),
        (
            "expert_routes_after_replay",
            params::MAX_EXPERT_ROUTES_AFTER_REPLAY,
        ),
        (
            "gate_less_retirement_accepted",
            params::MAX_GATE_LESS_RETIREMENT_ACCEPTED,
        ),
        (
            "input_revoked_readmissions_admitted",
            params::MAX_INPUT_REVOKED_READMISSIONS_ADMITTED,
        ),
        (
            "fenced_gate_retires_admitted",
            params::MAX_FENCED_GATE_RETIRES_ADMITTED,
        ),
        (
            "reactivations_from_revoked_admitted",
            params::MAX_REACTIVATIONS_FROM_REVOKED_ADMITTED,
        ),
        (
            "mismatched_readmission_evidence_admitted",
            params::MAX_MISMATCHED_READMISSION_EVIDENCE_ADMITTED,
        ),
        (
            "readmissions_without_gate_grant_admitted",
            params::MAX_READMISSIONS_WITHOUT_GATE_GRANT_ADMITTED,
        ),
        (
            "duplicate_result_commits",
            params::MAX_DUPLICATE_RESULT_COMMITS,
        ),
        ("self_fenced_results", params::MAX_SELF_FENCED_RESULTS),
        (
            "planned_successor_routes_rejected",
            params::MAX_PLANNED_SUCCESSOR_ROUTES_REJECTED,
        ),
        (
            "revoked_proposer_commits",
            params::MAX_REVOKED_PROPOSER_COMMITS,
        ),
        (
            "revoked_gate_grant_commits",
            params::MAX_REVOKED_GATE_GRANT_COMMITS,
        ),
        (
            "structural_commits_after_session_end",
            params::MAX_STRUCTURAL_COMMITS_AFTER_SESSION_END,
        ),
        (
            "structural_commits_after_identity_dependency_change",
            params::MAX_STRUCTURAL_COMMITS_AFTER_IDENTITY_DEPENDENCY_CHANGE,
        ),
        (
            "commits_with_revoked_input_accepted",
            params::MAX_COMMITS_WITH_REVOKED_INPUT_ACCEPTED,
        ),
        (
            "stale_input_generation_commits_admitted",
            params::MAX_STALE_INPUT_GENERATION_COMMITS_ADMITTED,
        ),
        (
            "stale_input_digest_commits_admitted",
            params::MAX_STALE_INPUT_DIGEST_COMMITS_ADMITTED,
        ),
        (
            "stale_eval_input_commits_admitted",
            params::MAX_STALE_EVAL_INPUT_COMMITS_ADMITTED,
        ),
        (
            "stale_eval_input_supersession_commits_admitted",
            params::MAX_STALE_EVAL_INPUT_SUPERSESSION_COMMITS_ADMITTED,
        ),
        (
            "structural_commits_with_stale_dependency",
            params::MAX_STRUCTURAL_COMMITS_WITH_STALE_DEPENDENCY,
        ),
        (
            "expert_routes_after_revocation_before_fence",
            params::MAX_EXPERT_ROUTES_AFTER_REVOCATION_BEFORE_FENCE,
        ),
        (
            "expert_routes_after_crash_between_append_and_apply",
            params::MAX_EXPERT_ROUTES_AFTER_CRASH_BETWEEN_APPEND_AND_APPLY,
        ),
        (
            "replay_fence_set_mismatch_served",
            params::MAX_REPLAY_FENCE_SET_MISMATCH_SERVED,
        ),
        (
            "experts_serving_after_revocation",
            params::MAX_EXPERTS_SERVING_AFTER_REVOCATION,
        ),
        (
            "fence_set_digest_mismatches",
            params::MAX_FENCE_SET_DIGEST_MISMATCHES,
        ),
        (
            "replacement_visible_without_fence",
            params::MAX_REPLACEMENT_VISIBLE_WITHOUT_FENCE,
        ),
        (
            "routable_after_eval_revocation",
            params::MAX_ROUTABLE_AFTER_EVAL_REVOCATION,
        ),
        (
            "routable_after_supersession",
            params::MAX_ROUTABLE_AFTER_SUPERSESSION,
        ),
        (
            "routable_after_structural_invalidation",
            params::MAX_ROUTABLE_AFTER_STRUCTURAL_INVALIDATION,
        ),
        (
            "derived_routable_after_transitive_invalidation",
            params::MAX_DERIVED_ROUTABLE_AFTER_TRANSITIVE_INVALIDATION,
        ),
        (
            "wrong_generation_revocations",
            params::MAX_WRONG_GENERATION_REVOCATIONS,
        ),
    ] {
        rules.push(at_most(name, bound));
    }
    rules.push(
        "fenced_and_probed_dependents_set_digest == derived_dependency_set_digest".to_string(),
    );
    rules.push("fenced_and_probed_dependents == derived_dependency_set_size".to_string());
    each_at_most(
        "invalidation_matrix_stale_hits",
        params::MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL,
        &mut rules,
    );
    rules.push(at_most(
        "inflight_expert_outputs_after_revocation",
        params::MAX_INFLIGHT_EXPERT_OUTPUTS_AFTER_REVOCATION,
    ));
    each_at_most(
        "inflight_outputs_after_input_invalidation",
        params::MAX_INFLIGHT_OUTPUTS_AFTER_INPUT_INVALIDATION,
        &mut rules,
    );
    each_at_most(
        "outputs_after_invalidation",
        params::MAX_OUTPUTS_AFTER_INVALIDATION,
        &mut rules,
    );
    for (name, bound) in [
        (
            "inflight_expert_call_fallback_output_mismatches",
            params::MAX_INFLIGHT_EXPERT_CALL_FALLBACK_OUTPUT_MISMATCHES,
        ),
        (
            "stale_policy_evidence_admissions",
            params::MAX_STALE_POLICY_EVIDENCE_ADMISSIONS,
        ),
        (
            "readmitted_without_current_epoch_evidence",
            params::MAX_READMITTED_WITHOUT_CURRENT_EPOCH_EVIDENCE,
        ),
        (
            "routable_after_authority_revocation",
            params::MAX_ROUTABLE_AFTER_AUTHORITY_REVOCATION,
        ),
        (
            "routable_after_tightening_rotation",
            params::MAX_ROUTABLE_AFTER_TIGHTENING_ROTATION,
        ),
    ] {
        rules.push(at_most(name, bound));
    }
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, resolve, Json, Status};

    /// Set the integer or string at the dotted `path` of an object tree, creating the objects.
    fn put(root: &mut Vec<(String, Json)>, path: &str, value: Json) {
        let (first, rest) = match path.split_once('.') {
            Some((first, rest)) => (first, Some(rest)),
            None => (path, None),
        };
        let at = root.iter().position(|(name, _)| name == first);
        match rest {
            None => match at {
                Some(index) => root[index].1 = value,
                None => root.push((first.to_string(), value)),
            },
            Some(rest) => {
                let index = at.unwrap_or_else(|| {
                    root.push((first.to_string(), Json::Obj(Vec::new())));
                    root.len() - 1
                });
                let Json::Obj(inner) = &mut root[index].1 else {
                    panic!("{path}: not an object");
                };
                put(inner, rest, value);
            }
        }
    }

    /// A result in which every rule holds exactly at its threshold: each `>=` and `<=` counter
    /// at its bound and each `==` pair of digests equal.
    fn passing() -> Json {
        let mut root: Vec<(String, Json)> = Vec::new();
        for rule in coverage().iter().chain(&falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            match tokens[1] {
                "==" => {
                    put(&mut root, tokens[0], Json::str("sha256:a"));
                    put(&mut root, tokens[2], Json::str("sha256:a"));
                }
                _ => {
                    let bound: i64 = tokens[2].parse().expect("a literal bound");
                    put(&mut root, tokens[0], Json::Int(bound));
                }
            }
        }
        put(&mut root, "fenced_and_probed_dependents", Json::Int(3));
        put(&mut root, "derived_dependency_set_size", Json::Int(3));
        Json::Obj(root)
    }

    fn failing_rules(root: &Json, rules: &[String]) -> Vec<String> {
        judge(root, &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        let root = passing();
        assert!(failing_rules(&root, &coverage()).is_empty());
        assert!(failing_rules(&root, &falsifiers()).is_empty());
    }

    #[test]
    fn the_rules_are_unique_and_three_tokens() {
        let all: Vec<String> = coverage().into_iter().chain(falsifiers()).collect();
        let unique: std::collections::BTreeSet<&String> = all.iter().collect();
        assert_eq!(unique.len(), all.len(), "a rule is listed twice");
        for rule in &all {
            assert_eq!(rule.split(' ').count(), 3, "{rule}");
        }
    }

    /// Every rule can fail on its own: moving exactly the counter it names past its bound (or
    /// making the two digests differ) fails that rule and no other.
    #[test]
    fn every_rule_can_fail_on_its_own() {
        for (rules, other) in [(coverage(), falsifiers()), (falsifiers(), coverage())] {
            for rule in &rules {
                let tokens: Vec<&str> = rule.split(' ').collect();
                let Json::Obj(mut pairs) = passing() else {
                    unreachable!()
                };
                match tokens[1] {
                    "==" => {
                        let moved = match resolve(&passing(), tokens[0]) {
                            Some(Json::Int(count)) => Json::Int(count + 1),
                            _ => Json::str("sha256:b"),
                        };
                        put(&mut pairs, tokens[0], moved);
                    }
                    op => {
                        let bound: i64 = tokens[2].parse().unwrap();
                        let moved = if op == ">=" { bound - 1 } else { bound + 1 };
                        put(&mut pairs, tokens[0], Json::Int(moved));
                    }
                }
                let root = Json::Obj(pairs);
                assert_eq!(failing_rules(&root, &rules), vec![rule.clone()], "{rule}");
                // A counter that only this rule names leaves the other list alone; a counter
                // that both lists name (a bound in each) is allowed to move the other too.
                let name = tokens[0];
                if !other.iter().any(|r| r.split(' ').next() == Some(name)) {
                    assert!(failing_rules(&root, &other).is_empty(), "{rule}");
                }
            }
        }
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for rule in coverage().iter().chain(&falsifiers()).step_by(7) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            for replacement in [
                None,
                Some(Json::Null),
                Some(Json::str("x")),
                Some(Json::Float(0.5)),
            ] {
                let Json::Obj(mut pairs) = passing() else {
                    unreachable!()
                };
                // Remove the leaf the rule names.
                match &replacement {
                    None => {
                        let (parent, leaf) = match tokens[0].rsplit_once('.') {
                            Some((parent, leaf)) => (Some(parent), leaf),
                            None => (None, tokens[0]),
                        };
                        match parent {
                            None => pairs.retain(|(key, _)| key != leaf),
                            Some(parent) => {
                                let mut at = &mut pairs;
                                for key in parent.split('.') {
                                    let index =
                                        at.iter().position(|(name, _)| name == key).unwrap();
                                    let Json::Obj(inner) = &mut at[index].1 else {
                                        panic!()
                                    };
                                    at = inner;
                                }
                                at.retain(|(key, _)| key != leaf);
                            }
                        }
                    }
                    Some(value) => put(&mut pairs, tokens[0], value.clone()),
                }
                let root = Json::Obj(pairs);
                let failed = failing_rules(&root, std::slice::from_ref(rule));
                // A string under `==` is a digest and legitimate; under >= / <= it is not.
                if tokens[1] == "==" && matches!(replacement, Some(Json::Str(_))) {
                    continue;
                }
                assert_eq!(failed, vec![rule.clone()], "{rule} with {replacement:?}");
            }
        }
    }

    #[test]
    fn every_rule_names_a_metric_of_the_manifest() {
        for rule in coverage().iter().chain(&falsifiers()) {
            for token in [
                rule.split(' ').next().unwrap(),
                rule.split(' ').nth(2).unwrap(),
            ] {
                if token.parse::<i64>().is_ok() {
                    continue;
                }
                let base = token.split('.').next().unwrap();
                assert!(METRIC_NAMES.contains(&base), "{rule}: {base}");
            }
        }
    }

    #[test]
    fn every_metric_of_the_manifest_is_judged() {
        let ruled: std::collections::BTreeSet<String> = coverage()
            .into_iter()
            .chain(falsifiers())
            .flat_map(|rule| {
                rule.split(' ')
                    .map(|token| token.split('.').next().unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        for name in METRIC_NAMES {
            assert!(ruled.contains(name), "{name} has no rule");
        }
    }

    #[test]
    fn the_nested_families_and_digests_are_metrics_of_the_manifest() {
        for name in NESTED.iter().chain(&DIGESTS) {
            assert!(METRIC_NAMES.contains(name), "{name}");
        }
        for name in NESTED {
            assert!(!cells(name).is_empty(), "{name} has no cells");
        }
    }

    #[test]
    fn the_cells_of_a_family_resolve_in_the_passing_result() {
        let root = passing();
        for name in NESTED {
            for cell in cells(name) {
                // A coverage family has a `>=` rule per cell; a violation family a `<=` one.
                let path = format!("{name}.{cell}");
                let judged = coverage()
                    .into_iter()
                    .chain(falsifiers())
                    .any(|rule| rule.split(' ').next() == Some(&path));
                assert!(judged, "{path} is not judged");
                assert!(resolve(&root, &path).is_some(), "{path}");
            }
        }
    }
}
