//! The metric list of R009 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R009).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`; the harness
//! judges its own output with them, and a test pins the proposed `criteria.toml` of the draft
//! preregistration to exactly this list.

use super::params::{self, CONFIGS};
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 45] = [
    "shared_row_fraction",
    "predicted_shared_row_fraction",
    "bitwise_mismatch_count",
    "base_pages_dropped_on_revoke",
    "alora_prefix_reuse_attempts",
    "adapter_revocations_performed_by_type",
    "adapter_reuse_attempts_by_type",
    "revoked_adapter_entry_hits",
    "post_revocation_lookups_on_adapter_entries",
    "alora_boundary_page_cases",
    "base_pages_before_revoke",
    "adapter_entries_before_revocation",
    "adapter_logits_bitwise_mismatch_count",
    "cross_adapter_attempts",
    "input_revocations_performed",
    "post_revocation_lookups_matching_tokens",
    "revoked_lineage_entry_hits",
    "same_tokens_different_generation_cases",
    "invalidation_suppressed_arms",
    "restore_arms",
    "post_event_lookups",
    "stale_hits",
    "stale_eligible_entries_before_event",
    "post_event_lookups_on_stale_eligible_entries",
    "shared_kv_rows",
    "total_kv_rows",
    "old_generation_as_new_attempts",
    "cross_adapter_hits",
    "old_generation_hits",
    "restored_pre_revocation_entries",
    "empty_restart_arm",
    "inflight_adapter_revocation_race_cases",
    "outputs_after_adapter_revocation",
    "inflight_input_race_cases",
    "outputs_after_input_invalidation",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "inflight_adapter_requests_cancelled",
    "inflight_adapter_requests_restarted_from_fresh_prefill",
    "adapter_restart_output_mismatches",
    "inflight_adapter_cancelled_request_id_set_digest",
    "inflight_adapter_restarted_request_id_set_digest",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "model_pin_mismatches",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 8] = [
    "adapter_revocations_performed_by_type",
    "adapter_reuse_attempts_by_type",
    "shared_kv_rows",
    "total_kv_rows",
    "inflight_adapter_revocation_race_cases",
    "inflight_input_race_cases",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// The adapter types of the pinned run.
pub fn adapter_types() -> Vec<&'static str> {
    params::ADAPTER_TYPES_RUN.to_vec()
}

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    let mut add = |rule: String| rules.push(rule);
    // The pinned configurations (QKV-LoRA and Q-only at f = L/2 and f = 0, MLP-only at L/2,
    // aLoRA at the pinned invocation position) were all evaluated, among them the interior one.
    for config in CONFIGS {
        add(format!("internal.config_runs.{} >= 1", config.name));
    }
    add(format!(
        "alora_prefix_reuse_attempts >= {}",
        params::MIN_ALORA_PREFIX_REUSE_ATTEMPTS
    ));
    add(format!(
        "alora_boundary_page_cases >= {}",
        params::MIN_ALORA_BOUNDARY_PAGE_CASES
    ));
    for ty in adapter_types() {
        add(format!(
            "adapter_revocations_performed_by_type.{ty} >= {}",
            params::MIN_ADAPTER_REVOCATIONS_PER_TYPE
        ));
        add(format!(
            "adapter_reuse_attempts_by_type.{ty} >= {}",
            params::MIN_ADAPTER_REUSE_ATTEMPTS_PER_TYPE
        ));
    }
    add(format!(
        "post_revocation_lookups_on_adapter_entries >= {}",
        params::MIN_POST_REVOCATION_LOOKUPS_ON_ADAPTER_ENTRIES
    ));
    add(format!(
        "base_pages_before_revoke >= {}",
        params::MIN_BASE_PAGES_BEFORE_REVOKE
    ));
    add(format!(
        "adapter_entries_before_revocation >= {}",
        params::MIN_ADAPTER_ENTRIES_BEFORE_REVOCATION
    ));
    add(format!(
        "cross_adapter_attempts >= {}",
        params::MIN_CROSS_ADAPTER_ATTEMPTS
    ));
    add(format!(
        "old_generation_as_new_attempts >= {}",
        params::MIN_OLD_GENERATION_AS_NEW_ATTEMPTS
    ));
    add(format!(
        "input_revocations_performed >= {}",
        params::MIN_INPUT_REVOCATIONS_PERFORMED
    ));
    add(format!(
        "post_revocation_lookups_matching_tokens >= {}",
        params::MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS
    ));
    add(format!(
        "same_tokens_different_generation_cases >= {}",
        params::MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES
    ));
    add(format!(
        "invalidation_suppressed_arms >= {}",
        params::MIN_INVALIDATION_SUPPRESSED_ARMS
    ));
    add(format!("restore_arms >= {}", params::MIN_RESTORE_ARMS));
    add(format!(
        "restored_pre_revocation_entries >= {}",
        params::MIN_RESTORED_PRE_REVOCATION_ENTRIES
    ));
    add("stale_eligible_entries_before_event >= 1".to_string());
    add(
        "post_event_lookups_on_stale_eligible_entries == stale_eligible_entries_before_event"
            .to_string(),
    );
    add(format!(
        "post_event_lookups >= {}",
        params::MIN_POST_EVENT_LOOKUPS
    ));
    for ty in params::INVALIDATION_TYPES {
        add(format!(
            "inflight_adapter_revocation_race_cases.{ty} >= {}",
            params::MIN_INFLIGHT_ADAPTER_REVOCATION_RACE_CASES
        ));
        add(format!(
            "inflight_input_race_cases.{ty} >= {}",
            params::MIN_INFLIGHT_INPUT_RACE_CASES
        ));
    }
    add(format!(
        "selection_digest_mismatches <= {}",
        params::MAX_SELECTION_DIGEST_MISMATCHES
    ));
    add(format!(
        "procedure_mismatches <= {}",
        params::MAX_PROCEDURE_MISMATCHES
    ));
    add(
        "inflight_adapter_requests_restarted_from_fresh_prefill >= inflight_adapter_requests_cancelled"
            .to_string(),
    );
    add(
        "inflight_adapter_cancelled_request_id_set_digest == inflight_adapter_restarted_request_id_set_digest"
            .to_string(),
    );
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        add(format!(
            "invalidation_matrix_cases.{cell} >= {}",
            params::MIN_INVALIDATION_MATRIX_CASES_PER_CELL
        ));
    }
    add(format!(
        "model_pin_mismatches <= {}",
        params::MAX_MODEL_PIN_MISMATCHES
    ));
    rules
}

/// The falsifiers of the manifest: each tripped one falsifies R009 on this seed.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    let mut add = |rule: String| rules.push(rule);
    add(format!(
        "bitwise_mismatch_count <= {}",
        params::MAX_ADAPTER_LOGITS_BITWISE_MISMATCH_COUNT
    ));
    add(format!(
        "adapter_logits_bitwise_mismatch_count <= {}",
        params::MAX_ADAPTER_LOGITS_BITWISE_MISMATCH_COUNT
    ));
    // Fractions are integer numerators and denominators; equality is the integer cross
    // difference, never a float comparison (tolerance 0.0).
    for config in CONFIGS {
        add(format!(
            "shared_row_fraction.{}.cross_difference == 0",
            config.name
        ));
    }
    add("internal.per_layer_fraction_mismatches <= 0".to_string());
    add("internal.prediction_structure_mismatches <= 0".to_string());
    add("internal.report_physical_mismatches <= 0".to_string());
    add("internal.adoption_mismatches <= 0".to_string());
    add("base_pages_dropped_on_revoke <= 0".to_string());
    add(format!(
        "revoked_adapter_entry_hits <= {}",
        params::MAX_REVOKED_ADAPTER_ENTRY_HITS
    ));
    add("internal.post_invocation_rows_under_base_key <= 0".to_string());
    add(format!(
        "cross_adapter_hits <= {}",
        params::MAX_CROSS_ADAPTER_HITS
    ));
    add(format!(
        "old_generation_hits <= {}",
        params::MAX_OLD_GENERATION_HITS
    ));
    add(format!(
        "revoked_lineage_entry_hits <= {}",
        params::MAX_REVOKED_LINEAGE_ENTRY_HITS
    ));
    add(format!("stale_hits <= {}", params::MAX_STALE_HITS));
    add(format!(
        "outputs_after_adapter_revocation <= {}",
        params::MAX_OUTPUTS_AFTER_ADAPTER_REVOCATION
    ));
    add(format!(
        "outputs_after_input_invalidation <= {}",
        params::MAX_OUTPUTS_AFTER_INPUT_INVALIDATION
    ));
    add(format!(
        "adapter_restart_output_mismatches <= {}",
        params::ADAPTER_RESTART_OUTPUT_MISMATCHES_MAX
    ));
    add("internal.restart_resolution_mismatches <= 0".to_string());
    add("internal.matrix_selectivity_mismatches <= 0".to_string());
    // A restored per-layer entry whose key scope disagrees with its layer tag.
    add("internal.layered_scope_mismatches <= 0".to_string());
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        add(format!(
            "invalidation_matrix_stale_hits.{cell} <= {}",
            params::MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL
        ));
    }
    rules
}
