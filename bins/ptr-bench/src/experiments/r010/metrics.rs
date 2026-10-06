//! The metric list of R010 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R010).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`; the harness
//! judges its own output with them, and a test pins the proposed `criteria.toml` of the draft
//! preregistration to exactly this list.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 46] = [
    "stale_pages_served",
    "wrongly_dropped_pages",
    "nondeterministic_exact_tags",
    "bitwise_mismatch_count",
    "nondeterministic_reuse_attempts",
    "deterministic_revocation_attempts",
    "pages_above_j_before_revocation",
    "pages_at_or_below_j_before_revocation",
    "deterministic_reuse_events",
    "post_revocation_reuse_probes",
    "retained_pages_reused_after_revocation",
    "recomputed_pages_above_j_served",
    "mismatching_runs_tagged_exact",
    "nondeterministic_divergence_observed",
    "input_revocations_performed",
    "post_revocation_lookups_matching_tokens",
    "revoked_lineage_entry_hits",
    "same_tokens_different_generation_cases",
    "invalidation_suppressed_arms",
    "restore_arms",
    "post_event_lookups",
    "stale_hits",
    "pages_above_j_probed_after_revocation",
    "stale_eligible_entries_before_event",
    "post_event_lookups_on_stale_eligible_entries",
    "expert_generation_revocations_performed",
    "pages_above_layer_cached_before_revocation",
    "matching_post_revocation_lookups",
    "stale_expert_generation_hits",
    "restored_pre_revocation_entries",
    "empty_restart_arm",
    "inflight_expert_revocation_race_cases",
    "outputs_after_expert_revocation",
    "inflight_input_race_cases",
    "outputs_after_input_invalidation",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "probed_page_set_digest",
    "cached_page_set_digest_above_j",
    "inflight_race_cases_by_cell",
    "inflight_outputs_after_invalidation",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "expert_quarantine_kv_cases",
    "kv_reuse_after_expert_fence",
    "model_pin_mismatches",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 6] = [
    "inflight_expert_revocation_race_cases",
    "inflight_input_race_cases",
    "inflight_race_cases_by_cell",
    "inflight_outputs_after_invalidation",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// The four in-flight cells `{revocation, supersession} x {expert, semantic input}`, as the
/// member names of the per-cell metrics.
pub fn inflight_cells() -> Vec<String> {
    params::INFLIGHT_RACE_CELLS
        .iter()
        .map(|[ty, kind]| format!("{ty}_{kind}"))
        .collect()
}

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    let mut add = |rule: String| rules.push(rule);
    add(format!(
        "deterministic_revocation_attempts >= {}",
        params::MIN_DETERMINISTIC_REVOCATION_ATTEMPTS
    ));
    add(format!(
        "pages_above_j_before_revocation >= {}",
        params::MIN_PAGES_ABOVE_J_BEFORE_REVOCATION
    ));
    add(format!(
        "pages_at_or_below_j_before_revocation >= {}",
        params::MIN_PAGES_AT_OR_BELOW_J_BEFORE_REVOCATION
    ));
    add(format!(
        "deterministic_reuse_events >= {}",
        params::MIN_DETERMINISTIC_REUSE_EVENTS
    ));
    add(format!(
        "nondeterministic_reuse_attempts >= {}",
        params::MIN_NONDETERMINISTIC_REUSE_ATTEMPTS
    ));
    add(format!(
        "post_revocation_reuse_probes >= {}",
        params::MIN_POST_REVOCATION_REUSE_PROBES
    ));
    add(format!(
        "retained_pages_reused_after_revocation >= {}",
        params::MIN_RETAINED_PAGES_REUSED_AFTER_REVOCATION
    ));
    add(format!(
        "recomputed_pages_above_j_served >= {}",
        params::MIN_RECOMPUTED_PAGES_ABOVE_J_SERVED
    ));
    add("probed_page_set_digest == cached_page_set_digest_above_j".to_string());
    add(format!(
        "nondeterministic_divergence_observed >= {}",
        params::MIN_NONDETERMINISTIC_DIVERGENCE_OBSERVED
    ));
    add("internal.revoked_expert_routed_tokens >= 1".to_string());
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
    add(format!(
        "expert_generation_revocations_performed >= {}",
        params::MIN_EXPERT_GENERATION_REVOCATIONS_PERFORMED
    ));
    add(format!(
        "pages_above_layer_cached_before_revocation >= {}",
        params::MIN_PAGES_ABOVE_LAYER_CACHED_BEFORE_REVOCATION
    ));
    add(format!(
        "matching_post_revocation_lookups >= {}",
        params::MIN_MATCHING_POST_REVOCATION_LOOKUPS
    ));
    for ty in params::INVALIDATION_TYPES {
        add(format!(
            "inflight_expert_revocation_race_cases.{ty} >= {}",
            params::MIN_INFLIGHT_EXPERT_REVOCATION_RACE_CASES
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
    for cell in inflight_cells() {
        add(format!(
            "inflight_race_cases_by_cell.{cell} >= {}",
            params::MIN_INFLIGHT_RACE_CASES_PER_CELL
        ));
    }
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        add(format!(
            "invalidation_matrix_cases.{cell} >= {}",
            params::MIN_INVALIDATION_MATRIX_CASES_PER_CELL
        ));
    }
    add(format!(
        "expert_quarantine_kv_cases >= {}",
        params::MIN_EXPERT_QUARANTINE_KV_CASES
    ));
    add(format!(
        "model_pin_mismatches <= {}",
        params::MAX_MODEL_PIN_MISMATCHES
    ));
    rules
}

/// The falsifiers of the manifest: each tripped one falsifies R010 on this seed.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    let mut add = |rule: String| rules.push(rule);
    add(format!(
        "stale_pages_served <= {}",
        params::MAX_STALE_PAGES_SERVED
    ));
    add(format!(
        "wrongly_dropped_pages <= {}",
        params::MAX_WRONGLY_DROPPED_PAGES
    ));
    add(format!(
        "nondeterministic_exact_tags <= {}",
        params::MAX_NONDETERMINISTIC_EXACT_TAGS
    ));
    add(format!(
        "mismatching_runs_tagged_exact <= {}",
        params::MAX_MISMATCHING_RUNS_TAGGED_EXACT
    ));
    add(format!(
        "bitwise_mismatch_count <= {}",
        params::MAX_BITWISE_MISMATCH_COUNT
    ));
    add(format!(
        "revoked_lineage_entry_hits <= {}",
        params::MAX_REVOKED_LINEAGE_ENTRY_HITS
    ));
    add(format!("stale_hits <= {}", params::MAX_STALE_HITS));
    add(format!(
        "stale_expert_generation_hits <= {}",
        params::MAX_STALE_EXPERT_GENERATION_HITS
    ));
    add(format!(
        "outputs_after_expert_revocation <= {}",
        params::MAX_OUTPUTS_AFTER_EXPERT_REVOCATION
    ));
    add(format!(
        "outputs_after_input_invalidation <= {}",
        params::MAX_OUTPUTS_AFTER_INPUT_INVALIDATION
    ));
    for cell in inflight_cells() {
        add(format!(
            "inflight_outputs_after_invalidation.{cell} <= {}",
            params::MAX_INFLIGHT_OUTPUTS_AFTER_INVALIDATION_PER_CELL
        ));
    }
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        add(format!(
            "invalidation_matrix_stale_hits.{cell} <= {}",
            params::MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL
        ));
    }
    add(format!(
        "kv_reuse_after_expert_fence <= {}",
        params::MAX_KV_REUSE_AFTER_EXPERT_FENCE
    ));
    add("internal.adoption_mismatches <= 0".to_string());
    add("internal.restart_resolution_mismatches <= 0".to_string());
    add("internal.matrix_selectivity_mismatches <= 0".to_string());
    // A restored per-layer entry whose key scope disagrees with its layer tag.
    add("internal.layered_scope_mismatches <= 0".to_string());
    add("internal.batch_invariance_mismatches <= 0".to_string());
    rules
}
