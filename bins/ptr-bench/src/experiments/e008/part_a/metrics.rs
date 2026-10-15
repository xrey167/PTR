//! The Part A metric list of E008 and the three judgements of a run: the coverage
//! prerequisites (a run that misses one is vacuous), the falsifiers (a run that trips one
//! falsifies E008) and the validity rules (a run that violates one does not describe the
//! planned procedure).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`). E008 is
//! `planned` and has no `criteria.toml`: these lists are the harness's reading of the
//! hypothesis and falsification text of `experiment.toml`, for the head to pin in the
//! preregistration.

use super::controls::CONTROLS;
use super::matrix;
use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The Part A metrics of the manifest, in its order, then the two this harness adds
/// (`procedure_mismatches` is the manifest's; `control_trips` is the harness's).
pub const METRIC_NAMES: [&str; 22] = [
    "reuse_after_raw_ancestor_removed",
    "raw_ancestor_omission_cases",
    "input_lineage_revalidation_cases",
    "stale_input_lineage_kv_served",
    "upstream_expert_fence_cases",
    "downstream_kv_reuse_after_upstream_fence",
    "upstream_adapter_invalidation_race_cases",
    "downstream_kv_reuse_after_upstream_adapter_invalidation",
    "upstream_base_model_invalidation_cases",
    "downstream_kv_reuse_after_upstream_base_model_invalidation",
    "approximate_frame_downstream_exact_lookup_cases",
    "approximate_frames_served_to_exact",
    "untainted_outputs_from_approximate_frames",
    "downstream_persist_rejection_cases",
    "downstream_restore_attempts_admitted",
    "downstream_entries_after_restart",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "matrix_frame_bijection_violations",
    "outputs_after_upstream_invalidation",
    "procedure_mismatches",
    "control_trips",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 3] = [
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "control_trips",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// Counters whose natural value is zero (violations, mismatches): a run in which none happens
/// still prints a zero.
pub const ZERO_BASED: [&str; 11] = [
    "reuse_after_raw_ancestor_removed",
    "stale_input_lineage_kv_served",
    "downstream_kv_reuse_after_upstream_fence",
    "downstream_kv_reuse_after_upstream_adapter_invalidation",
    "downstream_kv_reuse_after_upstream_base_model_invalidation",
    "approximate_frames_served_to_exact",
    "untainted_outputs_from_approximate_frames",
    "downstream_restore_attempts_admitted",
    "downstream_entries_after_restart",
    "matrix_frame_bijection_violations",
    "outputs_after_upstream_invalidation",
];

/// Harness-internal counters, seeded at zero so the rules that name them can be read; printed
/// under `internal`.
pub const INTERNAL: [&str; 4] = [
    "verdict_mismatches",
    "attempts_without_typed_rejection",
    "control_inconsistencies",
    "revoke_messages_dropped_or_delayed",
];

/// Counters whose value is a number of attempts or cases (never violations).
const CASES: [&str; 7] = [
    "raw_ancestor_omission_cases",
    "input_lineage_revalidation_cases",
    "upstream_expert_fence_cases",
    "upstream_adapter_invalidation_race_cases",
    "upstream_base_model_invalidation_cases",
    "approximate_frame_downstream_exact_lookup_cases",
    "downstream_persist_rejection_cases",
];

/// Reading notes. `matrix_frame_bijection_violations` is mainly a coverage check: it fails when
/// the slot/kind mapping stops being a bijection or no frame the harness builds reaches a kind,
/// which a code change to the closed lists makes visible; it is not a product falsifier of a
/// stale or tainted serve. The three controls are external baselines (see `controls.rs`).
///
/// Hard counters that mirror a per-cell stale hit as a top-level integer, for the mutation
/// checker (it reads top-level integers only).
pub fn cell_counter(cell: &str) -> String {
    format!("stale_hits_in_{cell}")
}

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    for name in CASES {
        rules.push(format!("{name} >= {}", params::MIN_CASES));
    }
    for cell in matrix::cell_names() {
        rules.push(format!(
            "invalidation_matrix_cases.{cell} >= {}",
            params::MIN_CASES
        ));
    }
    rules.push(format!(
        "internal.revoke_messages_dropped_or_delayed >= {}",
        params::MIN_CASES
    ));
    for control in CONTROLS {
        rules.push(format!("control_trips.{control} >= {}", params::MIN_CASES));
    }
    rules
}

/// The falsifiers: each rule that does not hold falsifies E008 on that seed.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    for name in ZERO_BASED {
        rules.push(format!("{name} <= {}", params::MAX_VIOLATIONS));
    }
    for cell in matrix::cell_names() {
        rules.push(format!(
            "invalidation_matrix_stale_hits.{cell} <= {}",
            params::MAX_VIOLATIONS
        ));
    }
    for name in ["verdict_mismatches", "attempts_without_typed_rejection"] {
        rules.push(format!("internal.{name} <= {}", params::MAX_VIOLATIONS));
    }
    rules
}

/// The counters that say the record does not describe the planned procedure. They are the
/// harness's own, never a finding about the product.
pub const HARNESS_COUNTERS: [&str; 2] = ["procedure_mismatches", "control_inconsistencies"];

/// The validity rules: each that does not hold makes the seed INVALID.
pub fn validity() -> Vec<String> {
    vec![
        format!("procedure_mismatches <= {}", params::MAX_VIOLATIONS),
        format!(
            "internal.control_inconsistencies <= {}",
            params::MAX_VIOLATIONS
        ),
    ]
}

/// The counters a mutation may be expected to fire: top-level integers the mutation checker
/// can read, namely every `<= 0` falsifier counter that is not nested, the harness counters
/// and the per-cell mirrors.
pub fn hard_counters() -> Vec<String> {
    let mut names: Vec<String> = ZERO_BASED.iter().map(|name| name.to_string()).collect();
    names.push("procedure_mismatches".to_string());
    names.push("verdict_mismatches".to_string());
    names.push("attempts_without_typed_rejection".to_string());
    names.push("control_inconsistencies".to_string());
    for cell in matrix::cell_names() {
        names.push(cell_counter(&cell));
    }
    names
}

/// The counters whose firing is an observed violation by the product: the hard counters less
/// the harness's own.
pub fn product_counters() -> Vec<String> {
    hard_counters()
        .into_iter()
        .filter(|name| !HARNESS_COUNTERS.contains(&name.as_str()))
        .collect()
}
