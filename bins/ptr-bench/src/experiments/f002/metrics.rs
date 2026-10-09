//! The metric list of F002 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies F002).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): a
//! prerequisite "counter < N" is written `>= N`, a violation "counter > 0" is written `<= 0`.
//! The harness judges its own output with them, and a test pins `criteria.toml` to exactly
//! these lists, in order.
//!
//! Fractions are integer permille or parts per million. Every metric is a JSON integer.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics, in the order the harness prints them.
pub const METRIC_NAMES: [&str; 157] = [
    "items_total",
    "worlds_run",
    "veto_contradictions",
    "model_oracle_mismatches",
    "resolution_oracle_mismatches",
    "metric_mismatches",
    "active_set_calibration_reported",
    "gold_guard_violations",
    "acquisition_violations",
    "sound_world_determined_errors",
    "unexpected_refusals",
    "determinism_mismatches",
    "product_panics",
    "gain_violations",
    "determined_seen",
    "estimated_seen",
    "unknown_seen",
    "disputed_seen",
    "vetoed_items",
    "gold_items",
    "few_function_warnings",
    "budget_cells_seen",
    "correlated_pair_items",
    "contaminated_vetoes",
    "active_set_probes",
    "gold_guard_probes",
    "acquisition_items",
    "negative_controls_tripped",
    "w1_census_accuracy_ds_permille",
    "w1_census_accuracy_mv_permille",
    "w1_census_accuracy_mvp_permille",
    "w1_census_accuracy_mvc_permille",
    "w1_census_accuracy_mvcp_permille",
    "w1_accuracy_ds_permille",
    "w1_accuracy_mv_permille",
    "w1_accuracy_mvp_permille",
    "w1_accuracy_mvc_permille",
    "w1_accuracy_mvcp_permille",
    "w1_brier_ds_ppm",
    "w1_brier_mv_ppm",
    "w1_brier_mvp_ppm",
    "w1_brier_mvc_ppm",
    "w1_brier_mvcp_ppm",
    "w1_ece_ds_ppm",
    "w1_ece_mv_ppm",
    "w1_ece_mvp_ppm",
    "w1_ece_mvc_ppm",
    "w1_ece_mvcp_ppm",
    "w1_brier_gain_ppm",
    "w1_ece_gain_ppm",
    "w1_brier_gain_lower_ppm",
    "w1_ece_gain_lower_ppm",
    "w1_brier_gain_mvc_ppm",
    "w1_ece_gain_mvc_ppm",
    "w1_brier_gain_mvcp_ppm",
    "w1_ece_gain_mvcp_ppm",
    "w1_gold_items",
    "w1_determined_items",
    "w1_estimated_items",
    "w1_unknown_items",
    "w1_disputed_items",
    "w1_determined_permille",
    "w1_estimated_permille",
    "w1_unknown_permille",
    "w1_disputed_permille",
    "w1_alpha_ppm",
    "w1_alpha_defined",
    "w1_vo_labelled_gold",
    "w1_vo_correct_gold",
    "w1_mv_veto_violating_items",
    "w1_mvp_veto_violating_items",
    "w2_census_accuracy_ds_permille",
    "w2_census_accuracy_mv_permille",
    "w2_census_accuracy_mvp_permille",
    "w2_census_accuracy_mvc_permille",
    "w2_census_accuracy_mvcp_permille",
    "w2_accuracy_ds_permille",
    "w2_accuracy_mv_permille",
    "w2_accuracy_mvp_permille",
    "w2_accuracy_mvc_permille",
    "w2_accuracy_mvcp_permille",
    "w2_brier_ds_ppm",
    "w2_brier_mv_ppm",
    "w2_brier_mvp_ppm",
    "w2_brier_mvc_ppm",
    "w2_brier_mvcp_ppm",
    "w2_ece_ds_ppm",
    "w2_ece_mv_ppm",
    "w2_ece_mvp_ppm",
    "w2_ece_mvc_ppm",
    "w2_ece_mvcp_ppm",
    "w2_brier_gain_ppm",
    "w2_ece_gain_ppm",
    "w2_brier_gain_lower_ppm",
    "w2_ece_gain_lower_ppm",
    "w2_brier_gain_mvc_ppm",
    "w2_ece_gain_mvc_ppm",
    "w2_brier_gain_mvcp_ppm",
    "w2_ece_gain_mvcp_ppm",
    "w2_gold_items",
    "w2_determined_items",
    "w2_estimated_items",
    "w2_unknown_items",
    "w2_disputed_items",
    "w2_determined_permille",
    "w2_estimated_permille",
    "w2_unknown_permille",
    "w2_disputed_permille",
    "w2_alpha_ppm",
    "w2_alpha_defined",
    "w2_vo_labelled_gold",
    "w2_vo_correct_gold",
    "w2_mv_veto_violating_items",
    "w2_mvp_veto_violating_items",
    "w3_census_accuracy_ds_permille",
    "w3_census_accuracy_mv_permille",
    "w3_census_accuracy_mvp_permille",
    "w3_census_accuracy_mvc_permille",
    "w3_census_accuracy_mvcp_permille",
    "w3_accuracy_ds_permille",
    "w3_accuracy_mv_permille",
    "w3_accuracy_mvp_permille",
    "w3_accuracy_mvc_permille",
    "w3_accuracy_mvcp_permille",
    "w3_brier_ds_ppm",
    "w3_brier_mv_ppm",
    "w3_brier_mvp_ppm",
    "w3_brier_mvc_ppm",
    "w3_brier_mvcp_ppm",
    "w3_ece_ds_ppm",
    "w3_ece_mv_ppm",
    "w3_ece_mvp_ppm",
    "w3_ece_mvc_ppm",
    "w3_ece_mvcp_ppm",
    "w3_brier_gain_ppm",
    "w3_ece_gain_ppm",
    "w3_brier_gain_lower_ppm",
    "w3_ece_gain_lower_ppm",
    "w3_brier_gain_mvc_ppm",
    "w3_ece_gain_mvc_ppm",
    "w3_brier_gain_mvcp_ppm",
    "w3_ece_gain_mvcp_ppm",
    "w3_gold_items",
    "w3_determined_items",
    "w3_estimated_items",
    "w3_unknown_items",
    "w3_disputed_items",
    "w3_determined_permille",
    "w3_estimated_permille",
    "w3_unknown_permille",
    "w3_disputed_permille",
    "w3_alpha_ppm",
    "w3_alpha_defined",
    "w3_vo_labelled_gold",
    "w3_vo_correct_gold",
    "w3_mv_veto_violating_items",
    "w3_mvp_veto_violating_items",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each has a
/// falsifier rule of its own (`gain_violations` restates the three gain rules as a count).
pub const HARD_COUNTERS: [&str; 12] = [
    "veto_contradictions",
    "model_oracle_mismatches",
    "resolution_oracle_mismatches",
    "metric_mismatches",
    "active_set_calibration_reported",
    "gold_guard_violations",
    "acquisition_violations",
    "sound_world_determined_errors",
    "unexpected_refusals",
    "determinism_mismatches",
    "product_panics",
    "gain_violations",
];

/// The counters the run sets to zero before it starts, so that a counter nothing incremented
/// prints `0` rather than being a harness gap.
pub const ZEROED: [&str; 25] = [
    "items_total",
    "acquisition_items",
    "worlds_run",
    "veto_contradictions",
    "model_oracle_mismatches",
    "resolution_oracle_mismatches",
    "metric_mismatches",
    "active_set_calibration_reported",
    "gold_guard_violations",
    "acquisition_violations",
    "sound_world_determined_errors",
    "unexpected_refusals",
    "determinism_mismatches",
    "product_panics",
    "determined_seen",
    "estimated_seen",
    "unknown_seen",
    "disputed_seen",
    "vetoed_items",
    "few_function_warnings",
    "budget_cells_seen",
    "correlated_pair_items",
    "contaminated_vetoes",
    "active_set_probes",
    "gold_guard_probes",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites: each unmet one makes the run vacuous. `vetoed_items`,
/// `budget_cells_seen`, `correlated_pair_items` and `contaminated_vetoes` are read back from
/// the vote matrix the product was given, `determined_seen` to `disputed_seen`,
/// `few_function_warnings`, `active_set_probes` and `acquisition_items` from what the product
/// returned; `worlds_run` and `gold_items` are counts of the run's own work.
pub fn coverage() -> Vec<String> {
    vec![
        format!("worlds_run >= {}", params::MIN_WORLDS_RUN),
        format!("determined_seen >= {}", params::MIN_DETERMINED_SEEN),
        format!("estimated_seen >= {}", params::MIN_ESTIMATED_SEEN),
        format!("unknown_seen >= {}", params::MIN_UNKNOWN_SEEN),
        format!("disputed_seen >= {}", params::MIN_DISPUTED_SEEN),
        format!("vetoed_items >= {}", params::MIN_VETOED_ITEMS),
        format!("gold_items >= {}", params::MIN_GOLD_ITEMS),
        format!(
            "few_function_warnings >= {}",
            params::MIN_FEWER_THAN_THREE_WARNINGS
        ),
        format!("budget_cells_seen >= {}", params::MIN_BUDGET_CELLS),
        format!(
            "correlated_pair_items >= {}",
            params::MIN_CORRELATED_PAIR_ITEMS
        ),
        format!("contaminated_vetoes >= {}", params::MIN_CONTAMINATED_VETOES),
        format!("active_set_probes >= {}", params::MIN_ACTIVE_SET_PROBES),
        format!("gold_guard_probes >= {}", params::MIN_GOLD_GUARD_PROBES),
        format!("acquisition_items >= {}", params::MIN_ACQUISITION_ITEMS),
    ]
}

/// The falsifiers: each tripped one falsifies F002 on this seed.
pub fn falsifiers() -> Vec<String> {
    let brier = if params::BRIER_GATE_ON_LOWER_BOUND {
        "w1_brier_gain_lower_ppm"
    } else {
        "w1_brier_gain_ppm"
    };
    let ece = if params::ECE_GATE_ON_LOWER_BOUND {
        "w1_ece_gain_lower_ppm"
    } else {
        "w1_ece_gain_ppm"
    };
    vec![
        format!("veto_contradictions <= {}", params::MAX_VIOLATIONS),
        format!("model_oracle_mismatches <= {}", params::MAX_VIOLATIONS),
        format!("resolution_oracle_mismatches <= {}", params::MAX_VIOLATIONS),
        format!("metric_mismatches <= {}", params::MAX_VIOLATIONS),
        format!(
            "active_set_calibration_reported <= {}",
            params::MAX_VIOLATIONS
        ),
        format!("gold_guard_violations <= {}", params::MAX_VIOLATIONS),
        format!("acquisition_violations <= {}", params::MAX_VIOLATIONS),
        format!(
            "sound_world_determined_errors <= {}",
            params::MAX_VIOLATIONS
        ),
        format!("unexpected_refusals <= {}", params::MAX_VIOLATIONS),
        format!("determinism_mismatches <= {}", params::MAX_VIOLATIONS),
        format!("product_panics <= {}", params::MAX_VIOLATIONS),
        format!("gain_violations <= {}", params::MAX_VIOLATIONS),
        format!("{brier} >= {}", params::MIN_GAIN_PPM),
        format!("{ece} >= {}", params::MIN_GAIN_PPM),
        "w1_census_accuracy_ds_permille >= w1_census_accuracy_mv_permille".to_string(),
    ]
}
