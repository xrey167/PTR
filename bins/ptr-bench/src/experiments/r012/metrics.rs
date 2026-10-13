//! The metric list of R012 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R012).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`), derived
//! from the manifest's falsification text and the thresholds of `config.toml`: every
//! "< N" prerequisite becomes `>= N`, every "> 0" violation becomes `<= 0`. The harness
//! judges its own output with them, and a test pins `criteria.toml` to exactly these lists.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 21] = [
    "approximate_run_actions_admitted",
    "approximate_action_admission_attempts",
    "approximate_semdb_promotions",
    "approximate_semdb_promotion_attempts",
    "approximate_candidate_path_attempts",
    "approximate_candidate_exact_admissions",
    "tainted_candidate_reload_attempts",
    "taint_lost",
    "fresh_exact_recomputation_controls",
    "candidate_taint_hops",
    "approximate_pod_result_promotion_attempts",
    "approximate_pod_result_promotions_accepted",
    "misclassified_revocable_as_raw_attempts",
    "misclassified_raw_admissions",
    "approximate_output_reintroduced_as_raw_attempts",
    "laundered_exact_admissions",
    "exact_echo_cases",
    "taint_cleared_by_echo",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous, in the
/// order of the falsification text.
pub fn coverage() -> Vec<String> {
    vec![
        format!(
            "approximate_action_admission_attempts >= {}",
            params::MIN_APPROXIMATE_ACTION_ADMISSION_ATTEMPTS
        ),
        format!(
            "approximate_semdb_promotion_attempts >= {}",
            params::MIN_APPROXIMATE_SEMDB_PROMOTION_ATTEMPTS
        ),
        format!(
            "approximate_candidate_path_attempts >= {}",
            params::MIN_APPROXIMATE_CANDIDATE_PATH_ATTEMPTS
        ),
        format!(
            "approximate_pod_result_promotion_attempts >= {}",
            params::MIN_APPROXIMATE_POD_RESULT_PROMOTION_ATTEMPTS
        ),
        format!(
            "tainted_candidate_reload_attempts >= {}",
            params::MIN_TAINTED_CANDIDATE_RELOAD_ATTEMPTS
        ),
        format!(
            "fresh_exact_recomputation_controls >= {}",
            params::MIN_FRESH_EXACT_RECOMPUTATION_CONTROLS
        ),
        format!(
            "candidate_taint_hops >= {}",
            params::MIN_CANDIDATE_TAINT_HOPS
        ),
        format!(
            "misclassified_revocable_as_raw_attempts >= {}",
            params::MIN_MISCLASSIFIED_REVOCABLE_AS_RAW_ATTEMPTS
        ),
        format!(
            "approximate_output_reintroduced_as_raw_attempts >= {}",
            params::MIN_APPROXIMATE_OUTPUT_REINTRODUCED_AS_RAW_ATTEMPTS
        ),
        format!("exact_echo_cases >= {}", params::MIN_EXACT_ECHO_CASES),
        format!(
            "selection_digest_mismatches <= {}",
            params::MAX_SELECTION_DIGEST_MISMATCHES
        ),
        format!(
            "procedure_mismatches <= {}",
            params::MAX_PROCEDURE_MISMATCHES
        ),
        format!(
            "model_pin_mismatches <= {}",
            params::MAX_MODEL_PIN_MISMATCHES
        ),
    ]
}

/// The falsifiers of the manifest: each tripped one falsifies R012 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        format!(
            "approximate_run_actions_admitted <= {}",
            params::MAX_APPROXIMATE_RUN_ACTIONS_ADMITTED
        ),
        format!(
            "approximate_semdb_promotions <= {}",
            params::MAX_APPROXIMATE_SEMDB_PROMOTIONS
        ),
        format!(
            "approximate_candidate_exact_admissions <= {}",
            params::MAX_APPROXIMATE_CANDIDATE_EXACT_ADMISSIONS
        ),
        format!(
            "approximate_pod_result_promotions_accepted <= {}",
            params::MAX_APPROXIMATE_POD_RESULT_PROMOTIONS_ACCEPTED
        ),
        format!("taint_lost <= {}", params::MAX_TAINT_LOST),
        format!(
            "misclassified_raw_admissions <= {}",
            params::MAX_MISCLASSIFIED_RAW_ADMISSIONS
        ),
        format!(
            "laundered_exact_admissions <= {}",
            params::MAX_LAUNDERED_EXACT_ADMISSIONS
        ),
        format!(
            "taint_cleared_by_echo <= {}",
            params::MAX_TAINT_CLEARED_BY_ECHO
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        let mut pairs = Vec::new();
        for name in METRIC_NAMES {
            let value = match name {
                "candidate_taint_hops" => 2,
                name if name.ends_with("_attempts")
                    || name.ends_with("_controls")
                    || name.ends_with("_cases") =>
                {
                    1
                }
                _ => 0,
            };
            pairs.push((name.to_string(), Json::Int(value)));
        }
        pairs
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
        let root = Json::Obj(passing());
        assert!(failing_rules(&root, &coverage()).is_empty());
        assert!(failing_rules(&root, &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_when_its_counter_drops_below_the_threshold() {
        for rule in coverage() {
            let name = rule.split(' ').next().unwrap();
            let mut pairs = passing();
            let slot = pairs.iter_mut().find(|(key, _)| key == name).unwrap();
            // The prerequisites (>=) drop by one; the mismatch counters (<=) rise by one.
            slot.1 = match (&slot.1, rule.contains(">=")) {
                (Json::Int(value), true) => Json::Int(value - 1),
                (Json::Int(value), false) => Json::Int(value + 1),
                _ => unreachable!(),
            };
            let failed = failing_rules(&Json::Obj(pairs), &coverage());
            assert_eq!(failed, vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn each_falsifier_trips_on_a_single_violation() {
        for rule in falsifiers() {
            let name = rule.split(' ').next().unwrap();
            let mut pairs = passing();
            pairs.iter_mut().find(|(key, _)| key == name).unwrap().1 = Json::Int(1);
            let failed = failing_rules(&Json::Obj(pairs), &falsifiers());
            assert_eq!(failed, vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn a_missing_or_null_counter_is_not_a_pass() {
        let mut pairs = passing();
        pairs.retain(|(key, _)| key != "taint_lost");
        assert_eq!(
            failing_rules(&Json::Obj(pairs), &falsifiers()),
            vec!["taint_lost <= 0".to_string()]
        );
        let mut pairs = passing();
        pairs
            .iter_mut()
            .find(|(key, _)| key == "taint_lost")
            .unwrap()
            .1 = Json::Null;
        assert_eq!(
            failing_rules(&Json::Obj(pairs), &falsifiers()),
            vec!["taint_lost <= 0".to_string()]
        );
    }
}
