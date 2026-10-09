//! The metric list of F003, its hard counters, and the two judgements of a run: the coverage
//! prerequisites (a run that misses one is vacuous) and the falsifiers (a run that trips one
//! falsifies F003 on that seed).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`). The
//! harness judges its own output with them, and a test pins `criteria.toml` to exactly these
//! lists, in order. Every metric is a top-level JSON integer; fractions are permille or ppm.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics, in the order the harness prints them.
pub const METRIC_NAMES: [&str; 48] = [
    "stream_branches",
    "eligible_branches",
    "adjudications",
    "policies_recorded",
    "policies_certified",
    "ineligible_kinds_seen",
    "nonslice_probes",
    "mc_replicates",
    "mc_certified_replicates",
    "ope_logs",
    "ope_estimates",
    "positivity_probes",
    "slice_rate_tests",
    "heldout_candidates_min_certified",
    "ltt_reviews_saved_permille",
    "uncalibrated_true_harm_ppm",
    "uncalibrated_baseline_rejected",
    "negative_controls_tripped",
    "ltt_coverage_violations",
    "ltt_coverage_critical",
    "ineligible_auto_proposals",
    "decision_mismatches",
    "unexplained_triages",
    "nonslice_adjudicated",
    "adjudication_mismatches",
    "calibration_set_mismatches",
    "held_out_overlap",
    "held_out_missing",
    "threshold_mismatches",
    "crc_mismatches",
    "bound_oracle_mismatches",
    "ltt_coverage_excess",
    "heldout_policy_violations",
    "slice_rate_deviation_violations",
    "ips_bias_violations",
    "dr_bias_violations",
    "positivity_unrefused",
    "ope_errors",
    "product_errors",
    "product_panics",
    "data_starvation",
    "hard_failures",
    "mc_never_replicates",
    "mc_naive_violations",
    "heldout_violations_literal_ucb",
    "stream_realized_harm_ltt_ppm",
    "stream_realized_harm_uncalibrated_ppm",
    "uncalibrated_reviews_saved_permille",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// `data_starvation` is one when the product left the harness without the data a negative
/// control or a coverage count needs (fewer adjudications than the first policy size, or fewer
/// than half the Monte-Carlo replicates completed): a product defect, not a harness fault.
///
/// The counters that must be zero: any of them above zero is a hard failure of the run
/// (`hard_failures` is their sum, and the mutation checker reads the ones that fired).
/// `ltt_coverage_excess` is the count of coverage violations beyond the critical value:
/// `ltt_coverage_violations` itself is not zero for a correct policy (it is at most delta of
/// the replicates).
pub const HARD_COUNTERS: [&str; 21] = [
    "ineligible_auto_proposals",
    "decision_mismatches",
    "unexplained_triages",
    "nonslice_adjudicated",
    "adjudication_mismatches",
    "calibration_set_mismatches",
    "held_out_overlap",
    "held_out_missing",
    "threshold_mismatches",
    "crc_mismatches",
    "bound_oracle_mismatches",
    "ltt_coverage_excess",
    "heldout_policy_violations",
    "slice_rate_deviation_violations",
    "ips_bias_violations",
    "dr_bias_violations",
    "positivity_unrefused",
    "ope_errors",
    "product_errors",
    "product_panics",
    "data_starvation",
];

/// The coverage prerequisites: each unmet one makes the run vacuous. Counters read below are
/// counted from events the run observed, with two exceptions that are by construction and named
/// so: `stream_branches` (the length of the generated stream) and `uncalibrated_true_harm_ppm`
/// (the model's exact harm rate at the uncalibrated threshold, which is what gives the
/// scenario teeth; its observed counterpart is `uncalibrated_baseline_rejected`).
pub fn coverage() -> Vec<String> {
    let shape = params::PINNED;
    let alpha_ppm = i64::from(params::ALPHA_PERMILLE) * 1000;
    vec![
        format!("stream_branches >= {}", shape.stream_branches),
        // The stream is 75 percent eligible and 15 percent of those are adjudicated; the
        // floors leave several standard deviations of room.
        "eligible_branches >= 20000".to_string(),
        "adjudications >= 3000".to_string(),
        // Every policy size was reached and recorded.
        format!("policies_recorded >= {}", shape.policy_sizes.len()),
        // At least one recorded policy certified a threshold: the held-out test has a subject.
        "policies_certified >= 1".to_string(),
        // Every ineligible kind was decided by verification alone at least once.
        "ineligible_kinds_seen >= 6".to_string(),
        "nonslice_probes >= 10000".to_string(),
        format!("mc_replicates >= {}", shape.mc_replicates),
        // At least half of the replicates certified something.
        format!("mc_certified_replicates >= {}", shape.mc_replicates / 2),
        format!("ope_logs >= {}", shape.ope_logs),
        format!("ope_estimates >= {}", shape.ope_logs * 4 * 2),
        format!("positivity_probes >= {}", shape.ope_logs * 2),
        // The overall share, three namespaces and two score halves.
        "slice_rate_tests >= 6".to_string(),
        // The held-out test has data: every certified policy had at least 100 candidates.
        "heldout_candidates_min_certified >= 100".to_string(),
        "ltt_reviews_saved_permille >= 1".to_string(),
        format!("uncalibrated_true_harm_ppm >= {}", alpha_ppm + 1),
        "uncalibrated_baseline_rejected >= 1".to_string(),
        "negative_controls_tripped >= 4".to_string(),
    ]
}

/// The falsifiers: each tripped one falsifies F003 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        "ineligible_auto_proposals <= 0".to_string(),
        "decision_mismatches <= 0".to_string(),
        "unexplained_triages <= 0".to_string(),
        "nonslice_adjudicated <= 0".to_string(),
        "adjudication_mismatches <= 0".to_string(),
        "calibration_set_mismatches <= 0".to_string(),
        "held_out_overlap <= 0".to_string(),
        "held_out_missing <= 0".to_string(),
        "threshold_mismatches <= 0".to_string(),
        "crc_mismatches <= 0".to_string(),
        "bound_oracle_mismatches <= 0".to_string(),
        "ltt_coverage_violations <= ltt_coverage_critical".to_string(),
        "ltt_coverage_excess <= 0".to_string(),
        "heldout_policy_violations <= 0".to_string(),
        "slice_rate_deviation_violations <= 0".to_string(),
        "ips_bias_violations <= 0".to_string(),
        "dr_bias_violations <= 0".to_string(),
        "positivity_unrefused <= 0".to_string(),
        "ope_errors <= 0".to_string(),
        "product_errors <= 0".to_string(),
        "product_panics <= 0".to_string(),
        "data_starvation <= 0".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        let thresholds: Vec<(String, i64)> = coverage()
            .iter()
            .filter_map(|rule| {
                let tokens: Vec<&str> = rule.split(' ').collect();
                (tokens[1] == ">=").then(|| (tokens[0].to_string(), tokens[2].parse().unwrap()))
            })
            .collect();
        METRIC_NAMES
            .iter()
            .map(|name| {
                let value = match *name {
                    "ltt_coverage_violations" | "ltt_coverage_critical" => 130,
                    other => thresholds
                        .iter()
                        .find(|(key, _)| key == other)
                        .map_or(0, |(_, bound)| *bound),
                };
                (name.to_string(), Json::Int(value))
            })
            .collect()
    }

    fn failing_rules(pairs: Vec<(String, Json)>, rules: &[String]) -> Vec<String> {
        judge(&Json::Obj(pairs), &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    fn set(pairs: &mut [(String, Json)], name: &str, value: Json) {
        pairs.iter_mut().find(|(key, _)| key == name).unwrap().1 = value;
    }

    #[test]
    fn the_hard_counters_are_metrics_and_names_are_unique() {
        for counter in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&counter), "{counter}");
        }
        let mut names = METRIC_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), METRIC_NAMES.len());
        for name in METRIC_NAMES {
            assert!(!name.contains("wall") && !name.contains("_ns"), "{name}");
        }
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        assert!(failing_rules(passing(), &coverage()).is_empty());
        assert!(failing_rules(passing(), &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_when_its_counter_leaves_the_threshold() {
        for rule in coverage() {
            let name = rule.split(' ').next().unwrap();
            let bound: i64 = rule.rsplit(' ').next().unwrap().parse().unwrap();
            let mut pairs = passing();
            set(&mut pairs, name, Json::Int(bound - 1));
            assert_eq!(
                failing_rules(pairs, &coverage()),
                vec![rule.clone()],
                "{rule}"
            );
        }
    }

    #[test]
    fn each_falsifier_trips_on_a_single_violation() {
        for rule in falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let mut pairs = passing();
            let current = match pairs.iter().find(|(key, _)| key == tokens[0]).unwrap().1 {
                Json::Int(value) => value,
                _ => unreachable!(),
            };
            set(&mut pairs, tokens[0], Json::Int(current + 1));
            assert_eq!(
                failing_rules(pairs, &falsifiers()),
                vec![rule.clone()],
                "{rule}"
            );
        }
    }

    #[test]
    fn the_coverage_bound_holds_at_the_critical_count_and_fails_one_above() {
        let mut pairs = passing();
        assert!(failing_rules(pairs.clone(), &falsifiers()).is_empty());
        set(&mut pairs, "ltt_coverage_critical", Json::Int(129));
        assert_eq!(
            failing_rules(pairs, &falsifiers()),
            vec!["ltt_coverage_violations <= ltt_coverage_critical".to_string()]
        );
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "ineligible_auto_proposals",
            "ltt_coverage_critical",
            "policies_certified",
        ] {
            for damage in [
                None,
                Some(Json::Null),
                Some(Json::str("0")),
                Some(Json::Float(0.0)),
            ] {
                let mut pairs = passing();
                match damage {
                    None => pairs.retain(|(key, _)| key != name),
                    Some(value) => set(&mut pairs, name, value),
                }
                let all: Vec<String> = coverage().into_iter().chain(falsifiers()).collect();
                assert!(!failing_rules(pairs, &all).is_empty(), "{name}");
            }
        }
    }

    #[test]
    fn every_rule_names_a_metric_and_every_hard_counter_has_a_rule() {
        let all: Vec<String> = coverage().into_iter().chain(falsifiers()).collect();
        for rule in &all {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            assert!(METRIC_NAMES.contains(&tokens[0]), "{rule}");
        }
        for counter in HARD_COUNTERS {
            assert!(
                all.iter()
                    .any(|rule| rule.starts_with(&format!("{counter} "))),
                "{counter} has no rule"
            );
        }
    }
}
