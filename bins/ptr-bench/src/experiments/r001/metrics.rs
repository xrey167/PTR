//! The metric list of R001, its hard counters, and the two judgements of a run: the coverage
//! prerequisites (a run that misses one is vacuous) and the falsifiers (a run that trips one
//! falsifies R001).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): a
//! prerequisite "counter < N" is written `>= N`, a violation "counter > 0" is written `<= 0`,
//! and the two bounds that compare two measured numbers are written reference against
//! reference. The harness judges its own output with them, and a test pins `criteria.toml` to
//! exactly these lists, in order.
//!
//! Gated quantities are counts and virtual ticks. Wall-clock latency, the resident set and
//! the threaded arm's peak depth timing live under `descriptive` in the output and no rule
//! reads them.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics, in the order the harness prints them.
pub const METRIC_NAMES: [&str; 54] = [
    "offered",
    "admitted",
    "refused_full",
    "refused_closed",
    "delivered",
    "silent_drops",
    "ownership_violations",
    "capacity_violations",
    "fifo_violations",
    "model_divergences",
    "spurious_refusals",
    "wait_bound_violations",
    "payload_bound_violations",
    "closed_path_mismatches",
    "determinism_mismatches",
    "canary_misses",
    "product_panics",
    "product_hangs",
    "threaded_ledger_violations",
    "threaded_capacity_violations",
    "threaded_admitted_after_close",
    "bounded_peak_depth",
    "unbounded_peak_depth",
    "bounded_peak_payload_bytes",
    "bounded_payload_bytes_bound",
    "unbounded_peak_payload_bytes",
    "bounded_p99_wait_ticks",
    "bounded_max_wait_ticks",
    "bounded_wait_bound_ticks",
    "unbounded_p99_wait_ticks",
    "refusal_permille_overload",
    "overload_phases",
    "unbounded_exceeding_cells",
    "capacities_exercised",
    "payload_classes",
    "recoveries_observed",
    "closed_path_cases",
    "closed_while_full_cells",
    "closed_while_nonempty_cells",
    "owner_drop_cases",
    "threaded_producers",
    "threaded_fill_rounds",
    "threaded_producers_with_admissions",
    "threaded_interleaved_fills",
    "threaded_round2_admitted",
    "threaded_round2_full",
    "threaded_round2_closed",
    "threaded_offered",
    "threaded_admitted",
    "threaded_refused_full",
    "threaded_refused_closed",
    "threaded_delivered",
    "negative_controls_tripped",
    "hard_failures",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The counters that must be zero: any of them above zero is a hard failure of the run
/// (`hard_failures` is their sum, and the mutation checker reads the ones that fired).
pub const HARD_COUNTERS: [&str; 15] = [
    "silent_drops",
    "ownership_violations",
    "capacity_violations",
    "fifo_violations",
    "model_divergences",
    "spurious_refusals",
    "wait_bound_violations",
    "payload_bound_violations",
    "closed_path_mismatches",
    "determinism_mismatches",
    "canary_misses",
    "product_panics",
    "product_hangs",
    "threaded_ledger_violations",
    "threaded_capacity_violations",
];

/// The cells of the sweep: every capacity with every payload class.
pub fn cell_count() -> usize {
    params::CAPACITIES.len() * params::PAYLOAD_CLASSES.len()
}

/// The coverage prerequisites: each unmet one makes the run vacuous. Every counter they read
/// is derived from events the run observed, not from a configured constant.
pub fn coverage() -> Vec<String> {
    let threaded = params::THREADED_PINNED;
    vec![
        // Every overload phase of every cell saw a refusal.
        format!("overload_phases >= {}", 3 * cell_count()),
        "refused_full >= 1".to_string(),
        "delivered >= 1".to_string(),
        // The unbounded baseline outgrows the capacity in every cell.
        format!("unbounded_exceeding_cells >= {}", cell_count()),
        // A capacity counts when the bounded queue was observed full at it; a payload class when
        // a body of that class was observed alive.
        format!("capacities_exercised >= {}", params::CAPACITIES.len()),
        format!("payload_classes >= {}", params::PAYLOAD_CLASSES.len()),
        // The bounded queue drains in the recovery phase and accepts again, in every cell.
        format!("recoveries_observed >= {}", cell_count()),
        // Six scheduled closes and the capacity-0 unit.
        format!("closed_path_cases >= {}", cell_count() + 1),
        // Closed beats Full: cells where an offer was refused Closed while the queue was full,
        // and cells where messages were delivered after the close (drain before Closed).
        "closed_while_full_cells >= 4".to_string(),
        format!("closed_while_nonempty_cells >= {}", cell_count()),
        "owner_drop_cases >= 1".to_string(),
        // The threaded arm: the exact fills ran, the contended stage saw all three outcomes, and
        // at least two producers actually got messages in.
        format!("threaded_fill_rounds >= {}", threaded.fill_rounds),
        "threaded_round2_admitted >= 1".to_string(),
        "threaded_round2_full >= 1".to_string(),
        "threaded_round2_closed >= 1".to_string(),
        "threaded_producers_with_admissions >= 2".to_string(),
        // Fills in which at least two producers got a message in: the producers met at the
        // queue, which a race in the capacity check needs (about 5% of the fills in practice).
        "threaded_interleaved_fills >= 20".to_string(),
        // The first fill alone delivers `capacity` messages and the close drain one more.
        format!("threaded_delivered >= {}", threaded.capacity + 1),
        // The oracle's own canaries (broken queues it must catch) were all caught.
        "canary_misses <= 0".to_string(),
    ]
}

/// The falsifiers: each tripped one falsifies R001 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        "capacity_violations <= 0".to_string(),
        "silent_drops <= 0".to_string(),
        "ownership_violations <= 0".to_string(),
        "fifo_violations <= 0".to_string(),
        "model_divergences <= 0".to_string(),
        "spurious_refusals <= 0".to_string(),
        "bounded_p99_wait_ticks <= bounded_wait_bound_ticks".to_string(),
        "wait_bound_violations <= 0".to_string(),
        "bounded_peak_payload_bytes <= bounded_payload_bytes_bound".to_string(),
        "payload_bound_violations <= 0".to_string(),
        "determinism_mismatches <= 0".to_string(),
        "closed_path_mismatches <= 0".to_string(),
        "threaded_ledger_violations <= 0".to_string(),
        "threaded_capacity_violations <= 0".to_string(),
        "product_panics <= 0".to_string(),
        "product_hangs <= 0".to_string(),
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
        let mut pairs: Vec<(String, Json)> = Vec::new();
        for name in METRIC_NAMES {
            let value = match name {
                "bounded_p99_wait_ticks" | "bounded_wait_bound_ticks" => 5,
                "bounded_peak_payload_bytes" | "bounded_payload_bytes_bound" => 100,
                other => thresholds
                    .iter()
                    .find(|(key, _)| key == other)
                    .map_or(0, |(_, bound)| *bound),
            };
            pairs.push((name.to_string(), Json::Int(value)));
        }
        pairs
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
    fn the_hard_counters_are_metrics_and_no_gated_name_is_a_wall_clock_one() {
        for counter in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&counter), "{counter}");
        }
        for name in METRIC_NAMES {
            assert!(
                !name.contains("wall") && !name.contains("_ns") && !name.contains("rss"),
                "{name}"
            );
        }
        let mut names = METRIC_NAMES.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), METRIC_NAMES.len());
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
            let broken = if rule.contains(">=") {
                bound - 1
            } else {
                bound + 1
            };
            set(&mut pairs, name, Json::Int(broken));
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
    fn the_cross_reference_bounds_fail_from_either_side() {
        let mut pairs = passing();
        set(&mut pairs, "bounded_wait_bound_ticks", Json::Int(4));
        assert_eq!(
            failing_rules(pairs, &falsifiers()),
            vec!["bounded_p99_wait_ticks <= bounded_wait_bound_ticks".to_string()]
        );
        let mut pairs = passing();
        set(&mut pairs, "bounded_payload_bytes_bound", Json::Int(99));
        assert_eq!(
            failing_rules(pairs, &falsifiers()),
            vec!["bounded_peak_payload_bytes <= bounded_payload_bytes_bound".to_string()]
        );
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "capacity_violations",
            "bounded_p99_wait_ticks",
            "canary_misses",
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
                let failed = failing_rules(
                    pairs,
                    &coverage()
                        .into_iter()
                        .chain(falsifiers())
                        .collect::<Vec<_>>(),
                );
                assert!(!failed.is_empty(), "{name}");
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
