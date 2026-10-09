//! The metric list of S002 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies S002).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`, every
//! "a > b" violation `a <= b`. The harness judges its own output with them, and a test pins
//! `criteria.toml` to exactly these lists, in order.
//!
//! Fractions are integer permille. Every metric is a JSON integer.

use super::params;
use crate::experiments::layered_kv::emit::{MetricList, Tally};
use std::collections::BTreeSet;

/// The two arms, as indexes into the per-arm counters.
pub const A0: usize = 0;
pub const A1: usize = 1;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 41] = [
    "runs",
    "stale_runs_by_schedule",
    "stale_runs_observed",
    "stale_actions_executed",
    "stale_effect_attempts_committed",
    "stale_checkpoint_admissions",
    "fresh_actions_refused",
    "stale_boundary_attempts_a0",
    "stale_boundary_attempts_a1",
    "stale_action_rate_permille_a0",
    "stale_action_rate_permille_a1",
    "boundary_refusals",
    "boundary_refusals_a0",
    "boundary_refusals_a1",
    "cancelled_runs",
    "false_cancellations",
    "missed_cancellations",
    "fresh_runs_completed",
    "fresh_runs_starved",
    "cancel_contract_violations",
    "boundary_arm_divergences",
    "cancel_latency_rounds_max",
    "cancel_latency_violation",
    "wasted_units_a0",
    "wasted_units_a1",
    "wasted_work_violation",
    "total_units_a0",
    "total_units_a1",
    "net_units_a1_minus_a0",
    "double_executions",
    "executor_vs_ledger_mismatches",
    "mirror_drift",
    "product_panics",
    "product_hangs",
    "effect_kinds_covered",
    "generation_stale_cases",
    "injection_positions_covered",
    "control_trips",
    "noop_writes_observed",
    "retries_replayed",
    "restarted_runs",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// metric above; the first twelve and the last two are falsifier rules on their own, the
/// others restate a rule that compares two numbers (`cancel_latency_rounds_max <= 1`,
/// `wasted_units_a1 <= wasted_units_a0`, `wasted_units_a1 <= 0`) as a count.
pub const HARD_COUNTERS: [&str; 16] = [
    "stale_actions_executed",
    "stale_effect_attempts_committed",
    "stale_checkpoint_admissions",
    "fresh_actions_refused",
    "false_cancellations",
    "missed_cancellations",
    "fresh_runs_starved",
    "cancel_contract_violations",
    "boundary_arm_divergences",
    "double_executions",
    "executor_vs_ledger_mismatches",
    "mirror_drift",
    "cancel_latency_violation",
    "wasted_work_violation",
    "product_panics",
    "product_hangs",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    vec![
        format!("runs >= {}", params::MIN_RUNS),
        format!(
            "stale_runs_by_schedule >= {}",
            params::MIN_STALE_RUNS_BY_SCHEDULE
        ),
        format!("stale_runs_observed >= {}", params::MIN_STALE_RUNS_OBSERVED),
        format!(
            "boundary_refusals_a0 >= {}",
            params::MIN_BOUNDARY_REFUSALS_A0
        ),
        format!(
            "boundary_refusals_a1 >= {}",
            params::MIN_BOUNDARY_REFUSALS_A1
        ),
        format!("cancelled_runs >= {}", params::MIN_CANCELLED_RUNS),
        format!(
            "effect_kinds_covered >= {}",
            params::MIN_EFFECT_KINDS_COVERED
        ),
        format!(
            "fresh_runs_completed >= {}",
            params::MIN_FRESH_RUNS_COMPLETED
        ),
        format!(
            "generation_stale_cases >= {}",
            params::MIN_GENERATION_STALE_CASES
        ),
        format!(
            "injection_positions_covered >= {}",
            params::MIN_INJECTION_POSITIONS_COVERED
        ),
        format!("control_trips >= {}", params::MIN_CONTROL_TRIPS),
        format!("wasted_units_a0 >= {}", params::MIN_WASTED_UNITS_A0),
        format!(
            "noop_writes_observed >= {}",
            params::MIN_NOOP_WRITES_OBSERVED
        ),
        format!("retries_replayed >= {}", params::MIN_RETRIES_REPLAYED),
        format!("restarted_runs >= {}", params::MIN_RESTARTED_RUNS),
    ]
}

/// The falsifiers: each tripped one falsifies S002 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        format!(
            "stale_actions_executed <= {}",
            params::MAX_STALE_ACTIONS_EXECUTED
        ),
        format!(
            "stale_effect_attempts_committed <= {}",
            params::MAX_STALE_EFFECT_ATTEMPTS_COMMITTED
        ),
        format!(
            "stale_checkpoint_admissions <= {}",
            params::MAX_STALE_CHECKPOINT_ADMISSIONS
        ),
        format!(
            "fresh_actions_refused <= {}",
            params::MAX_FRESH_ACTIONS_REFUSED
        ),
        format!("false_cancellations <= {}", params::MAX_FALSE_CANCELLATIONS),
        format!(
            "missed_cancellations <= {}",
            params::MAX_MISSED_CANCELLATIONS
        ),
        format!("fresh_runs_starved <= {}", params::MAX_FRESH_RUNS_STARVED),
        format!(
            "cancel_contract_violations <= {}",
            params::MAX_CANCEL_CONTRACT_VIOLATIONS
        ),
        format!(
            "boundary_arm_divergences <= {}",
            params::MAX_BOUNDARY_ARM_DIVERGENCES
        ),
        format!("double_executions <= {}", params::MAX_DOUBLE_EXECUTIONS),
        format!(
            "executor_vs_ledger_mismatches <= {}",
            params::MAX_EXECUTOR_VS_LEDGER_MISMATCHES
        ),
        format!("mirror_drift <= {}", params::MAX_MIRROR_DRIFT),
        format!(
            "cancel_latency_rounds_max <= {}",
            params::MAX_CANCEL_LATENCY_ROUNDS_MAX
        ),
        "wasted_units_a1 <= wasted_units_a0".to_string(),
        format!("wasted_units_a1 <= {}", params::MAX_WASTED_UNITS_A1),
        format!("product_panics <= {}", params::MAX_PRODUCT_PANICS),
        format!("product_hangs <= {}", params::MAX_PRODUCT_HANGS),
    ]
}

/// `1000 * numerator / denominator`, or `empty` when there is nothing to divide.
pub fn permille(numerator: i64, denominator: i64, empty: i64) -> i64 {
    if denominator == 0 {
        empty
    } else {
        1000 * numerator / denominator
    }
}

/// Everything a run counts. Per-arm numbers are indexed by [`A0`] and [`A1`].
#[derive(Clone, Debug, Default)]
pub struct Counters {
    pub runs: i64,
    pub stale_runs_by_schedule: i64,
    pub stale_runs_observed: i64,
    pub stale_executed: [i64; 2],
    pub stale_effect_attempts_committed: i64,
    pub stale_checkpoint_admissions: i64,
    pub fresh_actions_refused: i64,
    pub stale_boundary_attempts: [i64; 2],
    pub boundary_refusals: [i64; 2],
    pub cancelled_runs: i64,
    pub false_cancellations: i64,
    pub missed_cancellations: i64,
    pub fresh_runs_completed: i64,
    pub fresh_runs_starved: i64,
    pub cancel_contract_violations: i64,
    pub boundary_arm_divergences: i64,
    pub cancel_latency_rounds_max: i64,
    pub wasted_units: [i64; 2],
    pub total_units: [i64; 2],
    pub double_executions: i64,
    pub executor_vs_ledger_mismatches: i64,
    pub mirror_drift: i64,
    pub product_panics: i64,
    pub product_hangs: i64,
    pub effect_kinds: BTreeSet<&'static str>,
    pub generation_stale_cases: i64,
    pub positions: BTreeSet<params::Position>,
    pub control_trips: i64,
    pub noop_writes_observed: i64,
    pub retries_replayed: i64,
    pub restarted_runs: i64,
    // Descriptive counters, in no rule.
    pub executions: i64,
    pub retries_refused: i64,
    pub cancelled_step_work_units: i64,
    pub generation_boundary_refusals: i64,
    pub stale_permit_refusals: i64,
    pub revision_refusals: i64,
    pub late_stale_refusals_by_checkpoint: [i64; 3],
    pub cases_by_class: std::collections::BTreeMap<&'static str, i64>,
    pub cases_by_position: std::collections::BTreeMap<&'static str, i64>,
    pub cases_by_effect: std::collections::BTreeMap<&'static str, i64>,
    pub cases_by_writer: std::collections::BTreeMap<&'static str, i64>,
    pub latency_histogram: std::collections::BTreeMap<i64, i64>,
}

impl Counters {
    pub fn stale_actions_executed(&self) -> i64 {
        self.stale_executed[A0] + self.stale_executed[A1]
    }

    pub fn boundary_refusals_total(&self) -> i64 {
        self.boundary_refusals[A0] + self.boundary_refusals[A1]
    }

    pub fn cancel_latency_violation(&self) -> i64 {
        i64::from(self.cancel_latency_rounds_max > params::MAX_CANCEL_LATENCY_ROUNDS_MAX)
    }

    pub fn wasted_work_violation(&self) -> i64 {
        i64::from(
            self.wasted_units[A1] > self.wasted_units[A0]
                || self.wasted_units[A1] > params::MAX_WASTED_UNITS_A1,
        )
    }
}

/// The metrics of a run, from its counters, into `tally`.
pub fn fill(tally: &mut Tally, c: &Counters) {
    let values = [
        ("runs", c.runs),
        ("stale_runs_by_schedule", c.stale_runs_by_schedule),
        ("stale_runs_observed", c.stale_runs_observed),
        ("stale_actions_executed", c.stale_actions_executed()),
        (
            "stale_effect_attempts_committed",
            c.stale_effect_attempts_committed,
        ),
        ("stale_checkpoint_admissions", c.stale_checkpoint_admissions),
        ("fresh_actions_refused", c.fresh_actions_refused),
        ("stale_boundary_attempts_a0", c.stale_boundary_attempts[A0]),
        ("stale_boundary_attempts_a1", c.stale_boundary_attempts[A1]),
        (
            "stale_action_rate_permille_a0",
            permille(c.stale_executed[A0], c.stale_boundary_attempts[A0], 0),
        ),
        (
            "stale_action_rate_permille_a1",
            permille(c.stale_executed[A1], c.stale_boundary_attempts[A1], 0),
        ),
        ("boundary_refusals", c.boundary_refusals_total()),
        ("boundary_refusals_a0", c.boundary_refusals[A0]),
        ("boundary_refusals_a1", c.boundary_refusals[A1]),
        ("cancelled_runs", c.cancelled_runs),
        ("false_cancellations", c.false_cancellations),
        ("missed_cancellations", c.missed_cancellations),
        ("fresh_runs_completed", c.fresh_runs_completed),
        ("fresh_runs_starved", c.fresh_runs_starved),
        ("cancel_contract_violations", c.cancel_contract_violations),
        ("boundary_arm_divergences", c.boundary_arm_divergences),
        ("cancel_latency_rounds_max", c.cancel_latency_rounds_max),
        ("cancel_latency_violation", c.cancel_latency_violation()),
        ("wasted_units_a0", c.wasted_units[A0]),
        ("wasted_units_a1", c.wasted_units[A1]),
        ("wasted_work_violation", c.wasted_work_violation()),
        ("total_units_a0", c.total_units[A0]),
        ("total_units_a1", c.total_units[A1]),
        (
            "net_units_a1_minus_a0",
            c.total_units[A1] - c.total_units[A0],
        ),
        ("double_executions", c.double_executions),
        (
            "executor_vs_ledger_mismatches",
            c.executor_vs_ledger_mismatches,
        ),
        ("mirror_drift", c.mirror_drift),
        ("product_panics", c.product_panics),
        ("product_hangs", c.product_hangs),
        ("effect_kinds_covered", c.effect_kinds.len() as i64),
        ("generation_stale_cases", c.generation_stale_cases),
        ("injection_positions_covered", c.positions.len() as i64),
        ("control_trips", c.control_trips),
        ("noop_writes_observed", c.noop_writes_observed),
        ("retries_replayed", c.retries_replayed),
        ("restarted_runs", c.restarted_runs),
    ];
    for (name, value) in values {
        tally.set(name, value);
    }
}

/// The sum of the hard counters in `tally`.
pub fn hard_failures(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        METRIC_NAMES
            .iter()
            .map(|name| {
                let value = match *name {
                    "runs" => params::MIN_RUNS,
                    "stale_runs_by_schedule" => params::MIN_STALE_RUNS_BY_SCHEDULE,
                    "stale_runs_observed" => params::MIN_STALE_RUNS_OBSERVED,
                    "cancelled_runs" => params::MIN_CANCELLED_RUNS,
                    "effect_kinds_covered" => params::MIN_EFFECT_KINDS_COVERED,
                    "fresh_runs_completed" => params::MIN_FRESH_RUNS_COMPLETED,
                    "generation_stale_cases" => params::MIN_GENERATION_STALE_CASES,
                    "injection_positions_covered" => params::MIN_INJECTION_POSITIONS_COVERED,
                    "control_trips" => params::MIN_CONTROL_TRIPS,
                    "cancel_latency_rounds_max" => params::MAX_CANCEL_LATENCY_ROUNDS_MAX,
                    "wasted_units_a0" => 5,
                    "wasted_units_a1"
                    | "stale_actions_executed"
                    | "stale_effect_attempts_committed"
                    | "stale_checkpoint_admissions"
                    | "fresh_actions_refused"
                    | "false_cancellations"
                    | "missed_cancellations"
                    | "fresh_runs_starved"
                    | "cancel_contract_violations"
                    | "boundary_arm_divergences"
                    | "double_executions"
                    | "executor_vs_ledger_mismatches"
                    | "mirror_drift"
                    | "cancel_latency_violation"
                    | "wasted_work_violation"
                    | "product_panics"
                    | "product_hangs" => 0,
                    _ => 1,
                };
                (name.to_string(), Json::Int(value))
            })
            .collect()
    }

    fn failing(pairs: Vec<(String, Json)>, rules: &[String]) -> Vec<String> {
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
    fn a_result_at_every_threshold_holds_every_rule() {
        assert!(failing(passing(), &coverage()).is_empty());
        assert!(failing(passing(), &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_alone_when_its_counter_leaves_the_threshold() {
        for rule in coverage() {
            let name = rule.split(' ').next().unwrap();
            let bound: i64 = rule.rsplit(' ').next().unwrap().parse().unwrap();
            let mut pairs = passing();
            set(&mut pairs, name, Json::Int(bound - 1));
            assert_eq!(failing(pairs, &coverage()), vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn each_falsifier_trips_alone_on_a_single_violation() {
        for rule in falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let mut pairs = passing();
            if let Ok(bound) = tokens[2].parse::<i64>() {
                set(&mut pairs, tokens[0], Json::Int(bound + 1));
                // "wasted_units_a1 <= 0" also breaks the comparison with the other arm only when
                // it exceeds it: the reference has a0 = 5, so one unit breaks the first rule only.
                assert_eq!(failing(pairs, &falsifiers()), vec![rule.clone()], "{rule}");
            } else {
                // wasted_units_a1 <= wasted_units_a0: 6 > 5, and also > 0.
                set(&mut pairs, "wasted_units_a1", Json::Int(6));
                let tripped = failing(pairs, &falsifiers());
                assert!(tripped.contains(&rule), "{rule}: {tripped:?}");
            }
        }
    }

    #[test]
    fn the_comparison_with_the_baseline_is_a_rule_of_its_own() {
        // a1 = 3 <= a0 = 5 holds the comparison but not the absolute bound; a1 = 6 breaks both.
        let mut pairs = passing();
        set(&mut pairs, "wasted_units_a1", Json::Int(3));
        assert_eq!(
            failing(pairs, &falsifiers()),
            vec!["wasted_units_a1 <= 0".to_string()]
        );
        let mut pairs = passing();
        set(&mut pairs, "wasted_units_a1", Json::Int(6));
        assert_eq!(
            failing(pairs, &falsifiers()),
            vec![
                "wasted_units_a1 <= wasted_units_a0".to_string(),
                "wasted_units_a1 <= 0".to_string()
            ]
        );
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "stale_actions_executed",
            "cancel_latency_rounds_max",
            "wasted_units_a0",
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
                assert!(!failing(pairs, &falsifiers()).is_empty(), "{name}");
            }
        }
    }

    #[test]
    fn fractions_are_integer_permille_and_empty_denominators_take_the_given_value() {
        assert_eq!(permille(1, 3, 0), 333);
        assert_eq!(permille(3, 3, 0), 1000);
        assert_eq!(permille(0, 0, 1000), 1000);
        assert_eq!(permille(0, 0, 0), 0);
    }

    #[test]
    fn the_hard_counters_are_metrics_and_every_zero_bounded_falsifier_counter_is_hard() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        for rule in falsifiers() {
            let first = rule.split(' ').next().unwrap();
            if rule.ends_with("<= 0") && first != "wasted_units_a1" {
                assert!(HARD_COUNTERS.contains(&first), "{rule}");
            }
        }
    }

    #[test]
    fn the_derived_violations_restate_their_rules() {
        let mut c = Counters::default();
        assert_eq!(c.cancel_latency_violation(), 0);
        assert_eq!(c.wasted_work_violation(), 0);
        c.cancel_latency_rounds_max = 1;
        assert_eq!(c.cancel_latency_violation(), 0);
        c.cancel_latency_rounds_max = 2;
        assert_eq!(c.cancel_latency_violation(), 1);
        c.wasted_units = [4, 1];
        assert_eq!(c.wasted_work_violation(), 1, "a1 > 0 is a violation");
        c.wasted_units = [0, 1];
        assert_eq!(c.wasted_work_violation(), 1);
        c.wasted_units = [4, 0];
        assert_eq!(c.wasted_work_violation(), 0);
    }

    #[test]
    fn filling_a_tally_prints_every_metric_as_an_integer() {
        let mut tally = Tally::new(LIST);
        let mut counters = Counters {
            stale_boundary_attempts: [10, 4],
            ..Counters::default()
        };
        counters.stale_executed = [1, 0];
        counters.false_cancellations = 2;
        fill(&mut tally, &counters);
        let (printed, gaps) = tally.metrics();
        assert!(gaps.is_empty());
        assert!(printed
            .iter()
            .all(|(_, value)| matches!(value, Json::Int(_))));
        assert_eq!(tally.get("stale_action_rate_permille_a0"), 100);
        assert_eq!(hard_failures(&tally), 3);
    }
}
