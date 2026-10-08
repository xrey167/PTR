//! The metric list of S001 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies S001).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`, every
//! "a != b" violation `a == b`. The harness judges its own output with them, and a test pins
//! `criteria.toml` to exactly these lists, in order.
//!
//! Fractions are integer permille. Every metric is a JSON integer.

use super::params;
use super::runner::Counters;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 33] = [
    "deltas",
    "affected_total",
    "structural_closure_total",
    "value_changed_total",
    "missed_invalidations",
    "closure_mismatches",
    "semantic_divergences",
    "recall_permille",
    "precision_structural_permille",
    "precision_value_permille",
    "unnecessary_invalidations",
    "evals_global",
    "evals_incremental",
    "locality_ratio_permille_local",
    "locality_ratio_permille_hub",
    "noop_revision_bumps",
    "revision_mismatches",
    "unexpected_refusals",
    "cycle_attempts",
    "cycle_refusals",
    "cycle_unrefused",
    "missing_dependency_probes",
    "missing_dependency_refusals",
    "missing_dependency_unrefused",
    "locality_violation",
    "max_closure_depth",
    "rewires",
    "removals",
    "noop_writes",
    "hub_deltas",
    "lossy_class_deltas",
    "local_class_deltas",
    "topology_classes",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// metric above; the first six are falsifier rules on their own, the others restate a rule
/// that compares two counters (`cycle_attempts == cycle_refusals`) or a ratio as a count.
pub const HARD_COUNTERS: [&str; 9] = [
    "missed_invalidations",
    "closure_mismatches",
    "semantic_divergences",
    "noop_revision_bumps",
    "revision_mismatches",
    "unexpected_refusals",
    "cycle_unrefused",
    "missing_dependency_unrefused",
    "locality_violation",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    vec![
        format!("deltas >= {}", params::MIN_DELTAS),
        format!("topology_classes >= {}", params::MIN_TOPOLOGY_CLASSES),
        format!("max_closure_depth >= {}", params::MIN_MAX_CLOSURE_DEPTH),
        format!("rewires >= {}", params::MIN_REWIRES),
        format!("removals >= {}", params::MIN_REMOVALS),
        format!("noop_writes >= {}", params::MIN_NOOP_WRITES),
        format!("cycle_attempts >= {}", params::MIN_CYCLE_ATTEMPTS),
        format!(
            "missing_dependency_probes >= {}",
            params::MIN_MISSING_DEPENDENCY_PROBES
        ),
        format!("hub_deltas >= {}", params::MIN_HUB_DELTAS),
        format!("lossy_class_deltas >= {}", params::MIN_LOSSY_CLASS_DELTAS),
        format!("local_class_deltas >= {}", params::MIN_LOCAL_CLASS_DELTAS),
        format!("value_changed_total >= {}", params::MIN_VALUE_CHANGED_TOTAL),
        format!("evals_global >= {}", params::MIN_EVALS_GLOBAL),
    ]
}

/// The falsifiers: each tripped one falsifies S001 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        format!(
            "missed_invalidations <= {}",
            params::MAX_MISSED_INVALIDATIONS
        ),
        format!("closure_mismatches <= {}", params::MAX_CLOSURE_MISMATCHES),
        format!(
            "semantic_divergences <= {}",
            params::MAX_SEMANTIC_DIVERGENCES
        ),
        format!("noop_revision_bumps <= {}", params::MAX_NOOP_REVISION_BUMPS),
        format!("revision_mismatches <= {}", params::MAX_REVISION_MISMATCHES),
        format!("unexpected_refusals <= {}", params::MAX_UNEXPECTED_REFUSALS),
        "cycle_attempts == cycle_refusals".to_string(),
        "missing_dependency_probes == missing_dependency_refusals".to_string(),
        format!(
            "locality_ratio_permille_local <= {}",
            params::MAX_LOCALITY_RATIO_PERMILLE_LOCAL
        ),
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

/// The metrics of a pass, from its counters, into `tally`.
pub fn fill(tally: &mut Tally, c: &Counters, topology_classes: usize) {
    let locality_local = permille(c.local_incremental, c.local_global, 0);
    let values = [
        ("deltas", c.deltas),
        ("affected_total", c.affected_total),
        ("structural_closure_total", c.structural_closure_total),
        ("value_changed_total", c.value_changed_total),
        ("missed_invalidations", c.missed_invalidations),
        ("closure_mismatches", c.closure_mismatches),
        ("semantic_divergences", c.semantic_divergences),
        (
            "recall_permille",
            permille(c.value_changed_in_affected, c.value_changed_total, 1000),
        ),
        (
            "precision_structural_permille",
            permille(c.affected_in_closure, c.affected_total, 1000),
        ),
        (
            "precision_value_permille",
            permille(c.recomputed_changed, c.recomputed_total, 1000),
        ),
        ("unnecessary_invalidations", c.unnecessary_invalidations),
        ("evals_global", c.evals_global),
        ("evals_incremental", c.evals_incremental),
        ("locality_ratio_permille_local", locality_local),
        (
            "locality_ratio_permille_hub",
            permille(c.hub_incremental, c.hub_global, 0),
        ),
        ("noop_revision_bumps", c.noop_revision_bumps),
        ("revision_mismatches", c.revision_mismatches),
        ("unexpected_refusals", c.unexpected_refusals),
        ("cycle_attempts", c.cycle_attempts),
        ("cycle_refusals", c.cycle_refusals),
        ("cycle_unrefused", c.cycle_attempts - c.cycle_refusals),
        ("missing_dependency_probes", c.missing_dependency_probes),
        ("missing_dependency_refusals", c.missing_dependency_refusals),
        (
            "missing_dependency_unrefused",
            c.missing_dependency_probes - c.missing_dependency_refusals,
        ),
        (
            "locality_violation",
            i64::from(locality_local > params::MAX_LOCALITY_RATIO_PERMILLE_LOCAL),
        ),
        ("max_closure_depth", c.max_closure_depth),
        ("rewires", c.rewires),
        ("removals", c.removals),
        ("noop_writes", c.noop_writes),
        ("hub_deltas", c.hub_deltas),
        ("lossy_class_deltas", c.lossy_class_deltas),
        ("local_class_deltas", c.local_class_deltas),
        ("topology_classes", topology_classes as i64),
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
                    "deltas" => params::MIN_DELTAS,
                    "topology_classes" => params::MIN_TOPOLOGY_CLASSES,
                    "max_closure_depth" => params::MIN_MAX_CLOSURE_DEPTH,
                    "local_class_deltas" => params::MIN_LOCAL_CLASS_DELTAS,
                    "cycle_attempts"
                    | "cycle_refusals"
                    | "missing_dependency_probes"
                    | "missing_dependency_refusals" => 3,
                    "locality_ratio_permille_local" => params::MAX_LOCALITY_RATIO_PERMILLE_LOCAL,
                    "missed_invalidations"
                    | "closure_mismatches"
                    | "semantic_divergences"
                    | "noop_revision_bumps"
                    | "revision_mismatches"
                    | "unexpected_refusals"
                    | "cycle_unrefused"
                    | "missing_dependency_unrefused"
                    | "locality_violation" => 0,
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
            if tokens[1] == "==" {
                set(&mut pairs, tokens[0], Json::Int(4));
            } else {
                let bound: i64 = tokens[2].parse().unwrap();
                set(&mut pairs, tokens[0], Json::Int(bound + 1));
            }
            assert_eq!(failing(pairs, &falsifiers()), vec![rule.clone()], "{rule}");
        }
        // The other side of each equality.
        let mut pairs = passing();
        set(&mut pairs, "cycle_refusals", Json::Int(4));
        assert_eq!(
            failing(pairs, &falsifiers()),
            vec!["cycle_attempts == cycle_refusals".to_string()]
        );
        let mut pairs = passing();
        set(&mut pairs, "missing_dependency_refusals", Json::Int(2));
        assert_eq!(
            failing(pairs, &falsifiers()),
            vec!["missing_dependency_probes == missing_dependency_refusals".to_string()]
        );
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "missed_invalidations",
            "locality_ratio_permille_local",
            "cycle_refusals",
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
    fn the_hard_counters_are_metrics_and_the_falsifier_counters_are_hard() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        for rule in falsifiers() {
            let first = rule.split(' ').next().unwrap();
            if rule.contains("<= 0") {
                assert!(HARD_COUNTERS.contains(&first), "{rule}");
            }
        }
    }

    #[test]
    fn filling_a_tally_prints_every_metric_as_an_integer() {
        let mut tally = Tally::new(LIST);
        let counters = Counters {
            local_incremental: 10,
            local_global: 100,
            cycle_attempts: 5,
            cycle_refusals: 4,
            ..Counters::default()
        };
        fill(&mut tally, &counters, 5);
        let (printed, gaps) = tally.metrics();
        assert!(gaps.is_empty());
        assert!(printed
            .iter()
            .all(|(_, value)| matches!(value, Json::Int(_))));
        assert_eq!(tally.get("locality_ratio_permille_local"), 100);
        assert_eq!(tally.get("cycle_unrefused"), 1);
        assert_eq!(hard_failures(&tally), 1);
    }
}
