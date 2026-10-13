//! The metric list of Q002 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies Q002).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`, every
//! "a < b" violation `a >= b`. The harness judges its own output with them, and a test pins
//! `criteria.toml` to exactly these lists, in order.
//!
//! Fractions are integer permille. Every metric is a JSON integer (a grouped metric an object
//! of integers).

use super::arms::STAGE_SKIP_KINDS;
use super::params;
use super::world::Class;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 34] = [
    "questions",
    "answerable_questions",
    "unanswerable_questions",
    "hits_by_class",
    "poison_attempts",
    "true_claims",
    "true_known_promoted",
    "recall_permille",
    "recall_violation",
    "hard_honest_presented",
    "hard_honest_refused",
    "false_known_mechanism",
    "false_known_baseline_direct",
    "false_known_gen_filter_only",
    "false_known_bool_trust",
    "stale_or_revoked_promoted",
    "wrong_refusal_reason",
    "product_panics",
    "stage_skip_attempts",
    "stage_skip_by_kind",
    "stage_skip_accepted",
    "stage_skip_unexpected",
    "task_success_mechanism",
    "task_success_baseline",
    "task_success_gen_filter_only",
    "task_success_bool_trust",
    "task_success_violation",
    "answerable_success_permille",
    "answerable_success_baseline_permille",
    "abstentions_correct",
    "oracle_label_mismatches",
    "corpus_selection_digest_mismatches",
    "resolver_calls",
    "lifecycle_queries",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// metric above; the first six, `hard_honest_refused`, `oracle_label_mismatches` and
/// `corpus_selection_digest_mismatches` are rules on their own, `recall_violation` restates
/// the recall rule as a count and `task_success_violation` the three task-success rules.
pub const HARD_COUNTERS: [&str; 11] = [
    "false_known_mechanism",
    "stale_or_revoked_promoted",
    "stage_skip_accepted",
    "stage_skip_unexpected",
    "wrong_refusal_reason",
    "product_panics",
    "hard_honest_refused",
    "recall_violation",
    "task_success_violation",
    "oracle_label_mismatches",
    "corpus_selection_digest_mismatches",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &["hits_by_class", "poison_attempts", "stage_skip_by_kind"],
};

/// The class names of the grouped metrics: the honest hits and the nine poison classes.
pub fn class_names() -> Vec<&'static str> {
    [Class::Honest, Class::HardHonest]
        .into_iter()
        .chain(Class::POISON)
        .map(Class::name)
        .collect()
}

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Class::POISON
        .iter()
        .map(|class| {
            format!(
                "poison_attempts.{} >= {}",
                class.name(),
                params::MIN_POISON_ATTEMPTS_PER_CLASS
            )
        })
        .collect();
    rules.push(format!("true_claims >= {}", params::MIN_TRUE_CLAIMS));
    rules.push(format!(
        "hard_honest_presented >= {}",
        params::MIN_HARD_HONEST_PRESENTED
    ));
    rules.push(format!(
        "unanswerable_questions >= {}",
        params::MIN_UNANSWERABLE_QUESTIONS
    ));
    rules.extend(STAGE_SKIP_KINDS.iter().map(|kind| {
        format!(
            "stage_skip_by_kind.{kind} >= {}",
            params::MIN_STAGE_SKIP_ATTEMPTS_PER_KIND
        )
    }));
    rules.push(format!(
        "false_known_baseline_direct >= {}",
        params::MIN_FALSE_KNOWN_BASELINE_DIRECT
    ));
    rules.push(format!(
        "false_known_gen_filter_only >= {}",
        params::MIN_FALSE_KNOWN_GEN_FILTER_ONLY
    ));
    rules.push(format!(
        "false_known_bool_trust >= {}",
        params::MIN_FALSE_KNOWN_BOOL_TRUST
    ));
    rules.push(format!(
        "oracle_label_mismatches <= {}",
        params::MAX_ORACLE_LABEL_MISMATCHES
    ));
    rules.push(format!(
        "corpus_selection_digest_mismatches <= {}",
        params::MAX_CORPUS_SELECTION_DIGEST_MISMATCHES
    ));
    rules
}

/// The falsifiers: each tripped one falsifies Q002 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        format!(
            "false_known_mechanism <= {}",
            params::MAX_FALSE_KNOWN_MECHANISM
        ),
        format!("stage_skip_accepted <= {}", params::MAX_STAGE_SKIP_ACCEPTED),
        format!(
            "stale_or_revoked_promoted <= {}",
            params::MAX_STALE_OR_REVOKED_PROMOTED
        ),
        format!(
            "stage_skip_unexpected <= {}",
            params::MAX_STAGE_SKIP_UNEXPECTED
        ),
        format!(
            "wrong_refusal_reason <= {}",
            params::MAX_WRONG_REFUSAL_REASON
        ),
        format!("product_panics <= {}", params::MAX_PRODUCT_PANICS),
        format!("hard_honest_refused <= {}", params::MAX_HARD_HONEST_REFUSED),
        format!("recall_permille >= {}", params::MIN_RECALL_PERMILLE),
        "task_success_mechanism >= task_success_baseline".to_string(),
        "task_success_mechanism >= task_success_gen_filter_only".to_string(),
        "task_success_mechanism >= task_success_bool_trust".to_string(),
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

/// The metrics of a run, derived from the counters the arms and the cross-checks filled, into
/// `tally`. `questions` and `answerable` come from the generated world.
pub fn fill(tally: &mut Tally, questions: i64, answerable: i64) {
    tally.set("questions", questions);
    tally.set("answerable_questions", answerable);
    tally.set("unanswerable_questions", questions - answerable);
    let recall = permille(
        tally.get("true_known_promoted"),
        tally.get("true_claims"),
        0,
    );
    tally.set("recall_permille", recall);
    tally.set(
        "recall_violation",
        i64::from(recall < params::MIN_RECALL_PERMILLE),
    );
    let mechanism = tally.get("task_success_mechanism");
    tally.set(
        "task_success_violation",
        i64::from(
            mechanism < tally.get("task_success_baseline")
                || mechanism < tally.get("task_success_gen_filter_only")
                || mechanism < tally.get("task_success_bool_trust"),
        ),
    );
    tally.set(
        "answerable_success_permille",
        permille(tally.get("answerable_success_mechanism"), answerable, 0),
    );
    tally.set(
        "answerable_success_baseline_permille",
        permille(tally.get("answerable_success_baseline"), answerable, 0),
    );
    tally.seed_group("hits_by_class", &class_names());
    tally.seed_group(
        "poison_attempts",
        &Class::POISON.iter().map(|c| c.name()).collect::<Vec<_>>(),
    );
    tally.seed_group("stage_skip_by_kind", &STAGE_SKIP_KINDS);
    for name in [
        "false_known_mechanism",
        "false_known_baseline_direct",
        "false_known_gen_filter_only",
        "false_known_bool_trust",
        "stale_or_revoked_promoted",
        "wrong_refusal_reason",
        "product_panics",
        "stage_skip_attempts",
        "stage_skip_accepted",
        "stage_skip_unexpected",
        "true_claims",
        "true_known_promoted",
        "hard_honest_presented",
        "hard_honest_refused",
        "abstentions_correct",
        "task_success_mechanism",
        "task_success_baseline",
        "task_success_gen_filter_only",
        "task_success_bool_trust",
        "oracle_label_mismatches",
        "corpus_selection_digest_mismatches",
        "resolver_calls",
    ] {
        let value = tally.get(name);
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
    fn passing() -> Json {
        let mut pairs: Vec<(String, Json)> = METRIC_NAMES
            .iter()
            .map(|name| (name.to_string(), Json::Int(0)))
            .collect();
        let set = |pairs: &mut Vec<(String, Json)>, name: &str, value: Json| {
            pairs.iter_mut().find(|(key, _)| key == name).unwrap().1 = value;
        };
        set(
            &mut pairs,
            "hits_by_class",
            Json::Obj(
                class_names()
                    .into_iter()
                    .map(|name| (name.to_string(), Json::Int(30)))
                    .collect(),
            ),
        );
        set(
            &mut pairs,
            "poison_attempts",
            Json::Obj(
                Class::POISON
                    .iter()
                    .map(|c| {
                        (
                            c.name().to_string(),
                            Json::Int(params::MIN_POISON_ATTEMPTS_PER_CLASS),
                        )
                    })
                    .collect(),
            ),
        );
        set(
            &mut pairs,
            "stage_skip_by_kind",
            Json::Obj(
                STAGE_SKIP_KINDS
                    .iter()
                    .map(|k| {
                        (
                            k.to_string(),
                            Json::Int(params::MIN_STAGE_SKIP_ATTEMPTS_PER_KIND),
                        )
                    })
                    .collect(),
            ),
        );
        for (name, value) in [
            ("true_claims", params::MIN_TRUE_CLAIMS),
            ("hard_honest_presented", params::MIN_HARD_HONEST_PRESENTED),
            ("unanswerable_questions", params::MIN_UNANSWERABLE_QUESTIONS),
            ("false_known_baseline_direct", 1),
            ("false_known_gen_filter_only", 1),
            ("false_known_bool_trust", 1),
            ("recall_permille", params::MIN_RECALL_PERMILLE),
            ("task_success_mechanism", 5),
            ("task_success_baseline", 5),
        ] {
            set(&mut pairs, name, Json::Int(value));
        }
        Json::Obj(pairs)
    }

    fn failing(root: &Json, rules: &[String]) -> Vec<String> {
        judge(root, &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    fn with(root: &Json, path: &str, value: Json) -> Json {
        fn set(at: &mut Json, keys: &[&str], value: Json) {
            let Json::Obj(pairs) = at else {
                panic!("object")
            };
            let slot = pairs
                .iter_mut()
                .find(|(k, _)| k == keys[0])
                .expect("member");
            if keys.len() == 1 {
                slot.1 = value;
            } else {
                set(&mut slot.1, &keys[1..], value);
            }
        }
        let mut copy = root.clone();
        set(&mut copy, &path.split('.').collect::<Vec<_>>(), value);
        copy
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        assert!(failing(&passing(), &coverage()).is_empty());
        assert!(failing(&passing(), &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_alone_when_its_counter_leaves_the_threshold() {
        for rule in coverage() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let bound: i64 = tokens[2].parse().unwrap();
            let violating = if tokens[1] == ">=" {
                bound - 1
            } else {
                bound + 1
            };
            let broken = with(&passing(), tokens[0], Json::Int(violating));
            assert_eq!(failing(&broken, &coverage()), vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn each_falsifier_trips_alone_on_a_single_violation() {
        for rule in falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let broken = if let Ok(bound) = tokens[2].parse::<i64>() {
                let violating = if tokens[1] == ">=" {
                    bound - 1
                } else {
                    bound + 1
                };
                with(&passing(), tokens[0], Json::Int(violating))
            } else {
                // Three rules share the mechanism's counter: raise the one they differ in.
                with(&passing(), tokens[2], Json::Int(6))
            };
            assert_eq!(
                failing(&broken, &falsifiers()),
                vec![rule.clone()],
                "{rule}"
            );
        }
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in [
            "false_known_mechanism",
            "recall_permille",
            "task_success_baseline",
        ] {
            for damage in [Json::Null, Json::str("0"), Json::Float(0.0)] {
                let broken = with(&passing(), name, damage);
                assert!(!failing(&broken, &falsifiers()).is_empty(), "{name}");
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
    fn the_hard_counters_are_metrics_and_the_zero_bounded_falsifiers_are_hard() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        for rule in falsifiers().iter().chain(&coverage()) {
            if rule.contains("<= 0") {
                let first = rule.split(' ').next().unwrap();
                assert!(HARD_COUNTERS.contains(&first), "{rule}");
            }
        }
    }

    #[test]
    fn filling_a_tally_prints_every_metric_and_the_recall_violation_is_a_count() {
        let mut tally = Tally::new(LIST);
        tally.set("true_claims", 100);
        tally.set("true_known_promoted", 94);
        tally.set("task_success_mechanism", 200);
        tally.set("task_success_baseline", 100);
        tally.set("task_success_gen_filter_only", 200);
        tally.set("task_success_bool_trust", 200);
        fill(&mut tally, 256, 156);
        assert_eq!(tally.get("recall_permille"), 940);
        assert_eq!(tally.get("recall_violation"), 1);
        assert_eq!(tally.get("task_success_violation"), 0);
        assert_eq!(tally.get("unanswerable_questions"), 100);
        assert_eq!(hard_failures(&tally), 1);
    }

    #[test]
    fn the_task_success_violation_counts_a_mechanism_below_any_one_arm() {
        for (baseline, gen_filter, bool_trust) in
            [(201, 200, 200), (200, 201, 200), (200, 200, 201)]
        {
            let mut tally = Tally::new(LIST);
            tally.set("task_success_mechanism", 200);
            tally.set("task_success_baseline", baseline);
            tally.set("task_success_gen_filter_only", gen_filter);
            tally.set("task_success_bool_trust", bool_trust);
            fill(&mut tally, 256, 156);
            assert_eq!(tally.get("task_success_violation"), 1);
        }
    }
}
