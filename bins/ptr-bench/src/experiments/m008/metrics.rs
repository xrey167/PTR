//! The metric list of M008 and the two judgements of a run: the coverage prerequisites (a run that
//! misses one is vacuous) and the falsifiers (a run that trips one falsifies M008).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite becomes `>= N`, every "> 0" violation `<= 0`, every "a > b" violation
//! `a <= b`. The harness judges its own output with them, and a test pins `criteria.toml` to
//! exactly these lists, in order. Fractions are integer permille; every metric is a JSON integer
//! (a grouped metric an object of integers).

use super::params;
use super::score::Score;
use super::stats;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The arms scored against the oracle (the code under test and the three baselines).
pub const ARMS: [&str; 4] = ["fast", "recency", "hybrid_live", "hybrid_append"];

pub const METRIC_NAMES: [&str; 62] = [
    "queries",
    "answerable_queries",
    "unanswerable_queries",
    "updated_cue_queries",
    "stable_cue_queries",
    "old_stratum_queries",
    "written_facts",
    "recall_at_1_permille",
    "recall_at_k_permille",
    "recall_at_1_unthresholded_permille",
    "fast_exact_cue_recall_at_1_permille",
    "capacity_recall_at_1_permille",
    "sweep_recall_at_1_permille",
    "sweep_unanswerable_answered_permille",
    "false_recall_permille",
    "unknown_permille",
    "unanswerable_answered_permille",
    "tokens_presented",
    "recall_updated_permille",
    "recall_old_stratum_permille",
    "recall_recent_stratum_permille",
    "latency_ns_per_query",
    "false_recall_fast_permille",
    "false_recall_recency_permille",
    "recall_gain_vs_recency_permille",
    "recall_gain_vs_recency_lo_permille",
    "recall_gain_vs_hybrid_live_permille",
    "recall_gain_vs_hybrid_live_lo_permille",
    "recall_gain_vs_hybrid_append_permille",
    "recall_gain_vs_hybrid_append_lo_permille",
    "gain_violation_recency",
    "gain_violation_hybrid_live",
    "false_recall_violation",
    "token_budget_violations",
    "window_budget_violations",
    "product_panics",
    "product_errors",
    "stale_in_live_index",
    "stale_residual_violations",
    "readback_failures",
    "decode_contract_violations",
    "projection_mismatch_accepted",
    "oracle_label_mismatches",
    "readback_checks",
    "stale_residual_checks",
    "projection_probes",
    "contract_probes",
    "fast_score_abstentions",
    "fast_margin_abstentions",
    "contract_probe_margin_refusals",
    "control_oracle_cheat_recall_permille",
    "control_random_recall_permille",
    "control_stale_first_false_recall_permille",
    "control_positive_gain_vs_recency_lo_permille",
    "control_positive_gain_vs_hybrid_live_lo_permille",
    "control_positive_false_recall_headroom_permille",
    "stale_top1_permille",
    "hybrid_live_documents",
    "hybrid_append_documents",
    "recency_window_facts",
    "fast_candidates",
    "bootstrap_resamples",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// metric above; `gain_violation` and `false_recall_violation` restate the three outcome
/// falsifiers as counts for the mutation checker (the rules are the authority).
pub const HARD_COUNTERS: [&str; 13] = [
    "gain_violation_recency",
    "gain_violation_hybrid_live",
    "false_recall_violation",
    "token_budget_violations",
    "window_budget_violations",
    "product_panics",
    "product_errors",
    "stale_in_live_index",
    "stale_residual_violations",
    "readback_failures",
    "decode_contract_violations",
    "projection_mismatch_accepted",
    "oracle_label_mismatches",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[
        "recall_at_1_permille",
        "recall_at_k_permille",
        "false_recall_permille",
        "unknown_permille",
        "unanswerable_answered_permille",
        "tokens_presented",
        "recall_at_1_unthresholded_permille",
        "capacity_recall_at_1_permille",
        "sweep_recall_at_1_permille",
        "sweep_unanswerable_answered_permille",
        "recall_updated_permille",
        "recall_old_stratum_permille",
        "recall_recent_stratum_permille",
        "latency_ns_per_query",
    ],
};

/// The arms whose answers to the unanswerable queries must stay near the matched operating point.
const GATED_ARMS: [&str; 3] = ["fast", "recency", "hybrid_live"];

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules = vec![
        format!("updated_cue_queries >= {}", params::MIN_UPDATED_CUE_QUERIES),
        format!(
            "unanswerable_queries >= {}",
            params::MIN_UNANSWERABLE_QUERIES
        ),
        format!("old_stratum_queries >= {}", params::MIN_OLD_STRATUM_QUERIES),
        format!("written_facts >= {}", params::MIN_WRITTEN_FACTS),
        format!("readback_checks >= {}", params::MIN_READBACK_CHECKS),
        format!(
            "stale_residual_checks >= {}",
            params::MIN_STALE_RESIDUAL_CHECKS
        ),
        format!("projection_probes >= {}", params::MIN_PROJECTION_PROBES),
        format!("contract_probes >= {}", params::MIN_CONTRACT_PROBES),
        format!(
            "fast_score_abstentions >= {}",
            params::MIN_FAST_SCORE_ABSTENTIONS
        ),
        format!(
            "contract_probe_margin_refusals >= {}",
            params::MIN_CONTRACT_PROBE_MARGIN_REFUSALS
        ),
    ];
    for arm in GATED_ARMS {
        rules.push(format!(
            "unanswerable_answered_permille.{arm} <= {}",
            params::MAX_UNANSWERABLE_ANSWERED_PERMILLE
        ));
    }
    rules.push(format!(
        "control_oracle_cheat_recall_permille >= {}",
        params::MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE
    ));
    rules.push(format!(
        "control_random_recall_permille <= {}",
        params::MAX_CONTROL_RANDOM_RECALL_PERMILLE
    ));
    rules.push(format!(
        "control_stale_first_false_recall_permille >= {}",
        params::MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE
    ));
    for arm in ["recency", "hybrid_live"] {
        rules.push(format!(
            "control_positive_gain_vs_{arm}_lo_permille >= {}",
            params::MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE
        ));
    }
    rules.push(format!(
        "control_positive_false_recall_headroom_permille >= {}",
        params::MIN_CONTROL_POSITIVE_FALSE_RECALL_HEADROOM_PERMILLE
    ));
    rules.push(format!(
        "oracle_label_mismatches <= {}",
        params::MAX_ORACLE_LABEL_MISMATCHES
    ));
    // Harness and baseline integrity, judged INVALID by the aggregator, not falsified: a stale
    // version in the live index is the driver's or the harness's defect, and a recency window over
    // the budget is the harness's.
    rules.push(format!(
        "stale_in_live_index <= {}",
        params::MAX_STALE_IN_LIVE_INDEX
    ));
    rules.push(format!(
        "window_budget_violations <= {}",
        params::MAX_WINDOW_BUDGET_VIOLATIONS
    ));
    rules
}

/// The falsifiers: each tripped one falsifies M008 on this seed.
pub fn falsifiers() -> Vec<String> {
    vec![
        "false_recall_fast_permille <= false_recall_recency_permille".to_string(),
        format!(
            "recall_gain_vs_recency_lo_permille >= {}",
            params::MIN_RECALL_GAIN_LO_PERMILLE
        ),
        format!(
            "recall_gain_vs_hybrid_live_lo_permille >= {}",
            params::MIN_RECALL_GAIN_LO_PERMILLE
        ),
        format!(
            "token_budget_violations <= {}",
            params::MAX_TOKEN_BUDGET_VIOLATIONS
        ),
        format!("product_panics <= {}", params::MAX_PRODUCT_PANICS),
        format!("product_errors <= {}", params::MAX_PRODUCT_ERRORS),
        format!(
            "stale_residual_violations <= {}",
            params::MAX_STALE_RESIDUAL_VIOLATIONS
        ),
        format!("readback_failures <= {}", params::MAX_READBACK_FAILURES),
        format!(
            "decode_contract_violations <= {}",
            params::MAX_DECODE_CONTRACT_VIOLATIONS
        ),
        format!(
            "projection_mismatch_accepted <= {}",
            params::MAX_PROJECTION_MISMATCH_ACCEPTED
        ),
    ]
}

/// `1000 * numerator / denominator` rounded down, or 0 when there is nothing to divide.
pub fn permille(numerator: i64, denominator: i64) -> i64 {
    if denominator == 0 {
        0
    } else {
        1000 * numerator / denominator
    }
}

/// Everything scored in one run.
pub struct Scores {
    pub fast: Score,
    pub recency: Score,
    pub hybrid_live: Score,
    pub hybrid_append: Score,
    pub oracle_cheat: Score,
    pub random: Score,
    pub stale_first: Score,
}

impl Scores {
    fn arm(&self, name: &str) -> &Score {
        match name {
            "fast" => &self.fast,
            "recency" => &self.recency,
            "hybrid_live" => &self.hybrid_live,
            _ => &self.hybrid_append,
        }
    }
}

/// The inputs of the outcome rules for `candidate` standing in the memory's place: its false-recall
/// rate over all queries, the recency buffer's, and the paired gain of its recall@1 over each
/// baseline as `(point, lower bound)` in permille. The memory and the positive control both go
/// through this one function.
pub struct Outcome {
    pub false_recall_permille: i64,
    pub recency_false_recall_permille: i64,
    pub gain_recency: (i64, i64),
    pub gain_hybrid_live: (i64, i64),
    pub gain_hybrid_append: (i64, i64),
}

pub fn outcome(candidate: &Score, scores: &Scores, seed: u64) -> Outcome {
    let gain = |other: &Score, label: u64| {
        let diffs: Vec<i64> = candidate
            .hits
            .iter()
            .zip(&other.hits)
            .map(|(a, b)| a - b)
            .collect();
        stats::paired_gain(&diffs, seed, label)
    };
    Outcome {
        false_recall_permille: permille(candidate.false_recall(), candidate.queries),
        recency_false_recall_permille: permille(
            scores.recency.false_recall(),
            scores.recency.queries,
        ),
        gain_recency: gain(&scores.recency, 0),
        gain_hybrid_live: gain(&scores.hybrid_live, 1),
        gain_hybrid_append: gain(&scores.hybrid_append, 2),
    }
}

/// Fill the derived metrics of `tally` from the scores of the arms and the controls.
pub fn fill(tally: &mut Tally, scores: &Scores, seed: u64) {
    let fast = &scores.fast;
    tally.set("queries", fast.queries);
    tally.set("answerable_queries", fast.answerable);
    tally.set("unanswerable_queries", fast.unanswerable);
    tally.set("updated_cue_queries", fast.updated);
    tally.set("stable_cue_queries", fast.answerable - fast.updated);
    tally.set("old_stratum_queries", fast.old);
    for arm in ARMS {
        let score = scores.arm(arm);
        let set = |tally: &mut Tally, metric: &str, value: i64| {
            tally.set(&format!("{metric}.{arm}"), value);
        };
        set(
            tally,
            "recall_at_1_permille",
            permille(score.correct, score.answerable),
        );
        set(
            tally,
            "recall_at_k_permille",
            permille(score.correct_in_list, score.answerable),
        );
        set(
            tally,
            "false_recall_permille",
            permille(score.false_recall(), score.queries),
        );
        set(
            tally,
            "unknown_permille",
            permille(score.unknown, score.queries),
        );
        set(
            tally,
            "unanswerable_answered_permille",
            permille(score.answered_unanswerable, score.unanswerable),
        );
        set(tally, "tokens_presented", score.tokens);
        set(
            tally,
            "recall_updated_permille",
            permille(score.updated_correct, score.updated),
        );
        set(
            tally,
            "recall_old_stratum_permille",
            permille(score.old_correct, score.old),
        );
        set(
            tally,
            "recall_recent_stratum_permille",
            permille(score.recent_correct, score.recent),
        );
    }
    tally.set(
        "false_recall_fast_permille",
        permille(fast.false_recall(), fast.queries),
    );
    tally.set(
        "false_recall_recency_permille",
        permille(scores.recency.false_recall(), scores.recency.queries),
    );
    tally.set(
        "stale_top1_permille",
        permille(fast.stale_top1, fast.queries),
    );
    let result = outcome(fast, scores, seed);
    for (name, (point, lower)) in [
        ("recency", result.gain_recency),
        ("hybrid_live", result.gain_hybrid_live),
        ("hybrid_append", result.gain_hybrid_append),
    ] {
        tally.set(&format!("recall_gain_vs_{name}_permille"), point);
        tally.set(&format!("recall_gain_vs_{name}_lo_permille"), lower);
    }
    tally.set(
        "gain_violation_recency",
        i64::from(result.gain_recency.1 < params::MIN_RECALL_GAIN_LO_PERMILLE),
    );
    tally.set(
        "gain_violation_hybrid_live",
        i64::from(result.gain_hybrid_live.1 < params::MIN_RECALL_GAIN_LO_PERMILLE),
    );
    tally.set(
        "false_recall_violation",
        i64::from(result.false_recall_permille > result.recency_false_recall_permille),
    );
    // The positive control: the oracle-cheat answers stand in the memory's place and go through
    // the same gain, bootstrap and false-recall path. Every outcome rule must pass for it.
    let positive = outcome(&scores.oracle_cheat, scores, seed);
    tally.set(
        "control_positive_gain_vs_recency_lo_permille",
        positive.gain_recency.1,
    );
    tally.set(
        "control_positive_gain_vs_hybrid_live_lo_permille",
        positive.gain_hybrid_live.1,
    );
    tally.set(
        "control_positive_false_recall_headroom_permille",
        positive.recency_false_recall_permille - positive.false_recall_permille,
    );
    tally.set(
        "token_budget_violations",
        ARMS.iter()
            .map(|arm| scores.arm(arm).budget_violations)
            .sum(),
    );
    tally.set(
        "control_oracle_cheat_recall_permille",
        permille(scores.oracle_cheat.correct, scores.oracle_cheat.answerable),
    );
    tally.set(
        "control_random_recall_permille",
        permille(scores.random.correct, scores.random.answerable),
    );
    tally.set(
        "control_stale_first_false_recall_permille",
        permille(
            scores.stale_first.false_recall(),
            scores.stale_first.queries,
        ),
    );
    tally.set("bootstrap_resamples", params::BOOTSTRAP_RESAMPLES as i64);
    for name in [
        "product_panics",
        "product_errors",
        "stale_in_live_index",
        "stale_residual_violations",
        "readback_failures",
        "decode_contract_violations",
        "projection_mismatch_accepted",
        "oracle_label_mismatches",
        "window_budget_violations",
        "readback_checks",
        "stale_residual_checks",
        "projection_probes",
        "contract_probes",
        "fast_score_abstentions",
        "fast_margin_abstentions",
        "contract_probe_margin_refusals",
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
            "unanswerable_answered_permille",
            Json::Obj(
                ARMS.iter()
                    .map(|arm| {
                        (
                            arm.to_string(),
                            Json::Int(params::MAX_UNANSWERABLE_ANSWERED_PERMILLE),
                        )
                    })
                    .collect(),
            ),
        );
        for (name, value) in [
            ("updated_cue_queries", params::MIN_UPDATED_CUE_QUERIES),
            ("unanswerable_queries", params::MIN_UNANSWERABLE_QUERIES),
            ("old_stratum_queries", params::MIN_OLD_STRATUM_QUERIES),
            ("written_facts", params::MIN_WRITTEN_FACTS),
            ("readback_checks", params::MIN_READBACK_CHECKS),
            ("stale_residual_checks", params::MIN_STALE_RESIDUAL_CHECKS),
            ("projection_probes", params::MIN_PROJECTION_PROBES),
            ("contract_probes", params::MIN_CONTRACT_PROBES),
            ("fast_score_abstentions", params::MIN_FAST_SCORE_ABSTENTIONS),
            (
                "contract_probe_margin_refusals",
                params::MIN_CONTRACT_PROBE_MARGIN_REFUSALS,
            ),
            (
                "control_positive_gain_vs_recency_lo_permille",
                params::MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE,
            ),
            (
                "control_positive_gain_vs_hybrid_live_lo_permille",
                params::MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE,
            ),
            (
                "control_positive_false_recall_headroom_permille",
                params::MIN_CONTROL_POSITIVE_FALSE_RECALL_HEADROOM_PERMILLE,
            ),
            (
                "control_oracle_cheat_recall_permille",
                params::MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE,
            ),
            (
                "control_random_recall_permille",
                params::MAX_CONTROL_RANDOM_RECALL_PERMILLE,
            ),
            (
                "control_stale_first_false_recall_permille",
                params::MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE,
            ),
            (
                "recall_gain_vs_recency_lo_permille",
                params::MIN_RECALL_GAIN_LO_PERMILLE,
            ),
            (
                "recall_gain_vs_hybrid_live_lo_permille",
                params::MIN_RECALL_GAIN_LO_PERMILLE,
            ),
            ("false_recall_fast_permille", 100),
            ("false_recall_recency_permille", 100),
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
                // The false-recall rule compares two counters: raise the left one.
                with(&passing(), tokens[0], Json::Int(101))
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
            "product_panics",
            "recall_gain_vs_recency_lo_permille",
            "false_recall_recency_permille",
        ] {
            for damage in [Json::Null, Json::str("0"), Json::Float(0.0)] {
                let broken = with(&passing(), name, damage);
                assert!(!failing(&broken, &falsifiers()).is_empty(), "{name}");
            }
        }
    }

    #[test]
    fn the_hard_counters_are_metrics_and_the_zero_bounded_rules_are_hard() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        for rule in falsifiers().iter().chain(&coverage()) {
            if rule.ends_with("<= 0") {
                let first = rule.split(' ').next().unwrap();
                assert!(HARD_COUNTERS.contains(&first), "{rule}");
            }
        }
        // The three outcome falsifiers each have a restated count.
        assert!(HARD_COUNTERS.contains(&"gain_violation_recency"));
        assert!(HARD_COUNTERS.contains(&"gain_violation_hybrid_live"));
        assert!(HARD_COUNTERS.contains(&"false_recall_violation"));
    }

    #[test]
    fn fractions_are_integer_permille_and_empty_denominators_are_zero() {
        assert_eq!(permille(1, 3), 333);
        assert_eq!(permille(3, 3), 1000);
        assert_eq!(permille(0, 0), 0);
    }

    #[test]
    fn the_outcome_path_passes_for_a_perfect_arm_and_fails_for_a_blind_one() {
        use super::super::arms::controls;
        use super::super::oracle::Truth;
        use super::super::params::Shape;
        use super::super::pool::tests::shared;
        use super::super::score::score;
        use super::super::session::Session;
        let pool = shared();
        let session = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        let truth = Truth::fold(&session.events);
        let c = controls(&session, &truth, 1001);
        let scored = |answers: &Vec<Vec<usize>>| score(pool, &session, &truth, answers);
        let build = |fast: &Vec<Vec<usize>>, base: &Vec<Vec<usize>>| Scores {
            fast: scored(fast),
            recency: scored(base),
            hybrid_live: scored(base),
            hybrid_append: scored(base),
            oracle_cheat: scored(&c.oracle_cheat),
            random: scored(&c.random),
            stale_first: scored(&c.stale_first),
        };
        // The memory is perfect: every outcome rule passes through the real scoring and bootstrap.
        let mut tally = Tally::new(LIST);
        fill(&mut tally, &build(&c.oracle_cheat, &c.stale_first), 1001);
        for counter in [
            "gain_violation_recency",
            "gain_violation_hybrid_live",
            "false_recall_violation",
        ] {
            assert_eq!(tally.get(counter), 0, "{counter}");
        }
        assert!(tally.get("recall_gain_vs_recency_lo_permille") >= 1);
        assert!(tally.get("recall_gain_vs_hybrid_live_lo_permille") >= 1);
        // The memory answers at random: both gain rules and the false-recall rule fire.
        let mut blind = Tally::new(LIST);
        fill(&mut blind, &build(&c.random, &c.oracle_cheat), 1001);
        assert_eq!(blind.get("gain_violation_recency"), 1);
        assert_eq!(blind.get("gain_violation_hybrid_live"), 1);
        assert_eq!(blind.get("false_recall_violation"), 1);
        // The positive control does not depend on the memory: it is the oracle path.
        assert!(blind.get("control_positive_false_recall_headroom_permille") >= 0);
    }
}
