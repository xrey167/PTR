//! The metric list of F001 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies F001).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite becomes `>= N`, every "> 0" violation `<= 0`. The harness judges its own
//! output with them, and a test pins `criteria.toml` to exactly these lists, in order.
//!
//! Every value is a JSON integer. Fractions are integer milli-units.

use super::params::{self, BUDGETS, PROFILES};
use super::runner::{Arm, Pass};
use crate::experiments::layered_kv::emit::{Json, MetricList, Tally};

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 36] = [
    "tasks",
    "runs",
    "success",
    "verifier_calls",
    "generator_draws",
    "draws_per_success",
    "repair_gain",
    "draws_gain",
    "false_accepts",
    "unverified_returns",
    "double_verifications",
    "budget_violations",
    "collateral_edits",
    "exhausted_as_success",
    "verdict_mismatches",
    "verifier_call_mismatches",
    "expectation_violations",
    "product_panics",
    "calls_after_acceptance",
    "premature_exhaustions",
    "missed_acceptances",
    "trajectory_mismatches",
    "selection_mismatches",
    "repair_gain_shortfalls",
    "draws_gain_shortfalls",
    "altered_candidates",
    "status_pass",
    "status_fail",
    "status_unknown",
    "status_disputed",
    "initial_fail_tasks",
    "repaired_successes",
    "exhausted_runs",
    "lying_refused",
    "shallow_refused",
    "profiles",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names. Each is a
/// falsifier rule `<counter> <= 0` on its own.
pub const HARD_COUNTERS: [&str; 18] = [
    "false_accepts",
    "unverified_returns",
    "double_verifications",
    "budget_violations",
    "collateral_edits",
    "exhausted_as_success",
    "verdict_mismatches",
    "verifier_call_mismatches",
    "expectation_violations",
    "product_panics",
    "calls_after_acceptance",
    "premature_exhaustions",
    "missed_acceptances",
    "trajectory_mismatches",
    "selection_mismatches",
    "repair_gain_shortfalls",
    "draws_gain_shortfalls",
    "altered_candidates",
];

/// The budgets at which repair must be strictly better than rewriting. The budget-1 arms are
/// the same run on the same draws, so there is nothing to gain there.
pub const GAIN_BUDGETS: [u32; 4] = [2, 4, 8, 16];

/// The metrics that are plain integers (the others are objects).
const INTEGER_METRICS: [&str; 29] = [
    "tasks",
    "runs",
    "false_accepts",
    "unverified_returns",
    "double_verifications",
    "budget_violations",
    "collateral_edits",
    "exhausted_as_success",
    "verdict_mismatches",
    "verifier_call_mismatches",
    "expectation_violations",
    "product_panics",
    "calls_after_acceptance",
    "premature_exhaustions",
    "missed_acceptances",
    "trajectory_mismatches",
    "selection_mismatches",
    "repair_gain_shortfalls",
    "draws_gain_shortfalls",
    "altered_candidates",
    "status_pass",
    "status_fail",
    "status_unknown",
    "status_disputed",
    "initial_fail_tasks",
    "repaired_successes",
    "exhausted_runs",
    "lying_refused",
    "shallow_refused",
];

/// The draws per success when no run succeeded: larger than any real cost, so that an arm that
/// never succeeds is never cheaper than one that does.
pub const NO_SUCCESS_COST_MILLI: i64 = 1_000_000_000_000;

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &[],
};

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules = vec![
        format!("status_pass >= {}", params::MIN_STATUS_SEEN),
        format!("status_fail >= {}", params::MIN_STATUS_SEEN),
        format!("status_unknown >= {}", params::MIN_STATUS_SEEN),
        format!("status_disputed >= {}", params::MIN_STATUS_SEEN),
    ];
    for (name, _) in PROFILES {
        rules.push(format!(
            "profiles.{name}.tasks >= {}",
            params::MIN_TASKS_PER_PROFILE
        ));
    }
    rules.extend([
        format!("initial_fail_tasks >= {}", params::MIN_INITIAL_FAIL_TASKS),
        format!("repaired_successes >= {}", params::MIN_REPAIRED_SUCCESSES),
        format!("exhausted_runs >= {}", params::MIN_EXHAUSTED_RUNS),
        format!("lying_refused >= {}", params::MIN_LYING_REFUSED),
        format!("shallow_refused >= {}", params::MIN_SHALLOW_REFUSED),
    ]);
    rules
}

/// The falsifiers: each tripped one falsifies F001 on this seed. The first eighteen are the hard
/// counters. The other eight are the strict comparative rules: at every budget above one, repair
/// must succeed on at least one task more than rewrite-only (`repair_gain`) and spend at least
/// one milli-draw less per success (`draws_gain`). The exact analytic expectation guarantees
/// gains of thousands (see the preregistration), so a loop that silently stops repairing, for
/// which the two arms are the same run, trips them. They are CONSEQUENCES OF THE STUB'S SKILL:
/// they test the loop, not a model. `repair_gain_shortfalls` and `draws_gain_shortfalls` count
/// the gain rules that fail, so that a mutation can name them.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = HARD_COUNTERS
        .iter()
        .map(|name| format!("{name} <= {}", params::MAX_HARD_COUNTER))
        .collect();
    for budget in GAIN_BUDGETS {
        rules.push(format!("repair_gain.b{budget} >= {}", params::MIN_GAIN));
    }
    for budget in GAIN_BUDGETS {
        rules.push(format!("draws_gain.b{budget} >= {}", params::MIN_GAIN));
    }
    rules
}

/// `1000 * draws / successes`, or the no-success cost.
pub fn draws_per_success(draws: i64, successes: i64) -> i64 {
    if successes == 0 {
        NO_SUCCESS_COST_MILLI
    } else {
        1000 * draws / successes
    }
}

/// The pooled (all profiles) total of `pick` over the cells of one arm and budget.
fn pooled(pass: &Pass, arm: Arm, budget: u32, pick: fn(&super::runner::CellStats) -> i64) -> i64 {
    pass.cells
        .iter()
        .filter(|((_, a, b), _)| *a == arm && *b == budget)
        .map(|(_, stats)| pick(stats))
        .sum()
}

/// The grouped metric `pick` as `{single, rewrite: {bN}, repair: {bN}}`.
fn grouped(pass: &Pass, pick: fn(&super::runner::CellStats) -> i64) -> Json {
    let by_budget = |arm: Arm| {
        Json::Obj(
            BUDGETS
                .iter()
                .map(|budget| {
                    (
                        format!("b{budget}"),
                        Json::Int(pooled(pass, arm, *budget, pick)),
                    )
                })
                .collect(),
        )
    };
    Json::obj([
        ("single", Json::Int(pooled(pass, Arm::Single, 1, pick))),
        ("rewrite", by_budget(Arm::Rewrite)),
        ("repair", by_budget(Arm::Repair)),
    ])
}

/// The metrics of a pass into its tally.
pub fn fill(pass: &mut Pass) {
    for name in INTEGER_METRICS {
        let value = pass.tally.get(name);
        pass.tally.set(name, value);
    }
    let success = grouped(pass, |stats| stats.successes);
    let calls = grouped(pass, |stats| stats.calls);
    let draws = grouped(pass, |stats| stats.draws);
    let cost = |arm: Arm| {
        Json::Obj(
            BUDGETS
                .iter()
                .map(|budget| {
                    let d = pooled(pass, arm, *budget, |s| s.draws);
                    let s = pooled(pass, arm, *budget, |s| s.successes);
                    (format!("b{budget}"), Json::Int(draws_per_success(d, s)))
                })
                .collect(),
        )
    };
    let draws_per = Json::obj([
        ("rewrite", cost(Arm::Rewrite)),
        ("repair", cost(Arm::Repair)),
    ]);
    let profiles = Json::Obj(
        PROFILES
            .iter()
            .zip(&pass.profile_tasks)
            .map(|((name, _), tasks)| {
                (
                    (*name).to_string(),
                    Json::obj([("tasks", Json::Int(*tasks))]),
                )
            })
            .collect(),
    );
    let gains = |value: &dyn Fn(u32) -> i64| {
        Json::Obj(
            GAIN_BUDGETS
                .iter()
                .map(|budget| (format!("b{budget}"), Json::Int(value(*budget))))
                .collect(),
        )
    };
    let success_gain = |budget: u32| {
        pooled(pass, Arm::Repair, budget, |s| s.successes)
            - pooled(pass, Arm::Rewrite, budget, |s| s.successes)
    };
    let cost_of = |arm: Arm, budget: u32| {
        draws_per_success(
            pooled(pass, arm, budget, |s| s.draws),
            pooled(pass, arm, budget, |s| s.successes),
        )
    };
    let draws_gain = |budget: u32| cost_of(Arm::Rewrite, budget) - cost_of(Arm::Repair, budget);
    let repair_gain_json = gains(&success_gain);
    let draws_gain_json = gains(&draws_gain);
    let short = |value: &dyn Fn(u32) -> i64| {
        GAIN_BUDGETS
            .iter()
            .filter(|budget| value(**budget) < params::MIN_GAIN)
            .count() as i64
    };
    let repair_short = short(&success_gain);
    let draws_short = short(&draws_gain);
    pass.tally.set("repair_gain_shortfalls", repair_short);
    pass.tally.set("draws_gain_shortfalls", draws_short);
    pass.tally.set_json("repair_gain", repair_gain_json);
    pass.tally.set_json("draws_gain", draws_gain_json);
    pass.tally.set_json("success", success);
    pass.tally.set_json("verifier_calls", calls);
    pass.tally.set_json("generator_draws", draws);
    pass.tally.set_json("draws_per_success", draws_per);
    pass.tally.set_json("profiles", profiles);
}

/// The sum of the hard counters in `tally`.
pub fn hard_failures(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rule_lists_have_the_stated_shape() {
        assert_eq!(falsifiers().len(), 26);
        assert_eq!(coverage().len(), 12);
        for rule in coverage().iter().chain(&falsifiers()) {
            assert_eq!(rule.split(' ').count(), 3, "{rule}");
        }
        assert!(falsifiers()
            .iter()
            .take(HARD_COUNTERS.len())
            .all(|rule| rule.ends_with(" <= 0")));
    }

    #[test]
    fn every_hard_counter_is_a_metric_and_an_integer_metric() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
            assert!(INTEGER_METRICS.contains(&name), "{name}");
        }
        for name in INTEGER_METRICS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
    }

    #[test]
    fn a_arm_that_never_succeeds_is_never_cheaper_per_success() {
        assert_eq!(draws_per_success(10, 0), NO_SUCCESS_COST_MILLI);
        assert_eq!(draws_per_success(70, 10), 7000);
        assert!(draws_per_success(1_000_000, 1) < NO_SUCCESS_COST_MILLI);
    }
}
