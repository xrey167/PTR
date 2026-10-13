//! The preregistered parameters of S002, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`, which
//! is read at compile time; the drift test fails when a constant differs from the file, and when
//! the file holds a key no constant mirrors. S002 is frozen at its preregistration and has
//! completed; the file is frozen with it.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

use ptr_types::Effect;

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/semdb/S002-snapshot-cancellation/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/semdb/S002-snapshot-cancellation/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "snapshot-cancellation";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from
/// the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- snapshot-cancellation <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
pub const CASES_PER_SEED: usize = 1500;
pub const ROUNDS_MIN: u64 = 2;
pub const ROUNDS_MAX: u64 = 6;
pub const STEP_TICKS: u64 = 10;
pub const CLASS_NEVER_PERMILLE: u64 = 150;
pub const CLASS_BUMP_EARLY_PERMILLE: u64 = 300;
pub const CLASS_BUMP_LATE_PERMILLE: u64 = 200;
pub const CLASS_NOOP_PERMILLE: u64 = 100;
pub const CLASS_LIFECYCLE_PERMILLE: u64 = 250;
pub const LIFECYCLE_REVOKE_PERMILLE: u64 = 500;
pub const HANG_SECONDS: u64 = 60;
pub const MIN_RUNS: i64 = 1000;
pub const MIN_STALE_RUNS_BY_SCHEDULE: i64 = 100;
pub const MIN_STALE_RUNS_OBSERVED: i64 = 100;
pub const MIN_BOUNDARY_REFUSALS_A0: i64 = 1;
pub const MIN_BOUNDARY_REFUSALS_A1: i64 = 1;
pub const MIN_CANCELLED_RUNS: i64 = 100;
pub const MIN_EFFECT_KINDS_COVERED: i64 = 4;
pub const MIN_FRESH_RUNS_COMPLETED: i64 = 100;
pub const MIN_GENERATION_STALE_CASES: i64 = 50;
pub const MIN_INJECTION_POSITIONS_COVERED: i64 = 5;
pub const MIN_CONTROL_TRIPS: i64 = 5;
pub const MIN_WASTED_UNITS_A0: i64 = 1;
pub const MIN_NOOP_WRITES_OBSERVED: i64 = 1;
pub const MIN_RETRIES_REPLAYED: i64 = 1;
pub const MIN_RESTARTED_RUNS: i64 = 1;
pub const MAX_STALE_ACTIONS_EXECUTED: i64 = 0;
pub const MAX_STALE_EFFECT_ATTEMPTS_COMMITTED: i64 = 0;
pub const MAX_STALE_CHECKPOINT_ADMISSIONS: i64 = 0;
pub const MAX_FRESH_ACTIONS_REFUSED: i64 = 0;
pub const MAX_FALSE_CANCELLATIONS: i64 = 0;
pub const MAX_MISSED_CANCELLATIONS: i64 = 0;
pub const MAX_FRESH_RUNS_STARVED: i64 = 0;
pub const MAX_CANCEL_CONTRACT_VIOLATIONS: i64 = 0;
pub const MAX_BOUNDARY_ARM_DIVERGENCES: i64 = 0;
pub const MAX_DOUBLE_EXECUTIONS: i64 = 0;
pub const MAX_EXECUTOR_VS_LEDGER_MISMATCHES: i64 = 0;
pub const MAX_MIRROR_DRIFT: i64 = 0;
pub const MAX_CANCEL_LATENCY_ROUNDS_MAX: i64 = 1;
pub const MAX_WASTED_UNITS_A1: i64 = 0;
pub const MAX_PRODUCT_PANICS: i64 = 0;
pub const MAX_PRODUCT_HANGS: i64 = 0;

/// The four effect kinds a run can end in, in the order of `config.toml`.
pub const EFFECTS: [Effect; 4] = [
    Effect::Read,
    Effect::Mutation,
    Effect::External,
    Effect::Irreversible,
];

pub fn effect_name(effect: Effect) -> &'static str {
    match effect {
        Effect::Pure => "Pure",
        Effect::Read => "Read",
        Effect::Mutation => "Mutation",
        Effect::External => "External",
        Effect::Irreversible => "Irreversible",
    }
}

/// Where a writer acts relative to the run it interrupts.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Position {
    /// After `begin_resumable_run`, before the first step.
    BeforeRound1,
    /// Between two rounds.
    MidRun,
    /// After the last round, before the step that hands out the action.
    JustBeforeAction,
    /// After the run finished, before `authorize_action`.
    BetweenActionAndAuthorize,
    /// After `prepare_execution`, before `execute_prepared`.
    BetweenPrepareAndExecute,
}

impl Position {
    pub const ALL: [Position; 5] = [
        Self::BeforeRound1,
        Self::MidRun,
        Self::JustBeforeAction,
        Self::BetweenActionAndAuthorize,
        Self::BetweenPrepareAndExecute,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::BeforeRound1 => "before-round-1",
            Self::MidRun => "mid-run",
            Self::JustBeforeAction => "just-before-action",
            Self::BetweenActionAndAuthorize => "between-action-and-authorize",
            Self::BetweenPrepareAndExecute => "between-prepare-and-execute",
        }
    }

    /// Whether the run is still stepping when a writer acts here.
    pub fn is_early(self) -> bool {
        matches!(
            self,
            Self::BeforeRound1 | Self::MidRun | Self::JustBeforeAction
        )
    }
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: a few hundred cases.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// The mutation checker's second argument: this many cases; no evidence.
    Custom(usize),
}

pub const REDUCED_CASES: usize = 240;

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
            Self::Custom(_) => "custom",
        }
    }

    pub fn cases(self) -> usize {
        match self {
            Self::Pinned => CASES_PER_SEED,
            Self::Reduced => REDUCED_CASES,
            Self::Custom(cases) => cases,
        }
    }
}

/// One value of the compiled table, comparable with a value of `config.toml`.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Leaf {
    Int(i64),
    Bool(bool),
    Text(String),
    Ints(Vec<i64>),
    Texts(Vec<String>),
}

/// The compiled table, one entry per key of `[pinned_thresholds]`.
#[cfg(test)]
pub fn compiled() -> Vec<(&'static str, Leaf)> {
    fn int(value: impl TryInto<i64>) -> Leaf {
        Leaf::Int(value.try_into().ok().expect("fits"))
    }
    fn texts(items: &[&str]) -> Leaf {
        Leaf::Texts(items.iter().map(|item| item.to_string()).collect())
    }
    vec![
        ("seed_aggregation", Leaf::Text(SEED_AGGREGATION.into())),
        (
            "seeds",
            Leaf::Ints(SEEDS.iter().map(|seed| *seed as i64).collect()),
        ),
        ("cases_per_seed", int(CASES_PER_SEED)),
        (
            "effects",
            Leaf::Texts(EFFECTS.iter().map(|e| effect_name(*e).to_string()).collect()),
        ),
        ("rounds_min", int(ROUNDS_MIN)),
        ("rounds_max", int(ROUNDS_MAX)),
        ("step_ticks", int(STEP_TICKS)),
        ("class_never_permille", int(CLASS_NEVER_PERMILLE)),
        ("class_bump_early_permille", int(CLASS_BUMP_EARLY_PERMILLE)),
        ("class_bump_late_permille", int(CLASS_BUMP_LATE_PERMILLE)),
        ("class_noop_permille", int(CLASS_NOOP_PERMILLE)),
        ("class_lifecycle_permille", int(CLASS_LIFECYCLE_PERMILLE)),
        ("lifecycle_revoke_permille", int(LIFECYCLE_REVOKE_PERMILLE)),
        (
            "positions",
            Leaf::Texts(Position::ALL.iter().map(|p| p.name().to_string()).collect()),
        ),
        (
            "arms",
            texts(&["A0-cancellation-off", "A1-cancellation-on"]),
        ),
        ("controls", texts(&["bypass", "ledger_scan"])),
        ("hang_seconds", int(HANG_SECONDS)),
        ("min_runs", int(MIN_RUNS)),
        (
            "min_stale_runs_by_schedule",
            int(MIN_STALE_RUNS_BY_SCHEDULE),
        ),
        ("min_stale_runs_observed", int(MIN_STALE_RUNS_OBSERVED)),
        ("min_boundary_refusals_a0", int(MIN_BOUNDARY_REFUSALS_A0)),
        ("min_boundary_refusals_a1", int(MIN_BOUNDARY_REFUSALS_A1)),
        ("min_cancelled_runs", int(MIN_CANCELLED_RUNS)),
        ("min_effect_kinds_covered", int(MIN_EFFECT_KINDS_COVERED)),
        ("min_fresh_runs_completed", int(MIN_FRESH_RUNS_COMPLETED)),
        ("min_generation_stale_cases", int(MIN_GENERATION_STALE_CASES)),
        (
            "min_injection_positions_covered",
            int(MIN_INJECTION_POSITIONS_COVERED),
        ),
        ("min_control_trips", int(MIN_CONTROL_TRIPS)),
        ("min_wasted_units_a0", int(MIN_WASTED_UNITS_A0)),
        ("min_noop_writes_observed", int(MIN_NOOP_WRITES_OBSERVED)),
        ("min_retries_replayed", int(MIN_RETRIES_REPLAYED)),
        ("min_restarted_runs", int(MIN_RESTARTED_RUNS)),
        (
            "max_stale_actions_executed",
            int(MAX_STALE_ACTIONS_EXECUTED),
        ),
        (
            "max_stale_effect_attempts_committed",
            int(MAX_STALE_EFFECT_ATTEMPTS_COMMITTED),
        ),
        (
            "max_stale_checkpoint_admissions",
            int(MAX_STALE_CHECKPOINT_ADMISSIONS),
        ),
        ("max_fresh_actions_refused", int(MAX_FRESH_ACTIONS_REFUSED)),
        ("max_false_cancellations", int(MAX_FALSE_CANCELLATIONS)),
        ("max_missed_cancellations", int(MAX_MISSED_CANCELLATIONS)),
        ("max_fresh_runs_starved", int(MAX_FRESH_RUNS_STARVED)),
        (
            "max_cancel_contract_violations",
            int(MAX_CANCEL_CONTRACT_VIOLATIONS),
        ),
        (
            "max_boundary_arm_divergences",
            int(MAX_BOUNDARY_ARM_DIVERGENCES),
        ),
        ("max_double_executions", int(MAX_DOUBLE_EXECUTIONS)),
        (
            "max_executor_vs_ledger_mismatches",
            int(MAX_EXECUTOR_VS_LEDGER_MISMATCHES),
        ),
        ("max_mirror_drift", int(MAX_MIRROR_DRIFT)),
        (
            "max_cancel_latency_rounds_max",
            int(MAX_CANCEL_LATENCY_ROUNDS_MAX),
        ),
        ("max_wasted_units_a1", int(MAX_WASTED_UNITS_A1)),
        ("max_product_panics", int(MAX_PRODUCT_PANICS)),
        ("max_product_hangs", int(MAX_PRODUCT_HANGS)),
        ("wall_time_gated", Leaf::Bool(false)),
        ("total_units_gated", Leaf::Bool(false)),
        (
            "owner_decision_2026_10_09_cancellation",
            Leaf::Text("refactor run_resumable_inner into a steppable run (existing entry points stay wrappers, behaviour unchanged with cancellation off); coarse revision-granular cooperative cancellation before the effect boundary only; the boundary remains the safety barrier; gated claim: cancellation never weakens the boundary and cuts wasted work; coverage also requires the baseline to show at least one stale attempt reaching the boundary; invariant review before freeze; do last".into()),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
            toml::Value::Boolean(flag) => Leaf::Bool(*flag),
            toml::Value::String(text) => Leaf::Text(text.clone()),
            toml::Value::Array(items) if items.iter().all(toml::Value::is_integer) => Leaf::Ints(
                items
                    .iter()
                    .map(|item| item.as_integer().expect("integer"))
                    .collect(),
            ),
            toml::Value::Array(items) => Leaf::Texts(
                items
                    .iter()
                    .map(|item| item.as_str().expect("string").to_string())
                    .collect(),
            ),
            other => panic!("unexpected value {other:?}"),
        }
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each.
    fn drift(config: &str) -> Vec<String> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let file: BTreeMap<String, Leaf> = parsed["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .iter()
            .map(|(key, value)| (key.clone(), leaf_of(value)))
            .collect();
        let compiled: BTreeMap<String, Leaf> = compiled()
            .into_iter()
            .map(|(key, leaf)| (key.to_string(), leaf))
            .collect();
        let mut problems = Vec::new();
        for (key, value) in &file {
            match compiled.get(key) {
                None => problems.push(format!("{key}: no constant")),
                Some(constant) if constant != value => problems.push(format!("{key}: drifted")),
                Some(_) => {}
            }
        }
        for key in compiled.keys().filter(|key| !file.contains_key(*key)) {
            problems.push(format!("{key}: no key in the file"));
        }
        problems
    }

    #[test]
    fn the_compiled_constants_are_the_pinned_thresholds_of_config_toml() {
        assert_eq!(drift(CONFIG_TOML), Vec::<String>::new());
    }

    #[test]
    fn a_changed_added_or_removed_key_of_config_toml_is_drift() {
        let changed = CONFIG_TOML.replace("rounds_max = 6\n", "rounds_max = 7\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["rounds_max: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("rounds_max = 6\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["rounds_max: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_class_mix_adds_up_and_the_manifest_seeds_are_the_constants() {
        assert_eq!(
            CLASS_NEVER_PERMILLE
                + CLASS_BUMP_EARLY_PERMILLE
                + CLASS_BUMP_LATE_PERMILLE
                + CLASS_NOOP_PERMILLE
                + CLASS_LIFECYCLE_PERMILLE,
            1000
        );
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_size() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.cases(), CASES_PER_SEED);
        assert_ne!(Scale::Reduced.cases(), CASES_PER_SEED);
        assert_eq!(Scale::Custom(5).cases(), 5);
        assert_eq!(Scale::Custom(5).name(), "custom");
    }

    #[test]
    fn early_positions_are_the_three_before_the_run_finishes() {
        let early: Vec<&str> = Position::ALL
            .iter()
            .filter(|p| p.is_early())
            .map(|p| p.name())
            .collect();
        assert_eq!(early, ["before-round-1", "mid-run", "just-before-action"]);
    }
}
