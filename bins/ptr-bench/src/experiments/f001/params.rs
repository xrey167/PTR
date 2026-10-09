//! The preregistered parameters of F001, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. F001 is `planned`: nothing here is
//! frozen.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/feedback/F001-generate-verify-repair/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/feedback/F001-generate-verify-repair/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "generate-verify-repair";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from
/// the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- generate-verify-repair <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
/// The verifier-call budgets of the rewrite and repair arms.
pub const BUDGETS: [u32; 5] = [1, 2, 4, 8, 16];
pub const TASKS_PER_PROFILE: usize = 2000;
pub const CONTROL_TASKS_PER_PROFILE: usize = 300;
/// The defect slots of the task: target, operation, capability, input type, effect, payload size
/// and payload encoding.
pub const SLOTS: usize = 7;
pub const SLOT_CODES: [&str; SLOTS] = [
    "target",
    "operation",
    "capability",
    "input-type",
    "effect",
    "payload-size",
    "payload-utf8",
];
/// Per-slot probability (per mille) that the stub generator gets the slot right, by profile.
pub const PROFILES: [(&str, [u16; SLOTS]); 3] = [
    ("uniform", [900, 900, 900, 900, 900, 900, 900]),
    ("heterogeneous", [950, 900, 900, 800, 700, 500, 500]),
    ("hard", [650, 650, 600, 600, 550, 550, 500]),
];
/// Probability (per mille) that the repair stub fixes one slot the critique names.
pub const REPAIR_Q_PERMILLE: u64 = 800;
/// The verifier noise: per verifier call, the share (per mille) of calls on which one stub of
/// the fabric answers non-committally or lies. The rest of the calls are clean.
pub const MODE_UNKNOWN_PERMILLE: u64 = 40;
pub const MODE_DISPUTED_PERMILLE: u64 = 30;
pub const MODE_SHALLOW_PERMILLE: u64 = 30;
pub const MODE_LYING_PERMILLE: u64 = 30;
pub const MODE_FORGED_PERMILLE: u64 = 20;
pub const REQUIRED_LEVEL: &str = "Deterministic";
/// The stub-frequency self-tests (a harness fault, exit 2) use this many standard deviations.
pub const EXPECTATION_SIGMA: u32 = 5;
/// A measured cell whose exact tail probability under the analytic expectation is below this
/// (per 1e9) is a violation: 573 per billion is the two-sided tail of five standard deviations.
pub const EXPECTATION_ALPHA_PER_BILLION: u64 = 573;
pub const SELFTEST_SAMPLES: usize = 40_000;
pub const MIN_TASKS_PER_PROFILE: i64 = 1000;
pub const MIN_STATUS_SEEN: i64 = 1;
pub const MIN_INITIAL_FAIL_TASKS: i64 = 500;
pub const MIN_REPAIRED_SUCCESSES: i64 = 100;
pub const MIN_EXHAUSTED_RUNS: i64 = 50;
pub const MIN_LYING_REFUSED: i64 = 10;
pub const MIN_SHALLOW_REFUSED: i64 = 1;
pub const MAX_HARD_COUNTER: i64 = 0;
/// The least gain of repair over rewrite-only (successes; milli-draws per success) at each
/// budget above one.
pub const MIN_GAIN: i64 = 1;

/// The profiles as the preregistration states them: `name=p,p,...;name=...` (per mille).
pub fn profiles_text() -> String {
    PROFILES
        .iter()
        .map(|(name, p)| {
            let list: Vec<String> = p.iter().map(u16::to_string).collect();
            format!("{name}={}", list.join(","))
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Seeds for the unit tests, disjoint from the declared (confirmatory) seeds: no test may run
/// the harness on a declared seed other than the dry-run seed 17.
#[cfg(test)]
pub const PILOT_SEEDS: [u64; 5] = [1, 2, 3, 5, 8];

/// What a run covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: the same plan with fewer tasks.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// This many tasks per profile (the mutation checker's second argument); no evidence.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
            Self::Custom(_) => "custom",
        }
    }

    pub fn tasks_per_profile(self) -> usize {
        match self {
            Self::Pinned => TASKS_PER_PROFILE,
            Self::Reduced => 250,
            Self::Custom(tasks) => tasks,
        }
    }
}

/// One value of the compiled table, comparable with a value of `config.toml`.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Leaf {
    Int(i64),
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
    fn ints<T: Copy + TryInto<i64>>(values: &[T]) -> Leaf {
        Leaf::Ints(
            values
                .iter()
                .map(|value| (*value).try_into().ok().expect("fits"))
                .collect(),
        )
    }
    vec![
        ("seed_aggregation", Leaf::Text(SEED_AGGREGATION.into())),
        ("seeds", ints(&SEEDS)),
        ("budgets", ints(&BUDGETS)),
        ("tasks_per_profile", int(TASKS_PER_PROFILE)),
        ("control_tasks_per_profile", int(CONTROL_TASKS_PER_PROFILE)),
        ("slots", int(SLOTS)),
        (
            "slot_codes",
            Leaf::Texts(SLOT_CODES.iter().map(|code| code.to_string()).collect()),
        ),
        ("profiles", Leaf::Text(profiles_text())),
        ("repair_q", int(REPAIR_Q_PERMILLE)),
        ("mode_unknown_permille", int(MODE_UNKNOWN_PERMILLE)),
        ("mode_disputed_permille", int(MODE_DISPUTED_PERMILLE)),
        ("mode_shallow_permille", int(MODE_SHALLOW_PERMILLE)),
        ("mode_lying_permille", int(MODE_LYING_PERMILLE)),
        ("mode_forged_permille", int(MODE_FORGED_PERMILLE)),
        ("required_level", Leaf::Text(REQUIRED_LEVEL.into())),
        ("expectation_sigma", int(EXPECTATION_SIGMA)),
        (
            "expectation_alpha_per_billion",
            int(EXPECTATION_ALPHA_PER_BILLION),
        ),
        ("selftest_samples", int(SELFTEST_SAMPLES)),
        ("min_tasks_per_profile", int(MIN_TASKS_PER_PROFILE)),
        ("min_status_seen", int(MIN_STATUS_SEEN)),
        ("min_initial_fail_tasks", int(MIN_INITIAL_FAIL_TASKS)),
        ("min_repaired_successes", int(MIN_REPAIRED_SUCCESSES)),
        ("min_exhausted_runs", int(MIN_EXHAUSTED_RUNS)),
        ("min_lying_refused", int(MIN_LYING_REFUSED)),
        ("min_shallow_refused", int(MIN_SHALLOW_REFUSED)),
        ("max_hard_counter", int(MAX_HARD_COUNTER)),
        ("min_gain", int(MIN_GAIN)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
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
        let changed = CONFIG_TOML.replace("repair_q = 800\n", "repair_q = 801\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["repair_q: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("repair_q = 800\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["repair_q: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_manifest_seeds_are_the_constants_and_the_noise_leaves_clean_calls() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
        let noise = MODE_UNKNOWN_PERMILLE
            + MODE_DISPUTED_PERMILLE
            + MODE_SHALLOW_PERMILLE
            + MODE_LYING_PERMILLE
            + MODE_FORGED_PERMILLE;
        assert!(noise < 1000);
        assert!(BUDGETS.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(BUDGETS[0], 1, "the single-shot arm is the budget-1 run");
        for (_, p) in PROFILES {
            assert!(p.iter().all(|value| (1..1000).contains(value)));
        }
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_task_count() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.tasks_per_profile(), TASKS_PER_PROFILE);
        assert_ne!(Scale::Reduced.tasks_per_profile(), TASKS_PER_PROFILE);
        assert_eq!(Scale::Custom(5).tasks_per_profile(), 5);
        assert_eq!(Scale::Custom(5).name(), "custom");
    }
}
