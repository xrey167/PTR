//! The preregistered parameters of F002, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. F002 is `planned`: nothing here is
//! frozen.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/feedback/F002-weak-supervision/config.toml");

/// The experiment's manifest, read at compile time.
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/feedback/F002-weak-supervision/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "weak-supervision";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from
/// the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str = "cargo run --release --locked -p ptr-bench -- weak-supervision <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
/// Seeds the pilot (the decision of the ECE gate form) may use; never confirmatory.
pub const PILOT_SEEDS: [u64; 3] = [1001, 1002, 1003];

/// Four classes, five probabilistic functions, four verifier functions.
pub const CLASSES: usize = 4;
pub const PROBABILISTIC: usize = 5;
pub const VERIFIERS: usize = 4;
pub const ITEMS: usize = 2000;
pub const GOLD_ITEMS: usize = 600;
pub const CLASS_PRIOR_PERMILLE: [u64; CLASSES] = [400, 300, 200, 100];
/// Accuracy of each probabilistic function (permille) in the heterogeneous worlds (W1, W2).
pub const ACCURACY_HETEROGENEOUS_PERMILLE: [u64; PROBABILISTIC] = [850, 750, 650, 600, 550];
/// ... and in the homogeneous world (W3).
pub const ACCURACY_HOMOGENEOUS_PERMILLE: [u64; PROBABILISTIC] = [700, 700, 700, 700, 700];
/// Relative weight of each wrong class given the true class, row-major `[true][voted]`; the
/// diagonal is 0. A function's wrong votes are drawn from the row, so confusions are not
/// uniform.
pub const WRONG_CLASS_WEIGHTS: [u64; CLASSES * CLASSES] =
    [0, 5, 3, 2, 5, 0, 2, 3, 3, 2, 0, 5, 2, 3, 5, 0];
/// Number of probabilistic functions that vote on an item, and the weight (permille) of each.
pub const VOTES_PER_ITEM: [usize; 3] = [2, 3, 5];
pub const VOTES_PER_ITEM_WEIGHT_PERMILLE: [u64; 3] = [250, 500, 250];
/// The deliberately correlated pair of functions (indices), and the probability (permille) that
/// the second repeats the first's vote when both vote on an item.
pub const CORRELATED_PAIR: [usize; 2] = [3, 4];
pub const CORRELATED_COPY_PERMILLE: u64 = 600;
/// Probability (permille) that each verifier votes on an item, in W1 and W3 (W3 has none) ...
pub const VERIFIER_COVERAGE_SOUND_PERMILLE: [u64; VERIFIERS] = [500, 400, 300, 200];
pub const VERIFIER_COVERAGE_NONE_PERMILLE: [u64; VERIFIERS] = [0, 0, 0, 0];
/// ... and in the contaminated world W2, where a voting verifier vetoes the true class with
/// probability `CONTAMINATION_PERMILLE`.
pub const VERIFIER_COVERAGE_CONTAMINATED_PERMILLE: [u64; VERIFIERS] = [800, 700, 600, 500];
pub const CONTAMINATION_PERMILLE: u64 = 80;
pub const MIN_PROBABILITY_PERMILLE: u64 = 700;
pub const CALIBRATION_BINS: usize = 10;
pub const DS_MAX_ITERATIONS: usize = 1000;
pub const DS_TOLERANCE_PPM: u64 = 1;
pub const DS_SMOOTHING_MILLI: u64 = 1000;
/// Tolerance (ppm) between the product's fit and the independent EM, and the width (nano) for
/// the metric comparisons.
pub const MODEL_ORACLE_TOLERANCE_PPM: u64 = 1;
pub const METRIC_TOLERANCE_NANO: u64 = 1;
pub const WILSON_Z_MILLI: u64 = 1960;
pub const ACQUISITION_BUDGET: usize = 100;
/// The sharpening exponent (milli) of the calibrated vote share, fitted on the pilot seeds.
pub const MVC_BETA_MILLI: u64 = 1700;
pub const BOOTSTRAP_RESAMPLES: usize = 2000;
pub const BOOTSTRAP_SEED: u64 = 20_261_009;
pub const BOOTSTRAP_INTERVAL_PERMILLE: u64 = 950;
/// Which gain rules gate on the lower bound of the paired bootstrap interval instead of the
/// point gain (decided at the pilot; see PREREGISTRATION.md section 7).
pub const BRIER_GATE_ON_LOWER_BOUND: bool = true;
pub const ECE_GATE_ON_LOWER_BOUND: bool = true;

pub const MIN_WORLDS_RUN: i64 = 3;
pub const MIN_DETERMINED_SEEN: i64 = 1;
pub const MIN_ESTIMATED_SEEN: i64 = 1;
pub const MIN_UNKNOWN_SEEN: i64 = 1;
pub const MIN_DISPUTED_SEEN: i64 = 1;
pub const MIN_VETOED_ITEMS: i64 = 200;
pub const MIN_GOLD_ITEMS: i64 = 500;
pub const MIN_FEWER_THAN_THREE_WARNINGS: i64 = 1;
pub const MIN_BUDGET_CELLS: i64 = 3;
pub const MIN_CORRELATED_PAIR_ITEMS: i64 = 50;
pub const MIN_CONTAMINATED_VETOES: i64 = 1;
pub const MIN_ACTIVE_SET_PROBES: i64 = 1;
pub const MIN_GOLD_GUARD_PROBES: i64 = 4;
pub const MIN_ACQUISITION_ITEMS: i64 = 1;
pub const MAX_VIOLATIONS: i64 = 0;
pub const MIN_GAIN_PPM: i64 = 1;

/// The three worlds, in the order the harness runs them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum World {
    /// Sound verifiers, heterogeneous accuracies: a gain is expected by construction.
    Sound,
    /// Contaminated verifiers: used for the hard veto check and the Disputed rate.
    Contaminated,
    /// Homogeneous accuracies and no verifier: no gain is expected (a control).
    Homogeneous,
}

impl World {
    pub const ALL: [World; 3] = [World::Sound, World::Contaminated, World::Homogeneous];

    pub fn key(self) -> &'static str {
        match self {
            Self::Sound => "w1",
            Self::Contaminated => "w2",
            Self::Homogeneous => "w3",
        }
    }

    pub fn accuracy_permille(self) -> [u64; PROBABILISTIC] {
        match self {
            Self::Sound | Self::Contaminated => ACCURACY_HETEROGENEOUS_PERMILLE,
            Self::Homogeneous => ACCURACY_HOMOGENEOUS_PERMILLE,
        }
    }

    pub fn verifier_coverage_permille(self) -> [u64; VERIFIERS] {
        match self {
            Self::Sound => VERIFIER_COVERAGE_SOUND_PERMILLE,
            Self::Contaminated => VERIFIER_COVERAGE_CONTAMINATED_PERMILLE,
            Self::Homogeneous => VERIFIER_COVERAGE_NONE_PERMILLE,
        }
    }

    pub fn contamination_permille(self) -> u64 {
        match self {
            Self::Contaminated => CONTAMINATION_PERMILLE,
            Self::Sound | Self::Homogeneous => 0,
        }
    }

    pub fn correlated_copy_permille(self) -> u64 {
        match self {
            Self::Sound | Self::Contaminated => CORRELATED_COPY_PERMILLE,
            Self::Homogeneous => 0,
        }
    }

    /// Mixed into the seed so that the three worlds draw independent streams.
    pub fn salt(self) -> u64 {
        match self {
            Self::Sound => 0x0F00_0001,
            Self::Contaminated => 0x0F00_0002,
            Self::Homogeneous => 0x0F00_0003,
        }
    }
}

/// What a run covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub items: usize,
    pub gold: usize,
    pub budget: usize,
    pub resamples: usize,
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: small worlds, the same structure.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => Shape {
                items: ITEMS,
                gold: GOLD_ITEMS,
                budget: ACQUISITION_BUDGET,
                resamples: BOOTSTRAP_RESAMPLES,
            },
            Self::Reduced => Shape {
                items: 400,
                gold: 150,
                budget: 40,
                resamples: 100,
            },
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
        ("pilot_seeds", ints(&PILOT_SEEDS)),
        ("classes", int(CLASSES)),
        ("probabilistic_functions", int(PROBABILISTIC)),
        ("verifier_functions", int(VERIFIERS)),
        ("items", int(ITEMS)),
        ("gold_items", int(GOLD_ITEMS)),
        ("class_prior_permille", ints(&CLASS_PRIOR_PERMILLE)),
        (
            "accuracy_heterogeneous_permille",
            ints(&ACCURACY_HETEROGENEOUS_PERMILLE),
        ),
        (
            "accuracy_homogeneous_permille",
            ints(&ACCURACY_HOMOGENEOUS_PERMILLE),
        ),
        ("wrong_class_weights", ints(&WRONG_CLASS_WEIGHTS)),
        ("votes_per_item", ints(&VOTES_PER_ITEM)),
        (
            "votes_per_item_weight_permille",
            ints(&VOTES_PER_ITEM_WEIGHT_PERMILLE),
        ),
        ("correlated_pair", ints(&CORRELATED_PAIR)),
        ("correlated_copy_permille", int(CORRELATED_COPY_PERMILLE)),
        (
            "verifier_coverage_sound_permille",
            ints(&VERIFIER_COVERAGE_SOUND_PERMILLE),
        ),
        (
            "verifier_coverage_contaminated_permille",
            ints(&VERIFIER_COVERAGE_CONTAMINATED_PERMILLE),
        ),
        ("contamination_permille", int(CONTAMINATION_PERMILLE)),
        ("min_probability_permille", int(MIN_PROBABILITY_PERMILLE)),
        ("calibration_bins", int(CALIBRATION_BINS)),
        ("ds_max_iterations", int(DS_MAX_ITERATIONS)),
        ("ds_tolerance_ppm", int(DS_TOLERANCE_PPM)),
        ("ds_smoothing_milli", int(DS_SMOOTHING_MILLI)),
        (
            "model_oracle_tolerance_ppm",
            int(MODEL_ORACLE_TOLERANCE_PPM),
        ),
        ("metric_tolerance_nano", int(METRIC_TOLERANCE_NANO)),
        ("wilson_z_milli", int(WILSON_Z_MILLI)),
        ("acquisition_budget", int(ACQUISITION_BUDGET)),
        ("mvc_beta_milli", int(MVC_BETA_MILLI)),
        ("bootstrap_resamples", int(BOOTSTRAP_RESAMPLES)),
        ("bootstrap_seed", int(BOOTSTRAP_SEED)),
        (
            "bootstrap_interval_permille",
            int(BOOTSTRAP_INTERVAL_PERMILLE),
        ),
        (
            "brier_gate_on_lower_bound",
            Leaf::Bool(BRIER_GATE_ON_LOWER_BOUND),
        ),
        (
            "ece_gate_on_lower_bound",
            Leaf::Bool(ECE_GATE_ON_LOWER_BOUND),
        ),
        ("min_worlds_run", int(MIN_WORLDS_RUN)),
        ("min_determined_seen", int(MIN_DETERMINED_SEEN)),
        ("min_estimated_seen", int(MIN_ESTIMATED_SEEN)),
        ("min_unknown_seen", int(MIN_UNKNOWN_SEEN)),
        ("min_disputed_seen", int(MIN_DISPUTED_SEEN)),
        ("min_vetoed_items", int(MIN_VETOED_ITEMS)),
        ("min_gold_items", int(MIN_GOLD_ITEMS)),
        (
            "min_fewer_than_three_warnings",
            int(MIN_FEWER_THAN_THREE_WARNINGS),
        ),
        ("min_budget_cells", int(MIN_BUDGET_CELLS)),
        ("min_correlated_pair_items", int(MIN_CORRELATED_PAIR_ITEMS)),
        ("min_contaminated_vetoes", int(MIN_CONTAMINATED_VETOES)),
        ("min_active_set_probes", int(MIN_ACTIVE_SET_PROBES)),
        ("min_gold_guard_probes", int(MIN_GOLD_GUARD_PROBES)),
        ("min_acquisition_items", int(MIN_ACQUISITION_ITEMS)),
        ("max_violations", int(MAX_VIOLATIONS)),
        ("min_gain_ppm", int(MIN_GAIN_PPM)),
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
            toml::Value::Array(items) => Leaf::Ints(
                items
                    .iter()
                    .map(|item| item.as_integer().expect("integer"))
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
        let changed = CONFIG_TOML.replace("items = 2000\n", "items = 2001\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["items: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("items = 2000\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["items: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_distributions_add_up_and_the_manifest_seeds_are_the_constants() {
        assert_eq!(CLASS_PRIOR_PERMILLE.iter().sum::<u64>(), 1000);
        assert_eq!(VOTES_PER_ITEM_WEIGHT_PERMILLE.iter().sum::<u64>(), 1000);
        for class in 0..CLASSES {
            assert_eq!(WRONG_CLASS_WEIGHTS[class * CLASSES + class], 0);
            assert!(
                WRONG_CLASS_WEIGHTS[class * CLASSES..(class + 1) * CLASSES]
                    .iter()
                    .sum::<u64>()
                    > 0
            );
        }
        assert!(VOTES_PER_ITEM.iter().all(|m| *m <= PROBABILISTIC));
        assert!(CORRELATED_PAIR.iter().all(|f| *f < PROBABILISTIC));
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
        for pilot in PILOT_SEEDS {
            assert!(
                !SEEDS.contains(&pilot),
                "a pilot seed is never confirmatory"
            );
        }
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_shape() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.shape().items, ITEMS);
        assert_eq!(Scale::Pinned.shape().gold, GOLD_ITEMS);
        assert_ne!(Scale::Reduced.shape(), Scale::Pinned.shape());
        assert_eq!(Scale::Reduced.name(), "reduced");
    }
}
