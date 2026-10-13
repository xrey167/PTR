//! The preregistered parameters of F007, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file and
//! when the file holds a key no constant mirrors. F007 is `planned`: nothing here is frozen.
//! The decision margins (`criteria.toml`, `[decision]`) are not compiled in: the harness reports
//! counts, the aggregator decides.

#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/feedback/F007-vcw-loop/config.toml");

/// The experiment's manifest, read at compile time.
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/feedback/F007-vcw-loop/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "vcw-loop";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
/// Pilot seeds are at least this; a confirmatory seed is never one of them and no pilot seed is
/// ever confirmatory.
pub const PILOT_SEED_FLOOR: u64 = 1001;
/// Requests per world and seed, all kinds together.
pub const REQUESTS_PER_WORLD: usize = 4000;
/// Share of requests by task kind in permille: arithmetic, JSON object, code.
pub const KIND_WEIGHTS_PERMILLE: [u64; 3] = [400, 350, 250];
/// The verifier-call budget of a request, the same for every arm, and the round budget.
pub const VERIFIER_CALL_BUDGET: u32 = 3;
pub const ROUND_BUDGET: u32 = 3;
/// Requests per lane (task kind) of the three training streams: statistics (A), tuning (B) and
/// promotion (C), all under uniform routing and all on traffic of pilot seeds.
pub const TRAIN_A: usize = 1500;
pub const TRAIN_B: usize = 1000;
pub const TRAIN_C: usize = 1500;
/// The Beta prior of the learned router and the temperature (per mille) of the softmax gate and of
/// the gate-weighted proposers of best-of-N: nine tuning candidates each, chosen on stream B by the doubly robust estimate.
pub const PRIORS: [u64; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
pub const GATE_TAU_PERMILLE: [u64; 9] = [1, 2, 5, 10, 20, 50, 100, 200, 400];
/// A tuned parameter at the edge of its grid is a grid that was too narrow when it beats its inner
/// neighbour on stream B by more than this many standard errors of their PAIRED difference (both are
/// evaluated on the very same stream, so the pair's noise is the noise that matters, and it shrinks as
/// the two candidates become the same policy). 3 standard errors is a one-sided 0.13 percent level: the
/// rule is tested up to nine times per seed and must not invalidate a run on chance. In milli-units.
pub const EDGE_Z_MILLI: u64 = 3000;
/// The promotion rule shared by both learned arms: lower 95% bound of the doubly robust
/// difference to uniform routing above the margin, on at least this many decisions and this
/// effective sample size.
pub const PROMOTION_MARGIN_PERMILLE: u64 = 20;
pub const PROMOTION_MIN_RECORDS: usize = 300;
pub const PROMOTION_MIN_ESS: u64 = 100;
/// Requests of a lane per observation window of the structure policy, and the policy's
/// thresholds for a Birth (Birth is the only operation enabled).
pub const WINDOW_REQUESTS: u32 = 40;
pub const BIRTH_MIN_UNROUTABLE: u64 = 3;
pub const BIRTH_WINDOWS: u64 = 2;
/// Held-out tasks per code of the forgetting gate's suite.
pub const HELD_OUT_TASKS_PER_CODE: usize = 30;
/// The skill (per mille) of an existing Pod at the code nobody can repair, and of the born
/// specialist, by world.
pub const GAP_SKILL_PERMILLE: u64 = 0;
pub const BORN_SKILL_PERMILLE: u64 = 800;
/// The share (per mille) of failed repairs that a Pod which reads the workspace broadcast fixes
/// anyway, in the world that declares the assumption (`ctx`).
pub const BROADCAST_SECOND_CHANCE_PERMILLE: u64 = 500;
/// Proposal defect rate per code (per mille) of a Pod that is not good at the kind, and the
/// multiplier (per mille) for the Pod that is: names `code=rate,...`.
pub const ERROR_PERMILLE: &str = "arith.format=200,arith.value=450,arith.bounds=100,json.parse=200,json.type=350,json.required=350,json.extra=250,code.syntax=250,code.test_failed=450,code.timeout=250";
pub const PROPOSER_GOOD_MULTIPLIER_PERMILLE: u64 = 600;
pub const PROPOSER_OTHER_MULTIPLIER_PERMILLE: u64 = 1300;
/// Repair skill (per mille) by world: on-role specialist base and its offset span, the
/// generalist, the off-role specialist; names `world=on,span,generalist,off`.
pub const SKILL_PERMILLE: &str =
    "hetero=800,80,600,300;mild=720,40,620,400;homog=600,0,600,600;ctx=800,80,600,300";
/// Per-seed thresholds of the harness controls (permille of requests).
pub const SHUFFLED_MIN_LOSS_PERMILLE: i64 = 50;
pub const NULL_MAX_GAP_PERMILLE: i64 = 30;
/// Derived from the planning model alone (`power.py derive`, half of the modelled gain; the arithmetic
/// uses no pilot number, the choice of "half" is a judgement): gate over the untrained pool, best-of-N
/// over single shot.
pub const BASELINE_MIN_GAIN_PERMILLE: i64 = 43;
pub const BON_MIN_GAIN_PERMILLE: i64 = 176;
/// The oracle fixed-Pod reference may be below the full loop by at most this (it has no L3). A
/// judgement: two points is a fifth of the margin, small enough that a crippled oracle shows, large
/// enough for the noise of one seed.
pub const ORACLE_TOLERANCE_PERMILLE: i64 = 20;
/// Coverage prerequisites.
pub const MIN_REQUESTS_PER_KIND_PERMILLE: i64 = 200;
pub const MIN_HARD_EVENTS_PER_CODE: i64 = 10;
pub const MIN_REPAIRS_APPLIED: i64 = 200;
pub const MIN_EXHAUSTED_RUNS: i64 = 20;
pub const MIN_BIRTHS: i64 = 1;
pub const MAX_HARD_COUNTER: i64 = 0;
/// The descriptive worlds that are run but never gate a verdict.
pub const WORLDS: [&str; 4] = ["hetero", "homog", "mild", "ctx"];

/// What a run covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// This many requests per world (the mutation checker's second argument); no evidence.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Custom(_) => "custom",
        }
    }

    pub fn requests(self) -> usize {
        match self {
            Self::Pinned => REQUESTS_PER_WORLD,
            Self::Custom(requests) => requests,
        }
    }

    /// Requests per lane of the three training streams.
    pub fn training(self) -> [usize; 3] {
        match self {
            Self::Pinned => [TRAIN_A, TRAIN_B, TRAIN_C],
            Self::Custom(requests) => [requests, requests * 2 / 3, requests],
        }
    }
}

/// The seeds of the three training streams of a run with seed `seed`: always pilot seeds
/// (at least [`PILOT_SEED_FLOOR`]), different for every run seed and never a declared seed.
pub fn training_seeds(seed: u64) -> [u64; 3] {
    let base = PILOT_SEED_FLOOR + 1000 * (seed + 1);
    [base, base + 1, base + 2]
}

/// `name=value` pairs of a compiled table string.
pub fn parse_table(text: &str) -> Vec<(&str, &str)> {
    text.split(';')
        .flat_map(|group| group.split(','))
        .filter_map(|pair| pair.split_once('='))
        .collect()
}

#[cfg(test)]
pub use drift::*;

#[cfg(test)]
mod drift {
    use super::*;

    /// One value of the compiled table, comparable with a value of `config.toml`.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum Leaf {
        Int(i64),
        Text(String),
        Ints(Vec<i64>),
    }

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
            ("pilot_seed_floor", int(PILOT_SEED_FLOOR)),
            ("requests_per_world", int(REQUESTS_PER_WORLD)),
            ("kind_weights_permille", ints(&KIND_WEIGHTS_PERMILLE)),
            ("verifier_call_budget", int(VERIFIER_CALL_BUDGET)),
            ("round_budget", int(ROUND_BUDGET)),
            ("train_a", int(TRAIN_A)),
            ("train_b", int(TRAIN_B)),
            ("train_c", int(TRAIN_C)),
            ("priors", ints(&PRIORS)),
            ("gate_tau_permille", ints(&GATE_TAU_PERMILLE)),
            ("promotion_margin_permille", int(PROMOTION_MARGIN_PERMILLE)),
            ("promotion_min_records", int(PROMOTION_MIN_RECORDS)),
            ("promotion_min_ess", int(PROMOTION_MIN_ESS)),
            ("window_requests", int(WINDOW_REQUESTS)),
            ("birth_min_unroutable", int(BIRTH_MIN_UNROUTABLE)),
            ("birth_windows", int(BIRTH_WINDOWS)),
            ("held_out_tasks_per_code", int(HELD_OUT_TASKS_PER_CODE)),
            ("gap_skill_permille", int(GAP_SKILL_PERMILLE)),
            ("born_skill_permille", int(BORN_SKILL_PERMILLE)),
            (
                "broadcast_second_chance_permille",
                int(BROADCAST_SECOND_CHANCE_PERMILLE),
            ),
            ("error_permille", Leaf::Text(ERROR_PERMILLE.into())),
            (
                "proposer_good_multiplier_permille",
                int(PROPOSER_GOOD_MULTIPLIER_PERMILLE),
            ),
            (
                "proposer_other_multiplier_permille",
                int(PROPOSER_OTHER_MULTIPLIER_PERMILLE),
            ),
            ("skill_permille", Leaf::Text(SKILL_PERMILLE.into())),
            (
                "shuffled_min_loss_permille",
                int(SHUFFLED_MIN_LOSS_PERMILLE),
            ),
            ("null_max_gap_permille", int(NULL_MAX_GAP_PERMILLE)),
            (
                "baseline_min_gain_permille",
                int(BASELINE_MIN_GAIN_PERMILLE),
            ),
            ("bon_min_gain_permille", int(BON_MIN_GAIN_PERMILLE)),
            ("oracle_tolerance_permille", int(ORACLE_TOLERANCE_PERMILLE)),
            ("edge_z_milli", int(EDGE_Z_MILLI)),
            (
                "min_requests_per_kind_permille",
                int(MIN_REQUESTS_PER_KIND_PERMILLE),
            ),
            ("min_hard_events_per_code", int(MIN_HARD_EVENTS_PER_CODE)),
            ("min_repairs_applied", int(MIN_REPAIRS_APPLIED)),
            ("min_exhausted_runs", int(MIN_EXHAUSTED_RUNS)),
            ("min_births", int(MIN_BIRTHS)),
            ("max_hard_counter", int(MAX_HARD_COUNTER)),
        ]
    }

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
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
    pub fn drift(config: &str) -> Vec<String> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let file: std::collections::BTreeMap<String, Leaf> = parsed["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .iter()
            .map(|(key, value)| (key.clone(), leaf_of(value)))
            .collect();
        let compiled: std::collections::BTreeMap<String, Leaf> = compiled()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compiled_constants_are_the_pinned_thresholds_of_config_toml() {
        assert_eq!(drift(CONFIG_TOML), Vec::<String>::new());
    }

    #[test]
    fn a_changed_added_or_removed_key_of_config_toml_is_drift() {
        let changed = CONFIG_TOML.replace("window_requests = 40\n", "window_requests = 41\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["window_requests: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("window_requests = 40\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["window_requests: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_manifest_seeds_are_the_constants() {
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
    fn training_streams_use_pilot_seeds_only_and_never_a_declared_seed() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in SEEDS.into_iter().chain([1001, 1002, 1003, 1004]) {
            for train in training_seeds(seed) {
                assert!(train >= PILOT_SEED_FLOOR, "{train}");
                assert!(!SEEDS.contains(&train), "{train}");
                assert!(seen.insert(train), "{train} is shared between runs");
            }
        }
        // A pilot run seed is itself at least the floor and distinct from every training seed.
        for pilot in 1001..1010 {
            assert!(!seen.contains(&pilot), "{pilot}");
        }
    }

    #[test]
    fn the_weights_and_tables_are_well_formed() {
        assert_eq!(KIND_WEIGHTS_PERMILLE.iter().sum::<u64>(), 1000);
        assert_eq!(
            parse_table(SKILL_PERMILLE)
                .iter()
                .filter(|(name, _)| WORLDS.contains(name))
                .count(),
            4
        );
        assert_eq!(parse_table(ERROR_PERMILLE).len(), 10);
        const { assert!(VERIFIER_CALL_BUDGET >= 1 && ROUND_BUDGET >= 1) };
        assert_eq!(Scale::Pinned.requests(), REQUESTS_PER_WORLD);
        assert_eq!(Scale::Pinned.training(), [TRAIN_A, TRAIN_B, TRAIN_C]);
        assert_eq!(Scale::Custom(9).name(), "custom");
    }
}
