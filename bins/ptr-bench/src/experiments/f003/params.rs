//! The parameters of F003, compiled in.
//!
//! They mirror `[preregistration]` and `[pinned_thresholds]` of the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file or
//! the file holds a key no constant mirrors. The world model (the namespaces, their score and
//! harm tables) is code and `PREREGISTRATION.md`, frozen with the commit.

// The drift test and the echo of the procedure use items the harness itself does not read.
#![cfg_attr(not(test), allow(dead_code))]

pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/feedback/F003-calibrated-arbiter/config.toml");
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/feedback/F003-calibrated-arbiter/experiment.toml");
pub const PREREGISTRATION_TEXT: &str =
    include_str!("../../../../../experiments/feedback/F003-calibrated-arbiter/PREREGISTRATION.md");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "calibrated-arbiter";
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- calibrated-arbiter <seed>";
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const DECLARED_SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// Risk level alpha and confidence `1 - delta` of Learn-then-Test, in permille.
pub const ALPHA_PERMILLE: u32 = 50;
pub const DELTA_PERMILLE: u32 = 50;
pub const UNCALIBRATED_THRESHOLD_PERMILLE: u32 = 500;

/// One scale of a run. Only `Pinned` is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub stream_branches: usize,
    pub policy_sizes: &'static [usize],
    pub mc_replicates: usize,
    pub ope_logs: usize,
    pub ope_log_branches: usize,
}

pub const PINNED: Shape = Shape {
    stream_branches: 30_000,
    policy_sizes: &POLICY_SIZES,
    mc_replicates: 2_000,
    ope_logs: 200,
    ope_log_branches: 4_000,
};

/// The unit tests' scale: the same code on a shorter stream and fewer replicates.
pub const REDUCED: Shape = Shape {
    stream_branches: 9_000,
    policy_sizes: &[150, 300, 450, 700],
    mc_replicates: 200,
    ope_logs: 20,
    ope_log_branches: 1_500,
};

/// The adjudications after which a policy is recorded.
pub const POLICY_SIZES: [usize; 4] = [600, 1000, 1500, 2200];
/// The share of eligible branches reserved for adjudication, in permille.
pub const SLICE_RATE_PERMILLE: u32 = 150;
/// The simulated adjudicator's error rate, in permille.
pub const EPSILON_PERMILLE: u32 = 2;
/// The noisy adjudicator error rate of the ungated descriptive arm, in permille.
pub const NOISY_EPSILON_PERMILLE: u32 = 20;
/// Share of scores the scorer reports exactly on a grid point, in permille (`model::TIED_PERMILLE`).
pub const TIED_PERMILLE: u32 = 300;
/// Share of branches that are eligible, in permille.
pub const ELIGIBLE_PERMILLE: u32 = 750;
/// Share of the stream by verification kind, in permille, in the order of `model::Kind::ALL`.
pub const KIND_WEIGHTS_PERMILLE: [u32; 7] = [750, 60, 40, 40, 40, 30, 40];
/// Coverage test: replicates beyond the critical count are rejected at this level, in ppm.
pub const COVERAGE_TEST_LEVEL_PPM: u32 = 1_000;
/// Held-out tests: familywise level in permille and the number of tests it is split over
/// (the policies of a run times the declared seeds).
pub const HELDOUT_FAMILYWISE_PERMILLE: u32 = 50;
pub const HELDOUT_TESTS: u32 = 20;
/// The slice-rate z test rejects beyond this many thousandths of a standard deviation.
pub const SLICE_Z_LIMIT_MILLI: i64 = 4_000;
/// Every n-th replicate also runs the conformal-risk-control rule against its oracle.
pub const CRC_EVERY: usize = 20;
/// Replicates and logs whose outputs are re-derived by the determinism rerun.
pub const DETERMINISM_REPLICATES: usize = 100;
pub const DETERMINISM_LOGS: usize = 8;
/// Off-policy evaluation.
pub const OPE_LOGGING_THRESHOLD_PERMILLE: u32 = 500;
pub const OPE_BIAS_SIGMAS: f64 = 4.0;
pub const REWARD_SAFE: f64 = 1.0;
pub const REWARD_HARM: f64 = -9.0;
/// A negative control draws its held-out harm at this multiple of the model's.
pub const DRIFT_HARM_FACTOR: i64 = 3;
pub const DRIFT_SAMPLES: usize = 1_500;
pub const CONTROL_FALLBACK_THRESHOLD_PERMILLE: u32 = 700;

pub fn alpha() -> f64 {
    f64::from(ALPHA_PERMILLE) / 1000.0
}

pub fn delta() -> f64 {
    f64::from(DELTA_PERMILLE) / 1000.0
}

pub fn slice_rate() -> f64 {
    f64::from(SLICE_RATE_PERMILLE) / 1000.0
}

pub fn heldout_level() -> f64 {
    f64::from(HELDOUT_FAMILYWISE_PERMILLE) / 1000.0 / f64::from(HELDOUT_TESTS)
}

pub fn coverage_level() -> f64 {
    f64::from(COVERAGE_TEST_LEVEL_PPM) / 1_000_000.0
}
