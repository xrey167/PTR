//! The preregistered choices of M008, compiled in. `config.toml`'s `[pinned_thresholds]` holds
//! the same values and a drift test compares them, so a constant cannot change without the
//! experiment's configuration changing with it.
//!
//! Three groups. The pool (the frozen text pool and its frozen MiniLM vectors, from
//! `research/embeddings`) and the session shape; the budget (token budget, recency window,
//! ranked-list length) derived from the pool; and the operating points (memory shape, decode and
//! abstention thresholds), which were chosen on the pilot seeds only (`calibrate.rs`).

/// The subcommand and the benchmark name every result carries.
pub const BENCHMARK: &str = "fast-weight-memory";
pub const EXPERIMENT: &str = "M008";

/// The confirmatory seeds, as `experiment.toml` declares them. Nothing is tuned on them.
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
/// The pilot seeds: disjoint from the confirmatory ones, used for the memory shape and the
/// abstention thresholds only.
pub const PILOT_SEEDS: [u64; 5] = [1001, 1002, 1003, 1004, 1005];

// ---- the frozen artifact -------------------------------------------------------------------

pub const EMBEDDING_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const EMBEDDING_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
pub const POOL_FILE: &str = "research/embeddings/pool-m008.jsonl";
pub const VECTORS_FILE: &str = "research/embeddings/vectors-m008.f16";
pub const POOL_SHA256: &str = "0192b66e2e329c877cbce12dbbabe8cd0c0c930871729477e84852a19afd6ef7";
pub const VECTORS_SHA256: &str = "378d013fcd2840bd76e8debe67109200a28c86b96130dc9d4f1e1487146762ba";
pub const POOL_ROWS: usize = 14_000;
pub const EMBEDDING_DIM: usize = 384;
pub const POOL_CUES: usize = 1_900;
pub const POOL_VERSIONS: usize = 3;
pub const POOL_PARAPHRASES: usize = 4;
pub const POOL_OFFTOPIC: usize = 700;
/// Tokens are counted by this pattern (`[A-Za-z0-9]+|[^\sA-Za-z0-9]`) in the pool generator and
/// recounted here from the texts; a different count is a harness-integrity failure.
pub const TOKEN_PATTERN: &str = "[A-Za-z0-9]+|[^\\sA-Za-z0-9]";

// ---- the session ---------------------------------------------------------------------------

/// Cues written once, cues that get a later update, and the queries of each kind.
pub const CUES_WRITTEN: usize = 400;
pub const CUES_UPDATED: usize = 200;
pub const QUERIES_UPDATED: usize = 160;
pub const QUERIES_STABLE: usize = 160;
pub const QUERIES_UNWRITTEN: usize = 40;
pub const QUERIES_OFFTOPIC: usize = 40;

// ---- the budget ----------------------------------------------------------------------------

/// The equal token budget of every arm.
pub const TOKEN_BUDGET: usize = 512;
/// The longest fact of the pool in tokens (read from the pool at start-up and asserted).
pub const MAX_FACT_TOKENS: usize = 13;
/// N, the writes the recency buffer keeps: `floor(TOKEN_BUDGET / MAX_FACT_TOKENS)` (asserted).
pub const RECENCY_WINDOW: usize = 39;
/// Most facts an arm presents for one query (recall@k is recall within this list).
pub const RANKED_LIMIT: usize = 5;

// ---- the memory (chosen on the pilot seeds) ------------------------------------------------

pub const HEADS: usize = 8;
pub const KEY_DIM: usize = 32;
pub const VALUE_DIM: usize = 64;
/// `beta` of every write: 1000 permille, so an update replaces the association under its key.
pub const BETA_PERMILLE: i64 = 1000;
/// The scalar decay factor of every write, in permille (1000 is `Decay::None`).
pub const DECAY_PERMILLE: i64 = 999;
pub const CHECKPOINT_INTERVAL: u32 = 64;
pub const PROJECTION_SEED: u64 = 0x4d30_3038_0001;
pub const CODEBOOK_SEED: u64 = 0x4d30_3038_0002;
/// The projection the projection probes read with (not the memory's).
pub const OTHER_PROJECTION_SEED: u64 = 0x4d30_3038_0003;

// ---- the operating points (chosen on the pilot seeds) --------------------------------------

/// Decode thresholds of fast memory, in permille of the readout's score.
pub const FAST_MIN_SCORE_MILLI: i64 = 510;
pub const FAST_MIN_MARGIN_MILLI: i64 = 25;
/// The recency buffer abstains when the best cosine in its window is below this (permille).
pub const RECENCY_MIN_COSINE_MILLI: i64 = 580;
/// The hybrid arms abstain when the cosine between the query and their top hit is below this.
pub const HYBRID_LIVE_MIN_COSINE_MILLI: i64 = 670;
pub const HYBRID_APPEND_MIN_COSINE_MILLI: i64 = 670;
/// The false-recall rate on the unanswerable probes the thresholds are matched to (permille).
pub const MATCHED_FALSE_RECALL_PERMILLE: i64 = 50;

// ---- statistics ----------------------------------------------------------------------------

/// The paired bootstrap: resamples, the one-sided lower quantile (permille of the resamples), and
/// the pinned seed (the run's seed is mixed in, one stream per comparison).
pub const BOOTSTRAP_RESAMPLES: usize = 2_000;
pub const BOOTSTRAP_LOWER_QUANTILE_PERMILLE: usize = 50;
pub const BOOTSTRAP_SEED: u64 = 0x4d30_3038_b007;

// ---- product contract checks ---------------------------------------------------------------

/// A write read back through its own key must score its own code at least this (permille); the
/// delta rule at beta 1 reads back the value written under a unit key exactly.
pub const READBACK_MIN_MILLI: i64 = 990;
/// After an exact-key overwrite the stale version must score at most this (permille): the
/// delta rule erases it, what remains is crosstalk of other writes.
pub const STALE_RESIDUAL_MAX_MILLI: i64 = 400;
/// Tolerance between the product's decode and the harness's independent decode (scores).
pub const DECODE_SCORE_TOLERANCE: f32 = 1.0e-5;

// ---- coverage and controls -----------------------------------------------------------------

pub const MIN_UPDATED_CUE_QUERIES: i64 = 100;
pub const MIN_UNANSWERABLE_QUERIES: i64 = 50;
pub const MIN_OLD_STRATUM_QUERIES: i64 = 100;
pub const MIN_WRITTEN_FACTS: i64 = 500;
pub const MIN_READBACK_CHECKS: i64 = 500;
pub const MIN_STALE_RESIDUAL_CHECKS: i64 = 100;
pub const MIN_PROJECTION_PROBES: i64 = 50;
pub const MIN_FAST_SCORE_ABSTENTIONS: i64 = 20;
pub const MIN_CONTRACT_PROBE_MARGIN_REFUSALS: i64 = 20;
pub const MIN_CONTRACT_PROBES: i64 = 80;
/// The matched operating point must transfer: no gated arm answers more than this share of the
/// unanswerable queries of a confirmatory seed (permille). The target is 50.
pub const MAX_UNANSWERABLE_ANSWERED_PERMILLE: i64 = 150;
pub const MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE: i64 = 1000;
pub const MAX_CONTROL_RANDOM_RECALL_PERMILLE: i64 = 100;
pub const MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE: i64 = 300;
pub const MAX_ORACLE_LABEL_MISMATCHES: i64 = 0;
/// The positive control (the oracle-cheat answers in the memory's place) must clear the gain and
/// false-recall rules by at least this much (permille): the lower bounds of its gains, the headroom
/// of its false-recall rate under the recency buffer's.
pub const MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE: i64 = 1;
pub const MIN_CONTROL_POSITIVE_FALSE_RECALL_HEADROOM_PERMILLE: i64 = 0;

// ---- descriptive evidence (never judged) ----------------------------------------------------

/// Cue counts of the capacity table (the pinned session is the 400 of the main run) and the score
/// thresholds (permille) of the threshold sweep.
pub const CAPACITY_CASES: [usize; 4] = [16, 32, 64, 128];
pub const SWEEP_THRESHOLDS_MILLI: [i64; 10] = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900];

// ---- decision criteria ---------------------------------------------------------------------

pub const MIN_RECALL_GAIN_LO_PERMILLE: i64 = 1;
pub const MAX_TOKEN_BUDGET_VIOLATIONS: i64 = 0;
pub const MAX_WINDOW_BUDGET_VIOLATIONS: i64 = 0;
pub const MAX_PRODUCT_PANICS: i64 = 0;
pub const MAX_PRODUCT_ERRORS: i64 = 0;
pub const MAX_STALE_IN_LIVE_INDEX: i64 = 0;
pub const MAX_STALE_RESIDUAL_VIOLATIONS: i64 = 0;
pub const MAX_READBACK_FAILURES: i64 = 0;
pub const MAX_DECODE_CONTRACT_VIOLATIONS: i64 = 0;
pub const MAX_PROJECTION_MISMATCH_ACCEPTED: i64 = 0;

/// The scale of a run: the pinned one, or a reduced or mutation-checker one that is no evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// `<cases> <seed>`: `cases` cues written, the rest proportional.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Custom(_) => "custom",
        }
    }

    pub fn shape(self) -> Result<Shape, String> {
        match self {
            Self::Pinned => Ok(Shape::PINNED),
            Self::Custom(cases) => Shape::of_cases(cases),
        }
    }
}

/// The size of one session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub written: usize,
    pub updated: usize,
    pub queries_updated: usize,
    pub queries_stable: usize,
    pub queries_unwritten: usize,
    pub queries_offtopic: usize,
}

impl Shape {
    pub const PINNED: Shape = Shape {
        written: CUES_WRITTEN,
        updated: CUES_UPDATED,
        queries_updated: QUERIES_UPDATED,
        queries_stable: QUERIES_STABLE,
        queries_unwritten: QUERIES_UNWRITTEN,
        queries_offtopic: QUERIES_OFFTOPIC,
    };

    /// The pinned proportions at another size: half the written cues are updated and the queries
    /// are 40% updated cues, 40% never-updated cues, 10% unwritten cues and 10% off-topic.
    pub fn of_cases(cases: usize) -> Result<Self, String> {
        if !(16..=800).contains(&cases) {
            return Err(format!("cases must be in 16..=800, got {cases}"));
        }
        Ok(Self {
            written: cases,
            updated: cases / 2,
            queries_updated: cases * 2 / 5,
            queries_stable: cases * 2 / 5,
            queries_unwritten: cases / 10,
            queries_offtopic: cases / 10,
        })
    }

    pub fn queries(&self) -> usize {
        self.queries_updated + self.queries_stable + self.queries_unwritten + self.queries_offtopic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    pub(crate) fn config() -> toml::Table {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../experiments/model/M008-fast-weight-memory/config.toml");
        let text = std::fs::read_to_string(&path).expect("the experiment's config.toml");
        let table: toml::Table = text.parse().expect("config.toml parses");
        table["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .clone()
    }

    enum Pin {
        Int(i64),
        Str(&'static str),
        Ints(Vec<i64>),
    }

    fn ints<const N: usize>(values: [u64; N]) -> Pin {
        Pin::Ints(values.iter().map(|value| *value as i64).collect())
    }

    #[test]
    fn the_compiled_choices_are_the_configs() {
        let pins: Vec<(&str, Pin)> = vec![
            ("seeds", ints(SEEDS)),
            ("pilot_seeds", ints(PILOT_SEEDS)),
            ("embedding_model", Pin::Str(EMBEDDING_MODEL)),
            ("embedding_revision", Pin::Str(EMBEDDING_REVISION)),
            ("pool_file", Pin::Str(POOL_FILE)),
            ("vectors_file", Pin::Str(VECTORS_FILE)),
            ("pool_sha256", Pin::Str(POOL_SHA256)),
            ("vectors_sha256", Pin::Str(VECTORS_SHA256)),
            ("pool_rows", Pin::Int(POOL_ROWS as i64)),
            ("embedding_dim", Pin::Int(EMBEDDING_DIM as i64)),
            ("pool_cues", Pin::Int(POOL_CUES as i64)),
            ("pool_versions", Pin::Int(POOL_VERSIONS as i64)),
            ("pool_paraphrases", Pin::Int(POOL_PARAPHRASES as i64)),
            ("pool_offtopic", Pin::Int(POOL_OFFTOPIC as i64)),
            ("token_pattern", Pin::Str(TOKEN_PATTERN)),
            ("cues_written", Pin::Int(CUES_WRITTEN as i64)),
            ("cues_updated", Pin::Int(CUES_UPDATED as i64)),
            ("queries_updated", Pin::Int(QUERIES_UPDATED as i64)),
            ("queries_stable", Pin::Int(QUERIES_STABLE as i64)),
            ("queries_unwritten", Pin::Int(QUERIES_UNWRITTEN as i64)),
            ("queries_offtopic", Pin::Int(QUERIES_OFFTOPIC as i64)),
            ("token_budget", Pin::Int(TOKEN_BUDGET as i64)),
            ("max_fact_tokens", Pin::Int(MAX_FACT_TOKENS as i64)),
            ("recency_window", Pin::Int(RECENCY_WINDOW as i64)),
            ("ranked_limit", Pin::Int(RANKED_LIMIT as i64)),
            ("heads", Pin::Int(HEADS as i64)),
            ("key_dim", Pin::Int(KEY_DIM as i64)),
            ("value_dim", Pin::Int(VALUE_DIM as i64)),
            ("beta_permille", Pin::Int(BETA_PERMILLE)),
            ("decay_permille", Pin::Int(DECAY_PERMILLE)),
            (
                "checkpoint_interval",
                Pin::Int(i64::from(CHECKPOINT_INTERVAL)),
            ),
            ("projection_seed", Pin::Int(PROJECTION_SEED as i64)),
            ("codebook_seed", Pin::Int(CODEBOOK_SEED as i64)),
            (
                "other_projection_seed",
                Pin::Int(OTHER_PROJECTION_SEED as i64),
            ),
            ("fast_min_score_milli", Pin::Int(FAST_MIN_SCORE_MILLI)),
            ("fast_min_margin_milli", Pin::Int(FAST_MIN_MARGIN_MILLI)),
            (
                "recency_min_cosine_milli",
                Pin::Int(RECENCY_MIN_COSINE_MILLI),
            ),
            (
                "hybrid_live_min_cosine_milli",
                Pin::Int(HYBRID_LIVE_MIN_COSINE_MILLI),
            ),
            (
                "hybrid_append_min_cosine_milli",
                Pin::Int(HYBRID_APPEND_MIN_COSINE_MILLI),
            ),
            (
                "matched_false_recall_permille",
                Pin::Int(MATCHED_FALSE_RECALL_PERMILLE),
            ),
            ("bootstrap_resamples", Pin::Int(BOOTSTRAP_RESAMPLES as i64)),
            (
                "bootstrap_lower_quantile_permille",
                Pin::Int(BOOTSTRAP_LOWER_QUANTILE_PERMILLE as i64),
            ),
            ("bootstrap_seed", Pin::Int(BOOTSTRAP_SEED as i64)),
            ("readback_min_milli", Pin::Int(READBACK_MIN_MILLI)),
            (
                "stale_residual_max_milli",
                Pin::Int(STALE_RESIDUAL_MAX_MILLI),
            ),
            ("min_updated_cue_queries", Pin::Int(MIN_UPDATED_CUE_QUERIES)),
            (
                "min_unanswerable_queries",
                Pin::Int(MIN_UNANSWERABLE_QUERIES),
            ),
            ("min_old_stratum_queries", Pin::Int(MIN_OLD_STRATUM_QUERIES)),
            ("min_written_facts", Pin::Int(MIN_WRITTEN_FACTS)),
            ("min_readback_checks", Pin::Int(MIN_READBACK_CHECKS)),
            (
                "min_stale_residual_checks",
                Pin::Int(MIN_STALE_RESIDUAL_CHECKS),
            ),
            ("min_projection_probes", Pin::Int(MIN_PROJECTION_PROBES)),
            (
                "min_fast_score_abstentions",
                Pin::Int(MIN_FAST_SCORE_ABSTENTIONS),
            ),
            (
                "min_contract_probe_margin_refusals",
                Pin::Int(MIN_CONTRACT_PROBE_MARGIN_REFUSALS),
            ),
            ("min_contract_probes", Pin::Int(MIN_CONTRACT_PROBES)),
            (
                "max_unanswerable_answered_permille",
                Pin::Int(MAX_UNANSWERABLE_ANSWERED_PERMILLE),
            ),
            (
                "min_control_oracle_cheat_recall_permille",
                Pin::Int(MIN_CONTROL_ORACLE_CHEAT_RECALL_PERMILLE),
            ),
            (
                "max_control_random_recall_permille",
                Pin::Int(MAX_CONTROL_RANDOM_RECALL_PERMILLE),
            ),
            (
                "min_control_stale_first_false_recall_permille",
                Pin::Int(MIN_CONTROL_STALE_FIRST_FALSE_RECALL_PERMILLE),
            ),
            (
                "min_control_positive_gain_lo_permille",
                Pin::Int(MIN_CONTROL_POSITIVE_GAIN_LO_PERMILLE),
            ),
            (
                "min_control_positive_false_recall_headroom_permille",
                Pin::Int(MIN_CONTROL_POSITIVE_FALSE_RECALL_HEADROOM_PERMILLE),
            ),
            (
                "max_oracle_label_mismatches",
                Pin::Int(MAX_ORACLE_LABEL_MISMATCHES),
            ),
            (
                "min_recall_gain_lo_permille",
                Pin::Int(MIN_RECALL_GAIN_LO_PERMILLE),
            ),
            (
                "max_token_budget_violations",
                Pin::Int(MAX_TOKEN_BUDGET_VIOLATIONS),
            ),
            (
                "max_window_budget_violations",
                Pin::Int(MAX_WINDOW_BUDGET_VIOLATIONS),
            ),
            ("max_product_panics", Pin::Int(MAX_PRODUCT_PANICS)),
            ("max_product_errors", Pin::Int(MAX_PRODUCT_ERRORS)),
            ("max_stale_in_live_index", Pin::Int(MAX_STALE_IN_LIVE_INDEX)),
            (
                "max_stale_residual_violations",
                Pin::Int(MAX_STALE_RESIDUAL_VIOLATIONS),
            ),
            ("max_readback_failures", Pin::Int(MAX_READBACK_FAILURES)),
            (
                "max_decode_contract_violations",
                Pin::Int(MAX_DECODE_CONTRACT_VIOLATIONS),
            ),
            (
                "max_projection_mismatch_accepted",
                Pin::Int(MAX_PROJECTION_MISMATCH_ACCEPTED),
            ),
        ];
        let table = config();
        for (name, pin) in pins {
            let value = table
                .get(name)
                .unwrap_or_else(|| panic!("config.toml pins no {name}"));
            match pin {
                Pin::Int(expected) => {
                    assert_eq!(value.as_integer(), Some(expected), "{name}");
                }
                Pin::Str(expected) => {
                    assert_eq!(value.as_str(), Some(expected), "{name}");
                }
                Pin::Ints(expected) => {
                    let found: Vec<i64> = value
                        .as_array()
                        .unwrap_or_else(|| panic!("{name} is an array"))
                        .iter()
                        .map(|item| item.as_integer().expect("integer"))
                        .collect();
                    assert_eq!(found, expected, "{name}");
                }
            }
        }
    }

    #[test]
    fn the_pilot_seeds_are_disjoint_from_the_confirmatory_ones() {
        for pilot in PILOT_SEEDS {
            assert!(!SEEDS.contains(&pilot));
        }
    }

    #[test]
    fn the_window_is_what_the_budget_buys() {
        assert_eq!(RECENCY_WINDOW, TOKEN_BUDGET / MAX_FACT_TOKENS);
        const { assert!(RECENCY_WINDOW * MAX_FACT_TOKENS <= TOKEN_BUDGET) };
        const { assert!((RECENCY_WINDOW + 1) * MAX_FACT_TOKENS > TOKEN_BUDGET) };
    }

    #[test]
    fn the_pinned_shape_is_the_proportion_at_four_hundred_cases() {
        assert_eq!(Shape::of_cases(400), Ok(Shape::PINNED));
        assert!(Shape::of_cases(15).is_err());
        assert!(Shape::of_cases(801).is_err());
        let shape = Shape::of_cases(100).unwrap();
        assert_eq!(
            (shape.written, shape.updated, shape.queries()),
            (100, 50, 100)
        );
        assert_eq!(Shape::PINNED.queries(), 400);
        assert_eq!(Shape::PINNED.written + Shape::PINNED.updated, 600);
    }
}
