//! The preregistered parameters of Q003, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`, which
//! is read at compile time; the drift test fails when a constant differs from the file and when the
//! file holds a key no constant mirrors. The harness serves Q003-v3, the successor of the superseded Q003.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/retrieval/Q003-v3-postgres-hybrid/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/retrieval/Q003-v3-postgres-hybrid/experiment.toml");

/// The experiment this harness runs for: the id in the result's `experiment` field, which
/// `aggregate.py` and `criteria.toml` name. Q003-v3 succeeds Q003 (superseded) and keeps its harness.
pub const EXPERIMENT_ID: &str = "Q003-v3";

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "postgres-hybrid";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it from the
/// freeze on (until then its `entrypoint` is empty). The strong arm's interpreter and the models
/// directory come from `PTR_Q003_STRONG_PYTHON` and `PTR_Q003_MODELS_ROOT`.
pub const ENTRYPOINT: &str = "cargo run --release --locked -p ptr-bench --features postgres-experiments -- postgres-hybrid <seed> --strong-threads 1";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [43, 71, 101, 131, 173];

/// The frozen inputs, relative to the repository root, and their SHA-256. The harness hashes each
/// file it reads at start-up and exits 2 on a mismatch.
pub const POOL_PATH: &str = "research/embeddings/pool-q003.jsonl";
pub const POOL_SHA256: &str = "0a46e80db51a8a74596005a211acab9efad990303fba15fa6ea1f47aae432b05";
pub const VECTORS_PATH: &str = "research/embeddings/vectors-q003.f16";
pub const VECTORS_SHA256: &str = "4b5c350fcf2e5d59bd7bc77f7f8a671e1f0106e0630bd9423f7570733576a590";
pub const STRONG_VECTORS_PATH: &str = "research/embeddings/vectors-q003-bge-m3.f16";
pub const STRONG_VECTORS_SHA256: &str =
    "b663def2cc044c2ef096fe0eae46d0da210f91ab8d7513472a8ee25e482751c6";
pub const POOL_ROWS: usize = 12_000;
pub const EMBEDDING_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const EMBEDDING_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
pub const EMBEDDING_DIMENSIONS: usize = 384;
pub const STRONG_DIMENSIONS: usize = 1024;

/// The corpus: one capsule per cue of the pool, each with up to three generations (the three fact
/// versions of its cue). The lifecycle mix is in permille of the capsules: live at generation 1,
/// superseded once (live generation 2), superseded twice (live generation 3), and revoked at
/// generation 1, 2 or 3 (the last takes the remainder so the mix sums to the capsules).
pub const CAPSULES: usize = 2000;
pub const STAY_PERMILLE: usize = 350;
pub const SUPERSEDED_ONCE_PERMILLE: usize = 250;
pub const SUPERSEDED_TWICE_PERMILLE: usize = 200;
pub const REVOKED_AT_1_PERMILLE: usize = 75;
pub const REVOKED_AT_2_PERMILLE: usize = 62;

/// The queries: `QUERIES` per seed (the pilot's paired standard error of the recall difference was 14 to 16
/// per mille at 400 queries, so the count was raised to bring it to at most 10), one per distinct capsule,
/// paraphrase drawn by the seed.
/// `FIRST_CHECKPOINT_PERCENT` of them are asked at the first checkpoint (after every supersession,
/// before any revocation; all answerable), the rest at the second (after the revocations):
/// `REVOKED_TARGET_PERCENT` of all queries ask for a revoked capsule (unanswerable), the others for
/// a live one. Each group is split evenly over the live generations 1, 2 and 3 (revoked targets
/// over the revoked generations).
pub const QUERIES: usize = 1000;
pub const FIRST_CHECKPOINT_PERCENT: usize = 30;
pub const REVOKED_TARGET_PERCENT: usize = 10;
/// The strong arm answers this many of the queries (all of them when a shape has fewer): a seeded
/// subset of the answerable queries, stratified like them (by checkpoint and live generation), so
/// its cross-encoder (about 8 CPU-seconds a query, 16 to 40 under load) stays affordable while the
/// cheap arms see every query. The strong arm is reported on that subset and never gated.
pub const STRONG_QUERIES: usize = 200;
/// Rows planted by raw SQL (index lag or a failed delete): the immediate predecessor generation of
/// every superseded target queried, the revoked generation of every revoked target queried, and
/// `EXTRA_PLANT_PERCENT` of the queries' count again at each checkpoint of other capsules.
pub const EXTRA_PLANT_PERCENT: usize = 10;
/// Probe queries whose text is a planted row's own body (so the lexical mode matches it) and
/// probes whose text is a live document's body, as percent of `QUERIES`.
pub const STALE_PROBE_PERCENT: usize = 5;
pub const LIVE_PROBE_PERCENT: usize = 5;
/// Late writes of a non-live generation, as percent of `QUERIES`, per kind: a superseded
/// generation, a revoked generation, a generation ahead of the live one, a capsule never committed.
pub const LATE_SUPERSEDED_PERCENT: usize = 2;
pub const LATE_REVOKED_PERCENT: usize = 2;
pub const LATE_FUTURE_PERCENT: usize = 1;
pub const LATE_GHOST_PERCENT: usize = 1;
/// Queries repeated on the unchanged snapshot to check that a search is reproducible.
pub const REPEAT_QUERIES: usize = 20;

/// The product's query: candidates per mode before fusion, equal weights, the customary k.
pub const SEARCH_LIMIT: u32 = 40;
pub const LEXICAL_WEIGHT_PERMILLE: u32 = 1000;
pub const VECTOR_WEIGHT_PERMILLE: u32 = 1000;
pub const RANK_CONSTANT: u32 = 60;
/// `k` of recall@k and nDCG@k.
pub const K: usize = 10;

/// Coverage prerequisites (a run that misses one is vacuous) and falsifier bounds.
pub const MIN_PLANTED_STALE_ROWS: i64 = 200;
pub const MIN_REVOKED_TARGETS_QUERIED: i64 = 50;
pub const MIN_SUPERSEDED_TARGETS_QUERIED: i64 = 400;
pub const MIN_ANSWERABLE_QUERIES: i64 = 800;
pub const MIN_RETRIEVAL_HITS_PER_MODE: i64 = 1;
pub const MIN_LATE_WRITE_ATTEMPTS: i64 = 50;
pub const MIN_PROBE_QUERIES: i64 = 90;
pub const MIN_CONTROL_PLANTED_RETURNED: i64 = 50;
pub const MIN_CONTROL_VECTOR_RETURNED: i64 = 50;
/// Live-body probes whose own document the lexical mode must return (observed 50 of 50 on every
/// pilot and dry run); the hard counter `lexical_probe_violation` is 1 when fewer than
/// `LEXICAL_SELF_HIT_PERCENT` percent of the live probes were answered.
pub const MIN_LIVE_PROBE_LEXICAL_SELF_HITS: i64 = 45;
pub const LEXICAL_SELF_HIT_PERCENT: i64 = 90;
pub const MIN_REFERENCE_QUERIES: i64 = 1000;
pub const MIN_STRONG_QUERIES: i64 = 200;
pub const MIN_RECALL10_REFERENCE_PERMILLE: i64 = 900;
/// What the pilot had to confirm for the pinned query count (descriptive in a run: no rule reads it).
pub const PILOT_RECALL_PAIRED_SE_PERMILLE: i64 = 10;
pub const MAX_WORKLOAD_DIGEST_MISMATCHES: i64 = 0;
pub const MAX_STALE_HITS: i64 = 0;
pub const MAX_PLANTED_STALE_RETURNED: i64 = 0;
pub const MAX_PROJECTOR_RESIDUAL_ROWS: i64 = 0;
pub const MAX_NONDETERMINISTIC_SEARCHES: i64 = 0;
pub const MAX_LATE_WRITE_ACCEPTED: i64 = 0;
pub const MAX_LATE_WRITE_WRONG_REFUSAL: i64 = 0;
pub const MAX_PRODUCT_ERRORS: i64 = 0;
pub const MAX_LIFECYCLE_PROJECTION_MISMATCHES: i64 = 0;
pub const MAX_PRODUCT_PANICS: i64 = 0;
/// How far recall@10 may fall below the reference's before Q003 is rejected as the default (the
/// strong baseline's deficit is reported, not gated). The reference is a sanity floor.
pub const RECALL_MARGIN_PERMILLE: i64 = 30;
/// How far recall@10 may fall below the harness's own exact dense search over the live set, the
/// sensitivity control: the pilot seeds 1001 to 1003 showed no deficit (the HNSW list equalled the
/// exact search), so 30 sits clear of noise and well inside the reference's slack.
pub const RECALL_EXACT_MARGIN_PERMILLE: i64 = 30;

/// The SHA-256 of the generated workload (lifecycle, queries, plants, late writes, probes) of each
/// declared seed at the pinned shape. A run whose workload digest differs counts one
/// `workload_digest_mismatches`.
pub const WORKLOAD_DIGESTS: [(u64, &str); 5] = [
    (
        43,
        "31c46173e56d6eefb2f4e5336899f34dff452bd9f81f3d704aac623df90769dd",
    ),
    (
        71,
        "d98d3b3c9f7764d121191bea95d520bbf86d94894346cd08855f985d99f343e2",
    ),
    (
        101,
        "918a27a7337e01afb733d1f757222015423418a9e91a5a35dda0f603244c47c9",
    ),
    (
        131,
        "7cdbcfa0b102570b646fab2b03f67d145a915bdf822e89fbb1de9ec20175a2c4",
    ),
    (
        173,
        "81fd25a4599ae3175befff654daa16dfc3b0f2b9d34d04bd16f9517292e64582",
    ),
];

/// The harness's own limits (`[harness_limits]` of `config.toml`; a drift test compares them). They
/// are not thresholds: no rule reads them and no metric depends on them. A run that meets one ends
/// with exit status 2 and a typed reason, never as a finding about the product. The request
/// timeout bounds one exchange with a driver (write, flush and reply); the strong arm's
/// cross-encoder costs about 8 CPU-seconds a query, so an unloaded request is far below it, and
/// Q003-v2's completed seed 17 showed stretches of about 1,500 s a request on a loaded host. The
/// deadline bounds the whole seed: planning model 200 strong queries at 8 CPU-seconds is under an
/// hour, and the loaded host took 8.5 hours, so a day is the bound at which a run is stuck, not slow.
pub const REQUEST_TIMEOUT_SECONDS: u64 = 7_200;
pub const RUN_DEADLINE_SECONDS: u64 = 86_400;
/// How often a run prints where it is (phase, request in flight, slowest request).
pub const HEARTBEAT_SECONDS: u64 = 600;
/// The torch threads of the strong driver when `--strong-threads` is not given (the driver's own default).
pub const DEFAULT_STRONG_THREADS: u32 = 4;

/// The development seed: not a declared seed, the seed the mutation checker and the development
/// runs use (`seed` of `tests/mutations.toml`). Its workload digest is pinned (`[development_seed]`
/// of `config.toml`, compared by a drift test) so that the mutation of the workload generator is
/// seen at a seed that no experiment declares; it is Q003-v2's pin for seed 17, unchanged.
pub const DEVELOPMENT_SEED: u64 = 17;
pub const DEVELOPMENT_DIGESTS: [(u64, &str); 1] = [(
    17,
    "fdfe7b7405bf9bc09b1c53233d10a29bfa3c1ff1b67afb2f6f1e7972ac7d23cb",
)];

/// The pinned workload digest of a seed, if the seed is declared or is the development seed.
pub fn pinned_digest(seed: u64) -> Option<&'static str> {
    WORKLOAD_DIGESTS
        .iter()
        .chain(DEVELOPMENT_DIGESTS.iter())
        .find(|(pinned, _)| *pinned == seed)
        .map(|(_, digest)| *digest)
}

/// The shape of a workload: how many capsules and how many queries (the other quotas follow from
/// the percentages above).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub capsules: usize,
    pub queries: usize,
}

impl Shape {
    pub const PINNED: Shape = Shape {
        capsules: CAPSULES,
        queries: QUERIES,
    };

    /// The shape the unit tests run: small enough for a quick end-to-end run.
    pub const REDUCED: Shape = Shape {
        capsules: 90,
        queries: 36,
    };

    /// How many queries the strong arm answers.
    pub fn strong_queries(self) -> usize {
        self.queries.min(STRONG_QUERIES)
    }

    /// Whether the shape can be generated: every group of queries has capsules to draw from.
    pub fn check(self) -> Result<(), String> {
        if self.capsules < 30 || self.capsules > CAPSULES {
            return Err(format!(
                "capsules must be in 30..={CAPSULES}, got {}",
                self.capsules
            ));
        }
        if self.queries < 12 || self.queries > self.capsules {
            return Err(format!(
                "queries must be in 12..={} (one per capsule), got {}",
                self.capsules, self.queries
            ));
        }
        Ok(())
    }
}

/// What a run is: the pinned shape (evidence when run from the frozen commit) or another one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    /// `postgres-hybrid <seed>`.
    Pinned,
    /// `postgres-hybrid <queries> <seed>` (the mutation checker's form): no evidence.
    Custom(usize),
    /// The unit tests' small world.
    Reduced,
}

impl Scale {
    pub fn shape(self) -> Result<Shape, String> {
        let shape = match self {
            Scale::Pinned => Shape::PINNED,
            Scale::Custom(queries) => Shape {
                capsules: CAPSULES,
                queries,
            },
            Scale::Reduced => Shape::REDUCED,
        };
        shape.check()?;
        Ok(shape)
    }

    pub fn name(self) -> &'static str {
        match self {
            Scale::Pinned => "pinned",
            Scale::Custom(_) => "custom",
            Scale::Reduced => "reduced",
        }
    }
}
