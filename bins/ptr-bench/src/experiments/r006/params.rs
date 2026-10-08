//! The preregistered parameters of R006, compiled in.
//!
//! The constants of the first section mirror the keys of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values this harness chose where the file is silent (the corpus
//! selection, the per-seed role labels and the member names the grouped metrics print). R006 is
//! `planned`: nothing here is frozen.

use crate::experiments::layered_kv::corpus;

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R006-positional-reuse-oracle/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/runtime/R006-positional-reuse-oracle/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "positional-reuse-oracle";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it
/// from the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- positional-reuse-oracle <seed>";

// ---- one constant per key of [pinned_thresholds] ----------------------------------------------
pub const POSITION_SHIFT_TOKENS: [u64; 5] = [1, 7, 64, 513, 4096];
pub const MAX_BITWISE_MISMATCH_COUNT_PER_SHIFT: u64 = 0;
pub const MIN_REUSE_ATTEMPTS_PER_SHIFT: u64 = 1;
pub const MIN_SECONDARY_INDEX_LOOKUPS: u64 = 1;
pub const NONALIGNED_INSERTION_TOKENS: [u64; 2] = [1, 7];
pub const MIN_NONALIGNED_INSERTION_CASES_PER_SIZE: u64 = 1;
pub const MIN_UPSTREAM_REVOCATION_CASES: u64 = 1;
pub const MAX_REVOKED_UPSTREAM_PAGE_REACHABLE_COUNT: u64 = 0;
pub const TEST_MODEL_LAYERS: u64 = 1;
pub const MIN_FRESH_PREFILL_AFTER_REVOCATION: u64 = 1;
pub const MIN_REUSED_TOKENS_PER_SHIFT: u64 = 1;
pub const MIN_SECONDARY_INDEX_REUSE_CASES_PER_SHIFT: u64 = 1;
pub const MIN_POST_REVOCATION_SECONDARY_LOOKUPS_MATCHING_CONTENT: u64 = 1;
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const MIN_OLD_UPSTREAM_PAGES_SEEDED_BEFORE_REVOCATION: u64 = 1;
pub const MIN_DEAD_ORIGINAL_UPSTREAM_SHIFT_ATTEMPTS: u64 = 1;
pub const MAX_SHIFT_HITS_WITH_DEAD_ORIGINAL_UPSTREAM: u64 = 0;
pub const MIN_OMITTED_UPSTREAM_SEGMENT_CASES: u64 = 1;
pub const MAX_OMITTED_UPSTREAM_ADMISSIONS: u64 = 0;
pub const OMITTED_UPSTREAM_RULE: &str = "every original-prefix segment (identity, generation, digest) present in the requesting prefix as a multiset";
pub const MIN_UPSTREAM_SUPERSESSION_CASES: u64 = 1;
pub const MAX_SUPERSEDED_UPSTREAM_PAGE_REACHABLE_COUNT: u64 = 0;
pub const INFLIGHT_RACE_APPLICABLE: bool = true;
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 24] = [
    "seeds",
    "position_shift_tokens",
    "nonaligned_insertion_tokens",
    "test_model_layers",
    "weight_seed",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "corpus_generator_seed",
    "tiny_model_config_digest",
    "tokenizer_pin",
    "moe_topology",
    "declared_dependency_kinds",
    "production_modes",
    "derived_state_carriers",
    "rolling_window_tokens",
    "reused_segment_tokens",
    "tail_tokens",
    "nonaligned_origin_prefix_tokens",
    "harness_max_prompt_tokens",
    "harness_max_new_tokens",
    "base_checkpoint",
];
pub const MIN_SOURCE_GENERATION_MISMATCH_CASES: u64 = 1;
pub const MIN_SOURCE_DIGEST_MISMATCH_CASES: u64 = 1;
pub const MAX_SOURCE_LINEAGE_MISMATCH_ADMISSIONS: u64 = 0;
pub const FRESH_PREFILL_OUTPUT_MISMATCHES_MAX: u64 = 0;
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const MIN_POST_SUPERSESSION_SECONDARY_LOOKUPS_MATCHING_CONTENT: u64 = 1;
pub const IDENTITY_SET_CHECKS: [[&str; 2]; 1] = [[
    "invalidated_case_id_set_digest",
    "fresh_prefill_completed_case_id_set_digest",
]];
pub const INVALIDATION_COVERAGE_MATRIX_KINDS: [&str; 6] = [
    "semantic_lineage_segment",
    "memory_lineage_segment",
    "derived_lineage_segment",
    "base_model",
    "adapter_generation",
    "expert_generation",
];
pub const INVALIDATION_COVERAGE_MATRIX_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INVALIDATION_COVERAGE_MATRIX_PATHS: [&str; 4] = ["normal", "lag", "restore", "inflight"];
pub const INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION: [&str; 1] = ["fence"];
pub const INVALIDATION_COVERAGE_MATRIX_CELLS: [&str; 52] = [
    "semantic_lineage_segment_normal_revocation",
    "semantic_lineage_segment_normal_supersession",
    "semantic_lineage_segment_lag_revocation",
    "semantic_lineage_segment_lag_supersession",
    "semantic_lineage_segment_restore_revocation",
    "semantic_lineage_segment_restore_supersession",
    "semantic_lineage_segment_inflight_revocation",
    "semantic_lineage_segment_inflight_supersession",
    "memory_lineage_segment_normal_revocation",
    "memory_lineage_segment_normal_supersession",
    "memory_lineage_segment_lag_revocation",
    "memory_lineage_segment_lag_supersession",
    "memory_lineage_segment_restore_revocation",
    "memory_lineage_segment_restore_supersession",
    "memory_lineage_segment_inflight_revocation",
    "memory_lineage_segment_inflight_supersession",
    "derived_lineage_segment_normal_revocation",
    "derived_lineage_segment_normal_supersession",
    "derived_lineage_segment_lag_revocation",
    "derived_lineage_segment_lag_supersession",
    "derived_lineage_segment_restore_revocation",
    "derived_lineage_segment_restore_supersession",
    "derived_lineage_segment_inflight_revocation",
    "derived_lineage_segment_inflight_supersession",
    "base_model_normal_revocation",
    "base_model_normal_supersession",
    "base_model_lag_revocation",
    "base_model_lag_supersession",
    "base_model_restore_revocation",
    "base_model_restore_supersession",
    "base_model_inflight_revocation",
    "base_model_inflight_supersession",
    "adapter_generation_normal_revocation",
    "adapter_generation_normal_supersession",
    "adapter_generation_lag_revocation",
    "adapter_generation_lag_supersession",
    "adapter_generation_restore_revocation",
    "adapter_generation_restore_supersession",
    "adapter_generation_inflight_revocation",
    "adapter_generation_inflight_supersession",
    "expert_generation_normal_revocation",
    "expert_generation_normal_supersession",
    "expert_generation_normal_fence",
    "expert_generation_lag_revocation",
    "expert_generation_lag_supersession",
    "expert_generation_lag_fence",
    "expert_generation_restore_revocation",
    "expert_generation_restore_supersession",
    "expert_generation_restore_fence",
    "expert_generation_inflight_revocation",
    "expert_generation_inflight_supersession",
    "expert_generation_inflight_fence",
];
pub const MIN_INVALIDATION_MATRIX_CASES_PER_CELL: u64 = 1;
pub const MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL: u64 = 0;
pub const INVALIDATION_MATRIX_PATH_MEANING: &str = "normal = ordinary lookup after the invalidation; lag = lagging invalidation or index loss; restore = actual restore from snapshot or restart; inflight = in-flight race";
pub const MIN_SHIFTED_REUSE_INFLIGHT_RACE_CASES_PER_CELL: u64 = 1;
pub const MAX_SHIFTED_REUSE_INFLIGHT_OUTPUTS: u64 = 0;
pub const MAX_SHIFTED_REUSE_INFLIGHT_CACHE_INSERTS: u64 = 0;
pub const INFLIGHT_CONSUMPTION_INTERVAL: &str =
    "returned or repacked K/V still being consumed after the secondary-index lookup";
pub const INFLIGHT_RACE_REASON: &str = "consumption-time shifted-reuse races: returned or repacked K/V still being consumed after the secondary-index lookup (S4)";
pub const RESTORE_APPLICABLE: bool = true;
pub const MIN_EXACT_SHIFTED_PARITY_CASES: u64 = 1;
pub const MAX_TAINTED_EXACT_RESULTS: u64 = 0;
pub const MAX_NON_BIT_EXACT_EXACT_RESULTS: u64 = 0;
pub const SHIFTED_TAINT_SCOPE: &str =
    "multi-layer approximate M016 only; R006 is a one-layer Exact oracle";
pub const MIN_RAW_ANCESTOR_OMISSION_CASES: u64 = 1;
pub const MAX_REUSE_AFTER_RAW_ANCESTOR_REMOVED: u64 = 0;
pub const RAW_ANCESTOR_RULE: &str = "every raw ancestor (Raw, content_digest) recorded in the lineage must be present in the current request prefix, as a multiset";
pub const TINY_MODEL_CONFIG_LAYERS: u64 = 1;
pub const TINY_MODEL_CONFIG_HIDDEN_SIZE: u64 = 64;
pub const TINY_MODEL_CONFIG_NUM_HEADS: u64 = 4;
pub const TINY_MODEL_CONFIG_NUM_KV_HEADS: u64 = 2;
pub const TINY_MODEL_CONFIG_HEAD_DIM: u64 = 16;
pub const TINY_MODEL_CONFIG_VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_WEIGHT_SEED: u64 = 20240607;
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:a77e017c8ee3b56bb06a0b22512aae3a70a1b65d4ae283bf2d8bb02353029c2e";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY: &str = "none: dense tiny decoder";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const DECLARED_DEPENDENCY_KINDS: [&str; 6] = [
    "semantic",
    "memory",
    "derived",
    "base_model",
    "adapter",
    "expert",
];
pub const CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE: bool = true;
pub const CACHE_NAMESPACE_PARTITION: &str = "Exact secondary-index entries only; shifted results are Exact in the one-layer oracle and tagged separately from any approximate pages";
pub const PRODUCTION_MODES: [&str; 2] = ["Exact", "Approximate"];
pub const DERIVED_STATE_CARRIERS_SECONDARY_INDEX_ENTRY: [&str; 1] = ["mode"];
pub const ROLLING_WINDOW_TOKENS: u64 = 8;
pub const ROLLING_HASH: &str = "polynomial u64 fingerprint modulo 2^61-1 over stride-1 windows of rolling_window_tokens; a run shorter than the window is keyed once by the hash of its length and tokens; the caller verifies the actual tokens, so a collision cannot cause a hit";
pub const REUSE_UNIT: &str =
    "a whole admitted prompt segment, located at any offset by the rolling hash";
pub const EDIT_INSIDE_SEGMENT: &str = "an edit inside a segment changes its content digest: zero reuse and a typed source-digest refusal";
pub const SHIFT_CONSTRUCTION: &str = "origin run [C, tail]; baseline and shifted run [filler, C, tail] where filler is a Raw segment of exactly s tokens inserted upstream of C (s in position_shift_tokens), so C keeps its own lineage (kind, identity, generation, digest) while the ordered prefix lineage differs";
pub const SHIFT_FILLER_KIND: &str = "Raw";
pub const REUSED_SEGMENT_TOKENS: u64 = 48;
pub const TAIL_TOKENS: u64 = 16;
pub const NONALIGNED_INSERTION_CONSTRUCTION: &str = "origin run [A, C, tail] with A a Raw segment of nonaligned_origin_prefix_tokens tokens; the request inserts a Raw segment of n tokens (n in nonaligned_insertion_tokens) between A and C, so C starts at nonaligned_origin_prefix_tokens + n; neither start is a multiple of the 16-token page";
pub const NONALIGNED_ORIGIN_PREFIX_TOKENS: u64 = 24;
pub const SPLICE_RULE: &str = "a spliced span never includes the last prompt token (the stack needs that token's logits); a span with no whole-segment match is recomputed, never partly reused";
pub const HARNESS_MAX_PROMPT_TOKENS: u64 = 4352;
pub const HARNESS_MAX_NEW_TOKENS: u64 = 8;
pub const HARNESS_LIMITS_RULE: &str = "max_prompt_tokens must be at least max(position_shift_tokens) + reused_segment_tokens + tail_tokens + harness_max_new_tokens; raised via ReferenceKvLimits in the harness only";
pub const POSITIONAL_REUSE_DEFAULT: &str =
    "opt-in per backend and off by default, so the frozen R005 and R007 outputs do not change";
pub const STALENESS_MEASUREMENT: &str = "structural";
pub const STALE_HIT_DEFINITION: &str = "a secondary-index lookup that returned or admitted a run with a dead dependency: revoked, superseded (generation replaced or edited), never committed at the ledger anchor, or fenced";
pub const ESCAPED_OUTPUT_DEFINITION: &str = "tokens emitted or cache inserts made after the event step by a session whose dependency was invalidated";
pub const REACHABILITY_DEFINITION: &str = "revoked_upstream_page_reachable_count and superseded_upstream_page_reachable_count count lookups after the event that returned the page, whether or not the lookup was admitted";
pub const NEGATIVE_CONTROL_POSTROPE: &str =
    "a PostRope stack: the same shifted reuse must mismatch bitwise, else the run is vacuous";
pub const NEGATIVE_CONTROL_LIVENESS_BYPASS: &str = "a liveness-bypass fault mode of the secondary index: the reachability and dead-original admission counters must trip, else the run is vacuous";
pub const NEGATIVE_CONTROL_PRIMARY_ONLY: &str = "a primary-only run on the shifted content gets 0 reused tokens, showing the secondary index is what provides reuse";
pub const MULTISET_SCOPE: &str = "whole_request";
pub const MULTISET_SCOPE_RULE: &str = "a segment of a run's original prefix is present iff the whole request holds at least as many segments with the same (kind, identity, generation, digest); the position in the request does not matter (reordering allowed)";
pub const REFUSAL_REASONS_IN_CHECK_ORDER: [&str; 8] = [
    "SourceKindMismatch",
    "SourceIdentityMismatch",
    "SourceGenerationMismatch",
    "SourceDigestMismatch",
    "OriginalUpstreamDead",
    "BindingDependencyDead",
    "OriginalUpstreamOmitted",
    "RawAncestorOmitted",
];
pub const RAW_ANCESTOR_SCOPE: &str = "direct Raw segments of the requesting prefix (kind Raw, content digest) as a multiset; a Derived segment carries no ancestor list, so ancestors reached only through a derivation hop are not covered";
pub const EVIDENCE_LABELS_SEMANTIC_LINEAGE_SEGMENT: &str = "session_lookup";
pub const EVIDENCE_LABELS_MEMORY_LINEAGE_SEGMENT: &str = "session_lookup";
pub const EVIDENCE_LABELS_DERIVED_LINEAGE_SEGMENT: &str = "session_lookup";
pub const EVIDENCE_LABELS_BASE_MODEL: &str = "session_recorded_dependency";
pub const EVIDENCE_LABELS_ADAPTER_GENERATION: &str = "declared_dependency";
pub const EVIDENCE_LABELS_EXPERT_GENERATION: &str = "declared_dependency";
pub const DECLARED_DEPENDENCY_RULE: &str = "adapter and expert dependencies are declared by the caller with real liveness oracles (an adapter capsule; real expert records including the fence epoch); the dense one-layer model has no adapter or expert influence";
pub const INFLIGHT_RACE_REQUIRED_KINDS: [&str; 3] = [
    "semantic_lineage_segment",
    "memory_lineage_segment",
    "base_model",
];
pub const INFLIGHT_VARIANTS: [&str; 2] =
    ["event_before_splice", "event_after_splice_before_decode"];
pub const SNAPSHOT_ROWS: &str = "KV snapshots carry no K/V rows; restore re-attaches rows by an idempotent re-insert of the same content under a live oracle, with a rows-digest check (a mismatch is a typed error)";
pub const BASE_CHECKPOINT: &str = "none: seeded random";
pub const COMPARISON_RULE: &str = "bitwise on f32 bit patterns (to_bits), no tolerance";
pub const REUSE_LATENCY_SCOPE: &str =
    "reuse_latency_ms is timing only, never read by a rule; no wall-clock claim";
pub const OWNER_DECISION_2026_10_08_1_GRANULARITY: &str = "Granularity: rolling hash over sub-page spans in the secondary index, so non-aligned shifts (1- and 7-token insertions) and all preregistered shifts are reusable; a repack step assembles the reused PreRope rows at the new positions.";
pub const OWNER_DECISION_2026_10_08_2_MATRIX: &str = "Matrix: reuse the R005/R010 matrix machinery (layered_kv shared helpers; same cell semantics normal, lag, restore, in-flight); R005's frozen files stay untouched.";
pub const OWNER_DECISION_2026_10_08_3_REFUSALS: &str = "Refusals: one typed error per refusal reason of the secondary index (e.g. original upstream dead, upstream segment omitted, raw-ancestor multiset differs, own lineage differs), so criteria count per reason and negative cases fail for the named reason.";
pub const OWNER_DECISION_2026_10_08_4_GO: &str = "Go: implement R006 fully after R007 (size L).";
pub const OWNER_DECISION_2026_10_08_5_STALENESS: &str = "(2026-10-08) Staleness is measured structurally: a stale hit = a lookup that returned (or admitted) a dead run; an escaped output = tokens emitted or cache inserts after the event step. Stated in the preregistration. Negative controls: (a) a PostRope stack where the same shifted reuse must mismatch bitwise; (b) a liveness-bypass fault mode under which the reachability / dead-original admission counters must trip; (c) primary-only run gets 0 reused tokens.";
pub const OWNER_DECISION_2026_10_08_6_MULTISET: &str = "Multiset presence of original-prefix segments is checked over the whole request (reordering allowed).";
pub const OWNER_DECISION_2026_10_08_7_REUSE_UNIT: &str = "Reuse unit = a whole admitted segment, located at any offset by the rolling hash (W pinned in config); an edit inside a segment gives zero reuse and a typed refusal.";
pub const OWNER_DECISION_2026_10_08_8_DECLARED_AND_SNAPSHOT: &str = "Adapter and expert cells run as declared dependencies with real liveness oracles (adapter capsule; real expert records incl. fence), labelled evidence = declared_dependency. KV snapshots carry no rows; restore re-attaches rows by idempotent re-insert with a rows-digest check.";
pub const OWNER_DECISION_2026_10_08_9_CONFIG: &str = "(head) config.toml additions before the freeze are allowed (R006 is planned): rolling_window_tokens = 8, the shift construction (Raw filler of s tokens upstream), harness limits, evidence labels. Positional reuse is opt-in per backend and off by default so frozen R005/R007 outputs do not change.";

// ---- chosen by this harness (not in the config; echoed in every result and in the text) ------
/// The domain of the corpus selection digest.
pub const SELECTION_DOMAIN: &str = "ptr.r006.corpus-selection.v1";
/// Sequences of the corpus selection: sequence `i` for `i` in `0..SELECTION_SEQUENCES`.
pub const SELECTION_SEQUENCES: usize = 8;
/// Tokens of every selection sequence: the longest role (the 4096-token filler).
pub const SELECTION_SEQUENCE_TOKENS: usize = 4096;
/// The pinned digest of the corpus selection (`selection_digest` recomputes it).
pub const SELECTION_DIGEST: &str =
    "sha256:3bcd8736b20516454cae2f5007f24201c92756c9a2643e7f126adab6144ab066";
/// The domain of the seeded shuffle that decides which corpus sequence fills which role.
pub const ROLE_ORDER_LABEL: u64 = 60;
/// The per-seed role, echoed in the result.
pub const PER_SEED_ROLE: &str = "the seed picks which corpus sequence fills the filler, reused segment, tail, inserted and upstream roles; the corpus and the weights are fixed by their pinned seeds";
/// The upstream segment `U` of the revocation and supersession cases and its generation.
pub const UPSTREAM_IDENTITY: &str = "upstream:r006-u";
pub const UPSTREAM_GENERATION: u64 = 1;
/// Tokens of one KV page and of one catalog chunk (the unaligned starts are measured against it).
pub const PAGE_TOKENS: u64 = 16;

/// The corpus selection digest: the sequences in index order under [`SELECTION_DOMAIN`].
pub fn selection_digest() -> String {
    let sequences: Vec<Vec<u32>> = (0..SELECTION_SEQUENCES as u64)
        .map(|index| corpus::sequence(CORPUS_GENERATOR_SEED, index, SELECTION_SEQUENCE_TOKENS))
        .collect();
    corpus::selection_digest(SELECTION_DOMAIN, &sequences)
}

/// The seeds the manifest declares, read from its `seeds = [...]` line.
pub fn declared_seeds() -> Vec<u64> {
    EXPERIMENT_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("seeds"))
        .and_then(|line| line.split_once('['))
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| {
            inner
                .split(',')
                .filter_map(|seed| seed.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// The member of a per-shift metric (`reuse_attempts_per_shift.shift_64`).
pub fn shift_member(shift: u64) -> String {
    format!("shift_{shift}")
}

/// The member of a per-size metric (`nonaligned_insertion_cases.tokens_7`).
pub fn insertion_member(tokens: u64) -> String {
    format!("tokens_{tokens}")
}

/// The matrix cell name of a dependency kind, path and invalidation type.
pub fn cell_name(kind: &str, path: &str, ty: &str) -> String {
    format!("{kind}_{path}_{ty}")
}

/// The member of `shifted_reuse_inflight_race_cases` for a kind and an invalidation type (one
/// segment, because the grouped printer keeps a member name flat and a dot would be a path).
pub fn inflight_member(kind: &str, ty: &str) -> String {
    format!("{kind}_{ty}")
}

/// The in-flight members the manifest requires: `{semantic lineage segment, memory lineage
/// segment, base model} x {revocation, supersession}`.
pub fn inflight_members() -> Vec<String> {
    let mut members = Vec::new();
    for kind in INFLIGHT_RACE_REQUIRED_KINDS {
        for ty in INVALIDATION_TYPES {
            members.push(inflight_member(kind, ty));
        }
    }
    members
}

/// The tokens of the prompt `[filler of s][C][tail]` of a shift case.
pub fn shifted_prompt_tokens(shift: u64) -> u64 {
    shift + REUSED_SEGMENT_TOKENS + TAIL_TOKENS
}

// ---- the compiled table, as the drift test compares it -------------------------------------

/// A leaf of the flattened config, for the drift test.
#[derive(Clone, Debug, PartialEq)]
enum Leaf {
    Int(i64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    Ints(Vec<i64>),
}

fn int(value: u64) -> Leaf {
    Leaf::Int(i64::try_from(value).expect("a pinned value fits"))
}

fn text(value: &str) -> Leaf {
    Leaf::Str(value.to_string())
}

fn strs(values: &[&str]) -> Leaf {
    Leaf::Strs(values.iter().map(|value| (*value).to_string()).collect())
}

/// Every key of `[pinned_thresholds]` as a compiled constant (a table flattens to its dotted
/// path, a list of lists to its rows joined with `|`).
fn compiled() -> Vec<(&'static str, Leaf)> {
    vec![
        (
            "position_shift_tokens",
            Leaf::Ints(
                POSITION_SHIFT_TOKENS
                    .iter()
                    .map(|value| *value as i64)
                    .collect(),
            ),
        ),
        (
            "max_bitwise_mismatch_count_per_shift",
            int(MAX_BITWISE_MISMATCH_COUNT_PER_SHIFT),
        ),
        (
            "min_reuse_attempts_per_shift",
            int(MIN_REUSE_ATTEMPTS_PER_SHIFT),
        ),
        (
            "min_secondary_index_lookups",
            int(MIN_SECONDARY_INDEX_LOOKUPS),
        ),
        (
            "nonaligned_insertion_tokens",
            Leaf::Ints(
                NONALIGNED_INSERTION_TOKENS
                    .iter()
                    .map(|value| *value as i64)
                    .collect(),
            ),
        ),
        (
            "min_nonaligned_insertion_cases_per_size",
            int(MIN_NONALIGNED_INSERTION_CASES_PER_SIZE),
        ),
        (
            "min_upstream_revocation_cases",
            int(MIN_UPSTREAM_REVOCATION_CASES),
        ),
        (
            "max_revoked_upstream_page_reachable_count",
            int(MAX_REVOKED_UPSTREAM_PAGE_REACHABLE_COUNT),
        ),
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
        (
            "min_fresh_prefill_after_revocation",
            int(MIN_FRESH_PREFILL_AFTER_REVOCATION),
        ),
        (
            "min_reused_tokens_per_shift",
            int(MIN_REUSED_TOKENS_PER_SHIFT),
        ),
        (
            "min_secondary_index_reuse_cases_per_shift",
            int(MIN_SECONDARY_INDEX_REUSE_CASES_PER_SHIFT),
        ),
        (
            "min_post_revocation_secondary_lookups_matching_content",
            int(MIN_POST_REVOCATION_SECONDARY_LOOKUPS_MATCHING_CONTENT),
        ),
        ("seed_aggregation", text(SEED_AGGREGATION)),
        ("weights", text(WEIGHTS)),
        ("weight_seed", int(WEIGHT_SEED)),
        ("hidden_size", int(HIDDEN_SIZE)),
        ("num_heads", int(NUM_HEADS)),
        ("num_kv_heads", int(NUM_KV_HEADS)),
        ("head_dim", int(HEAD_DIM)),
        ("vocab_size", int(VOCAB_SIZE)),
        (
            "min_old_upstream_pages_seeded_before_revocation",
            int(MIN_OLD_UPSTREAM_PAGES_SEEDED_BEFORE_REVOCATION),
        ),
        (
            "min_dead_original_upstream_shift_attempts",
            int(MIN_DEAD_ORIGINAL_UPSTREAM_SHIFT_ATTEMPTS),
        ),
        (
            "max_shift_hits_with_dead_original_upstream",
            int(MAX_SHIFT_HITS_WITH_DEAD_ORIGINAL_UPSTREAM),
        ),
        (
            "min_omitted_upstream_segment_cases",
            int(MIN_OMITTED_UPSTREAM_SEGMENT_CASES),
        ),
        (
            "max_omitted_upstream_admissions",
            int(MAX_OMITTED_UPSTREAM_ADMISSIONS),
        ),
        ("omitted_upstream_rule", text(OMITTED_UPSTREAM_RULE)),
        (
            "min_upstream_supersession_cases",
            int(MIN_UPSTREAM_SUPERSESSION_CASES),
        ),
        (
            "max_superseded_upstream_page_reachable_count",
            int(MAX_SUPERSEDED_UPSTREAM_PAGE_REACHABLE_COUNT),
        ),
        (
            "inflight_race_applicable",
            Leaf::Bool(INFLIGHT_RACE_APPLICABLE),
        ),
        ("corpus", text(CORPUS)),
        ("corpus_generator_seed", int(CORPUS_GENERATOR_SEED)),
        ("corpus_selection_rule", text(CORPUS_SELECTION_RULE)),
        (
            "corpus_selection_digest_recorded_in_run_json",
            Leaf::Bool(CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON),
        ),
        (
            "max_selection_digest_mismatches",
            int(MAX_SELECTION_DIGEST_MISMATCHES),
        ),
        ("max_procedure_mismatches", int(MAX_PROCEDURE_MISMATCHES)),
        (
            "procedure_parameters_recorded_per_cell",
            strs(&PROCEDURE_PARAMETERS_RECORDED_PER_CELL),
        ),
        (
            "min_source_generation_mismatch_cases",
            int(MIN_SOURCE_GENERATION_MISMATCH_CASES),
        ),
        (
            "min_source_digest_mismatch_cases",
            int(MIN_SOURCE_DIGEST_MISMATCH_CASES),
        ),
        (
            "max_source_lineage_mismatch_admissions",
            int(MAX_SOURCE_LINEAGE_MISMATCH_ADMISSIONS),
        ),
        (
            "fresh_prefill_output_mismatches_max",
            int(FRESH_PREFILL_OUTPUT_MISMATCHES_MAX),
        ),
        ("invalidation_types", strs(&INVALIDATION_TYPES)),
        (
            "min_post_supersession_secondary_lookups_matching_content",
            int(MIN_POST_SUPERSESSION_SECONDARY_LOOKUPS_MATCHING_CONTENT),
        ),
        (
            "identity_set_checks",
            Leaf::Strs(
                IDENTITY_SET_CHECKS
                    .iter()
                    .map(|row| row.join("|"))
                    .collect(),
            ),
        ),
        (
            "invalidation_coverage_matrix.kinds",
            strs(&INVALIDATION_COVERAGE_MATRIX_KINDS),
        ),
        (
            "invalidation_coverage_matrix.types",
            strs(&INVALIDATION_COVERAGE_MATRIX_TYPES),
        ),
        (
            "invalidation_coverage_matrix.paths",
            strs(&INVALIDATION_COVERAGE_MATRIX_PATHS),
        ),
        (
            "invalidation_coverage_matrix.extra_types.expert_generation",
            strs(&INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION),
        ),
        (
            "invalidation_coverage_matrix.cells",
            strs(&INVALIDATION_COVERAGE_MATRIX_CELLS),
        ),
        (
            "min_invalidation_matrix_cases_per_cell",
            int(MIN_INVALIDATION_MATRIX_CASES_PER_CELL),
        ),
        (
            "max_invalidation_matrix_stale_hits_per_cell",
            int(MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL),
        ),
        (
            "invalidation_matrix_path_meaning",
            text(INVALIDATION_MATRIX_PATH_MEANING),
        ),
        (
            "min_shifted_reuse_inflight_race_cases_per_cell",
            int(MIN_SHIFTED_REUSE_INFLIGHT_RACE_CASES_PER_CELL),
        ),
        (
            "max_shifted_reuse_inflight_outputs",
            int(MAX_SHIFTED_REUSE_INFLIGHT_OUTPUTS),
        ),
        (
            "max_shifted_reuse_inflight_cache_inserts",
            int(MAX_SHIFTED_REUSE_INFLIGHT_CACHE_INSERTS),
        ),
        (
            "inflight_consumption_interval",
            text(INFLIGHT_CONSUMPTION_INTERVAL),
        ),
        ("inflight_race_reason", text(INFLIGHT_RACE_REASON)),
        ("restore_applicable", Leaf::Bool(RESTORE_APPLICABLE)),
        (
            "min_exact_shifted_parity_cases",
            int(MIN_EXACT_SHIFTED_PARITY_CASES),
        ),
        ("max_tainted_exact_results", int(MAX_TAINTED_EXACT_RESULTS)),
        (
            "max_non_bit_exact_exact_results",
            int(MAX_NON_BIT_EXACT_EXACT_RESULTS),
        ),
        ("shifted_taint_scope", text(SHIFTED_TAINT_SCOPE)),
        (
            "min_raw_ancestor_omission_cases",
            int(MIN_RAW_ANCESTOR_OMISSION_CASES),
        ),
        (
            "max_reuse_after_raw_ancestor_removed",
            int(MAX_REUSE_AFTER_RAW_ANCESTOR_REMOVED),
        ),
        ("raw_ancestor_rule", text(RAW_ANCESTOR_RULE)),
        ("tiny_model_config.layers", int(TINY_MODEL_CONFIG_LAYERS)),
        (
            "tiny_model_config.hidden_size",
            int(TINY_MODEL_CONFIG_HIDDEN_SIZE),
        ),
        (
            "tiny_model_config.num_heads",
            int(TINY_MODEL_CONFIG_NUM_HEADS),
        ),
        (
            "tiny_model_config.num_kv_heads",
            int(TINY_MODEL_CONFIG_NUM_KV_HEADS),
        ),
        (
            "tiny_model_config.head_dim",
            int(TINY_MODEL_CONFIG_HEAD_DIM),
        ),
        (
            "tiny_model_config.vocab_size",
            int(TINY_MODEL_CONFIG_VOCAB_SIZE),
        ),
        (
            "tiny_model_config.weight_seed",
            int(TINY_MODEL_CONFIG_WEIGHT_SEED),
        ),
        ("tiny_model_config_digest", text(TINY_MODEL_CONFIG_DIGEST)),
        (
            "tiny_model_config_digest_rule",
            text(TINY_MODEL_CONFIG_DIGEST_RULE),
        ),
        ("tokenizer_pin", text(TOKENIZER_PIN)),
        ("moe_topology", text(MOE_TOPOLOGY)),
        (
            "model_pins_recorded_in_run_json",
            Leaf::Bool(MODEL_PINS_RECORDED_IN_RUN_JSON),
        ),
        ("max_model_pin_mismatches", int(MAX_MODEL_PIN_MISMATCHES)),
        (
            "declared_dependency_kinds",
            strs(&DECLARED_DEPENDENCY_KINDS),
        ),
        (
            "cache_namespace_partitioned_by_production_mode",
            Leaf::Bool(CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE),
        ),
        ("cache_namespace_partition", text(CACHE_NAMESPACE_PARTITION)),
        ("production_modes", strs(&PRODUCTION_MODES)),
        (
            "derived_state_carriers.secondary_index_entry",
            strs(&DERIVED_STATE_CARRIERS_SECONDARY_INDEX_ENTRY),
        ),
        ("rolling_window_tokens", int(ROLLING_WINDOW_TOKENS)),
        ("rolling_hash", text(ROLLING_HASH)),
        ("reuse_unit", text(REUSE_UNIT)),
        ("edit_inside_segment", text(EDIT_INSIDE_SEGMENT)),
        ("shift_construction", text(SHIFT_CONSTRUCTION)),
        ("shift_filler_kind", text(SHIFT_FILLER_KIND)),
        ("reused_segment_tokens", int(REUSED_SEGMENT_TOKENS)),
        ("tail_tokens", int(TAIL_TOKENS)),
        (
            "nonaligned_insertion_construction",
            text(NONALIGNED_INSERTION_CONSTRUCTION),
        ),
        (
            "nonaligned_origin_prefix_tokens",
            int(NONALIGNED_ORIGIN_PREFIX_TOKENS),
        ),
        ("splice_rule", text(SPLICE_RULE)),
        ("harness_max_prompt_tokens", int(HARNESS_MAX_PROMPT_TOKENS)),
        ("harness_max_new_tokens", int(HARNESS_MAX_NEW_TOKENS)),
        ("harness_limits_rule", text(HARNESS_LIMITS_RULE)),
        ("positional_reuse_default", text(POSITIONAL_REUSE_DEFAULT)),
        ("staleness_measurement", text(STALENESS_MEASUREMENT)),
        ("stale_hit_definition", text(STALE_HIT_DEFINITION)),
        ("escaped_output_definition", text(ESCAPED_OUTPUT_DEFINITION)),
        ("reachability_definition", text(REACHABILITY_DEFINITION)),
        ("negative_control_postrope", text(NEGATIVE_CONTROL_POSTROPE)),
        (
            "negative_control_liveness_bypass",
            text(NEGATIVE_CONTROL_LIVENESS_BYPASS),
        ),
        (
            "negative_control_primary_only",
            text(NEGATIVE_CONTROL_PRIMARY_ONLY),
        ),
        ("multiset_scope", text(MULTISET_SCOPE)),
        ("multiset_scope_rule", text(MULTISET_SCOPE_RULE)),
        (
            "refusal_reasons_in_check_order",
            strs(&REFUSAL_REASONS_IN_CHECK_ORDER),
        ),
        ("raw_ancestor_scope", text(RAW_ANCESTOR_SCOPE)),
        (
            "evidence_labels.semantic_lineage_segment",
            text(EVIDENCE_LABELS_SEMANTIC_LINEAGE_SEGMENT),
        ),
        (
            "evidence_labels.memory_lineage_segment",
            text(EVIDENCE_LABELS_MEMORY_LINEAGE_SEGMENT),
        ),
        (
            "evidence_labels.derived_lineage_segment",
            text(EVIDENCE_LABELS_DERIVED_LINEAGE_SEGMENT),
        ),
        (
            "evidence_labels.base_model",
            text(EVIDENCE_LABELS_BASE_MODEL),
        ),
        (
            "evidence_labels.adapter_generation",
            text(EVIDENCE_LABELS_ADAPTER_GENERATION),
        ),
        (
            "evidence_labels.expert_generation",
            text(EVIDENCE_LABELS_EXPERT_GENERATION),
        ),
        ("declared_dependency_rule", text(DECLARED_DEPENDENCY_RULE)),
        (
            "inflight_race_required_kinds",
            strs(&INFLIGHT_RACE_REQUIRED_KINDS),
        ),
        ("inflight_variants", strs(&INFLIGHT_VARIANTS)),
        ("snapshot_rows", text(SNAPSHOT_ROWS)),
        ("base_checkpoint", text(BASE_CHECKPOINT)),
        ("comparison_rule", text(COMPARISON_RULE)),
        ("reuse_latency_scope", text(REUSE_LATENCY_SCOPE)),
        (
            "owner_decision_2026_10_08_1_granularity",
            text(OWNER_DECISION_2026_10_08_1_GRANULARITY),
        ),
        (
            "owner_decision_2026_10_08_2_matrix",
            text(OWNER_DECISION_2026_10_08_2_MATRIX),
        ),
        (
            "owner_decision_2026_10_08_3_refusals",
            text(OWNER_DECISION_2026_10_08_3_REFUSALS),
        ),
        (
            "owner_decision_2026_10_08_4_go",
            text(OWNER_DECISION_2026_10_08_4_GO),
        ),
        (
            "owner_decision_2026_10_08_5_staleness",
            text(OWNER_DECISION_2026_10_08_5_STALENESS),
        ),
        (
            "owner_decision_2026_10_08_6_multiset",
            text(OWNER_DECISION_2026_10_08_6_MULTISET),
        ),
        (
            "owner_decision_2026_10_08_7_reuse_unit",
            text(OWNER_DECISION_2026_10_08_7_REUSE_UNIT),
        ),
        (
            "owner_decision_2026_10_08_8_declared_and_snapshot",
            text(OWNER_DECISION_2026_10_08_8_DECLARED_AND_SNAPSHOT),
        ),
        (
            "owner_decision_2026_10_08_9_config",
            text(OWNER_DECISION_2026_10_08_9_CONFIG),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The leaves of a parsed TOML value, flattened as `compiled` flattens them.
    fn flatten(prefix: &str, value: &toml::Value, out: &mut BTreeMap<String, Leaf>) {
        match value {
            toml::Value::Table(table) => {
                for (key, inner) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    flatten(&path, inner, out);
                }
            }
            toml::Value::Integer(number) => {
                out.insert(prefix.to_string(), Leaf::Int(*number));
            }
            toml::Value::Boolean(flag) => {
                out.insert(prefix.to_string(), Leaf::Bool(*flag));
            }
            toml::Value::String(item) => {
                out.insert(prefix.to_string(), Leaf::Str(item.clone()));
            }
            toml::Value::Array(items) => {
                let leaf = if items.iter().all(toml::Value::is_integer) {
                    Leaf::Ints(items.iter().filter_map(toml::Value::as_integer).collect())
                } else {
                    Leaf::Strs(
                        items
                            .iter()
                            .map(|item| match item {
                                toml::Value::String(text) => text.clone(),
                                toml::Value::Array(inner) => inner
                                    .iter()
                                    .map(|part| part.as_str().expect("a string").to_string())
                                    .collect::<Vec<_>>()
                                    .join("|"),
                                other => panic!("{prefix}: unexpected list element {other:?}"),
                            })
                            .collect(),
                    )
                };
                out.insert(prefix.to_string(), leaf);
            }
            other => panic!("{prefix}: unexpected value {other:?}"),
        }
    }

    fn file_table(config: &str) -> BTreeMap<String, Leaf> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let thresholds = parsed
            .get("pinned_thresholds")
            .expect("[pinned_thresholds]")
            .clone();
        let mut out = BTreeMap::new();
        flatten("", &thresholds, &mut out);
        out
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each.
    fn drift(config: &str) -> Vec<String> {
        let file = file_table(config);
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
        let changed =
            CONFIG_TOML.replace("rolling_window_tokens = 8\n", "rolling_window_tokens = 9\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["rolling_window_tokens: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("rolling_window_tokens = 8\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(
            drift(&removed),
            ["rolling_window_tokens: no key in the file"]
        );
        let shifts = CONFIG_TOML.replace(
            "position_shift_tokens = [1, 7, 64, 513, 4096]",
            "position_shift_tokens = [1, 7, 64, 513, 4095]",
        );
        assert_ne!(shifts, CONFIG_TOML);
        assert_eq!(drift(&shifts), ["position_shift_tokens: drifted"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_matrix_cells_are_the_full_cross_product_of_the_configured_axes() {
        let mut expected = Vec::new();
        for kind in INVALIDATION_COVERAGE_MATRIX_KINDS {
            for path in INVALIDATION_COVERAGE_MATRIX_PATHS {
                for ty in INVALIDATION_COVERAGE_MATRIX_TYPES {
                    expected.push(cell_name(kind, path, ty));
                }
                if kind == "expert_generation" {
                    for ty in INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION {
                        expected.push(cell_name(kind, path, ty));
                    }
                }
            }
        }
        let mut cells: Vec<String> = INVALIDATION_COVERAGE_MATRIX_CELLS
            .iter()
            .map(|cell| (*cell).to_string())
            .collect();
        expected.sort();
        cells.sort();
        assert_eq!(cells, expected);
        assert_eq!(cells.len(), 52);
        // The file's axes are the ones this table was built from.
        assert_eq!(INVALIDATION_TYPES, INVALIDATION_COVERAGE_MATRIX_TYPES);
        assert_eq!(
            DECLARED_DEPENDENCY_KINDS.len(),
            INVALIDATION_COVERAGE_MATRIX_KINDS.len()
        );
    }

    #[test]
    fn every_matrix_kind_has_an_evidence_label_and_the_declared_ones_are_the_adapter_and_expert() {
        let labels = [
            (
                "semantic_lineage_segment",
                EVIDENCE_LABELS_SEMANTIC_LINEAGE_SEGMENT,
            ),
            (
                "memory_lineage_segment",
                EVIDENCE_LABELS_MEMORY_LINEAGE_SEGMENT,
            ),
            (
                "derived_lineage_segment",
                EVIDENCE_LABELS_DERIVED_LINEAGE_SEGMENT,
            ),
            ("base_model", EVIDENCE_LABELS_BASE_MODEL),
            ("adapter_generation", EVIDENCE_LABELS_ADAPTER_GENERATION),
            ("expert_generation", EVIDENCE_LABELS_EXPERT_GENERATION),
        ];
        let kinds: Vec<&str> = labels.iter().map(|(kind, _)| *kind).collect();
        assert_eq!(kinds, INVALIDATION_COVERAGE_MATRIX_KINDS);
        let declared: Vec<&str> = labels
            .iter()
            .filter(|(_, label)| *label == "declared_dependency")
            .map(|(kind, _)| *kind)
            .collect();
        assert_eq!(declared, ["adapter_generation", "expert_generation"]);
    }

    #[test]
    fn the_inflight_members_are_the_six_the_manifest_requires() {
        assert_eq!(
            inflight_members(),
            [
                "semantic_lineage_segment_revocation",
                "semantic_lineage_segment_supersession",
                "memory_lineage_segment_revocation",
                "memory_lineage_segment_supersession",
                "base_model_revocation",
                "base_model_supersession",
            ]
        );
        for kind in INFLIGHT_RACE_REQUIRED_KINDS {
            assert!(INVALIDATION_COVERAGE_MATRIX_KINDS.contains(&kind));
            assert_ne!(
                evidence_label(kind),
                "declared_dependency",
                "a declared dependency has no session to race"
            );
        }
    }

    fn evidence_label(kind: &str) -> &'static str {
        match kind {
            "semantic_lineage_segment" => EVIDENCE_LABELS_SEMANTIC_LINEAGE_SEGMENT,
            "memory_lineage_segment" => EVIDENCE_LABELS_MEMORY_LINEAGE_SEGMENT,
            "derived_lineage_segment" => EVIDENCE_LABELS_DERIVED_LINEAGE_SEGMENT,
            "base_model" => EVIDENCE_LABELS_BASE_MODEL,
            "adapter_generation" => EVIDENCE_LABELS_ADAPTER_GENERATION,
            "expert_generation" => EVIDENCE_LABELS_EXPERT_GENERATION,
            other => panic!("unknown kind {other}"),
        }
    }

    #[test]
    fn the_shift_and_insertion_constructions_fit_the_pinned_limits() {
        let longest = POSITION_SHIFT_TOKENS.iter().max().copied().unwrap();
        assert_eq!(longest, 4096);
        assert!(
            HARNESS_MAX_PROMPT_TOKENS
                >= longest + REUSED_SEGMENT_TOKENS + TAIL_TOKENS + HARNESS_MAX_NEW_TOKENS,
            "harness_limits_rule"
        );
        assert_eq!(shifted_prompt_tokens(4096), 4096 + 48 + 16);
        // The reused segment is at least one rolling window, so it is keyed by windows, and the
        // tail is at least one token, so a splice never has to include the last prompt token.
        const { assert!(REUSED_SEGMENT_TOKENS >= ROLLING_WINDOW_TOKENS) };
        const { assert!(TAIL_TOKENS >= 1) };
        // The unaligned insertion: neither the origin start nor any shifted start is a page.
        for n in NONALIGNED_INSERTION_TOKENS {
            assert_ne!(NONALIGNED_ORIGIN_PREFIX_TOKENS % PAGE_TOKENS, 0);
            assert_ne!(
                (NONALIGNED_ORIGIN_PREFIX_TOKENS + n) % PAGE_TOKENS,
                0,
                "{n}"
            );
        }
        // The two inserted sizes are the 1- and 7-token shifts of the shift list.
        for n in NONALIGNED_INSERTION_TOKENS {
            assert!(POSITION_SHIFT_TOKENS.contains(&n));
        }
        // The 7-token insertion is shorter than the window, the reused segment is not.
        assert!(NONALIGNED_INSERTION_TOKENS
            .iter()
            .all(|n| *n <= ROLLING_WINDOW_TOKENS));
        assert_eq!(
            ptr_runtime::ReferenceKvLimits::default().max_prompt_tokens,
            256,
            "the crate default stays 256 (the harness raises it, owner decision 9)"
        );
        assert_eq!(
            HARNESS_MAX_NEW_TOKENS as usize,
            ptr_runtime::ReferenceKvLimits::default().max_new_tokens
        );
        assert_eq!(PAGE_TOKENS as usize, ptr_runtime::KV_CHUNK_TOKENS);
    }

    #[test]
    fn the_refusal_reasons_are_eight_distinct_names_in_the_documented_order() {
        let unique: std::collections::BTreeSet<&str> =
            REFUSAL_REASONS_IN_CHECK_ORDER.iter().copied().collect();
        assert_eq!(unique.len(), REFUSAL_REASONS_IN_CHECK_ORDER.len());
        assert_eq!(REFUSAL_REASONS_IN_CHECK_ORDER[0], "SourceKindMismatch");
        // A dead upstream is reported before an omitted one (a superseded U is dead, not omitted).
        let position = |name: &str| {
            REFUSAL_REASONS_IN_CHECK_ORDER
                .iter()
                .position(|reason| *reason == name)
                .unwrap()
        };
        assert!(position("OriginalUpstreamDead") < position("OriginalUpstreamOmitted"));
        assert!(position("SourceDigestMismatch") < position("OriginalUpstreamDead"));
        assert!(position("OriginalUpstreamOmitted") < position("RawAncestorOmitted"));
    }

    #[test]
    fn the_pinned_selection_digest_is_what_the_corpus_generator_produces() {
        assert_eq!(selection_digest(), SELECTION_DIGEST);
        assert!(SELECTION_SEQUENCE_TOKENS as u64 >= *POSITION_SHIFT_TOKENS.iter().max().unwrap());
    }

    #[test]
    fn the_member_names_are_flat_and_stable() {
        assert_eq!(shift_member(513), "shift_513");
        assert_eq!(insertion_member(7), "tokens_7");
        assert_eq!(
            cell_name("expert_generation", "lag", "fence"),
            "expert_generation_lag_fence"
        );
        for member in POSITION_SHIFT_TOKENS
            .iter()
            .map(|shift| shift_member(*shift))
        {
            assert!(!member.contains('.'));
        }
    }

    #[test]
    fn the_pins_the_harness_shares_are_the_configs_and_the_model_has_one_layer() {
        use crate::experiments::r006::pins;
        assert_eq!(
            crate::experiments::layered_kv::SEQUENCE_SEED,
            CORPUS_GENERATOR_SEED
        );
        let shape = pins::SHAPE;
        assert_eq!(shape.layers, TINY_MODEL_CONFIG_LAYERS);
        assert_eq!(shape.layers, 1);
        assert_eq!(shape.layers, TEST_MODEL_LAYERS);
        assert_eq!(shape.hidden_size, TINY_MODEL_CONFIG_HIDDEN_SIZE);
        assert_eq!(shape.num_heads, TINY_MODEL_CONFIG_NUM_HEADS);
        assert_eq!(shape.num_kv_heads, TINY_MODEL_CONFIG_NUM_KV_HEADS);
        assert_eq!(shape.head_dim, TINY_MODEL_CONFIG_HEAD_DIM);
        assert_eq!(shape.vocab_size, TINY_MODEL_CONFIG_VOCAB_SIZE);
        assert_eq!(shape.weight_seed, TINY_MODEL_CONFIG_WEIGHT_SEED);
        assert_eq!(shape.weight_seed, WEIGHT_SEED);
        assert_eq!(shape.hidden_size, HIDDEN_SIZE);
        assert_eq!(WEIGHT_SEED, ptr_runtime::REFERENCE_KV_WEIGHT_SEED);
        assert_eq!(
            pins::canonical_digest(&pins::shape_fields()),
            TINY_MODEL_CONFIG_DIGEST
        );
    }

    #[test]
    fn the_owner_decisions_are_recorded_and_the_staleness_definition_is_structural() {
        for decision in [
            OWNER_DECISION_2026_10_08_1_GRANULARITY,
            OWNER_DECISION_2026_10_08_2_MATRIX,
            OWNER_DECISION_2026_10_08_3_REFUSALS,
            OWNER_DECISION_2026_10_08_4_GO,
            OWNER_DECISION_2026_10_08_5_STALENESS,
            OWNER_DECISION_2026_10_08_6_MULTISET,
            OWNER_DECISION_2026_10_08_7_REUSE_UNIT,
            OWNER_DECISION_2026_10_08_8_DECLARED_AND_SNAPSHOT,
            OWNER_DECISION_2026_10_08_9_CONFIG,
        ] {
            assert!(decision.len() > 20);
        }
        assert_eq!(STALENESS_MEASUREMENT, "structural");
        assert_eq!(MULTISET_SCOPE, "whole_request");
        assert_eq!(ROLLING_WINDOW_TOKENS, 8);
        assert_eq!(SHIFT_FILLER_KIND, "Raw");
    }
}
