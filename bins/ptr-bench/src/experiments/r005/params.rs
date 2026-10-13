//! The preregistered parameters of R005 (Exact half), compiled in.
//!
//! Every constant of the first two sections mirrors a key of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! third section holds the values the file leaves open (see `chosen`): they are this
//! harness's choice and are echoed in every result, and the preregistration lists
//! them. R005 is frozen at its preregistration and has completed; the file is frozen with it.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R005-exact-prefix-reuse/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/runtime/R005-exact-prefix-reuse/experiment.toml");

// ---- scalar and list keys, one constant each (generated from config.toml) ------------------

pub const MIN_REVOKED_FORK_ATTEMPTS: u64 = 1;
pub const MAX_REVOKED_FORK_COUNT: u64 = 0;
pub const MAX_TRIE_NODES: u64 = 4096;
pub const MAX_KV_PAGE_BYTES: u64 = 134217728;
pub const MAX_ACTIVE_FORK_PAGE_EVICTIONS: u64 = 0;
pub const MIN_EVICTIONS_WITH_ACTIVE_FORK_PINNED: u64 = 1;
pub const HIT_RATE_DEFINITION: &str = "reused_prefix_tokens / total_prompt_tokens";
pub const MIN_INVALIDATION_SUPPRESSED_ARMS: u64 = 1;
pub const MIN_RESTORE_ARMS: u64 = 1;
pub const MIN_POST_EVENT_LOOKUPS: u64 = 1;
pub const MAX_STALE_HITS: u64 = 0;
pub const MIN_INPUT_REVOCATIONS_PERFORMED: u64 = 1;
pub const MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS: u64 = 1;
pub const MAX_REVOKED_LINEAGE_ENTRY_HITS: u64 = 0;
pub const MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES: u64 = 1;
pub const MIN_REVOKED_PINNED_PAGE_ATTEMPTS: u64 = 1;
pub const MAX_REVOKED_PINNED_PAGE_SERVED_AFTER_REVOCATION: u64 = 0;
pub const REVOKED_PINNED_PAGE_BYTES_RECLAIMED_AFTER_RELEASE: &str = "all";
pub const MIN_CROSS_PRINCIPAL_ATTEMPTS: u64 = 1;
pub const MAX_CROSS_PRINCIPAL_HITS: u64 = 0;
pub const MIN_DISCARDED_OVERLAY_ENTRY_ATTEMPTS: u64 = 1;
pub const MAX_DISCARDED_OVERLAY_ENTRY_HITS: u64 = 0;
pub const MAX_METADATA_BYTES: u64 = 8388608;
pub const N_REQUESTS: u64 = 64;
pub const PROMPT_TOKENS: u64 = 256;
pub const SHARED_PREFIX_TOKENS: u64 = 128;
pub const COLD_FIRST_REQUEST_COUNTS_IN_HIT_RATE: bool = true;
pub const CONCURRENCY: u64 = 1;
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const MIN_SAME_TOKENS_DISTINCT_LINEAGE_ENTRIES: u64 = 2;
pub const MAX_CROSS_LINEAGE_HITS: u64 = 0;
pub const MIN_ACTIVE_FORK_REVOCATION_ATTEMPTS: u64 = 1;
pub const MAX_OUTPUTS_ESCAPED_AFTER_REVOCATION: u64 = 0;
pub const MIN_HIT_RATE_AT_50PCT_SHARED_PREFIX: f64 = 0.4;
pub const METADATA_BYTES_PER_NODE: u64 = 1024;
pub const KV_PAGE_BYTES_PER_NODE: u64 = 16384;
pub const NODES_ONLY_RUN_MAX_KV_PAGE_BYTES: u64 = 268435456;
pub const BYTES_ONLY_RUN_MAX_TRIE_NODES: u64 = 16384;
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const DTYPE: &str = "f32";
pub const MIN_CROSS_PROJECT_ATTEMPTS: u64 = 1;
pub const MAX_CROSS_PROJECT_HITS: u64 = 0;
pub const MIN_FORGED_WITNESS_FORK_ATTEMPTS: u64 = 1;
pub const MAX_FORGED_WITNESS_FORKS_ACCEPTED: u64 = 0;
pub const MIN_CANCELLED_RUN_EFFECT_ATTEMPTS: u64 = 1;
pub const MAX_CANCELLED_RUN_EFFECTS_ADMITTED: u64 = 0;
pub const MIN_EVICTIONS_NODES_ONLY_RUN: u64 = 1;
pub const MIN_EVICTIONS_BYTES_ONLY_RUN: u64 = 1;
pub const MIN_STALE_ELIGIBLE_ENTRIES_BEFORE_EVENT: u64 = 1;
pub const OVER_CAPACITY_WORKLOADS: &str =
    "independent nodes-only and bytes-only workloads of unique prompts";
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const MIN_CANCELLED_FORKS: u64 = 1;
pub const CANCELLED_FORKS_RESTARTED_FROM_FRESH_PREFILL_AT_LEAST_CANCELLED_FORKS: bool = true;
pub const MIN_RESTORED_PRE_REVOCATION_ENTRIES: u64 = 1;
pub const EMPTY_RESTART_ARM_ROLE: &str = "reported_separate_arm";
pub const MIN_RAW_PREFIX_REUSE_CASES: u64 = 1;
pub const MIN_RAW_IDENTICAL_PREFIX_HITS: u64 = 1;
pub const RAW_SEGMENT_ADMISSION: &str =
    "content-bound: (kind Raw, content digest); no identity, generation or liveness check";
pub const CAPACITY_EXHAUSTED_BEHAVIOUR: &str =
    "typed CapacityExhausted; uncached full prefill; no insert";
pub const MIN_SUPERSESSION_ACTIVE_FORK_RACE_CASES: u64 = 1;
pub const MIN_MEMORY_SUPERSESSION_ACTIVE_FORK_RACE_CASES: u64 = 1;
pub const MIN_MEMORY_REVOCATION_ACTIVE_FORK_RACE_CASES: u64 = 1;
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 31] = [
    "seeds",
    "n_requests",
    "prompt_tokens",
    "shared_prefix_tokens",
    "concurrency",
    "max_trie_nodes",
    "max_kv_page_bytes",
    "test_model_layers",
    "weight_seed",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "corpus_generator_seed",
    "kv_page_bytes_per_node",
    "metadata_bytes_per_node",
    "nodes_only_run_max_kv_page_bytes",
    "bytes_only_run_max_trie_nodes",
    "workload_limits",
    "invalidation_coverage_matrix",
    "superseded_lookup_cells",
    "saturation_definition",
    "chain_layer_covers",
    "tiny_model_config_digest",
    "tokenizer_pin",
    "moe_topology",
    "declared_dependency_kinds",
    "production_modes",
    "derived_state_carriers",
    "numerics_binding_environment_fields",
];
pub const CAPACITY_REFUSED_REQUESTS_COMPLETED_EQUALS_EXHAUSTED_REFUSALS: bool = true;
pub const MAX_CAPACITY_FALLBACK_OUTPUT_MISMATCHES: u64 = 0;
pub const MAX_CAPACITY_REFUSAL_CACHE_INSERTS: u64 = 0;
pub const MIN_SUPERSESSION_CANCELLED_FORKS: u64 = 1;
pub const SUPERSESSION_FORKS_RESTARTED_FROM_FRESH_PREFILL_EQUALS_CANCELLED: bool = true;
pub const MAX_CACHE_INSERTS_AFTER_SUPERSESSION: u64 = 0;
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const DEPENDENCY_KINDS: [&str; 2] = ["semantic_input", "memory_input"];
pub const COVERAGE_REPORTED_PER_MEMBER: &str = "dependency kind x invalidation type";
pub const MAX_OUTPUTS_AFTER_SUPERSESSION_SEMANTIC: u64 = 0;
pub const MAX_OUTPUTS_AFTER_SUPERSESSION_MEMORY: u64 = 0;
pub const WORKLOAD_PEAKS_REPORTED: &str =
    "per workload, evaluated against that workload's own limits";
pub const OUTPUTS_AFTER_SUPERSESSION_FIELD_FAMILY: &str = "outputs_after_supersession";
pub const MIN_BASE_MODEL_INFLIGHT_RACE_CASES_PER_TYPE: u64 = 1;
pub const MAX_BASE_MODEL_INFLIGHT_OUTPUTS_AFTER_INVALIDATION: u64 = 0;
pub const MAX_BASE_MODEL_INFLIGHT_CACHE_INSERTS: u64 = 0;
pub const BASE_MODEL_INFLIGHT_RACE_CASES_REPORTED: &str = "per invalidation type";
pub const MIN_INVALIDATION_MATRIX_CASES_PER_CELL: u64 = 1;
pub const MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL: u64 = 0;
pub const INVALIDATION_MATRIX_PATH_MEANING: &str = "normal = ordinary lookup after the invalidation; lag = lagging invalidation or index loss; restore = actual restore from snapshot or restart; inflight = in-flight race";
pub const MIN_SUPERSEDED_LOOKUP_CASES_PER_CELL: u64 = 1;
pub const MAX_SUPERSEDED_STALE_HITS: u64 = 0;
pub const MIN_SATURATION_DEMAND_CASES: u64 = 1;
pub const MIN_CAPACITY_EXHAUSTED_REFUSALS: u64 = 1;
pub const SATURATION_DEFINITION: &str = "attempted demand of concurrent forks exceeds capacity; the accepted pinned set stays within bounds; the excess insertion is refused with CapacityExhausted";
pub const MIN_CURRENT_LAYER_KV_TRANSFORM_CHANGE_CASES: u64 = 1;
pub const MAX_STALE_EXACT_HITS: u64 = 0;
pub const MIN_STALE_WITNESS_REPLAY_CASES: u64 = 1;
pub const MIN_WITNESS_REUSE_STATIC_EVIDENCE_PASSING: u64 = 2;
pub const WITNESS_REUSE_EVIDENCE: &str = "owner decision 2026-10-06: reuse of a consumed KvForkWitness is a compile-time error. Evidence is (1) the compile_fail doctest on KvForkWitness in crates/ptr-pods/src/paged_kv.rs (a second fork_prefix_admitted with the same witness must not compile) and (2) the unit test witness_tests::second_consume_of_the_same_witness_state_is_refused in the same file; no runtime witness_reuse_attempts is required";
pub const MAX_STALE_OR_REUSED_WITNESS_FORKS_ADMITTED: u64 = 0;
pub const CHAIN_LAYER_COVERS: [&str; 6] = [
    "layers_below",
    "pre_attention_norm",
    "kv_projection_weights_and_biases",
    "qk_norm_weights_and_eps",
    "stored_post_rope_rotary_parameters",
    "kv_quantization_parameters",
];
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY: &str = "none: dense tiny decoder";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const ROPE_REGIME_FIXED_AT_ADMISSION_FROM: &str = "prompt + max_new_tokens";
pub const ROPE_REGIME_INPUTS: [&str; 2] = ["prompt_tokens", "max_new_tokens"];
pub const DECLARED_DEPENDENCY_KINDS: [&str; 6] = [
    "semantic",
    "memory",
    "derived",
    "base_model",
    "adapter",
    "expert",
];
pub const CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE: bool = true;
pub const CACHE_NAMESPACE_PARTITION: &str =
    "Exact catalog separate from the tainted candidate namespace";
pub const PRODUCTION_MODES: [&str; 2] = ["Exact", "Approximate"];
pub const MIN_FIXED_REGIME_DECODE_CROSSING_CASES: u64 = 1;
pub const MAX_FIXED_REGIME_CROSSING_PARITY_FAILURES: u64 = 0;
pub const MIN_EFFECTIVE_REGIME_TRANSITION_CASES: u64 = 1;
pub const MAX_EXACT_OUTPUTS_ACROSS_EFFECTIVE_REGIME_TRANSITION_WITHOUT_RECOMPUTE: u64 = 0;
pub const FORCED_TRANSITION_EXAMPLES: [&str; 2] = [
    "admitted max_new_tokens exceeded",
    "runtime switches regimes",
];
pub const MIN_MISMATCHED_POSITION_CASES: u64 = 1;
pub const MAX_EXACT_HITS_WITH_MISMATCHED_POSITIONS: u64 = 0;
pub const CANONICAL_POSITIONS: &str = "position ids = trie depth offset from 0; no explicit position ids or offset; otherwise Exact lookup is bypassed (miss)";
pub const NUMERICS_BINDING_ENVIRONMENT_FIELDS: [&str; 5] = [
    "gpu_or_cpu_architecture",
    "driver_version",
    "cuda_runtime_version",
    "attention_gemm_kernel_library",
    "backend_build_identity",
];
pub const PRODUCER_ENVIRONMENT_RECORDED_IN_RUN_JSON: bool = true;
pub const MIN_CROSS_ENVIRONMENT_RESTORE_CASES: u64 = 1;
pub const MAX_EXACT_HITS_ACROSS_ENVIRONMENT: u64 = 0;
pub const CROSS_ENVIRONMENT_SIMULATION: &str = "a changed build-identity field";
pub const APPROXIMATE_FAMILY_SUCCESSOR: &str = "R012-approximate-taint-containment";

// ---- structured keys ----------------------------------------------------------------------

/// One workload's limits (`workload_limits.<name>`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkloadLimits {
    pub max_trie_nodes: u64,
    pub max_kv_page_bytes: u64,
    pub max_metadata_bytes: u64,
}

pub const LIMITS_MAIN: WorkloadLimits = WorkloadLimits {
    max_trie_nodes: 4096,
    max_kv_page_bytes: 134_217_728,
    max_metadata_bytes: 8_388_608,
};
pub const LIMITS_NODES_ONLY: WorkloadLimits = WorkloadLimits {
    max_trie_nodes: 4096,
    max_kv_page_bytes: 268_435_456,
    max_metadata_bytes: 67_108_864,
};
pub const LIMITS_BYTES_ONLY: WorkloadLimits = WorkloadLimits {
    max_trie_nodes: 16_384,
    max_kv_page_bytes: 67_108_864,
    max_metadata_bytes: 16_777_216,
};

/// The three workloads whose peaks are judged against their own limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Workload {
    Main,
    NodesOnly,
    BytesOnly,
}

impl Workload {
    pub const ALL: [Workload; 3] = [Workload::Main, Workload::NodesOnly, Workload::BytesOnly];

    pub fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::NodesOnly => "nodes_only",
            Self::BytesOnly => "bytes_only",
        }
    }

    pub fn limits(self) -> WorkloadLimits {
        match self {
            Self::Main => LIMITS_MAIN,
            Self::NodesOnly => LIMITS_NODES_ONLY,
            Self::BytesOnly => LIMITS_BYTES_ONLY,
        }
    }

    /// `binding_limit_per_workload.<name>`.
    pub fn binding_limit(self) -> &'static str {
        match self {
            Self::Main | Self::NodesOnly => "max_trie_nodes",
            Self::BytesOnly => "max_kv_page_bytes",
        }
    }
}

/// `tiny_model_config`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TinyModel {
    pub layers: u64,
    pub hidden_size: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub vocab_size: u64,
    pub weight_seed: u64,
}

pub const TINY_MODEL: TinyModel = TinyModel {
    layers: 4,
    hidden_size: 64,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 16,
    vocab_size: 256,
    weight_seed: 20_240_607,
};

/// `derived_state_carriers.ptr_catalog_key`.
pub const DERIVED_STATE_CARRIERS_PTR_CATALOG_KEY: [&str; 1] = ["mode"];

/// `identity_set_checks`: pairs of digests that must be equal.
pub const IDENTITY_SET_CHECKS: [[&str; 2]; 2] = [
    [
        "cancelled_fork_id_set_digest",
        "restarted_fork_id_set_digest",
    ],
    [
        "capacity_refused_request_id_set_digest",
        "capacity_completed_request_id_set_digest",
    ],
];

/// `superseded_lookup_cells`: (dependency kind, path).
pub const SUPERSEDED_LOOKUP_CELLS: [[&str; 2]; 6] = [
    ["semantic", "normal"],
    ["semantic", "lag"],
    ["semantic", "restore"],
    ["memory", "normal"],
    ["memory", "lag"],
    ["memory", "restore"],
];

/// `invalidation_coverage_matrix.*`.
pub const MATRIX_KINDS: [&str; 6] = [
    "semantic_input",
    "memory_input",
    "derived_input",
    "base_model",
    "adapter_generation",
    "expert_generation",
];
pub const MATRIX_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const MATRIX_PATHS: [&str; 4] = ["normal", "lag", "restore", "inflight"];
pub const MATRIX_EXTRA_EXPERT_TYPES: [&str; 1] = ["fence"];
pub const MATRIX_CELLS: [&str; 52] = [
    "semantic_input_normal_revocation",
    "semantic_input_normal_supersession",
    "semantic_input_lag_revocation",
    "semantic_input_lag_supersession",
    "semantic_input_restore_revocation",
    "semantic_input_restore_supersession",
    "semantic_input_inflight_revocation",
    "semantic_input_inflight_supersession",
    "memory_input_normal_revocation",
    "memory_input_normal_supersession",
    "memory_input_lag_revocation",
    "memory_input_lag_supersession",
    "memory_input_restore_revocation",
    "memory_input_restore_supersession",
    "memory_input_inflight_revocation",
    "memory_input_inflight_supersession",
    "derived_input_normal_revocation",
    "derived_input_normal_supersession",
    "derived_input_lag_revocation",
    "derived_input_lag_supersession",
    "derived_input_restore_revocation",
    "derived_input_restore_supersession",
    "derived_input_inflight_revocation",
    "derived_input_inflight_supersession",
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

// ---- chosen: values config.toml leaves open --------------------------------------------------
//
// These are the harness's own choices. Most are what the code under test hard-codes
// (`crates/ptr-runtime/src/kv_infer.rs`, `stack_config`); `pin::verify` proves the equality
// by reproducing a cold run with a stack built from these values and comparing the logits
// digest with the backend's. They are not in `config.toml` and so not covered by the drift
// test against it; the preregistration pins them (`PREREGISTRATION.md`, chosen values).

/// Tokens generated per request.
pub const MAX_NEW_TOKENS: usize = 8;
/// Longest prompt a backend accepts (the main workload's prompt length).
pub const MAX_PROMPT_TOKENS: usize = 256;
pub const INTERMEDIATE_SIZE: usize = 128;
pub const ROPE_THETA: f32 = 10_000.0;
pub const RMS_NORM_EPS: f32 = 1e-6;
/// The main arms run the dense tiny decoder without QK-norm.
pub const QK_NORM_MAIN: &str = "disabled";
/// The current-layer transform-change arm runs the same tiny decoder with QK-norm enabled,
/// so a key-norm weight exists to change.
pub const QK_NORM_TRANSFORM_ARM: &str = "enabled";
pub const KEY_ENCODING: &str = "pre-rope";
/// Dynamic-NTK scaling of the regime arms.
pub const NTK_FACTOR: f32 = 4.0;
///
/// 30 positions: a 16-token request (24 with its new tokens) and a 17-token one (25) stay in
/// the plain regime, a 24-token prompt (32 with its new tokens) is below the boundary yet
/// crosses it while decoding, and a 36-token request (44) is admitted into a scaled regime.
pub const NTK_ORIGINAL_MAX_POSITIONS: usize = 30;
/// Tokens per catalog chunk and per page.
pub const CHUNK_TOKENS: usize = 16;
/// Tokens per unique prompt of the over-capacity workloads: eight chunks, so 512 prompts
/// fill 4096 nodes exactly.
pub const OVER_CAPACITY_PROMPT_TOKENS: usize = 128;
/// Unique prompts that fill a catalog exactly to its binding limit.
pub const OVER_CAPACITY_FILL_PROMPTS: usize = 512;
/// Unique prompts offered after the fill, so every one of them evicts.
pub const OVER_CAPACITY_EXTRA_PROMPTS: usize = 64;
/// Sessions open at once in the saturation phase (one per filled prompt, with headroom).
pub const SATURATION_MAX_SESSIONS: usize = 520;
/// Fresh unique requests offered to the saturated catalog.
pub const SATURATION_EXCESS_REQUESTS: usize = 8;
pub const PRINCIPAL: &str = "r005-principal-a";
pub const OTHER_PRINCIPAL: &str = "r005-principal-b";
pub const PROJECT: &str = "r005-project-a";
pub const OTHER_PROJECT: &str = "r005-project-b";
/// Domain of the corpus selection digests.
pub const SELECTION_DOMAIN: &str = "ptr.r005.corpus-selection.v1";
/// The digests of the corpus selection of each workload (`workload::selection_digest`),
/// which the preregistration pins; a test reproduces them.
pub const SELECTION_DIGEST_MAIN: &str =
    "sha256:d06fd633832bb4bdb7b50e21473dcde2882ff234855aa9e32da6a97f67a7968d";
pub const SELECTION_DIGEST_NODES_ONLY: &str =
    "sha256:38b1b1da0c23c6a6fea31aa53b52bb92bce2e82df3674f928d87387fc2251b63";
pub const SELECTION_DIGEST_BYTES_ONLY: &str =
    "sha256:4f5a40e2b33245b3a99deda30331571c563a986438ea17d4ece9c592856465f9";

// ---- the compiled table, as the drift test compares it -------------------------------------

/// One leaf of the table.
#[derive(Clone, Debug, PartialEq)]
pub enum Leaf {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
}

impl Leaf {
    fn strs(values: &[&str]) -> Self {
        Self::Strs(values.iter().map(|value| (*value).to_string()).collect())
    }
}

fn int(value: u64) -> Leaf {
    Leaf::Int(value as i64)
}

/// Every key of `[pinned_thresholds]` as the harness compiles it, structured keys flattened
/// to dotted paths and lists of lists to `a|b` strings.
pub fn compiled() -> Vec<(String, Leaf)> {
    let mut table: Vec<(String, Leaf)> = vec![
        (
            "min_revoked_fork_attempts",
            Leaf::Int(MIN_REVOKED_FORK_ATTEMPTS as i64),
        ),
        (
            "max_revoked_fork_count",
            Leaf::Int(MAX_REVOKED_FORK_COUNT as i64),
        ),
        ("max_trie_nodes", Leaf::Int(MAX_TRIE_NODES as i64)),
        ("max_kv_page_bytes", Leaf::Int(MAX_KV_PAGE_BYTES as i64)),
        (
            "max_active_fork_page_evictions",
            Leaf::Int(MAX_ACTIVE_FORK_PAGE_EVICTIONS as i64),
        ),
        (
            "min_evictions_with_active_fork_pinned",
            Leaf::Int(MIN_EVICTIONS_WITH_ACTIVE_FORK_PINNED as i64),
        ),
        (
            "hit_rate_definition",
            Leaf::Str(HIT_RATE_DEFINITION.to_string()),
        ),
        (
            "min_invalidation_suppressed_arms",
            Leaf::Int(MIN_INVALIDATION_SUPPRESSED_ARMS as i64),
        ),
        ("min_restore_arms", Leaf::Int(MIN_RESTORE_ARMS as i64)),
        (
            "min_post_event_lookups",
            Leaf::Int(MIN_POST_EVENT_LOOKUPS as i64),
        ),
        ("max_stale_hits", Leaf::Int(MAX_STALE_HITS as i64)),
        (
            "min_input_revocations_performed",
            Leaf::Int(MIN_INPUT_REVOCATIONS_PERFORMED as i64),
        ),
        (
            "min_post_revocation_lookups_matching_tokens",
            Leaf::Int(MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS as i64),
        ),
        (
            "max_revoked_lineage_entry_hits",
            Leaf::Int(MAX_REVOKED_LINEAGE_ENTRY_HITS as i64),
        ),
        (
            "min_same_tokens_different_generation_cases",
            Leaf::Int(MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES as i64),
        ),
        (
            "min_revoked_pinned_page_attempts",
            Leaf::Int(MIN_REVOKED_PINNED_PAGE_ATTEMPTS as i64),
        ),
        (
            "max_revoked_pinned_page_served_after_revocation",
            Leaf::Int(MAX_REVOKED_PINNED_PAGE_SERVED_AFTER_REVOCATION as i64),
        ),
        (
            "revoked_pinned_page_bytes_reclaimed_after_release",
            Leaf::Str(REVOKED_PINNED_PAGE_BYTES_RECLAIMED_AFTER_RELEASE.to_string()),
        ),
        (
            "min_cross_principal_attempts",
            Leaf::Int(MIN_CROSS_PRINCIPAL_ATTEMPTS as i64),
        ),
        (
            "max_cross_principal_hits",
            Leaf::Int(MAX_CROSS_PRINCIPAL_HITS as i64),
        ),
        (
            "min_discarded_overlay_entry_attempts",
            Leaf::Int(MIN_DISCARDED_OVERLAY_ENTRY_ATTEMPTS as i64),
        ),
        (
            "max_discarded_overlay_entry_hits",
            Leaf::Int(MAX_DISCARDED_OVERLAY_ENTRY_HITS as i64),
        ),
        ("max_metadata_bytes", Leaf::Int(MAX_METADATA_BYTES as i64)),
        ("n_requests", Leaf::Int(N_REQUESTS as i64)),
        ("prompt_tokens", Leaf::Int(PROMPT_TOKENS as i64)),
        (
            "shared_prefix_tokens",
            Leaf::Int(SHARED_PREFIX_TOKENS as i64),
        ),
        (
            "cold_first_request_counts_in_hit_rate",
            Leaf::Bool(COLD_FIRST_REQUEST_COUNTS_IN_HIT_RATE),
        ),
        ("concurrency", Leaf::Int(CONCURRENCY as i64)),
        ("seed_aggregation", Leaf::Str(SEED_AGGREGATION.to_string())),
        (
            "min_same_tokens_distinct_lineage_entries",
            Leaf::Int(MIN_SAME_TOKENS_DISTINCT_LINEAGE_ENTRIES as i64),
        ),
        (
            "max_cross_lineage_hits",
            Leaf::Int(MAX_CROSS_LINEAGE_HITS as i64),
        ),
        (
            "min_active_fork_revocation_attempts",
            Leaf::Int(MIN_ACTIVE_FORK_REVOCATION_ATTEMPTS as i64),
        ),
        (
            "max_outputs_escaped_after_revocation",
            Leaf::Int(MAX_OUTPUTS_ESCAPED_AFTER_REVOCATION as i64),
        ),
        (
            "min_hit_rate_at_50pct_shared_prefix",
            Leaf::Float(MIN_HIT_RATE_AT_50PCT_SHARED_PREFIX),
        ),
        (
            "metadata_bytes_per_node",
            Leaf::Int(METADATA_BYTES_PER_NODE as i64),
        ),
        (
            "kv_page_bytes_per_node",
            Leaf::Int(KV_PAGE_BYTES_PER_NODE as i64),
        ),
        (
            "nodes_only_run_max_kv_page_bytes",
            Leaf::Int(NODES_ONLY_RUN_MAX_KV_PAGE_BYTES as i64),
        ),
        (
            "bytes_only_run_max_trie_nodes",
            Leaf::Int(BYTES_ONLY_RUN_MAX_TRIE_NODES as i64),
        ),
        ("test_model_layers", Leaf::Int(TEST_MODEL_LAYERS as i64)),
        ("dtype", Leaf::Str(DTYPE.to_string())),
        (
            "min_cross_project_attempts",
            Leaf::Int(MIN_CROSS_PROJECT_ATTEMPTS as i64),
        ),
        (
            "max_cross_project_hits",
            Leaf::Int(MAX_CROSS_PROJECT_HITS as i64),
        ),
        (
            "min_forged_witness_fork_attempts",
            Leaf::Int(MIN_FORGED_WITNESS_FORK_ATTEMPTS as i64),
        ),
        (
            "max_forged_witness_forks_accepted",
            Leaf::Int(MAX_FORGED_WITNESS_FORKS_ACCEPTED as i64),
        ),
        (
            "min_cancelled_run_effect_attempts",
            Leaf::Int(MIN_CANCELLED_RUN_EFFECT_ATTEMPTS as i64),
        ),
        (
            "max_cancelled_run_effects_admitted",
            Leaf::Int(MAX_CANCELLED_RUN_EFFECTS_ADMITTED as i64),
        ),
        (
            "min_evictions_nodes_only_run",
            Leaf::Int(MIN_EVICTIONS_NODES_ONLY_RUN as i64),
        ),
        (
            "min_evictions_bytes_only_run",
            Leaf::Int(MIN_EVICTIONS_BYTES_ONLY_RUN as i64),
        ),
        (
            "min_stale_eligible_entries_before_event",
            Leaf::Int(MIN_STALE_ELIGIBLE_ENTRIES_BEFORE_EVENT as i64),
        ),
        (
            "over_capacity_workloads",
            Leaf::Str(OVER_CAPACITY_WORKLOADS.to_string()),
        ),
        ("weights", Leaf::Str(WEIGHTS.to_string())),
        ("weight_seed", Leaf::Int(WEIGHT_SEED as i64)),
        ("hidden_size", Leaf::Int(HIDDEN_SIZE as i64)),
        ("num_heads", Leaf::Int(NUM_HEADS as i64)),
        ("num_kv_heads", Leaf::Int(NUM_KV_HEADS as i64)),
        ("head_dim", Leaf::Int(HEAD_DIM as i64)),
        ("vocab_size", Leaf::Int(VOCAB_SIZE as i64)),
        ("min_cancelled_forks", Leaf::Int(MIN_CANCELLED_FORKS as i64)),
        (
            "cancelled_forks_restarted_from_fresh_prefill_at_least_cancelled_forks",
            Leaf::Bool(CANCELLED_FORKS_RESTARTED_FROM_FRESH_PREFILL_AT_LEAST_CANCELLED_FORKS),
        ),
        (
            "min_restored_pre_revocation_entries",
            Leaf::Int(MIN_RESTORED_PRE_REVOCATION_ENTRIES as i64),
        ),
        (
            "empty_restart_arm_role",
            Leaf::Str(EMPTY_RESTART_ARM_ROLE.to_string()),
        ),
        (
            "min_raw_prefix_reuse_cases",
            Leaf::Int(MIN_RAW_PREFIX_REUSE_CASES as i64),
        ),
        (
            "min_raw_identical_prefix_hits",
            Leaf::Int(MIN_RAW_IDENTICAL_PREFIX_HITS as i64),
        ),
        (
            "raw_segment_admission",
            Leaf::Str(RAW_SEGMENT_ADMISSION.to_string()),
        ),
        (
            "capacity_exhausted_behaviour",
            Leaf::Str(CAPACITY_EXHAUSTED_BEHAVIOUR.to_string()),
        ),
        (
            "min_supersession_active_fork_race_cases",
            Leaf::Int(MIN_SUPERSESSION_ACTIVE_FORK_RACE_CASES as i64),
        ),
        (
            "min_memory_supersession_active_fork_race_cases",
            Leaf::Int(MIN_MEMORY_SUPERSESSION_ACTIVE_FORK_RACE_CASES as i64),
        ),
        (
            "min_memory_revocation_active_fork_race_cases",
            Leaf::Int(MIN_MEMORY_REVOCATION_ACTIVE_FORK_RACE_CASES as i64),
        ),
        ("corpus", Leaf::Str(CORPUS.to_string())),
        (
            "corpus_generator_seed",
            Leaf::Int(CORPUS_GENERATOR_SEED as i64),
        ),
        (
            "corpus_selection_rule",
            Leaf::Str(CORPUS_SELECTION_RULE.to_string()),
        ),
        (
            "corpus_selection_digest_recorded_in_run_json",
            Leaf::Bool(CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON),
        ),
        (
            "max_selection_digest_mismatches",
            Leaf::Int(MAX_SELECTION_DIGEST_MISMATCHES as i64),
        ),
        (
            "max_procedure_mismatches",
            Leaf::Int(MAX_PROCEDURE_MISMATCHES as i64),
        ),
        (
            "procedure_parameters_recorded_per_cell",
            Leaf::strs(&PROCEDURE_PARAMETERS_RECORDED_PER_CELL),
        ),
        (
            "capacity_refused_requests_completed_equals_exhausted_refusals",
            Leaf::Bool(CAPACITY_REFUSED_REQUESTS_COMPLETED_EQUALS_EXHAUSTED_REFUSALS),
        ),
        (
            "max_capacity_fallback_output_mismatches",
            Leaf::Int(MAX_CAPACITY_FALLBACK_OUTPUT_MISMATCHES as i64),
        ),
        (
            "max_capacity_refusal_cache_inserts",
            Leaf::Int(MAX_CAPACITY_REFUSAL_CACHE_INSERTS as i64),
        ),
        (
            "min_supersession_cancelled_forks",
            Leaf::Int(MIN_SUPERSESSION_CANCELLED_FORKS as i64),
        ),
        (
            "supersession_forks_restarted_from_fresh_prefill_equals_cancelled",
            Leaf::Bool(SUPERSESSION_FORKS_RESTARTED_FROM_FRESH_PREFILL_EQUALS_CANCELLED),
        ),
        (
            "max_cache_inserts_after_supersession",
            Leaf::Int(MAX_CACHE_INSERTS_AFTER_SUPERSESSION as i64),
        ),
        ("invalidation_types", Leaf::strs(&INVALIDATION_TYPES)),
        ("dependency_kinds", Leaf::strs(&DEPENDENCY_KINDS)),
        (
            "coverage_reported_per_member",
            Leaf::Str(COVERAGE_REPORTED_PER_MEMBER.to_string()),
        ),
        (
            "max_outputs_after_supersession_semantic",
            Leaf::Int(MAX_OUTPUTS_AFTER_SUPERSESSION_SEMANTIC as i64),
        ),
        (
            "max_outputs_after_supersession_memory",
            Leaf::Int(MAX_OUTPUTS_AFTER_SUPERSESSION_MEMORY as i64),
        ),
        (
            "workload_peaks_reported",
            Leaf::Str(WORKLOAD_PEAKS_REPORTED.to_string()),
        ),
        (
            "outputs_after_supersession_field_family",
            Leaf::Str(OUTPUTS_AFTER_SUPERSESSION_FIELD_FAMILY.to_string()),
        ),
        (
            "min_base_model_inflight_race_cases_per_type",
            Leaf::Int(MIN_BASE_MODEL_INFLIGHT_RACE_CASES_PER_TYPE as i64),
        ),
        (
            "max_base_model_inflight_outputs_after_invalidation",
            Leaf::Int(MAX_BASE_MODEL_INFLIGHT_OUTPUTS_AFTER_INVALIDATION as i64),
        ),
        (
            "max_base_model_inflight_cache_inserts",
            Leaf::Int(MAX_BASE_MODEL_INFLIGHT_CACHE_INSERTS as i64),
        ),
        (
            "base_model_inflight_race_cases_reported",
            Leaf::Str(BASE_MODEL_INFLIGHT_RACE_CASES_REPORTED.to_string()),
        ),
        (
            "min_invalidation_matrix_cases_per_cell",
            Leaf::Int(MIN_INVALIDATION_MATRIX_CASES_PER_CELL as i64),
        ),
        (
            "max_invalidation_matrix_stale_hits_per_cell",
            Leaf::Int(MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL as i64),
        ),
        (
            "invalidation_matrix_path_meaning",
            Leaf::Str(INVALIDATION_MATRIX_PATH_MEANING.to_string()),
        ),
        (
            "min_superseded_lookup_cases_per_cell",
            Leaf::Int(MIN_SUPERSEDED_LOOKUP_CASES_PER_CELL as i64),
        ),
        (
            "max_superseded_stale_hits",
            Leaf::Int(MAX_SUPERSEDED_STALE_HITS as i64),
        ),
        (
            "min_saturation_demand_cases",
            Leaf::Int(MIN_SATURATION_DEMAND_CASES as i64),
        ),
        (
            "min_capacity_exhausted_refusals",
            Leaf::Int(MIN_CAPACITY_EXHAUSTED_REFUSALS as i64),
        ),
        (
            "saturation_definition",
            Leaf::Str(SATURATION_DEFINITION.to_string()),
        ),
        (
            "min_current_layer_kv_transform_change_cases",
            Leaf::Int(MIN_CURRENT_LAYER_KV_TRANSFORM_CHANGE_CASES as i64),
        ),
        (
            "max_stale_exact_hits",
            Leaf::Int(MAX_STALE_EXACT_HITS as i64),
        ),
        (
            "min_stale_witness_replay_cases",
            Leaf::Int(MIN_STALE_WITNESS_REPLAY_CASES as i64),
        ),
        (
            "min_witness_reuse_static_evidence_passing",
            Leaf::Int(MIN_WITNESS_REUSE_STATIC_EVIDENCE_PASSING as i64),
        ),
        (
            "witness_reuse_evidence",
            Leaf::Str(WITNESS_REUSE_EVIDENCE.to_string()),
        ),
        (
            "max_stale_or_reused_witness_forks_admitted",
            Leaf::Int(MAX_STALE_OR_REUSED_WITNESS_FORKS_ADMITTED as i64),
        ),
        ("chain_layer_covers", Leaf::strs(&CHAIN_LAYER_COVERS)),
        (
            "tiny_model_config_digest",
            Leaf::Str(TINY_MODEL_CONFIG_DIGEST.to_string()),
        ),
        (
            "tiny_model_config_digest_rule",
            Leaf::Str(TINY_MODEL_CONFIG_DIGEST_RULE.to_string()),
        ),
        ("tokenizer_pin", Leaf::Str(TOKENIZER_PIN.to_string())),
        ("moe_topology", Leaf::Str(MOE_TOPOLOGY.to_string())),
        (
            "model_pins_recorded_in_run_json",
            Leaf::Bool(MODEL_PINS_RECORDED_IN_RUN_JSON),
        ),
        (
            "max_model_pin_mismatches",
            Leaf::Int(MAX_MODEL_PIN_MISMATCHES as i64),
        ),
        (
            "rope_regime_fixed_at_admission_from",
            Leaf::Str(ROPE_REGIME_FIXED_AT_ADMISSION_FROM.to_string()),
        ),
        ("rope_regime_inputs", Leaf::strs(&ROPE_REGIME_INPUTS)),
        (
            "declared_dependency_kinds",
            Leaf::strs(&DECLARED_DEPENDENCY_KINDS),
        ),
        (
            "cache_namespace_partitioned_by_production_mode",
            Leaf::Bool(CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE),
        ),
        (
            "cache_namespace_partition",
            Leaf::Str(CACHE_NAMESPACE_PARTITION.to_string()),
        ),
        ("production_modes", Leaf::strs(&PRODUCTION_MODES)),
        (
            "min_fixed_regime_decode_crossing_cases",
            Leaf::Int(MIN_FIXED_REGIME_DECODE_CROSSING_CASES as i64),
        ),
        (
            "max_fixed_regime_crossing_parity_failures",
            Leaf::Int(MAX_FIXED_REGIME_CROSSING_PARITY_FAILURES as i64),
        ),
        (
            "min_effective_regime_transition_cases",
            Leaf::Int(MIN_EFFECTIVE_REGIME_TRANSITION_CASES as i64),
        ),
        (
            "max_exact_outputs_across_effective_regime_transition_without_recompute",
            Leaf::Int(
                MAX_EXACT_OUTPUTS_ACROSS_EFFECTIVE_REGIME_TRANSITION_WITHOUT_RECOMPUTE as i64,
            ),
        ),
        (
            "forced_transition_examples",
            Leaf::strs(&FORCED_TRANSITION_EXAMPLES),
        ),
        (
            "min_mismatched_position_cases",
            Leaf::Int(MIN_MISMATCHED_POSITION_CASES as i64),
        ),
        (
            "max_exact_hits_with_mismatched_positions",
            Leaf::Int(MAX_EXACT_HITS_WITH_MISMATCHED_POSITIONS as i64),
        ),
        (
            "canonical_positions",
            Leaf::Str(CANONICAL_POSITIONS.to_string()),
        ),
        (
            "numerics_binding_environment_fields",
            Leaf::strs(&NUMERICS_BINDING_ENVIRONMENT_FIELDS),
        ),
        (
            "producer_environment_recorded_in_run_json",
            Leaf::Bool(PRODUCER_ENVIRONMENT_RECORDED_IN_RUN_JSON),
        ),
        (
            "min_cross_environment_restore_cases",
            Leaf::Int(MIN_CROSS_ENVIRONMENT_RESTORE_CASES as i64),
        ),
        (
            "max_exact_hits_across_environment",
            Leaf::Int(MAX_EXACT_HITS_ACROSS_ENVIRONMENT as i64),
        ),
        (
            "cross_environment_simulation",
            Leaf::Str(CROSS_ENVIRONMENT_SIMULATION.to_string()),
        ),
        (
            "approximate_family_successor",
            Leaf::Str(APPROXIMATE_FAMILY_SUCCESSOR.to_string()),
        ),
    ]
    .into_iter()
    .map(|(key, leaf)| (key.to_string(), leaf))
    .collect();
    for workload in Workload::ALL {
        let limits = workload.limits();
        let name = workload.name();
        table.push((
            format!("workload_limits.{name}.max_trie_nodes"),
            int(limits.max_trie_nodes),
        ));
        table.push((
            format!("workload_limits.{name}.max_kv_page_bytes"),
            int(limits.max_kv_page_bytes),
        ));
        table.push((
            format!("workload_limits.{name}.max_metadata_bytes"),
            int(limits.max_metadata_bytes),
        ));
        table.push((
            format!("binding_limit_per_workload.{name}"),
            Leaf::Str(workload.binding_limit().to_string()),
        ));
    }
    for (key, value) in [
        ("layers", TINY_MODEL.layers),
        ("hidden_size", TINY_MODEL.hidden_size),
        ("num_heads", TINY_MODEL.num_heads),
        ("num_kv_heads", TINY_MODEL.num_kv_heads),
        ("head_dim", TINY_MODEL.head_dim),
        ("vocab_size", TINY_MODEL.vocab_size),
        ("weight_seed", TINY_MODEL.weight_seed),
    ] {
        table.push((format!("tiny_model_config.{key}"), int(value)));
    }
    table.push((
        "derived_state_carriers.ptr_catalog_key".to_string(),
        Leaf::strs(&DERIVED_STATE_CARRIERS_PTR_CATALOG_KEY),
    ));
    table.push((
        "identity_set_checks".to_string(),
        Leaf::Strs(
            IDENTITY_SET_CHECKS
                .iter()
                .map(|pair| pair.join("|"))
                .collect(),
        ),
    ));
    table.push((
        "superseded_lookup_cells".to_string(),
        Leaf::Strs(
            SUPERSEDED_LOOKUP_CELLS
                .iter()
                .map(|pair| pair.join("|"))
                .collect(),
        ),
    ));
    table.push((
        "invalidation_coverage_matrix.kinds".to_string(),
        Leaf::strs(&MATRIX_KINDS),
    ));
    table.push((
        "invalidation_coverage_matrix.types".to_string(),
        Leaf::strs(&MATRIX_TYPES),
    ));
    table.push((
        "invalidation_coverage_matrix.paths".to_string(),
        Leaf::strs(&MATRIX_PATHS),
    ));
    table.push((
        "invalidation_coverage_matrix.extra_types.expert_generation".to_string(),
        Leaf::strs(&MATRIX_EXTRA_EXPERT_TYPES),
    ));
    table.push((
        "invalidation_coverage_matrix.cells".to_string(),
        Leaf::strs(&MATRIX_CELLS),
    ));
    table
}

/// The matrix cell name of a dependency kind, path and invalidation type.
pub fn cell_name(kind: &str, path: &str, kind_type: &str) -> String {
    format!("{kind}_{path}_{kind_type}")
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
            toml::Value::Float(number) => {
                out.insert(prefix.to_string(), Leaf::Float(*number));
            }
            toml::Value::Boolean(flag) => {
                out.insert(prefix.to_string(), Leaf::Bool(*flag));
            }
            toml::Value::String(text) => {
                out.insert(prefix.to_string(), Leaf::Str(text.clone()));
            }
            toml::Value::Array(items) => {
                let texts: Vec<String> = items
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
                    .collect();
                out.insert(prefix.to_string(), Leaf::Strs(texts));
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
        let compiled: BTreeMap<String, Leaf> = compiled().into_iter().collect();
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
        let changed = CONFIG_TOML.replace("max_trie_nodes = 4096\n", "max_trie_nodes = 4097\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["max_trie_nodes: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("min_cancelled_forks = 1\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["min_cancelled_forks: no key in the file"]);
        let floated = CONFIG_TOML.replace(
            "min_hit_rate_at_50pct_shared_prefix = 0.40",
            "min_hit_rate_at_50pct_shared_prefix = 0.41",
        );
        assert_eq!(
            drift(&floated),
            ["min_hit_rate_at_50pct_shared_prefix: drifted"]
        );
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&String> = all.iter().map(|(key, _)| key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_matrix_cells_are_the_full_cross_product_of_the_configured_axes() {
        let mut expected = Vec::new();
        for kind in MATRIX_KINDS {
            for path in MATRIX_PATHS {
                for kind_type in MATRIX_TYPES {
                    expected.push(cell_name(kind, path, kind_type));
                }
                if kind == "expert_generation" {
                    for kind_type in MATRIX_EXTRA_EXPERT_TYPES {
                        expected.push(cell_name(kind, path, kind_type));
                    }
                }
            }
        }
        let mut cells: Vec<String> = MATRIX_CELLS.iter().map(|cell| cell.to_string()).collect();
        expected.sort();
        cells.sort();
        assert_eq!(cells, expected);
        assert_eq!(MATRIX_CELLS.len(), 52);
    }

    #[test]
    fn the_workload_limits_follow_the_pinned_node_and_byte_budgets() {
        assert_eq!(LIMITS_MAIN.max_trie_nodes, MAX_TRIE_NODES);
        assert_eq!(LIMITS_MAIN.max_kv_page_bytes, MAX_KV_PAGE_BYTES);
        assert_eq!(LIMITS_MAIN.max_metadata_bytes, MAX_METADATA_BYTES);
        assert_eq!(
            LIMITS_NODES_ONLY.max_kv_page_bytes,
            NODES_ONLY_RUN_MAX_KV_PAGE_BYTES
        );
        assert_eq!(
            LIMITS_BYTES_ONLY.max_trie_nodes,
            BYTES_ONLY_RUN_MAX_TRIE_NODES
        );
        // The metadata limits are at least a node's 1 KiB each; only the bytes-only limit is
        // exactly that (main 8 MiB and nodes-only 64 MiB are looser than 4096 x 1 KiB).
        for workload in Workload::ALL {
            let limits = workload.limits();
            assert!(limits.max_metadata_bytes >= limits.max_trie_nodes * METADATA_BYTES_PER_NODE);
        }
        assert_eq!(
            LIMITS_BYTES_ONLY.max_metadata_bytes,
            LIMITS_BYTES_ONLY.max_trie_nodes * METADATA_BYTES_PER_NODE
        );
        // Exactly one binding limit per workload: 16 KiB a node.
        assert_eq!(
            LIMITS_MAIN.max_trie_nodes * KV_PAGE_BYTES_PER_NODE,
            MAX_KV_PAGE_BYTES / 2
        );
        const {
            assert!(
                LIMITS_NODES_ONLY.max_trie_nodes * KV_PAGE_BYTES_PER_NODE
                    < LIMITS_NODES_ONLY.max_kv_page_bytes
            )
        };
        const {
            assert!(
                LIMITS_BYTES_ONLY.max_kv_page_bytes / KV_PAGE_BYTES_PER_NODE
                    < LIMITS_BYTES_ONLY.max_trie_nodes
            )
        };
        // The over-capacity fill reaches the binding limit exactly.
        let nodes = OVER_CAPACITY_FILL_PROMPTS * (OVER_CAPACITY_PROMPT_TOKENS / CHUNK_TOKENS);
        assert_eq!(nodes as u64, LIMITS_NODES_ONLY.max_trie_nodes);
        assert_eq!(
            nodes as u64 * KV_PAGE_BYTES_PER_NODE,
            LIMITS_BYTES_ONLY.max_kv_page_bytes
        );
    }

    #[test]
    fn the_page_shape_gives_sixteen_kib_a_node() {
        let per_node = CHUNK_TOKENS as u64
            * TINY_MODEL.num_kv_heads
            * TINY_MODEL.head_dim
            * 4
            * 2
            * TINY_MODEL.layers;
        assert_eq!(per_node, KV_PAGE_BYTES_PER_NODE);
        assert_eq!(
            TINY_MODEL.hidden_size,
            TINY_MODEL.num_heads * TINY_MODEL.head_dim
        );
    }

    #[test]
    fn the_hit_rate_target_is_reachable_with_a_cold_first_request() {
        // 63 of 64 requests reuse the 128 shared tokens of a 256-token prompt.
        let reachable =
            ((N_REQUESTS - 1) * SHARED_PREFIX_TOKENS) as f64 / (N_REQUESTS * PROMPT_TOKENS) as f64;
        assert!(
            reachable >= MIN_HIT_RATE_AT_50PCT_SHARED_PREFIX,
            "{reachable}"
        );
        assert_eq!(PROMPT_TOKENS % CHUNK_TOKENS as u64, 0);
        assert_eq!(SHARED_PREFIX_TOKENS % CHUNK_TOKENS as u64, 0);
    }

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, crate::experiments::r005::metrics::METRIC_NAMES);
    }

    #[test]
    fn the_manifest_names_this_harness_as_its_entrypoint() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        assert_eq!(
            manifest["entrypoint"].as_str(),
            Some("cargo run --release --locked -p ptr-bench -- exact-prefix-reuse <seed>")
        );
    }

    #[test]
    fn the_preregistered_table_is_the_compiled_choices() {
        let config: toml::Table = CONFIG_TOML.parse().expect("config.toml parses");
        let table = config["preregistration"]
            .as_table()
            .expect("[preregistration]");
        let int = |key: &str| table[key].as_integer().expect(key);
        let text = |key: &str| table[key].as_str().expect(key);
        assert_eq!(int("max_new_tokens"), MAX_NEW_TOKENS as i64);
        assert_eq!(int("intermediate_size"), INTERMEDIATE_SIZE as i64);
        assert_eq!(text("rope_theta").parse::<f32>(), Ok(ROPE_THETA));
        assert_eq!(text("rms_norm_eps").parse::<f32>(), Ok(RMS_NORM_EPS));
        assert_eq!(text("qk_norm_main"), QK_NORM_MAIN);
        assert_eq!(text("qk_norm_transform_arm"), QK_NORM_TRANSFORM_ARM);
        assert_eq!(text("ntk_factor").parse::<f32>(), Ok(NTK_FACTOR));
        assert_eq!(
            int("ntk_original_max_positions"),
            NTK_ORIGINAL_MAX_POSITIONS as i64
        );
        assert_eq!(
            int("over_capacity_prompt_tokens"),
            OVER_CAPACITY_PROMPT_TOKENS as i64
        );
        assert_eq!(
            int("over_capacity_fill_prompts"),
            OVER_CAPACITY_FILL_PROMPTS as i64
        );
        assert_eq!(
            int("over_capacity_extra_prompts"),
            OVER_CAPACITY_EXTRA_PROMPTS as i64
        );
        assert_eq!(
            int("saturation_max_sessions"),
            SATURATION_MAX_SESSIONS as i64
        );
        assert_eq!(
            int("saturation_excess_requests"),
            SATURATION_EXCESS_REQUESTS as i64
        );
        assert_eq!(text("corpus_selection_digest_main"), SELECTION_DIGEST_MAIN);
        assert_eq!(
            text("corpus_selection_digest_nodes_only"),
            SELECTION_DIGEST_NODES_ONLY
        );
        assert_eq!(
            text("corpus_selection_digest_bytes_only"),
            SELECTION_DIGEST_BYTES_ONLY
        );
        assert_eq!(text("harness"), "exact-prefix-reuse");
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        assert_eq!(table["seeds"], manifest["seeds"]);
    }
}
