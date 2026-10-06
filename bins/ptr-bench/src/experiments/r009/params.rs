//! The preregistered parameters of R009, compiled in.
//!
//! The constants of the first section mirror the keys of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values the file leaves open (the draft preregistration pins them
//! as the `[preregistration]` table). R009 is `planned`: nothing here is frozen.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

use ptr_pods::Qwen2AdapterKind;

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R009-layer-shared-lora-kv/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/runtime/R009-layer-shared-lora-kv/experiment.toml");

// ---- one constant per key of [pinned_thresholds] (a structured key flattens to its dotted path) ----
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const REQUIRED_INTERIOR_FIRST_ADAPTED_LAYER: u64 = 2;
pub const MIN_ALORA_PREFIX_REUSE_ATTEMPTS: u64 = 1;
pub const MIN_ADAPTER_REVOCATIONS_PERFORMED: u64 = 1;
pub const SHARED_FRACTION_TOLERANCE: f64 = 0.0;
pub const ADAPTER_TYPES_RUN: [&str; 4] = ["qkv_lora", "q_only_lora", "mlp_only_lora", "alora"];
pub const MAX_REVOKED_ADAPTER_ENTRY_HITS: u64 = 0;
pub const ALORA_INVOCATION_OFFSET_IN_PAGE: u64 = 7;
pub const MIN_ALORA_BOUNDARY_PAGE_CASES: u64 = 1;
pub const PAGE_TOKENS: u64 = 16;
pub const MIN_POST_REVOCATION_LOOKUPS_ON_ADAPTER_ENTRIES: u64 = 1;
pub const SHARING_MEASURED_IN: &str = "kv_rows";
pub const SHARED_ROW_FRACTION_DEFINITION: &str = "shared_kv_rows / total_kv_rows";
pub const MIN_ADAPTER_REVOCATIONS_PER_TYPE: u64 = 1;
pub const MIN_ADAPTER_REUSE_ATTEMPTS_PER_TYPE: u64 = 1;
pub const MIN_INPUT_REVOCATIONS_PERFORMED: u64 = 1;
pub const MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS: u64 = 1;
pub const MAX_REVOKED_LINEAGE_ENTRY_HITS: u64 = 0;
pub const MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES: u64 = 1;
pub const MIN_INVALIDATION_SUPPRESSED_ARMS: u64 = 1;
pub const MIN_RESTORE_ARMS: u64 = 1;
pub const MIN_POST_EVENT_LOOKUPS: u64 = 1;
pub const MAX_STALE_HITS: u64 = 0;
pub const MIN_BASE_PAGES_BEFORE_REVOKE: u64 = 1;
pub const MIN_ADAPTER_ENTRIES_BEFORE_REVOCATION: u64 = 1;
pub const MAX_ADAPTER_LOGITS_BITWISE_MISMATCH_COUNT: u64 = 0;
pub const QKV_LORA_FIRST_LAYERS: [u64; 2] = [0, 2];
pub const Q_ONLY_LORA_FIRST_LAYERS: [u64; 2] = [0, 2];
pub const MLP_ONLY_LORA_FIRST_LAYERS: [u64; 1] = [2];
pub const ALORA_INVOCATION_POSITION_TOKENS: u64 = 39;
pub const PROMPT_TOKENS: u64 = 64;
pub const CLAIM_SCOPE: &str = "kv_memory_only";
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const MIN_CROSS_ADAPTER_ATTEMPTS: u64 = 1;
pub const MIN_OLD_GENERATION_AS_NEW_ATTEMPTS: u64 = 1;
pub const MAX_CROSS_ADAPTER_HITS: u64 = 0;
pub const MAX_OLD_GENERATION_HITS: u64 = 0;
pub const MIN_RESTORED_PRE_REVOCATION_ENTRIES: u64 = 1;
pub const EMPTY_RESTART_ARM_ROLE: &str = "reported_separate_arm";
pub const MIN_INFLIGHT_ADAPTER_REVOCATION_RACE_CASES: u64 = 1;
pub const MAX_OUTPUTS_AFTER_ADAPTER_REVOCATION: u64 = 0;
pub const MIN_INFLIGHT_INPUT_RACE_CASES: u64 = 1;
pub const MAX_OUTPUTS_AFTER_INPUT_INVALIDATION: u64 = 0;
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 23] = [
    "seeds",
    "adapter_types_run",
    "qkv_lora_first_layers",
    "alora_invocation_position_tokens",
    "prompt_tokens",
    "test_model_layers",
    "weight_seed",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "corpus_generator_seed",
    "required_interior_first_adapted_layer",
    "alora_invocation_offset_in_page",
    "page_tokens",
    "q_only_lora_first_layers",
    "mlp_only_lora_first_layers",
    "invalidation_coverage_matrix",
    "tiny_model_config_digest",
    "tokenizer_pin",
    "moe_topology",
    "declared_dependency_kinds",
];
pub const ADAPTER_RESTART_OUTPUT_MISMATCHES_MAX: u64 = 0;
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INFLIGHT_RACE_COUNTERS_REPORTED: &str = "per dependency kind x invalidation type";
pub const IDENTITY_SET_CHECKS: [[&str; 2]; 1] = [[
    "inflight_adapter_cancelled_request_id_set_digest",
    "inflight_adapter_restarted_request_id_set_digest",
]];
pub const INVALIDATION_COVERAGE_MATRIX_KINDS: [&str; 5] = [
    "base_model",
    "adapter_generation",
    "semantic_input",
    "memory_input",
    "derived_input",
];
pub const INVALIDATION_COVERAGE_MATRIX_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INVALIDATION_COVERAGE_MATRIX_PATHS: [&str; 4] = ["normal", "lag", "restore", "inflight"];
pub const INVALIDATION_COVERAGE_MATRIX_CELLS: [&str; 40] = [
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
];
pub const MIN_INVALIDATION_MATRIX_CASES_PER_CELL: u64 = 1;
pub const MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL: u64 = 0;
pub const INVALIDATION_MATRIX_PATH_MEANING: &str = "normal = ordinary lookup after the invalidation; lag = lagging invalidation or index loss; restore = actual restore from snapshot or restart; inflight = in-flight race";
pub const TINY_MODEL_CONFIG_LAYERS: u64 = 4;
pub const TINY_MODEL_CONFIG_HIDDEN_SIZE: u64 = 64;
pub const TINY_MODEL_CONFIG_NUM_HEADS: u64 = 4;
pub const TINY_MODEL_CONFIG_NUM_KV_HEADS: u64 = 2;
pub const TINY_MODEL_CONFIG_HEAD_DIM: u64 = 16;
pub const TINY_MODEL_CONFIG_VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_WEIGHT_SEED: u64 = 20240607;
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY: &str = "none: dense tiny decoder";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const DECLARED_DEPENDENCY_KINDS: [&str; 5] =
    ["semantic", "memory", "derived", "base_model", "adapter"];
pub const DEPENDENCY_LIST_NARROWING_REASON: &str =
    "narrowed: the dense LoRA workload has no experts";

// ---- the values the config leaves open (the draft preregistration's `[preregistration]`) ---

/// The subcommand and the manifest's entrypoint.
pub const BENCHMARK: &str = "layer-shared-lora-kv";
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- layer-shared-lora-kv <seed>";

/// Generated tokens per request: the backend default (`ReferenceKvLimits::default`).
pub const MAX_NEW_TOKENS: u64 = 8;
/// LoRA rank and scale, and the seed A and B are drawn from (with the adapter's identity and
/// generation): the weight seed.
pub const LORA_RANK: u64 = 4;
pub const LORA_SCALE: &str = "2.0";
pub const ADAPTER_SEED: u64 = WEIGHT_SEED;
/// Where the adapters attach: every layer from the first adapted layer on.
pub const ADAPTER_LAYERS: &str = "every layer >= first adapted layer";
/// aLoRA adapts the QKV projections at every layer, active from the invocation position on.
pub const ALORA_FIRST_LAYER: u64 = 0;
pub const ALORA_PROJECTIONS: &str = "q, k, v";
/// The MLP adapter is a residual-space correction of the MLP output: y = mlp(x) + s B (A x).
pub const MLP_ADAPTER_FORM: &str = "residual-space correction of the MLP output";
/// Partial hits: the request is computed in full and the live layers are adopted afterwards
/// in checked mode (a transiently higher peak memory than a true partial computation).
pub const PARTIAL_HIT_MODE: &str = "compute-then-adopt, checked";
/// Restart after an invalidation re-resolves to the live binding and is judged against a full
/// prefill of that same binding.
pub const RESTART_RESOLUTION: &str = "re-resolve to the live binding: supersession to the new generation, revocation to the base model";
/// The catalog counts rows of the prompt only: the generated tokens' rows are not shared.
pub const TOTAL_KV_ROWS_COUNTS: &str = "prompt rows of every layer";
/// Identities of the adapters of a configuration run.
pub const ADAPTER_A: &str = "lora-a";
pub const ADAPTER_B: &str = "lora-b";
pub const ADAPTER_G: &str = "lora-g";
pub const MATRIX_ADAPTER: &str = "lora-m";
/// The matrix's binding: a QKV-LoRA first attached at layer 2.
pub const MATRIX_ADAPTER_FIRST_LAYER: u64 = 2;
/// The seed names the order of the six configurations and places the in-flight race points;
/// the corpus is fixed by `corpus_generator_seed`.
pub const PER_SEED_ROLE: &str = "request order and race points";
/// The domain of the corpus selection digest and its pinned value (a test reproduces it).
pub const SELECTION_DOMAIN: &str = "ptr.r009.corpus-selection.v1";
pub const SELECTION_DIGEST: &str =
    "sha256:8093b0b8b9859c83a0a145a993e2158be39c31e2b80812a4f7da4ab0a6bb117e";

// ---- configurations ------------------------------------------------------------------------

/// One pinned adapter configuration of the run.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub name: &'static str,
    /// The member of `adapter_types_run` it belongs to.
    pub adapter_type: &'static str,
    pub kind: Qwen2AdapterKind,
    pub first_layer: usize,
    /// Stream of the corpus prompt of this configuration.
    pub stream: u64,
}

impl Config {
    /// The first layer whose K and V differ from the base model's: `f` for a QKV-LoRA and for
    /// an aLoRA (whose invocation position then bounds the unchanged rows), `f + 1` for a
    /// Q-only or MLP-only adapter, whose delta reaches layer `f + 1` through the residual
    /// stream only.
    pub fn first_changed_layer(&self) -> usize {
        match self.kind {
            Qwen2AdapterKind::Qkv | Qwen2AdapterKind::Activated { .. } => self.first_layer,
            Qwen2AdapterKind::QOnly | Qwen2AdapterKind::MlpOnly => self.first_layer + 1,
        }
    }

    /// The rows of `layer` (of a `prompt`-token prompt) the preregistration predicts to be
    /// shared with the base model: all of them below the first changed layer, none from it
    /// on; for an aLoRA every layer shares the rows before the invocation position.
    pub fn predicted_rows(&self, layer: usize, prompt: usize) -> usize {
        match self.kind {
            Qwen2AdapterKind::Activated { invocation } => invocation.min(prompt),
            _ if layer < self.first_changed_layer() => prompt,
            _ => 0,
        }
    }

    pub fn is_alora(&self) -> bool {
        matches!(self.kind, Qwen2AdapterKind::Activated { .. })
    }
}

pub const INVOCATION: usize = ALORA_INVOCATION_POSITION_TOKENS as usize;

/// The six configurations the pinned run covers, in the manifest's order of adapter types.
pub const CONFIGS: [Config; 6] = [
    Config {
        name: "qkv_lora_f2",
        adapter_type: "qkv_lora",
        kind: Qwen2AdapterKind::Qkv,
        first_layer: 2,
        stream: 0,
    },
    Config {
        name: "qkv_lora_f0",
        adapter_type: "qkv_lora",
        kind: Qwen2AdapterKind::Qkv,
        first_layer: 0,
        stream: 1,
    },
    Config {
        name: "q_only_lora_f2",
        adapter_type: "q_only_lora",
        kind: Qwen2AdapterKind::QOnly,
        first_layer: 2,
        stream: 2,
    },
    Config {
        name: "q_only_lora_f0",
        adapter_type: "q_only_lora",
        kind: Qwen2AdapterKind::QOnly,
        first_layer: 0,
        stream: 3,
    },
    Config {
        name: "mlp_only_lora_f2",
        adapter_type: "mlp_only_lora",
        kind: Qwen2AdapterKind::MlpOnly,
        first_layer: 2,
        stream: 4,
    },
    Config {
        name: "alora_inv39",
        adapter_type: "alora",
        kind: Qwen2AdapterKind::Activated {
            invocation: INVOCATION,
        },
        first_layer: ALORA_FIRST_LAYER as usize,
        stream: 5,
    },
];

/// The matrix cell name of a dependency kind, path and invalidation type.
pub fn cell_name(kind: &str, path: &str, ty: &str) -> String {
    format!("{kind}_{path}_{ty}")
}

// ---- the compiled table, as the drift test compares it -------------------------------------

/// One leaf of the table.
#[derive(Clone, Debug, PartialEq)]
pub enum Leaf {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    Ints(Vec<i64>),
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
pub fn compiled() -> Vec<(&'static str, Leaf)> {
    vec![
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
        (
            "required_interior_first_adapted_layer",
            int(REQUIRED_INTERIOR_FIRST_ADAPTED_LAYER),
        ),
        (
            "min_alora_prefix_reuse_attempts",
            int(MIN_ALORA_PREFIX_REUSE_ATTEMPTS),
        ),
        (
            "min_adapter_revocations_performed",
            int(MIN_ADAPTER_REVOCATIONS_PERFORMED),
        ),
        (
            "shared_fraction_tolerance",
            Leaf::Float(SHARED_FRACTION_TOLERANCE),
        ),
        ("adapter_types_run", Leaf::strs(&ADAPTER_TYPES_RUN)),
        (
            "max_revoked_adapter_entry_hits",
            int(MAX_REVOKED_ADAPTER_ENTRY_HITS),
        ),
        (
            "alora_invocation_offset_in_page",
            int(ALORA_INVOCATION_OFFSET_IN_PAGE),
        ),
        (
            "min_alora_boundary_page_cases",
            int(MIN_ALORA_BOUNDARY_PAGE_CASES),
        ),
        ("page_tokens", int(PAGE_TOKENS)),
        (
            "min_post_revocation_lookups_on_adapter_entries",
            int(MIN_POST_REVOCATION_LOOKUPS_ON_ADAPTER_ENTRIES),
        ),
        (
            "sharing_measured_in",
            Leaf::Str(SHARING_MEASURED_IN.to_string()),
        ),
        (
            "shared_row_fraction_definition",
            Leaf::Str(SHARED_ROW_FRACTION_DEFINITION.to_string()),
        ),
        (
            "min_adapter_revocations_per_type",
            int(MIN_ADAPTER_REVOCATIONS_PER_TYPE),
        ),
        (
            "min_adapter_reuse_attempts_per_type",
            int(MIN_ADAPTER_REUSE_ATTEMPTS_PER_TYPE),
        ),
        (
            "min_input_revocations_performed",
            int(MIN_INPUT_REVOCATIONS_PERFORMED),
        ),
        (
            "min_post_revocation_lookups_matching_tokens",
            int(MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS),
        ),
        (
            "max_revoked_lineage_entry_hits",
            int(MAX_REVOKED_LINEAGE_ENTRY_HITS),
        ),
        (
            "min_same_tokens_different_generation_cases",
            int(MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES),
        ),
        (
            "min_invalidation_suppressed_arms",
            int(MIN_INVALIDATION_SUPPRESSED_ARMS),
        ),
        ("min_restore_arms", int(MIN_RESTORE_ARMS)),
        ("min_post_event_lookups", int(MIN_POST_EVENT_LOOKUPS)),
        ("max_stale_hits", int(MAX_STALE_HITS)),
        (
            "min_base_pages_before_revoke",
            int(MIN_BASE_PAGES_BEFORE_REVOKE),
        ),
        (
            "min_adapter_entries_before_revocation",
            int(MIN_ADAPTER_ENTRIES_BEFORE_REVOCATION),
        ),
        (
            "max_adapter_logits_bitwise_mismatch_count",
            int(MAX_ADAPTER_LOGITS_BITWISE_MISMATCH_COUNT),
        ),
        (
            "qkv_lora_first_layers",
            Leaf::Ints(
                QKV_LORA_FIRST_LAYERS
                    .iter()
                    .map(|value| *value as i64)
                    .collect(),
            ),
        ),
        (
            "q_only_lora_first_layers",
            Leaf::Ints(
                Q_ONLY_LORA_FIRST_LAYERS
                    .iter()
                    .map(|value| *value as i64)
                    .collect(),
            ),
        ),
        (
            "mlp_only_lora_first_layers",
            Leaf::Ints(
                MLP_ONLY_LORA_FIRST_LAYERS
                    .iter()
                    .map(|value| *value as i64)
                    .collect(),
            ),
        ),
        (
            "alora_invocation_position_tokens",
            int(ALORA_INVOCATION_POSITION_TOKENS),
        ),
        ("prompt_tokens", int(PROMPT_TOKENS)),
        ("claim_scope", Leaf::Str(CLAIM_SCOPE.to_string())),
        ("seed_aggregation", Leaf::Str(SEED_AGGREGATION.to_string())),
        ("weights", Leaf::Str(WEIGHTS.to_string())),
        ("weight_seed", int(WEIGHT_SEED)),
        ("hidden_size", int(HIDDEN_SIZE)),
        ("num_heads", int(NUM_HEADS)),
        ("num_kv_heads", int(NUM_KV_HEADS)),
        ("head_dim", int(HEAD_DIM)),
        ("vocab_size", int(VOCAB_SIZE)),
        (
            "min_cross_adapter_attempts",
            int(MIN_CROSS_ADAPTER_ATTEMPTS),
        ),
        (
            "min_old_generation_as_new_attempts",
            int(MIN_OLD_GENERATION_AS_NEW_ATTEMPTS),
        ),
        ("max_cross_adapter_hits", int(MAX_CROSS_ADAPTER_HITS)),
        ("max_old_generation_hits", int(MAX_OLD_GENERATION_HITS)),
        (
            "min_restored_pre_revocation_entries",
            int(MIN_RESTORED_PRE_REVOCATION_ENTRIES),
        ),
        (
            "empty_restart_arm_role",
            Leaf::Str(EMPTY_RESTART_ARM_ROLE.to_string()),
        ),
        (
            "min_inflight_adapter_revocation_race_cases",
            int(MIN_INFLIGHT_ADAPTER_REVOCATION_RACE_CASES),
        ),
        (
            "max_outputs_after_adapter_revocation",
            int(MAX_OUTPUTS_AFTER_ADAPTER_REVOCATION),
        ),
        (
            "min_inflight_input_race_cases",
            int(MIN_INFLIGHT_INPUT_RACE_CASES),
        ),
        (
            "max_outputs_after_input_invalidation",
            int(MAX_OUTPUTS_AFTER_INPUT_INVALIDATION),
        ),
        ("corpus", Leaf::Str(CORPUS.to_string())),
        ("corpus_generator_seed", int(CORPUS_GENERATOR_SEED)),
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
            int(MAX_SELECTION_DIGEST_MISMATCHES),
        ),
        ("max_procedure_mismatches", int(MAX_PROCEDURE_MISMATCHES)),
        (
            "procedure_parameters_recorded_per_cell",
            Leaf::strs(&PROCEDURE_PARAMETERS_RECORDED_PER_CELL),
        ),
        (
            "adapter_restart_output_mismatches_max",
            int(ADAPTER_RESTART_OUTPUT_MISMATCHES_MAX),
        ),
        ("invalidation_types", Leaf::strs(&INVALIDATION_TYPES)),
        (
            "inflight_race_counters_reported",
            Leaf::Str(INFLIGHT_RACE_COUNTERS_REPORTED.to_string()),
        ),
        (
            "identity_set_checks",
            Leaf::Strs(
                IDENTITY_SET_CHECKS
                    .iter()
                    .map(|pair| pair.join("|"))
                    .collect(),
            ),
        ),
        (
            "invalidation_coverage_matrix.kinds",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_KINDS),
        ),
        (
            "invalidation_coverage_matrix.types",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_TYPES),
        ),
        (
            "invalidation_coverage_matrix.paths",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_PATHS),
        ),
        (
            "invalidation_coverage_matrix.cells",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_CELLS),
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
            Leaf::Str(INVALIDATION_MATRIX_PATH_MEANING.to_string()),
        ),
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
        ("max_model_pin_mismatches", int(MAX_MODEL_PIN_MISMATCHES)),
        (
            "declared_dependency_kinds",
            Leaf::strs(&DECLARED_DEPENDENCY_KINDS),
        ),
        (
            "dependency_list_narrowing_reason",
            Leaf::Str(DEPENDENCY_LIST_NARROWING_REASON.to_string()),
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
                let leaf = if items.iter().all(|item| item.is_integer()) {
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
        let changed = CONFIG_TOML.replace("page_tokens = 16\n", "page_tokens = 17\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["page_tokens: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("page_tokens = 16\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["page_tokens: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_matrix_cells_are_the_full_cross_product_of_the_configured_axes() {
        let mut expected = Vec::new();
        for kind in INVALIDATION_COVERAGE_MATRIX_KINDS {
            for path in INVALIDATION_COVERAGE_MATRIX_PATHS {
                for ty in INVALIDATION_COVERAGE_MATRIX_TYPES {
                    expected.push(cell_name(kind, path, ty));
                }
            }
        }
        let mut cells: Vec<String> = INVALIDATION_COVERAGE_MATRIX_CELLS
            .iter()
            .map(|cell| cell.to_string())
            .collect();
        expected.sort();
        cells.sort();
        assert_eq!(cells, expected);
        assert_eq!(cells.len(), 40);
    }

    #[test]
    fn the_configurations_are_the_pinned_adapter_types_and_first_layers() {
        let first_layers = |ty: &str| -> Vec<u64> {
            let mut layers: Vec<u64> = CONFIGS
                .iter()
                .filter(|config| config.adapter_type == ty)
                .map(|config| config.first_layer as u64)
                .collect();
            layers.sort_unstable();
            layers
        };
        assert_eq!(first_layers("qkv_lora"), QKV_LORA_FIRST_LAYERS);
        assert_eq!(first_layers("q_only_lora"), Q_ONLY_LORA_FIRST_LAYERS);
        assert_eq!(first_layers("mlp_only_lora"), MLP_ONLY_LORA_FIRST_LAYERS);
        let types: Vec<&str> = CONFIGS
            .iter()
            .map(|config| config.adapter_type)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut pinned = ADAPTER_TYPES_RUN.to_vec();
        pinned.sort_unstable();
        assert_eq!(types, pinned);
        // At least one configuration is interior (0 < f < L), and aLoRA's invocation is inside
        // a page at the pinned offset.
        assert!(CONFIGS
            .iter()
            .any(|config| config.first_layer as u64 == REQUIRED_INTERIOR_FIRST_ADAPTED_LAYER));
        assert_eq!(
            INVOCATION as u64 % PAGE_TOKENS,
            ALORA_INVOCATION_OFFSET_IN_PAGE
        );
        assert_eq!(PROMPT_TOKENS % PAGE_TOKENS, 0);
    }

    #[test]
    fn the_predicted_fractions_follow_the_preregistered_rules() {
        let rows = |name: &str| -> usize {
            let config = CONFIGS.iter().find(|config| config.name == name).unwrap();
            (0..TEST_MODEL_LAYERS as usize)
                .map(|layer| config.predicted_rows(layer, PROMPT_TOKENS as usize))
                .sum()
        };
        let total = (TEST_MODEL_LAYERS * PROMPT_TOKENS) as usize;
        assert_eq!(rows("qkv_lora_f2") * 2, total);
        assert_eq!(rows("qkv_lora_f0"), 0);
        assert_eq!(rows("q_only_lora_f2") * 4, total * 3);
        assert_eq!(rows("q_only_lora_f0") * 4, total);
        assert_eq!(rows("mlp_only_lora_f2") * 4, total * 3);
        assert_eq!(rows("alora_inv39"), 4 * 39);
    }

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, crate::experiments::r009::metrics::METRIC_NAMES);
    }

    #[test]
    fn the_pins_the_harness_shares_are_the_configs() {
        assert_eq!(
            crate::experiments::layered_kv::SEQUENCE_SEED,
            CORPUS_GENERATOR_SEED
        );
        let shape = crate::experiments::layered_kv::pins::SHAPE;
        assert_eq!(shape.layers, TINY_MODEL_CONFIG_LAYERS);
        assert_eq!(shape.hidden_size, TINY_MODEL_CONFIG_HIDDEN_SIZE);
        assert_eq!(shape.num_heads, TINY_MODEL_CONFIG_NUM_HEADS);
        assert_eq!(shape.num_kv_heads, TINY_MODEL_CONFIG_NUM_KV_HEADS);
        assert_eq!(shape.head_dim, TINY_MODEL_CONFIG_HEAD_DIM);
        assert_eq!(shape.vocab_size, TINY_MODEL_CONFIG_VOCAB_SIZE);
        assert_eq!(shape.weight_seed, TINY_MODEL_CONFIG_WEIGHT_SEED);
        assert_eq!(shape.weight_seed, WEIGHT_SEED);
        assert_eq!(shape.layers, TEST_MODEL_LAYERS);
        assert_eq!(shape.hidden_size, HIDDEN_SIZE);
        assert_eq!(
            crate::experiments::layered_kv::pins::canonical_digest(
                &crate::experiments::layered_kv::pins::shape_fields()
            ),
            TINY_MODEL_CONFIG_DIGEST
        );
        assert_eq!(LORA_RANK as usize, ptr_pods::QWEN2_ADAPTER_PINNED_RANK);
        assert_eq!(
            LORA_SCALE.parse::<f32>(),
            Ok(ptr_pods::QWEN2_ADAPTER_PINNED_SCALE)
        );
        assert_eq!(
            MAX_NEW_TOKENS as usize,
            ptr_runtime::ReferenceKvLimits::default().max_new_tokens
        );
        assert_eq!(PAGE_TOKENS as usize, ptr_runtime::KV_CHUNK_TOKENS);
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<i64> = manifest["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(toml::Value::as_integer)
            .collect();
        assert_eq!(seeds, vec![17, 29, 43, 71, 101]);
    }
}
