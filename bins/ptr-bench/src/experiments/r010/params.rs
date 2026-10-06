//! The preregistered parameters of R010, compiled in.
//!
//! The constants of the first section mirror the keys of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values the file leaves open (the draft preregistration pins them
//! as the `[preregistration]` table). R010 is `planned`: nothing here is frozen.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R010-moe-expert-kv-revocation/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R010-moe-expert-kv-revocation/experiment.toml"
);

// ---- one constant per key of [pinned_thresholds] (a structured key flattens to its dotted path) ----
pub const MAX_STALE_PAGES_SERVED: u64 = 0;
pub const MAX_WRONGLY_DROPPED_PAGES: u64 = 0;
pub const MAX_BITWISE_MISMATCH_COUNT: u64 = 0;
pub const DETERMINISTIC_ROUTING: &str = "dropless, batch-invariant";
pub const MIN_NONDETERMINISTIC_REUSE_ATTEMPTS: u64 = 1;
pub const MAX_NONDETERMINISTIC_EXACT_TAGS: u64 = 0;
pub const MIN_DETERMINISTIC_REVOCATION_ATTEMPTS: u64 = 1;
pub const MIN_PAGES_ABOVE_J_BEFORE_REVOCATION: u64 = 1;
pub const MIN_PAGES_AT_OR_BELOW_J_BEFORE_REVOCATION: u64 = 1;
pub const MIN_DETERMINISTIC_REUSE_EVENTS: u64 = 1;
pub const MIN_POST_REVOCATION_REUSE_PROBES: u64 = 1;
pub const MIN_RETAINED_PAGES_REUSED_AFTER_REVOCATION: u64 = 1;
pub const MIN_RECOMPUTED_PAGES_ABOVE_J_SERVED: u64 = 1;
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const REVOKED_EXPERT_LAYER_J: u64 = 1;
pub const MIN_INVALIDATION_SUPPRESSED_ARMS: u64 = 1;
pub const MIN_RESTORE_ARMS: u64 = 1;
pub const MIN_POST_EVENT_LOOKUPS: u64 = 1;
pub const MAX_STALE_HITS: u64 = 0;
pub const MIN_INPUT_REVOCATIONS_PERFORMED: u64 = 1;
pub const MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS: u64 = 1;
pub const MAX_REVOKED_LINEAGE_ENTRY_HITS: u64 = 0;
pub const MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES: u64 = 1;
pub const MAX_MISMATCHING_RUNS_TAGGED_EXACT: u64 = 0;
pub const MIN_NONDETERMINISTIC_DIVERGENCE_OBSERVED: u64 = 1;
pub const CLAIM_SCOPE: &str = "kv_memory_only";
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const MIN_EXPERT_GENERATION_REVOCATIONS_PERFORMED: u64 = 1;
pub const MIN_PAGES_ABOVE_LAYER_CACHED_BEFORE_REVOCATION: u64 = 1;
pub const MIN_MATCHING_POST_REVOCATION_LOOKUPS: u64 = 1;
pub const MAX_STALE_EXPERT_GENERATION_HITS: u64 = 0;
pub const MIN_RESTORED_PRE_REVOCATION_ENTRIES: u64 = 1;
pub const EMPTY_RESTART_ARM_ROLE: &str = "reported_separate_arm";
pub const MIN_INFLIGHT_EXPERT_REVOCATION_RACE_CASES: u64 = 1;
pub const MAX_OUTPUTS_AFTER_EXPERT_REVOCATION: u64 = 0;
pub const MIN_INFLIGHT_INPUT_RACE_CASES: u64 = 1;
pub const MAX_OUTPUTS_AFTER_INPUT_INVALIDATION: u64 = 0;
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 14] = [
    "seeds",
    "test_model_layers",
    "revoked_expert_layer_j",
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
];
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const PROBED_PAGE_SET_DIGEST_MUST_EQUAL: &str = "cached_page_set_digest_above_j";
pub const INFLIGHT_RACE_CELLS: [[&str; 2]; 4] = [
    ["revocation", "expert"],
    ["revocation", "semantic_input"],
    ["supersession", "expert"],
    ["supersession", "semantic_input"],
];
pub const MIN_INFLIGHT_RACE_CASES_PER_CELL: u64 = 1;
pub const MAX_INFLIGHT_OUTPUTS_AFTER_INVALIDATION_PER_CELL: u64 = 0;
pub const INVALIDATION_COVERAGE_MATRIX_KINDS: [&str; 5] = [
    "base_model",
    "expert_generation",
    "semantic_input",
    "memory_input",
    "derived_input",
];
pub const INVALIDATION_COVERAGE_MATRIX_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INVALIDATION_COVERAGE_MATRIX_PATHS: [&str; 4] = ["normal", "lag", "restore", "inflight"];
pub const INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION: [&str; 1] = ["fence"];
pub const INVALIDATION_COVERAGE_MATRIX_CELLS: [&str; 44] = [
    "base_model_normal_revocation",
    "base_model_normal_supersession",
    "base_model_lag_revocation",
    "base_model_lag_supersession",
    "base_model_restore_revocation",
    "base_model_restore_supersession",
    "base_model_inflight_revocation",
    "base_model_inflight_supersession",
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
pub const MIN_EXPERT_QUARANTINE_KV_CASES: u64 = 1;
pub const MAX_KV_REUSE_AFTER_EXPERT_FENCE: u64 = 0;
pub const EXPERT_FENCE_INVALIDATION: &str = "any-cause fence of a chain expert invalidates entries at the fence commit index; entries record the per-expert fence epoch";
pub const TINY_MODEL_CONFIG_LAYERS: u64 = 4;
pub const TINY_MODEL_CONFIG_HIDDEN_SIZE: u64 = 64;
pub const TINY_MODEL_CONFIG_NUM_HEADS: u64 = 4;
pub const TINY_MODEL_CONFIG_NUM_KV_HEADS: u64 = 2;
pub const TINY_MODEL_CONFIG_HEAD_DIM: u64 = 16;
pub const TINY_MODEL_CONFIG_VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_WEIGHT_SEED: u64 = 20240607;
pub const TINY_MODEL_CONFIG_NUM_EXPERTS: u64 = 4;
pub const TINY_MODEL_CONFIG_TOP_K: u64 = 2;
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:fccf7bd315c89853010d486c2f1e11344f5d1e29fa0c8791eef69b5e246f84f0";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY_NUM_EXPERTS: u64 = 4;
pub const MOE_TOPOLOGY_TOP_K: u64 = 2;
pub const MOE_TOPOLOGY_ROUTER_LAYOUT: &str = "linear softmax top-k, dropless, batch-invariant";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const DECLARED_DEPENDENCY_KINDS: [&str; 5] =
    ["semantic", "memory", "derived", "base_model", "expert"];
pub const DEPENDENCY_LIST_NARROWING_REASON: &str =
    "narrowed: the MoE expert-revocation workload has no adapters";

// ---- the values the config leaves open (the draft preregistration's `[preregistration]`) ---

/// The subcommand and the manifest's entrypoint.
pub const BENCHMARK: &str = "moe-expert-kv-revocation";
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- moe-expert-kv-revocation <seed>";

/// Generated tokens per request: the backend default (`ReferenceKvLimits::default`).
pub const MAX_NEW_TOKENS: u64 = 8;
/// Prompt tokens of a run request and the page size (the backend's chunk).
pub const PROMPT_TOKENS: u64 = 64;
pub const PAGE_TOKENS: u64 = 16;
/// The seed of the MoE weight domain (separate from the dense stream by its domain string).
pub const MOE_SEED: u64 = WEIGHT_SEED;
pub const MOE_LAYERS: &str = "every decoder layer";
pub const EXPERT_INTERMEDIATE_SIZE: u64 = 64;
pub const TOP_K_NORMALISATION: &str = "none: the selected probabilities are used as is";
pub const TIE_BREAK: &str = "lowest expert index wins";
pub const MASKED_EXPERT: &str = "stays in the softmax denominator with selection score -inf";
/// The revoked expert index `i` of layer `j = 1`: the most-routed expert of layer 1 on the
/// probe prompt (the harness recomputes it and requires it to be routed at least once).
pub const REVOKED_EXPERT_INDEX: u64 = 0;
/// The batch-variant negative control: capacity in tokens per expert per step, over a batch of
/// two co-batched requests.
pub const BATCH_VARIANT_CAPACITY: u64 = 1;
pub const COBATCH_REQUESTS: u64 = 2;
/// A binding tracks every live expert of every MoE layer.
pub const EXPERT_TRACKING: &str = "every live expert of every MoE layer";
/// Partial hits are computed in full and adopted afterwards in checked mode.
pub const PARTIAL_HIT_MODE: &str = "compute-then-adopt, checked";
/// Restart after an invalidation re-resolves to the live binding: a revoked or fenced expert
/// is masked, a superseded one moves to its live successor generation.
pub const RESTART_RESOLUTION: &str = "re-resolve to the live binding: revoked or fenced expert masked, superseded expert to its successor generation";
/// The seed names the order of the arms and places the in-flight race points; the corpus is
/// fixed by `corpus_generator_seed`.
pub const PER_SEED_ROLE: &str = "request order and race points";
pub const SELECTION_DOMAIN: &str = "ptr.r010.corpus-selection.v1";
pub const SELECTION_DIGEST: &str =
    "sha256:55a21badbe657d7bbee30769013d801c4f53136f6ea504e69cbefd3329652a05";

/// The layer of the revoked expert, as the config names it.
pub const J: usize = REVOKED_EXPERT_LAYER_J as usize;

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
        ("max_stale_pages_served", int(MAX_STALE_PAGES_SERVED)),
        ("max_wrongly_dropped_pages", int(MAX_WRONGLY_DROPPED_PAGES)),
        (
            "max_bitwise_mismatch_count",
            int(MAX_BITWISE_MISMATCH_COUNT),
        ),
        (
            "deterministic_routing",
            Leaf::Str(DETERMINISTIC_ROUTING.to_string()),
        ),
        (
            "min_nondeterministic_reuse_attempts",
            int(MIN_NONDETERMINISTIC_REUSE_ATTEMPTS),
        ),
        (
            "max_nondeterministic_exact_tags",
            int(MAX_NONDETERMINISTIC_EXACT_TAGS),
        ),
        (
            "min_deterministic_revocation_attempts",
            int(MIN_DETERMINISTIC_REVOCATION_ATTEMPTS),
        ),
        (
            "min_pages_above_j_before_revocation",
            int(MIN_PAGES_ABOVE_J_BEFORE_REVOCATION),
        ),
        (
            "min_pages_at_or_below_j_before_revocation",
            int(MIN_PAGES_AT_OR_BELOW_J_BEFORE_REVOCATION),
        ),
        (
            "min_deterministic_reuse_events",
            int(MIN_DETERMINISTIC_REUSE_EVENTS),
        ),
        (
            "min_post_revocation_reuse_probes",
            int(MIN_POST_REVOCATION_REUSE_PROBES),
        ),
        (
            "min_retained_pages_reused_after_revocation",
            int(MIN_RETAINED_PAGES_REUSED_AFTER_REVOCATION),
        ),
        (
            "min_recomputed_pages_above_j_served",
            int(MIN_RECOMPUTED_PAGES_ABOVE_J_SERVED),
        ),
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
        ("revoked_expert_layer_j", int(REVOKED_EXPERT_LAYER_J)),
        (
            "min_invalidation_suppressed_arms",
            int(MIN_INVALIDATION_SUPPRESSED_ARMS),
        ),
        ("min_restore_arms", int(MIN_RESTORE_ARMS)),
        ("min_post_event_lookups", int(MIN_POST_EVENT_LOOKUPS)),
        ("max_stale_hits", int(MAX_STALE_HITS)),
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
            "max_mismatching_runs_tagged_exact",
            int(MAX_MISMATCHING_RUNS_TAGGED_EXACT),
        ),
        (
            "min_nondeterministic_divergence_observed",
            int(MIN_NONDETERMINISTIC_DIVERGENCE_OBSERVED),
        ),
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
            "min_expert_generation_revocations_performed",
            int(MIN_EXPERT_GENERATION_REVOCATIONS_PERFORMED),
        ),
        (
            "min_pages_above_layer_cached_before_revocation",
            int(MIN_PAGES_ABOVE_LAYER_CACHED_BEFORE_REVOCATION),
        ),
        (
            "min_matching_post_revocation_lookups",
            int(MIN_MATCHING_POST_REVOCATION_LOOKUPS),
        ),
        (
            "max_stale_expert_generation_hits",
            int(MAX_STALE_EXPERT_GENERATION_HITS),
        ),
        (
            "min_restored_pre_revocation_entries",
            int(MIN_RESTORED_PRE_REVOCATION_ENTRIES),
        ),
        (
            "empty_restart_arm_role",
            Leaf::Str(EMPTY_RESTART_ARM_ROLE.to_string()),
        ),
        (
            "min_inflight_expert_revocation_race_cases",
            int(MIN_INFLIGHT_EXPERT_REVOCATION_RACE_CASES),
        ),
        (
            "max_outputs_after_expert_revocation",
            int(MAX_OUTPUTS_AFTER_EXPERT_REVOCATION),
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
        ("invalidation_types", Leaf::strs(&INVALIDATION_TYPES)),
        (
            "probed_page_set_digest_must_equal",
            Leaf::Str(PROBED_PAGE_SET_DIGEST_MUST_EQUAL.to_string()),
        ),
        (
            "inflight_race_cells",
            Leaf::Strs(
                INFLIGHT_RACE_CELLS
                    .iter()
                    .map(|pair| pair.join("|"))
                    .collect(),
            ),
        ),
        (
            "min_inflight_race_cases_per_cell",
            int(MIN_INFLIGHT_RACE_CASES_PER_CELL),
        ),
        (
            "max_inflight_outputs_after_invalidation_per_cell",
            int(MAX_INFLIGHT_OUTPUTS_AFTER_INVALIDATION_PER_CELL),
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
            "invalidation_coverage_matrix.extra_types.expert_generation",
            Leaf::strs(&INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION),
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
        (
            "min_expert_quarantine_kv_cases",
            int(MIN_EXPERT_QUARANTINE_KV_CASES),
        ),
        (
            "max_kv_reuse_after_expert_fence",
            int(MAX_KV_REUSE_AFTER_EXPERT_FENCE),
        ),
        (
            "expert_fence_invalidation",
            Leaf::Str(EXPERT_FENCE_INVALIDATION.to_string()),
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
            "tiny_model_config.num_experts",
            int(TINY_MODEL_CONFIG_NUM_EXPERTS),
        ),
        ("tiny_model_config.top_k", int(TINY_MODEL_CONFIG_TOP_K)),
        (
            "tiny_model_config_digest",
            Leaf::Str(TINY_MODEL_CONFIG_DIGEST.to_string()),
        ),
        (
            "tiny_model_config_digest_rule",
            Leaf::Str(TINY_MODEL_CONFIG_DIGEST_RULE.to_string()),
        ),
        ("tokenizer_pin", Leaf::Str(TOKENIZER_PIN.to_string())),
        ("moe_topology.num_experts", int(MOE_TOPOLOGY_NUM_EXPERTS)),
        ("moe_topology.top_k", int(MOE_TOPOLOGY_TOP_K)),
        (
            "moe_topology.router_layout",
            Leaf::Str(MOE_TOPOLOGY_ROUTER_LAYOUT.to_string()),
        ),
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
        let changed = CONFIG_TOML.replace("weight_seed = 20240607\n", "weight_seed = 20240608\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["weight_seed: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("weight_seed = 20240607\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["weight_seed: no key in the file"]);
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
                if kind == "expert_generation" {
                    for ty in INVALIDATION_COVERAGE_MATRIX_EXTRA_TYPES_EXPERT_GENERATION {
                        expected.push(cell_name(kind, path, ty));
                    }
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
        assert_eq!(cells.len(), 44);
    }

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, crate::experiments::r010::metrics::METRIC_NAMES);
    }

    #[test]
    fn the_pins_the_harness_shares_are_the_configs() {
        use crate::experiments::layered_kv::pins;
        assert_eq!(
            crate::experiments::layered_kv::SEQUENCE_SEED,
            CORPUS_GENERATOR_SEED
        );
        let shape = pins::SHAPE;
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
        let mut fields = pins::shape_fields();
        fields.push(("num_experts", TINY_MODEL_CONFIG_NUM_EXPERTS));
        fields.push(("top_k", TINY_MODEL_CONFIG_TOP_K));
        assert_eq!(pins::canonical_digest(&fields), TINY_MODEL_CONFIG_DIGEST);
        assert_eq!(TINY_MODEL_CONFIG_NUM_EXPERTS, MOE_TOPOLOGY_NUM_EXPERTS);
        assert_eq!(TINY_MODEL_CONFIG_TOP_K, MOE_TOPOLOGY_TOP_K);
        let spec = ptr_pods::Qwen2MoeSpec::pinned(MOE_SEED, TEST_MODEL_LAYERS as usize);
        assert_eq!(spec.experts as u64, MOE_TOPOLOGY_NUM_EXPERTS);
        assert_eq!(spec.top_k as u64, MOE_TOPOLOGY_TOP_K);
        assert_eq!(spec.expert_intermediate as u64, EXPERT_INTERMEDIATE_SIZE);
        assert_eq!(spec.mode, ptr_pods::Qwen2RoutingMode::Dropless);
        assert_eq!(
            MAX_NEW_TOKENS as usize,
            ptr_runtime::ReferenceKvLimits::default().max_new_tokens
        );
        assert_eq!(PAGE_TOKENS as usize, ptr_runtime::KV_CHUNK_TOKENS);
        assert!(J < TEST_MODEL_LAYERS as usize - 1);
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
