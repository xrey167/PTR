//! The preregistered parameters of R012, compiled in.
//!
//! The constants of the first section mirror the keys of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values of the `[preregistration]` table the draft preregistration
//! pins (also compared with the file) and the values this harness chose where the file is
//! silent. R012 is `planned`: nothing here is frozen.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R012-approximate-taint-containment/config.toml"
);

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R012-approximate-taint-containment/experiment.toml"
);

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "approximate-taint-containment";

/// The runner's command, as the draft preregistration states it. `experiment.toml` carries it
/// from the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- approximate-taint-containment <seed>";

// ---- one constant per key of [pinned_thresholds] ----------------------------------------------
pub const MAX_APPROXIMATE_RUN_ACTIONS_ADMITTED: u64 = 0;
pub const MAX_APPROXIMATE_SEMDB_PROMOTIONS: u64 = 0;
pub const MIN_APPROXIMATE_CANDIDATE_PATH_ATTEMPTS: u64 = 1;
pub const MAX_APPROXIMATE_CANDIDATE_EXACT_ADMISSIONS: u64 = 0;
pub const MIN_APPROXIMATE_ACTION_ADMISSION_ATTEMPTS: u64 = 1;
pub const MIN_APPROXIMATE_SEMDB_PROMOTION_ATTEMPTS: u64 = 1;
pub const MIN_TAINTED_CANDIDATE_RELOAD_ATTEMPTS: u64 = 1;
pub const MAX_TAINT_LOST: u64 = 0;
pub const MIN_FRESH_EXACT_RECOMPUTATION_CONTROLS: u64 = 1;
pub const MIN_CANDIDATE_TAINT_HOPS: u64 = 2;
pub const MIN_APPROXIMATE_POD_RESULT_PROMOTION_ATTEMPTS: u64 = 1;
pub const MAX_APPROXIMATE_POD_RESULT_PROMOTIONS_ACCEPTED: u64 = 0;
pub const APPROXIMATE_PRODUCER: &str = "pinned stub backend returning KvReuseMode::Approximate";
pub const MIN_MISCLASSIFIED_REVOCABLE_AS_RAW_ATTEMPTS: u64 = 1;
pub const MAX_MISCLASSIFIED_RAW_ADMISSIONS: u64 = 0;
pub const SEGMENT_KIND_AUTHORITY: &str = "ingress record";
pub const MIN_APPROXIMATE_OUTPUT_REINTRODUCED_AS_RAW_ATTEMPTS: u64 = 1;
pub const MAX_LAUNDERED_EXACT_ADMISSIONS: u64 = 0;
pub const RUNTIME_PRODUCED_CONTENT_SEGMENT_KIND: &str =
    "Derived (never Raw), provenance and Approximate taint propagate until exact recomputation";
pub const MIN_EXACT_ECHO_CASES: u64 = 1;
pub const MAX_TAINT_CLEARED_BY_ECHO: u64 = 0;
pub const TAINT_CLEARANCE_RULE: &str = "only exact recomputation of the same derived value from its original untainted inputs with the same operation";
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const DTYPE: &str = "f32";
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
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
pub const CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE: bool = true;
pub const CACHE_NAMESPACE_PARTITION: &str = "Exact-only catalog guard: no Approximate-produced entry is ever inserted or looked up (owner decision 2026-10-07); tainted candidates live in the runtime taint registry, not in a catalog namespace";
pub const PRODUCTION_MODES: [&str; 2] = ["Exact", "Approximate"];
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 9] = [
    "seeds",
    "test_model_layers",
    "weight_seed",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "corpus_generator_seed",
];
pub const TAINT_CLEARING_RULE_2026_10_07: &str = "only a cold exact recomputation (kv_report == None) with the same recipe and clean inputs; always clears the Derivation subject; clears Content and Action subjects only on byte-identical recomputed digests; an echo never clears";
pub const RAW_ADMISSION_RULE_2026_10_07: &str = "Raw is admitted only when it equals the request's own ingress record, otherwise MisclassifiedRawSegment; tainted content resubmitted as Raw keeps its taint (fail closed, false positives accepted); exact whole-segment digest only";
pub const RAW_LAUNDERING_NOT_DETECTED: [&str; 3] = ["paraphrase", "substring", "whitespace edit"];
pub const MAX_LIVE_TAINT_SUBJECTS: u64 = 65536;
pub const TAINT_REGISTRY_WHEN_FULL: &str =
    "new Approximate ingest refused before anything is appended; never evict";
pub const TAINT_LEDGER_EVENT: &str = "KvTaintCommitted (tag 29), written taint-first";
pub const POD_OUTPUT_KIND_SPLIT_REFUSED: [&str; 4] = [
    "StateDelta",
    "VerifiedResult",
    "ActionProposal",
    "promote_verified_pod_output",
];
pub const POD_OUTPUT_KIND_SPLIT_STORED: [&str; 5] = [
    "Observation",
    "Candidate",
    "ToolResult",
    "EnvironmentObservation",
    "Hypothesis",
];
pub const MERGE_AND_SEMDB_PROMOTION_OF_TAINTED_CONTENT: &str =
    "refused under every MergeAuthority, including Triage score 1.0";
pub const KV_CATALOG_GUARD: &str = "Exact-only guard, no KvSalt change (protects R005 results)";
pub const APPROXIMATE_PRODUCER_LOCATION: &str =
    "harness-only (bins/ptr-bench/src/experiments/r012/stub.rs)";
pub const HOP_DEFINITION: &str = "one derivation step: a run whose input includes a segment derived from a previous run's output; candidate_taint_hops counts the longest such chain";
pub const ATTEMPT_DEFINITION: &str = "an attempt counts only if it is made after the taint record is persisted (ledger append returned) and before any cold exact recomputation of the same recipe";

// ---- [preregistration] (the table frozen at the freeze) ---------------------------------------
pub const MAX_NEW_TOKENS: u64 = 8;
pub const INTERMEDIATE_SIZE: u64 = 128;
pub const ROPE_THETA: &str = "10000.0";
pub const RMS_NORM_EPS: &str = "1e-6";
pub const QK_NORM_MAIN: &str = "disabled";
pub const BASE_CHECKPOINT: &str = "none (seeded random)";
/// Tokens of every request the harness builds from the corpus (a Raw segment of the whole
/// ingress text, or an own-text Raw segment completed by one Derived segment).
pub const PROMPT_TOKENS: u64 = 64;
/// The prefill chunk of the backend (a catalog page): checked against `KV_CHUNK_TOKENS`.
pub const SEGMENT_TOKENS: u64 = 16;
/// Derivation hops of the candidate chain the run builds (`candidate_taint_hops` target).
pub const HOPS_PER_CHAIN: u64 = 3;

// ---- chosen by this harness (not in the config; echoed in every result and in the text) ------
/// The domain of the corpus selection digest.
pub const SELECTION_DOMAIN: &str = "ptr.r012.corpus-selection.v1";
/// Sequences of the corpus selection: sequence `i` for `i` in `0..SELECTION_SEQUENCES`.
pub const SELECTION_SEQUENCES: usize = 48;
/// The pinned digest of the corpus selection (`arms::selection_digest` recomputes it).
pub const SELECTION_DIGEST: &str =
    "sha256:9f48b499e6ec8735a762d8c746fbf6972d02c86c0b17f351a6680856350e3eef";
/// The domain of the shuffles that order the cases and assign the prompts to them.
pub const CASE_ORDER_LABEL: u64 = 12;
pub const PROMPT_ORDER_LABEL: u64 = 13;
/// Prompt slots one case may use (distinct corpus entries per case).
pub const PROMPT_SLOTS_PER_CASE: usize = 5;
/// The per-seed role, echoed in the result.
pub const PER_SEED_ROLE: &str =
    "the seed orders the cases and assigns corpus prompts to them; the corpus is fixed by corpus_generator_seed";
/// The producer identity bound into every recipe (the stub, also for the cold recomputation).
pub const STUB_NAME: &str = "r012-stub-approximate-producer";
pub const STUB_VERSION: &str = "1";
/// The canonical generation parameters bound into every recipe.
pub const GENERATION_PARAMETERS: &str = "max_new_tokens=8;prompt_tokens=64;greedy;byte-tokenizer";

/// The `[pinned_thresholds]` integer keys the harness judges (a key and its value), in the
/// order of the manifest's falsification text.
pub const THRESHOLDS: [(&str, u64); 21] = [
    (
        "min_approximate_action_admission_attempts",
        MIN_APPROXIMATE_ACTION_ADMISSION_ATTEMPTS,
    ),
    (
        "min_approximate_semdb_promotion_attempts",
        MIN_APPROXIMATE_SEMDB_PROMOTION_ATTEMPTS,
    ),
    (
        "min_approximate_candidate_path_attempts",
        MIN_APPROXIMATE_CANDIDATE_PATH_ATTEMPTS,
    ),
    (
        "min_approximate_pod_result_promotion_attempts",
        MIN_APPROXIMATE_POD_RESULT_PROMOTION_ATTEMPTS,
    ),
    (
        "min_tainted_candidate_reload_attempts",
        MIN_TAINTED_CANDIDATE_RELOAD_ATTEMPTS,
    ),
    (
        "min_fresh_exact_recomputation_controls",
        MIN_FRESH_EXACT_RECOMPUTATION_CONTROLS,
    ),
    ("min_candidate_taint_hops", MIN_CANDIDATE_TAINT_HOPS),
    (
        "min_misclassified_revocable_as_raw_attempts",
        MIN_MISCLASSIFIED_REVOCABLE_AS_RAW_ATTEMPTS,
    ),
    (
        "min_approximate_output_reintroduced_as_raw_attempts",
        MIN_APPROXIMATE_OUTPUT_REINTRODUCED_AS_RAW_ATTEMPTS,
    ),
    ("min_exact_echo_cases", MIN_EXACT_ECHO_CASES),
    (
        "max_selection_digest_mismatches",
        MAX_SELECTION_DIGEST_MISMATCHES,
    ),
    ("max_procedure_mismatches", MAX_PROCEDURE_MISMATCHES),
    ("max_model_pin_mismatches", MAX_MODEL_PIN_MISMATCHES),
    (
        "max_approximate_run_actions_admitted",
        MAX_APPROXIMATE_RUN_ACTIONS_ADMITTED,
    ),
    (
        "max_approximate_semdb_promotions",
        MAX_APPROXIMATE_SEMDB_PROMOTIONS,
    ),
    (
        "max_approximate_candidate_exact_admissions",
        MAX_APPROXIMATE_CANDIDATE_EXACT_ADMISSIONS,
    ),
    (
        "max_approximate_pod_result_promotions_accepted",
        MAX_APPROXIMATE_POD_RESULT_PROMOTIONS_ACCEPTED,
    ),
    ("max_taint_lost", MAX_TAINT_LOST),
    (
        "max_misclassified_raw_admissions",
        MAX_MISCLASSIFIED_RAW_ADMISSIONS,
    ),
    (
        "max_laundered_exact_admissions",
        MAX_LAUNDERED_EXACT_ADMISSIONS,
    ),
    ("max_taint_cleared_by_echo", MAX_TAINT_CLEARED_BY_ECHO),
];

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

/// A leaf of the flattened config, for the drift test.
#[derive(Clone, Debug, PartialEq)]
enum Leaf {
    Int(i64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
}

impl Leaf {
    fn strs(items: &[&str]) -> Self {
        Self::Strs(items.iter().map(|item| (*item).to_string()).collect())
    }
}

fn int(value: u64) -> Leaf {
    Leaf::Int(i64::try_from(value).expect("a pinned value fits"))
}

fn text(value: &str) -> Leaf {
    Leaf::Str(value.to_string())
}

/// Every key of `[pinned_thresholds]` as a compiled constant (a table flattens to its dotted
/// path).
fn compiled() -> Vec<(&'static str, Leaf)> {
    let mut all: Vec<(&'static str, Leaf)> = THRESHOLDS
        .iter()
        .map(|(key, value)| (*key, int(*value)))
        .collect();
    all.extend([
        ("approximate_producer", text(APPROXIMATE_PRODUCER)),
        ("segment_kind_authority", text(SEGMENT_KIND_AUTHORITY)),
        (
            "runtime_produced_content_segment_kind",
            text(RUNTIME_PRODUCED_CONTENT_SEGMENT_KIND),
        ),
        ("taint_clearance_rule", text(TAINT_CLEARANCE_RULE)),
        (
            "derived_state_carriers.ptr_catalog_entry",
            Leaf::strs(&["exact_only_guard"]),
        ),
        (
            "derived_state_carriers.taint_registry_subject",
            Leaf::strs(&["mode", "taint"]),
        ),
        (
            "derived_state_carriers.tainted_candidate",
            Leaf::strs(&["mode", "taint"]),
        ),
        ("seed_aggregation", text(SEED_AGGREGATION)),
        ("weights", text(WEIGHTS)),
        ("weight_seed", int(WEIGHT_SEED)),
        ("hidden_size", int(HIDDEN_SIZE)),
        ("num_heads", int(NUM_HEADS)),
        ("num_kv_heads", int(NUM_KV_HEADS)),
        ("head_dim", int(HEAD_DIM)),
        ("vocab_size", int(VOCAB_SIZE)),
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
        ("dtype", text(DTYPE)),
        ("corpus", text(CORPUS)),
        ("corpus_generator_seed", int(CORPUS_GENERATOR_SEED)),
        ("corpus_selection_rule", text(CORPUS_SELECTION_RULE)),
        (
            "corpus_selection_digest_recorded_in_run_json",
            Leaf::Bool(CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON),
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
        (
            "cache_namespace_partitioned_by_production_mode",
            Leaf::Bool(CACHE_NAMESPACE_PARTITIONED_BY_PRODUCTION_MODE),
        ),
        ("cache_namespace_partition", text(CACHE_NAMESPACE_PARTITION)),
        ("production_modes", Leaf::strs(&PRODUCTION_MODES)),
        (
            "procedure_parameters_recorded_per_cell",
            Leaf::strs(&PROCEDURE_PARAMETERS_RECORDED_PER_CELL),
        ),
        (
            "taint_clearing_rule_2026_10_07",
            text(TAINT_CLEARING_RULE_2026_10_07),
        ),
        (
            "raw_admission_rule_2026_10_07",
            text(RAW_ADMISSION_RULE_2026_10_07),
        ),
        (
            "raw_laundering_not_detected",
            Leaf::strs(&RAW_LAUNDERING_NOT_DETECTED),
        ),
        ("max_live_taint_subjects", int(MAX_LIVE_TAINT_SUBJECTS)),
        ("taint_registry_when_full", text(TAINT_REGISTRY_WHEN_FULL)),
        ("taint_ledger_event", text(TAINT_LEDGER_EVENT)),
        (
            "pod_output_kind_split.refused_with_typed_error",
            Leaf::strs(&POD_OUTPUT_KIND_SPLIT_REFUSED),
        ),
        (
            "pod_output_kind_split.stored_as_tainted",
            Leaf::strs(&POD_OUTPUT_KIND_SPLIT_STORED),
        ),
        (
            "merge_and_semdb_promotion_of_tainted_content",
            text(MERGE_AND_SEMDB_PROMOTION_OF_TAINTED_CONTENT),
        ),
        ("kv_catalog_guard", text(KV_CATALOG_GUARD)),
        (
            "approximate_producer_location",
            text(APPROXIMATE_PRODUCER_LOCATION),
        ),
        ("hop_definition", text(HOP_DEFINITION)),
        ("attempt_definition", text(ATTEMPT_DEFINITION)),
    ]);
    all
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
                out.insert(
                    prefix.to_string(),
                    Leaf::Strs(
                        items
                            .iter()
                            .map(|item| item.as_str().expect("a string list").to_string())
                            .collect(),
                    ),
                );
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
            // The dated owner-decision keys carry their date in the file name; the compiled
            // name drops it only for the clearance rule, which the file keeps as a plain key.
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
    fn the_preregistration_table_is_the_compiled_choices() {
        let config: toml::Table = CONFIG_TOML.parse().expect("config.toml parses");
        let table = config["preregistration"]
            .as_table()
            .expect("[preregistration]");
        let integer = |key: &str| table[key].as_integer().unwrap_or_else(|| panic!("{key}"));
        let string = |key: &str| table[key].as_str().unwrap_or_else(|| panic!("{key}"));
        assert_eq!(string("harness"), BENCHMARK);
        assert_eq!(integer("max_new_tokens"), MAX_NEW_TOKENS as i64);
        assert_eq!(integer("intermediate_size"), INTERMEDIATE_SIZE as i64);
        assert_eq!(string("rope_theta"), ROPE_THETA);
        assert_eq!(string("rms_norm_eps"), RMS_NORM_EPS);
        assert_eq!(string("qk_norm_main"), QK_NORM_MAIN);
        assert_eq!(string("base_checkpoint"), BASE_CHECKPOINT);
        assert_eq!(integer("prompt_tokens"), PROMPT_TOKENS as i64);
        assert_eq!(integer("segment_tokens"), SEGMENT_TOKENS as i64);
        assert_eq!(integer("hops_per_chain"), HOPS_PER_CHAIN as i64);
        assert_eq!(
            integer("corpus_generator_seed"),
            CORPUS_GENERATOR_SEED as i64
        );
        let seeds: Vec<i64> = table["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .filter_map(toml::Value::as_integer)
            .collect();
        assert_eq!(
            seeds,
            declared_seeds()
                .iter()
                .map(|s| *s as i64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_thresholds_the_harness_judges_are_the_configs() {
        let file = file_table(CONFIG_TOML);
        for (key, value) in THRESHOLDS {
            assert_eq!(file.get(key), Some(&int(value)), "{key}");
        }
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
        assert_eq!(shape.layers, TEST_MODEL_LAYERS);
        assert_eq!(pins::INTERMEDIATE_SIZE as u64, INTERMEDIATE_SIZE);
        assert_eq!(pins::MAX_NEW_TOKENS as u64, MAX_NEW_TOKENS);
        assert_eq!(
            pins::canonical_digest(&pins::shape_fields()),
            TINY_MODEL_CONFIG_DIGEST
        );
        assert_eq!(
            MAX_NEW_TOKENS as usize,
            ptr_runtime::ReferenceKvLimits::default().max_new_tokens
        );
        assert_eq!(SEGMENT_TOKENS as usize, ptr_runtime::KV_CHUNK_TOKENS);
        assert_eq!(
            SEGMENT_TOKENS as usize,
            crate::experiments::layered_kv::corpus::CHUNK_TOKENS
        );
    }
}
