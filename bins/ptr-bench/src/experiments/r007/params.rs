//! The preregistered parameters of R007, compiled in.
//!
//! The constants of the first section mirror the keys of `[pinned_thresholds]` in the
//! experiment's `config.toml`, which is read at compile time; the drift test fails when a
//! constant differs from the file, and when the file holds a key no constant mirrors. The
//! second section holds the values this harness chose where the file is silent (the corpus
//! selection, the run order of the probes, the span's identity, the raised limits). R007 is
//! `planned`: nothing here is frozen.

// The thresholds complete the compiled table the drift test compares with the file; the
// harness evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/runtime/R007-span-forget/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/runtime/R007-span-forget/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "span-forget";

/// The runner's command, as the preregistration states it. `experiment.toml` carries it
/// from the freeze on (until then its `entrypoint` is empty).
pub const ENTRYPOINT: &str = "cargo run --release --locked -p ptr-bench -- span-forget <seed>";

// ---- one constant per key of [pinned_thresholds] ----------------------------------------------
pub const CANARY_BEARING_SPAN_CASES: u64 = 1;
pub const MIN_SPAN_INSERTIONS: u64 = 1;
pub const MIN_SPAN_EVICTIONS: u64 = 1;
pub const MIN_DOWNSTREAM_RECOMPUTATIONS: u64 = 1;
pub const MIN_CANARY_PROBES: u64 = 1;
pub const MAX_STALE_DOWNSTREAM_ROWS_REUSED: u64 = 0;
pub const RECOMPUTE_TOKENS_EQUALS_TOKENS_AFTER_SPAN: bool = true;
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const MAX_CANARY_LOGIT_SHIFT: u64 = 0;
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const WEIGHT_SEED: u64 = 20240607;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const MIN_CANARY_SPAN_TOKENS: u64 = 1;
pub const MIN_DOWNSTREAM_ROWS_BEFORE_FORGET: u64 = 1;
pub const MIN_PREFIX_REUSE_TOKENS: u64 = 1;
pub const MIN_POST_FORGET_PROBES: u64 = 1;
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20240607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const CORPUS_SELECTION_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const PROCEDURE_PARAMETERS_RECORDED_PER_CELL: [&str; 23] = [
    "seeds",
    "span_tokens",
    "canary_probes",
    "weight_seed",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "corpus_generator_seed",
    "canary_bearing_span_cases",
    "span_position",
    "span_start_offset_tokens",
    "system_prompt_tokens",
    "document_tokens",
    "question_tokens",
    "canary_strings_distinct",
    "tiny_model_config_digest",
    "tokenizer_pin",
    "moe_topology",
    "test_model_layers",
    "canary_generator_seed",
    "canary_expected_digest",
];
pub const SPAN_TOKENS: u64 = 256;
pub const SPAN_START_OFFSET_TOKENS: u64 = 1024;
pub const SYSTEM_PROMPT_TOKENS: u64 = 64;
pub const DOCUMENT_TOKENS: u64 = 4096;
pub const QUESTION_TOKENS: u64 = 64;
pub const CANARY_PROBES: u64 = 32;
pub const CANARY_STRINGS_DISTINCT: bool = true;
pub const CANARY_IDENTITY_SET_DIGEST_RECORDED_IN_RUN_JSON: bool = true;
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const TINY_MODEL_CONFIG_LAYERS: u64 = 4;
pub const TINY_MODEL_CONFIG_HIDDEN_SIZE: u64 = 64;
pub const TINY_MODEL_CONFIG_NUM_HEADS: u64 = 4;
pub const TINY_MODEL_CONFIG_NUM_KV_HEADS: u64 = 2;
pub const TINY_MODEL_CONFIG_HEAD_DIM: u64 = 16;
pub const TINY_MODEL_CONFIG_VOCAB_SIZE: u64 = 256;
pub const TINY_MODEL_CONFIG_WEIGHT_SEED: u64 = 20240607;
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY: &str = "none: dense tiny decoder";
pub const MODEL_PINS_RECORDED_IN_RUN_JSON: bool = true;
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;
pub const CANARY_GENERATOR_SEED: u64 = 20240607;
pub const CANARY_EXPECTED_DIGEST: &str =
    "65f9b8cd60e57cb38a8588c23d5bb494c7817fc10511353980f8be1c690efe19";
pub const CANARY_PROBE_VECTOR_ENTRIES: u64 = 256;
pub const REFERENCE_LIMITS_RAISED_IN_HARNESS_ONLY: bool = true;
pub const PREFIX_REUSE_TOKENS_EXPECTED: u64 = 1088;
pub const PREFIX_REUSE_CHUNKS_EXPECTED: u64 = 68;
pub const TOKENS_AFTER_SPAN_EXPECTED: u64 = 2880;
pub const OWNER_DECISION_2026_10_08_FORGET_MECHANISM: &str =
    "the span is a revocable PromptSegment (Derived or Memory); forget = revoke its lineage segment, then run the prompt without it (prefix reused, everything after the span recomputed); the README states the mechanism is revocation plus parent-chained chunk keys; a negative control with injected stale reuse (set_faults) shows the criterion is not vacuous";
pub const OWNER_DECISION_2026_10_08_CANARY_PROBE: &str =
    "per probe, the full next-token logits vector (all 256 entries) after the probe suffix; canary_logit_shift = max absolute difference over all 256 entries and all 32 probes; frozen before the preregistration";
pub const OWNER_DECISION_2026_10_08_DEFERRED: &str =
    "R008 and M016 stay planned and are deferred to the GPU phase (pretrained Qwen2.5-0.5B); no re-pinning to the tiny model";
pub const OWNER_DECISION_2026_10_08_REFERENCE_LIMITS: &str =
    "reference limits are raised in the R007 harness only via ReferenceKvLimits; the default stays 256";
pub const CANARY_SHIFT_REDUCTION: &str =
    "max over all 256 logits entries and all 32 probes of the absolute difference to the fresh prefill without the span";
pub const COMPARISON_RULE: &str = "bitwise on f32 bit patterns (to_bits), no tolerance";
pub const FORGET_MECHANISM: &str =
    "revoke the span's lineage segment, then run the prompt without it; not a mid-sequence eviction primitive";
pub const NEGATIVE_CONTROL: &str =
    "injected stale reuse (set_faults) must change the digests and trip the criteria; otherwise the run is vacuous";
pub const CANARY_GENERATOR_RULE: &str =
    "canary_i = first 16 hex chars of sha256(seed + colon + i) for i in 0 to 31; digest = sha256 of the canaries joined by newlines";
pub const NOISE_THRESHOLD: &str = "none (bitwise only)";
pub const SPAN_POSITION: &str = "middle";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";

// ---- chosen by this harness (not in the config; echoed in every result and in the text) ------
/// The domain of the corpus selection digest.
pub const SELECTION_DOMAIN: &str = "ptr.r007.corpus-selection.v1";
/// Sequences of the corpus selection: sequence `i` for `i` in `0..SELECTION_SEQUENCES`.
pub const SELECTION_SEQUENCES: usize = 8;
/// Tokens of every selection sequence: the longest role (the document tail).
pub const SELECTION_SEQUENCE_TOKENS: usize = 2816;
/// The pinned digest of the corpus selection (`cases::selection_digest` recomputes it).
pub const SELECTION_DIGEST: &str =
    "sha256:48de01d2f91315b6d124dce668e01f783e599a00677f92b7325cf1467d3308c1";
/// The domains of the seeded shuffles: which corpus sequence fills which role, and the order
/// in which the 32 canaries are probed.
pub const ROLE_ORDER_LABEL: u64 = 7;
pub const PROBE_ORDER_LABEL: u64 = 107;
/// The per-seed role, echoed in the result.
pub const PER_SEED_ROLE: &str = "the seed picks which corpus sequence fills the system, document head, document tail and question roles and the order in which the 32 canaries are probed; the corpus, the weights and the canaries are fixed by their pinned seeds";
/// Bytes of one canary string (16 hex characters, one byte token each).
pub const CANARY_BYTES: usize = 16;
/// Canaries 0 to 15 are embedded in the span (16 x 16 = 256 span tokens); 16 to 31 never are.
pub const CANARIES_IN_SPAN: usize = 16;
/// The span's lineage identity (a `Derived` prompt segment) and generation.
pub const SPAN_IDENTITY: &str = "span:canary-1";
pub const SPAN_GENERATION: u64 = 1;
/// Tokens a probe request may add to the longest prompt (the probe suffix is one chunk).
pub const PROBE_HEADROOM_TOKENS: usize = 16;

/// The token counts of the segments of a prompt: system, document head, span, document tail,
/// question. The prompt with the span is all five, the one without omits the span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub system: usize,
    pub head: usize,
    pub span: usize,
    pub tail: usize,
    pub question: usize,
}

/// The preregistered shape: 64 + (1024 + 256 + 2816 = 4096) + 64 tokens.
pub const PINNED_SHAPE: Shape = Shape {
    system: SYSTEM_PROMPT_TOKENS as usize,
    head: SPAN_START_OFFSET_TOKENS as usize,
    span: SPAN_TOKENS as usize,
    tail: DOCUMENT_TOKENS as usize - SPAN_START_OFFSET_TOKENS as usize - SPAN_TOKENS as usize,
    question: QUESTION_TOKENS as usize,
};

/// The reduced shape of the smoke test: the same span (256 tokens, so all 32 canaries and the
/// 16 embedded ones keep their roles) in a short context. No evidence.
pub const REDUCED_SHAPE: Shape = Shape {
    system: 16,
    head: 16,
    span: SPAN_TOKENS as usize,
    tail: 16,
    question: 16,
};

impl Shape {
    /// Tokens before the span (the prefix every run reuses).
    pub fn prefix(self) -> usize {
        self.system + self.head
    }

    /// Tokens of the document (head, span and tail).
    pub fn document(self) -> usize {
        self.head + self.span + self.tail
    }

    /// Tokens after the span (document tail and question).
    pub fn after_span(self) -> usize {
        self.tail + self.question
    }

    pub fn with_span(self) -> usize {
        self.prefix() + self.span + self.after_span()
    }

    pub fn without_span(self) -> usize {
        self.prefix() + self.after_span()
    }
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

/// A leaf of the flattened config, for the drift test.
#[derive(Clone, Debug, PartialEq)]
enum Leaf {
    Int(i64),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
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
    vec![
        ("canary_bearing_span_cases", int(CANARY_BEARING_SPAN_CASES)),
        ("min_span_insertions", int(MIN_SPAN_INSERTIONS)),
        ("min_span_evictions", int(MIN_SPAN_EVICTIONS)),
        (
            "min_downstream_recomputations",
            int(MIN_DOWNSTREAM_RECOMPUTATIONS),
        ),
        ("min_canary_probes", int(MIN_CANARY_PROBES)),
        (
            "max_stale_downstream_rows_reused",
            int(MAX_STALE_DOWNSTREAM_ROWS_REUSED),
        ),
        (
            "recompute_tokens_equals_tokens_after_span",
            Leaf::Bool(RECOMPUTE_TOKENS_EQUALS_TOKENS_AFTER_SPAN),
        ),
        ("seed_aggregation", text(SEED_AGGREGATION)),
        ("max_canary_logit_shift", int(MAX_CANARY_LOGIT_SHIFT)),
        ("noise_threshold", text(NOISE_THRESHOLD)),
        ("weights", text(WEIGHTS)),
        ("weight_seed", int(WEIGHT_SEED)),
        ("hidden_size", int(HIDDEN_SIZE)),
        ("num_heads", int(NUM_HEADS)),
        ("num_kv_heads", int(NUM_KV_HEADS)),
        ("head_dim", int(HEAD_DIM)),
        ("vocab_size", int(VOCAB_SIZE)),
        ("min_canary_span_tokens", int(MIN_CANARY_SPAN_TOKENS)),
        (
            "min_downstream_rows_before_forget",
            int(MIN_DOWNSTREAM_ROWS_BEFORE_FORGET),
        ),
        ("min_prefix_reuse_tokens", int(MIN_PREFIX_REUSE_TOKENS)),
        ("min_post_forget_probes", int(MIN_POST_FORGET_PROBES)),
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
            Leaf::Strs(
                PROCEDURE_PARAMETERS_RECORDED_PER_CELL
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
        ),
        ("span_tokens", int(SPAN_TOKENS)),
        ("span_position", text(SPAN_POSITION)),
        ("span_start_offset_tokens", int(SPAN_START_OFFSET_TOKENS)),
        ("system_prompt_tokens", int(SYSTEM_PROMPT_TOKENS)),
        ("document_tokens", int(DOCUMENT_TOKENS)),
        ("question_tokens", int(QUESTION_TOKENS)),
        ("canary_probes", int(CANARY_PROBES)),
        (
            "canary_strings_distinct",
            Leaf::Bool(CANARY_STRINGS_DISTINCT),
        ),
        (
            "canary_identity_set_digest_recorded_in_run_json",
            Leaf::Bool(CANARY_IDENTITY_SET_DIGEST_RECORDED_IN_RUN_JSON),
        ),
        ("test_model_layers", int(TEST_MODEL_LAYERS)),
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
        ("canary_generator_rule", text(CANARY_GENERATOR_RULE)),
        ("canary_generator_seed", int(CANARY_GENERATOR_SEED)),
        ("canary_expected_digest", text(CANARY_EXPECTED_DIGEST)),
        (
            "owner_decision_2026_10_08_forget_mechanism",
            text(OWNER_DECISION_2026_10_08_FORGET_MECHANISM),
        ),
        (
            "owner_decision_2026_10_08_canary_probe",
            text(OWNER_DECISION_2026_10_08_CANARY_PROBE),
        ),
        (
            "owner_decision_2026_10_08_deferred",
            text(OWNER_DECISION_2026_10_08_DEFERRED),
        ),
        (
            "owner_decision_2026_10_08_reference_limits",
            text(OWNER_DECISION_2026_10_08_REFERENCE_LIMITS),
        ),
        (
            "canary_probe_vector_entries",
            int(CANARY_PROBE_VECTOR_ENTRIES),
        ),
        ("canary_shift_reduction", text(CANARY_SHIFT_REDUCTION)),
        ("comparison_rule", text(COMPARISON_RULE)),
        (
            "prefix_reuse_tokens_expected",
            int(PREFIX_REUSE_TOKENS_EXPECTED),
        ),
        (
            "prefix_reuse_chunks_expected",
            int(PREFIX_REUSE_CHUNKS_EXPECTED),
        ),
        (
            "tokens_after_span_expected",
            int(TOKENS_AFTER_SPAN_EXPECTED),
        ),
        ("forget_mechanism", text(FORGET_MECHANISM)),
        ("negative_control", text(NEGATIVE_CONTROL)),
        (
            "reference_limits_raised_in_harness_only",
            Leaf::Bool(REFERENCE_LIMITS_RAISED_IN_HARNESS_ONLY),
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
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_pinned_shape_is_the_configs_and_the_derived_counts_are_the_expected_ones() {
        let shape = PINNED_SHAPE;
        assert_eq!(shape.system as u64, SYSTEM_PROMPT_TOKENS);
        assert_eq!(shape.head as u64, SPAN_START_OFFSET_TOKENS);
        assert_eq!(shape.span as u64, SPAN_TOKENS);
        assert_eq!(shape.document() as u64, DOCUMENT_TOKENS);
        assert_eq!(shape.question as u64, QUESTION_TOKENS);
        assert_eq!(shape.prefix() as u64, PREFIX_REUSE_TOKENS_EXPECTED);
        assert_eq!(shape.after_span() as u64, TOKENS_AFTER_SPAN_EXPECTED);
        assert_eq!(
            shape.prefix() / ptr_runtime::KV_CHUNK_TOKENS,
            PREFIX_REUSE_CHUNKS_EXPECTED as usize
        );
        assert_eq!(shape.with_span(), 4224);
        assert_eq!(shape.without_span(), 3968);
        assert_eq!(CANARIES_IN_SPAN * CANARY_BYTES, shape.span);
        assert_eq!(CANARY_PROBES as usize, 32);
        // Every segment boundary of both shapes is a chunk boundary.
        for shape in [PINNED_SHAPE, REDUCED_SHAPE] {
            for tokens in [
                shape.system,
                shape.head,
                shape.span,
                shape.tail,
                shape.question,
            ] {
                assert_eq!(tokens % ptr_runtime::KV_CHUNK_TOKENS, 0);
            }
        }
        const { assert!(SELECTION_SEQUENCE_TOKENS >= PINNED_SHAPE.tail) };
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
        assert_eq!(
            pins::canonical_digest(&pins::shape_fields()),
            TINY_MODEL_CONFIG_DIGEST
        );
        assert_eq!(WEIGHT_SEED, ptr_runtime::REFERENCE_KV_WEIGHT_SEED);
        assert_eq!(
            ptr_runtime::ReferenceKvLimits::default().max_prompt_tokens,
            256,
            "the crate default stays 256 (owner decision 5)"
        );
    }
}
