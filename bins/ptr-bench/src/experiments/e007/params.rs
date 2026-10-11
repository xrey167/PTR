//! The preregistered parameters of E007, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file
//! and when the file holds a key no constant mirrors. E007 is `planned`: nothing here is
//! frozen.

#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/system/E007-split-pod-kv/config.toml");

/// The experiment's manifest, read at compile time (the metric names and seeds come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/system/E007-split-pod-kv/experiment.toml");

pub const BENCHMARK: &str = "split-pod-kv";

// ---- the model (the R005 tiny reference stack) ----
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];
pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const TEST_MODEL_LAYERS: u64 = 4;
pub const SPLIT_LAYER: u64 = 2;
pub const HIDDEN_SIZE: u64 = 64;
pub const NUM_HEADS: u64 = 4;
pub const NUM_KV_HEADS: u64 = 2;
pub const HEAD_DIM: u64 = 16;
pub const VOCAB_SIZE: u64 = 256;
pub const WEIGHT_SEED: u64 = 20_240_607;
pub const WEIGHTS: &str = "seeded random (pinned seed and shape)";
pub const BOUNDARY_DTYPE: &str = "F32";
pub const TOKENIZER_PIN: &str = "ByteTokenizer, vocabulary 256 (no external revision)";
pub const MOE_TOPOLOGY: &str = "none: dense tiny decoder";
pub const TINY_MODEL_CONFIG_DIGEST: &str =
    "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a";
pub const TINY_MODEL_CONFIG_DIGEST_RULE: &str =
    "sha256 of canonical JSON (sorted keys, no spaces) of tiny_model_config";
pub const MAX_MODEL_PIN_MISMATCHES: u64 = 0;

// ---- the corpus and the model-computation cases ----
pub const CORPUS: &str = "synthetic token sequences from a seeded generator";
pub const CORPUS_GENERATOR_SEED: u64 = 20_240_607;
pub const CORPUS_SELECTION_RULE: &str =
    "deterministic generator, sequence i derived from seed and index i, in index order";
pub const SELECTION_DOMAIN: &str = "ptr.e007.corpus-selection.v1";
pub const SELECTION_DIGEST: &str =
    "sha256:52efb0dee3e036edc77172632c28a2ba072ce353fdce3cee55d380402e530aba";
pub const MAX_SELECTION_DIGEST_MISMATCHES: u64 = 0;
pub const MAX_PROCEDURE_MISMATCHES: u64 = 0;
pub const SEQUENCES: u64 = 24;
pub const PROMPT_TOKENS_MIN: u64 = 8;
pub const PROMPT_TOKENS_MAX: u64 = 64;
pub const DECODE_STEPS: u64 = 8;
pub const PAGE_TOKENS: u64 = 16;
pub const INTERMEDIATE_SIZE: u64 = 128;
pub const ROPE_THETA: &str = "10000.0";
pub const RMS_NORM_EPS: &str = "1e-6";
pub const KEY_ENCODING: &str = "PreRope";
pub const OUTPUT_DEFINITION: &str = "final logits bytes plus per-layer KV bytes on each pod";

// ---- frames, chunks and the receiver's bounds ----
pub const SMALL_FRAME_CHUNK_BYTES: u64 = 2048;
pub const PROBE_CHUNK_BYTES: u64 = 512;
pub const PROBE_FRAME_ROWS: u64 = 12;
pub const FRAME_BUDGET_MS: u64 = 100;
pub const FRAME_SPACING_MS: u64 = 25;
pub const LARGE_FRAME_TOKENS: u64 = 2048;
pub const LARGE_FRAME_WIDTH: u64 = 2048;
pub const LARGE_FRAME_DTYPE: &str = "f32";
pub const LARGE_FRAME_BYTES: u64 = 16_777_216;
pub const MAX_BODY_BYTES: u64 = 4_194_304;
pub const MAX_CHUNK_BYTES: u64 = 3_670_016;
pub const LARGE_FRAME_CHUNKS: u64 = 5;
pub const LARGE_FRAME_SEND_ORDER: [u64; 5] = [1, 0, 3, 2, 4];
pub const REORDER_WINDOW: u64 = 4;
pub const FRAME_DEADLINE_MS: u64 = 5000;
pub const MAX_INCOMPLETE_FRAMES_PER_PEER: u64 = 8;
pub const MAX_CHUNKS_IN_FLIGHT: u64 = 32;
pub const MAX_FRAME_TOMBSTONES_PER_PRINCIPAL: u64 = 64;
pub const CLOCK_SKEW_BOUND_MS: u64 = 1000;
pub const MAX_CLOCK_SKEW_MS: u64 = 1000;
pub const MAX_FRAME_DEADLINE_LIFETIME_MS: u64 = 5000;
pub const REPLAY_WINDOW_MS: u64 = 10_000;
pub const REPLAY_WINDOW_RULE: &str =
    "replay_window >= max_frame_deadline_lifetime + 2 * max_clock_skew (10000 >= 5000 + 2 * 1000 = 7000)";
pub const MAX_ASSEMBLED_FRAME_BYTES: u64 = 33_554_432;
/// Bytes buffered over all principals together: four maximal frames (invariant 8).
pub const MAX_TOTAL_BUFFERED_BYTES: u64 = 134_217_728;
pub const TOMBSTONE_OVERCAPACITY_FRAMES: u64 = 65;
pub const TOMBSTONE_STAGGER_MS: u64 = 10;
pub const LATE_WINDOW_REPLAY_DELAY_RANGE_MS: [u64; 2] = [9000, 10_000];
pub const LATE_WINDOW_REPLAY_DELAY_RANGE_RULE: &str =
    "[0.9 * replay_window_ms, replay_window_ms), half-open";
pub const CHUNK_ENVELOPE_AUTHENTICATION: &str =
    "Ed25519 signature of the upstream runtime over the complete envelope";
pub const UPSTREAM_PRINCIPAL: &str = "stage-a-runtime";
pub const ADDRESSED_TO: &str = "stage-b-host";
pub const STAGE_IDENTITY: &str = "stage-a";
/// The downstream stage that computes from the frames and keeps the cache.
pub const DOWNSTREAM_STAGE_IDENTITY: &str = "stage-b";
pub const SIGNING_KEY_SEED: u64 = 20_240_607;
pub const CHUNK_HEADER_FIELDS: [&str; 8] = [
    "frame_id",
    "chunk_index",
    "chunk_total",
    "chunk_digest",
    "frame_digest",
    "upstream_chain",
    "signed_absolute_deadline",
    "parent_operation_id",
];
pub const CHUNK_DIGEST_COVERS: [&str; 9] = [
    "frame_id",
    "chunk_index",
    "chunk_total",
    "frame_digest",
    "parent_operation_id",
    "signed_absolute_deadline",
    "upstream_chain",
    "chunk_bytes",
    "domain",
];
pub const CHUNK_ENVELOPE_SIGNED_FIELDS: [&str; 8] = [
    "frame_id",
    "parent_operation_id",
    "chunk_index",
    "chunk_count",
    "signed_absolute_deadline",
    "frame_digest",
    "chunk_digest",
    "upstream_chain",
];
pub const REASSEMBLY_IDENTITY: &str = "(principal, parent_operation_id, frame_id)";
pub const TOMBSTONE_BOUND: &str =
    "admitted_incomplete_frames + retained_tombstones (abandoned and completed) <= max_frame_tombstones_per_principal";
pub const COMPLETION_TOMBSTONE_RETENTION: &str = "signed_absolute_deadline + clock skew bound";
pub const ABANDONMENT_TOMBSTONE_RETENTION: &str = "abandonment_time + replay_window_ms";
pub const SATURATION_BEHAVIOUR: &str = "fail closed with typed FrameTableFull";
pub const SESSION_POLICY: &str =
    "chunks of one (principal, parent_operation_id, frame_id) from any session of the principal join one assembly; exactly one delivery results";

// ---- invalidation of the upstream stage ----
pub const INVALIDATION_TYPES: [&str; 2] = ["revocation", "supersession"];
pub const INVALIDATION_PATHS: [&str; 3] = ["normal", "lag", "inflight"];
pub const INVALIDATION_COVERAGE_MATRIX_KINDS: [&str; 1] = ["upstream_stage_generation"];
pub const INVALIDATION_COVERAGE_MATRIX_CELLS: [&str; 6] = [
    "upstream_stage_generation_normal_revocation",
    "upstream_stage_generation_normal_supersession",
    "upstream_stage_generation_lag_revocation",
    "upstream_stage_generation_lag_supersession",
    "upstream_stage_generation_inflight_revocation",
    "upstream_stage_generation_inflight_supersession",
];
pub const INVALIDATION_MATRIX_PATH_MEANING: &str = "normal = the Revoke message is delivered, then an ordinary lookup; lag = the Revoke message is dropped (never delivered) or delayed (delivered after the lookup); inflight = a use begun before the invalidation finishes after it";
pub const INVALIDATION_MATRIX_NOT_APPLICABLE: &str =
    "restore: no KV is restored on a Pod; the downstream cache has no persistence";
pub const REVALIDATED_AT_USE: [&str; 3] = [
    "stage generation",
    "stage manifest digest",
    "placement epoch",
];
pub const ADVERSARIAL_ATTEMPT_KINDS: [&str; 19] = [
    "chain_mismatch",
    "duplicate_chunk",
    "out_of_window",
    "bad_chunk_digest",
    "mixed_version_frame",
    "incomplete_frames_over_peer_limit",
    "queue_overflow",
    "splice",
    "cross_operation_splice",
    "non_allowlisted_type",
    "mislabelled_kv_payload",
    "kv_sized_allowlisted_smuggling",
    "wrong_shape",
    "positions_length",
    "mislabelled_frame_digest",
    "late_chunk_after_abandon",
    "overcapacity_frame",
    "full_frame_replay",
    "stale_upstream_request",
];
pub const VALID_FRAME_CASES: [&str; 3] = [
    "large_frame_16mib_chunked_control",
    "concurrent_valid_reassembly",
    "activation_control",
];
pub const SMUGGLING_PROBE_KINDS: [&str; 3] = [
    "mislabelled_kv_payload",
    "non_allowlisted_type",
    "kv_sized_allowlisted_smuggling",
];
pub const WIRE_KV_PAGE_BYTES_DEFINITION: &str = "bytes of any admitted payload that does not decode as an allowlisted schema, or that matches the digest of a KV page held by either pod or carries a window of one; legitimate traffic only: the activation control and every adversarial probe are excluded; every payload is scanned as sent (each chunk) and every valid frame as reassembled, because a page can straddle two chunks";
pub const WIRE_CAPTURE: &str =
    "the run's JSON, key wire_capture: every payload that crossed the PodWire frame codec, with its role, type id, size, digest and verdict";
pub const TRANSPORT: &str = "the real PodWire request frame codec (encode_request, decode_request, MAX_BODY_BYTES enforced) in process; no network";

// ---- the thresholds (all attempts and cases are counted from observed events) ----
pub const MIN_ATTEMPTS: u64 = 1;
pub const MAX_VIOLATIONS: u64 = 0;
pub const MIN_ACTIVATION_FRAMES_CROSSED: u64 = 192;
pub const MIN_VALID_FRAMES_SENT: u64 = 192;
pub const MIN_LARGE_FRAME_CASES: u64 = 1;
pub const MIN_INTERLEAVED_FRAME_PAIRS: u64 = 1;
pub const MIN_INTERLEAVED_CHUNK_SWITCHES: u64 = 1;
pub const TOMBSTONES_RETAINED_AT_SATURATION: u64 = 64;
pub const ABANDONED_FRAMES_IN_SATURATION_CASE: u64 = 64;
pub const MIN_FRAMES_ACCEPTED_AFTER_TOMBSTONE_EXPIRY: u64 = 1;
pub const MIN_STALE_UPSTREAM_ENTRIES_BEFORE_REVOCATION: u64 = 1;
pub const MIN_REVOKE_MESSAGES_DROPPED_OR_DELAYED: u64 = 2;
pub const MIN_CONTROL_TRIPS: u64 = 1;
pub const BUFFERED_BYTES_BASELINE: u64 = 0;
pub const MIN_STALE_WORKER_FRAME_RACE_CASES: u64 = 1;
pub const MIN_SEQUENCES_PER_SEED: u64 = 24;

/// A value of `config.toml`, flattened.
#[derive(Clone, Debug, PartialEq)]
pub enum Leaf {
    Int(i64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Strs(Vec<String>),
}

fn i(value: u64) -> Leaf {
    Leaf::Int(value as i64)
}

fn s(value: &str) -> Leaf {
    Leaf::Str(value.to_string())
}

fn ss(values: &[&str]) -> Leaf {
    Leaf::Strs(values.iter().map(|value| value.to_string()).collect())
}

fn is(values: &[u64]) -> Leaf {
    Leaf::Ints(values.iter().map(|value| *value as i64).collect())
}

/// Every key of `[pinned_thresholds]` as the harness compiles it, structured keys flattened
/// to dotted paths.
pub fn compiled() -> Vec<(&'static str, Leaf)> {
    vec![
        ("seed_aggregation", s(SEED_AGGREGATION)),
        ("seeds", is(&SEEDS)),
        ("test_model_layers", i(TEST_MODEL_LAYERS)),
        ("split_layer", i(SPLIT_LAYER)),
        ("hidden_size", i(HIDDEN_SIZE)),
        ("num_heads", i(NUM_HEADS)),
        ("num_kv_heads", i(NUM_KV_HEADS)),
        ("head_dim", i(HEAD_DIM)),
        ("vocab_size", i(VOCAB_SIZE)),
        ("weight_seed", i(WEIGHT_SEED)),
        ("weights", s(WEIGHTS)),
        ("boundary_dtype", s(BOUNDARY_DTYPE)),
        ("tokenizer_pin", s(TOKENIZER_PIN)),
        ("moe_topology", s(MOE_TOPOLOGY)),
        ("tiny_model_config_digest", s(TINY_MODEL_CONFIG_DIGEST)),
        (
            "tiny_model_config_digest_rule",
            s(TINY_MODEL_CONFIG_DIGEST_RULE),
        ),
        ("max_model_pin_mismatches", i(MAX_MODEL_PIN_MISMATCHES)),
        ("corpus", s(CORPUS)),
        ("corpus_generator_seed", i(CORPUS_GENERATOR_SEED)),
        ("corpus_selection_rule", s(CORPUS_SELECTION_RULE)),
        ("corpus_selection_domain", s(SELECTION_DOMAIN)),
        ("corpus_selection_digest", s(SELECTION_DIGEST)),
        (
            "max_selection_digest_mismatches",
            i(MAX_SELECTION_DIGEST_MISMATCHES),
        ),
        ("max_procedure_mismatches", i(MAX_PROCEDURE_MISMATCHES)),
        ("sequences", i(SEQUENCES)),
        ("prompt_tokens_min", i(PROMPT_TOKENS_MIN)),
        ("prompt_tokens_max", i(PROMPT_TOKENS_MAX)),
        ("decode_steps", i(DECODE_STEPS)),
        ("page_tokens", i(PAGE_TOKENS)),
        ("intermediate_size", i(INTERMEDIATE_SIZE)),
        ("rope_theta", s(ROPE_THETA)),
        ("rms_norm_eps", s(RMS_NORM_EPS)),
        ("key_encoding", s(KEY_ENCODING)),
        ("output_definition", s(OUTPUT_DEFINITION)),
        ("small_frame_chunk_bytes", i(SMALL_FRAME_CHUNK_BYTES)),
        ("probe_chunk_bytes", i(PROBE_CHUNK_BYTES)),
        ("probe_frame_rows", i(PROBE_FRAME_ROWS)),
        ("frame_budget_ms", i(FRAME_BUDGET_MS)),
        ("frame_spacing_ms", i(FRAME_SPACING_MS)),
        ("large_frame_tokens", i(LARGE_FRAME_TOKENS)),
        ("large_frame_width", i(LARGE_FRAME_WIDTH)),
        ("large_frame_dtype", s(LARGE_FRAME_DTYPE)),
        ("large_frame_bytes", i(LARGE_FRAME_BYTES)),
        ("max_body_bytes", i(MAX_BODY_BYTES)),
        ("max_chunk_bytes", i(MAX_CHUNK_BYTES)),
        ("large_frame_chunks", i(LARGE_FRAME_CHUNKS)),
        ("large_frame_send_order", is(&LARGE_FRAME_SEND_ORDER)),
        ("reorder_window", i(REORDER_WINDOW)),
        ("frame_deadline_ms", i(FRAME_DEADLINE_MS)),
        (
            "max_incomplete_frames_per_peer",
            i(MAX_INCOMPLETE_FRAMES_PER_PEER),
        ),
        ("max_chunks_in_flight", i(MAX_CHUNKS_IN_FLIGHT)),
        (
            "max_frame_tombstones_per_principal",
            i(MAX_FRAME_TOMBSTONES_PER_PRINCIPAL),
        ),
        ("clock_skew_bound_ms", i(CLOCK_SKEW_BOUND_MS)),
        ("max_clock_skew_ms", i(MAX_CLOCK_SKEW_MS)),
        (
            "max_frame_deadline_lifetime_ms",
            i(MAX_FRAME_DEADLINE_LIFETIME_MS),
        ),
        ("replay_window_ms", i(REPLAY_WINDOW_MS)),
        ("replay_window_rule", s(REPLAY_WINDOW_RULE)),
        ("max_assembled_frame_bytes", i(MAX_ASSEMBLED_FRAME_BYTES)),
        ("max_total_buffered_bytes", i(MAX_TOTAL_BUFFERED_BYTES)),
        (
            "tombstone_overcapacity_frames",
            i(TOMBSTONE_OVERCAPACITY_FRAMES),
        ),
        ("tombstone_stagger_ms", i(TOMBSTONE_STAGGER_MS)),
        (
            "late_window_replay_delay_range_ms",
            is(&LATE_WINDOW_REPLAY_DELAY_RANGE_MS),
        ),
        (
            "late_window_replay_delay_range_rule",
            s(LATE_WINDOW_REPLAY_DELAY_RANGE_RULE),
        ),
        (
            "chunk_envelope_authentication",
            s(CHUNK_ENVELOPE_AUTHENTICATION),
        ),
        ("upstream_principal", s(UPSTREAM_PRINCIPAL)),
        ("addressed_to", s(ADDRESSED_TO)),
        ("stage_identity", s(STAGE_IDENTITY)),
        ("signing_key_seed", i(SIGNING_KEY_SEED)),
        ("chunk_header_fields", ss(&CHUNK_HEADER_FIELDS)),
        ("chunk_digest_covers", ss(&CHUNK_DIGEST_COVERS)),
        (
            "chunk_envelope_signed_fields",
            ss(&CHUNK_ENVELOPE_SIGNED_FIELDS),
        ),
        ("reassembly_identity", s(REASSEMBLY_IDENTITY)),
        ("tombstone_bound", s(TOMBSTONE_BOUND)),
        (
            "completion_tombstone_retention",
            s(COMPLETION_TOMBSTONE_RETENTION),
        ),
        (
            "abandonment_tombstone_retention",
            s(ABANDONMENT_TOMBSTONE_RETENTION),
        ),
        ("saturation_behaviour", s(SATURATION_BEHAVIOUR)),
        ("session_policy", s(SESSION_POLICY)),
        ("invalidation_types", ss(&INVALIDATION_TYPES)),
        ("invalidation_paths", ss(&INVALIDATION_PATHS)),
        (
            "invalidation_coverage_matrix.kinds",
            ss(&INVALIDATION_COVERAGE_MATRIX_KINDS),
        ),
        (
            "invalidation_coverage_matrix.types",
            ss(&INVALIDATION_TYPES),
        ),
        (
            "invalidation_coverage_matrix.paths",
            ss(&INVALIDATION_PATHS),
        ),
        (
            "invalidation_coverage_matrix.cells",
            ss(&INVALIDATION_COVERAGE_MATRIX_CELLS),
        ),
        (
            "invalidation_matrix_path_meaning",
            s(INVALIDATION_MATRIX_PATH_MEANING),
        ),
        (
            "invalidation_matrix_not_applicable",
            s(INVALIDATION_MATRIX_NOT_APPLICABLE),
        ),
        ("revalidated_at_use", ss(&REVALIDATED_AT_USE)),
        ("adversarial_attempt_kinds", ss(&ADVERSARIAL_ATTEMPT_KINDS)),
        ("valid_frame_cases", ss(&VALID_FRAME_CASES)),
        ("smuggling_probe_kinds", ss(&SMUGGLING_PROBE_KINDS)),
        (
            "wire_kv_page_bytes_definition",
            s(WIRE_KV_PAGE_BYTES_DEFINITION),
        ),
        ("wire_capture", s(WIRE_CAPTURE)),
        ("transport", s(TRANSPORT)),
        ("min_attempts", i(MIN_ATTEMPTS)),
        ("max_violations", i(MAX_VIOLATIONS)),
        (
            "min_activation_frames_crossed",
            i(MIN_ACTIVATION_FRAMES_CROSSED),
        ),
        ("min_valid_frames_sent", i(MIN_VALID_FRAMES_SENT)),
        ("min_large_frame_cases", i(MIN_LARGE_FRAME_CASES)),
        (
            "min_interleaved_frame_pairs",
            i(MIN_INTERLEAVED_FRAME_PAIRS),
        ),
        (
            "min_interleaved_chunk_switches",
            i(MIN_INTERLEAVED_CHUNK_SWITCHES),
        ),
        (
            "tombstones_retained_at_saturation",
            i(TOMBSTONES_RETAINED_AT_SATURATION),
        ),
        (
            "abandoned_frames_in_saturation_case",
            i(ABANDONED_FRAMES_IN_SATURATION_CASE),
        ),
        (
            "min_frames_accepted_after_tombstone_expiry",
            i(MIN_FRAMES_ACCEPTED_AFTER_TOMBSTONE_EXPIRY),
        ),
        (
            "min_stale_upstream_entries_before_revocation",
            i(MIN_STALE_UPSTREAM_ENTRIES_BEFORE_REVOCATION),
        ),
        (
            "min_revoke_messages_dropped_or_delayed",
            i(MIN_REVOKE_MESSAGES_DROPPED_OR_DELAYED),
        ),
        ("min_control_trips", i(MIN_CONTROL_TRIPS)),
        ("buffered_bytes_baseline", i(BUFFERED_BYTES_BASELINE)),
        (
            "min_stale_worker_frame_race_cases",
            i(MIN_STALE_WORKER_FRAME_RACE_CASES),
        ),
        ("min_sequences_per_seed", i(MIN_SEQUENCES_PER_SEED)),
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
            toml::Value::String(text) => {
                out.insert(prefix.to_string(), Leaf::Str(text.clone()));
            }
            toml::Value::Array(items) => {
                let leaf = if !items.is_empty() && items.iter().all(|item| item.is_integer()) {
                    Leaf::Ints(items.iter().filter_map(toml::Value::as_integer).collect())
                } else {
                    Leaf::Strs(
                        items
                            .iter()
                            .map(|item| item.as_str().expect("a string").to_string())
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
            // Keys that begin with owner_decision_ or stated_limit_ are prose and mirror
            // no constant.
            if key.starts_with("owner_decision_") || key.starts_with("stated_limit_") {
                continue;
            }
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
    fn the_matrix_cells_are_the_cross_product_of_the_axes() {
        let mut expected = Vec::new();
        for kind in INVALIDATION_COVERAGE_MATRIX_KINDS {
            for path in INVALIDATION_PATHS {
                for ty in INVALIDATION_TYPES {
                    expected.push(format!("{kind}_{path}_{ty}"));
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
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // the point is to pin relations between constants
    fn the_relations_the_design_needs_hold() {
        assert!(
            SPLIT_LAYER > 0 && SPLIT_LAYER < TEST_MODEL_LAYERS,
            "an internal split"
        );
        assert!(
            REPLAY_WINDOW_MS >= MAX_FRAME_DEADLINE_LIFETIME_MS + 2 * MAX_CLOCK_SKEW_MS,
            "{REPLAY_WINDOW_RULE}"
        );
        assert_eq!(CLOCK_SKEW_BOUND_MS, MAX_CLOCK_SKEW_MS);
        assert_eq!(
            LARGE_FRAME_BYTES,
            LARGE_FRAME_TOKENS * LARGE_FRAME_WIDTH * 4,
            "the F32 frame"
        );
        assert!(LARGE_FRAME_BYTES > MAX_BODY_BYTES, "above one PodWire body");
        assert!(MAX_CHUNK_BYTES < MAX_BODY_BYTES);
        assert_eq!(
            LARGE_FRAME_CHUNKS,
            LARGE_FRAME_BYTES.div_ceil(MAX_CHUNK_BYTES),
            "5 chunks of at most 3.5 MiB"
        );
        assert_eq!(MAX_CHUNK_BYTES, 3584 * 1024);
        let mut order = LARGE_FRAME_SEND_ORDER;
        order.sort_unstable();
        assert_eq!(order, [0, 1, 2, 3, 4]);
        assert!(
            LARGE_FRAME_SEND_ORDER
                .iter()
                .enumerate()
                .all(|(position, index)| (*index as usize) < position + REORDER_WINDOW as usize),
            "the large frame's send order is inside the reorder window"
        );
        assert_eq!(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM);
        assert_eq!(
            MIN_ACTIVATION_FRAMES_CROSSED,
            SEQUENCES * DECODE_STEPS,
            "one prompt frame and DECODE_STEPS - 1 decode frames per sequence"
        );
        assert!(MIN_VALID_FRAMES_SENT <= MIN_ACTIVATION_FRAMES_CROSSED);
        assert_eq!(MIN_SEQUENCES_PER_SEED, SEQUENCES);
        assert!(
            FRAME_BUDGET_MS + CLOCK_SKEW_BOUND_MS >= FRAME_SPACING_MS,
            "tombstones outlive a frame period"
        );
        assert!(
            (FRAME_BUDGET_MS + CLOCK_SKEW_BOUND_MS).div_ceil(FRAME_SPACING_MS)
                < MAX_FRAME_TOMBSTONES_PER_PRINCIPAL,
            "the pipeline's frame rate fits the principal's tombstone table"
        );
        assert!(LATE_WINDOW_REPLAY_DELAY_RANGE_MS[0] * 10 == REPLAY_WINDOW_MS * 9);
        assert_eq!(LATE_WINDOW_REPLAY_DELAY_RANGE_MS[1], REPLAY_WINDOW_MS);
        assert_eq!(
            TOMBSTONE_OVERCAPACITY_FRAMES,
            MAX_FRAME_TOMBSTONES_PER_PRINCIPAL + 1
        );
        assert_eq!(
            ABANDONED_FRAMES_IN_SATURATION_CASE,
            MAX_FRAME_TOMBSTONES_PER_PRINCIPAL
        );
        assert!(TOMBSTONE_STAGGER_MS > 0);
        assert!(
            (MAX_FRAME_TOMBSTONES_PER_PRINCIPAL - 1) * TOMBSTONE_STAGGER_MS
                < REPLAY_WINDOW_MS - FRAME_BUDGET_MS,
            "the staggered abandonments are all within one replay window"
        );
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

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, crate::experiments::e007::metrics::METRIC_NAMES);
    }

    #[test]
    fn the_pins_the_harness_shares_are_the_configs() {
        use crate::experiments::layered_kv::pins;
        assert_eq!(
            crate::experiments::layered_kv::SEQUENCE_SEED,
            CORPUS_GENERATOR_SEED
        );
        let shape = pins::SHAPE;
        assert_eq!(shape.layers, TEST_MODEL_LAYERS);
        assert_eq!(shape.hidden_size, HIDDEN_SIZE);
        assert_eq!(shape.num_heads, NUM_HEADS);
        assert_eq!(shape.num_kv_heads, NUM_KV_HEADS);
        assert_eq!(shape.head_dim, HEAD_DIM);
        assert_eq!(shape.vocab_size, VOCAB_SIZE);
        assert_eq!(shape.weight_seed, WEIGHT_SEED);
        assert_eq!(
            pins::canonical_digest(&pins::shape_fields()),
            TINY_MODEL_CONFIG_DIGEST
        );
        assert_eq!(pins::INTERMEDIATE_SIZE as u64, INTERMEDIATE_SIZE);
        assert_eq!(
            crate::experiments::layered_kv::corpus::CHUNK_TOKENS as u64,
            PAGE_TOKENS
        );
        assert_eq!(ptr_podwire::MAX_BODY_BYTES as u64, MAX_BODY_BYTES);
    }
}
