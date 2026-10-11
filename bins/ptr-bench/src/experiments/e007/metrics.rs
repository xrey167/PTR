//! The metric list of E007 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies E007).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`); the
//! harness judges its own output with them and a test pins `criteria.toml` to exactly these
//! lists, in order.

use super::params;
use crate::experiments::layered_kv::emit::MetricList;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 108] = [
    "bitwise_mismatch_count",
    "activation_frames_crossed",
    "valid_frames_sent",
    "frames_with_consistent_whole_frame_digest",
    "valid_sent_frame_id_set_digest",
    "consistent_digest_frame_id_set_digest",
    "valid_frame_bitwise_mismatches",
    "downstream_reuse_cases",
    "large_frame_cases",
    "large_frame_reassembled_bytes_mismatch_count",
    "large_frame_digest_mismatches",
    "wire_kv_handle_count",
    "wire_kv_page_bytes",
    "kv_pages_installed_from_wire",
    "non_allowlisted_payloads_admitted",
    "mislabelled_kv_payload_attempts",
    "non_allowlisted_type_attempts",
    "kv_sized_allowlisted_smuggling_attempts",
    "smuggling_probe_attempts",
    "smuggling_probes_admitted",
    "activation_control_cases",
    "activation_control_kv_installs",
    "shape_validation_rejections",
    "positions_length_rejection_attempts",
    "positions_length_mismatches_admitted",
    "unsigned_chunk_attempts",
    "unsigned_chunks_admitted",
    "chain_mismatch_attempts",
    "unbound_stage_attempts",
    "global_cap_attempts",
    "non_exact_upstream_attempts",
    "non_finite_activation_attempts",
    "chain_frame_mismatch_attempts",
    "boundary_tick_replay_attempts",
    "lost_revoke_full_cache_cases",
    "chain_mismatch_stores",
    "duplicate_chunk_attempts",
    "out_of_window_attempts",
    "bad_chunk_digest_attempts",
    "mixed_version_frame_attempts",
    "incomplete_frames_over_peer_limit",
    "queue_overflow_attempts",
    "splice_attempts",
    "cross_operation_splice_attempts",
    "cross_operation_splices_admitted",
    "digest_inconsistency_attempts",
    "mislabelled_frame_digest_attempts",
    "mislabelled_frame_digests_admitted",
    "interleaved_frame_pairs",
    "interleaved_chunk_switches",
    "cross_frame_mixing_count",
    "abandoned_frame_cases",
    "abandoned_chunks_sent",
    "abandoned_chunks_released",
    "buffered_bytes_before_abandon",
    "buffered_bytes_after_release",
    "reassembly_attempts_after_abandon",
    "abandoned_chunk_id_set_digest",
    "released_chunk_id_set_digest",
    "late_chunks_after_abandon_attempts",
    "late_chunks_after_abandon_admitted",
    "concurrent_abandonment_cases",
    "tombstone_saturation_cases",
    "new_frames_refused_typed",
    "tombstone_evictions_before_retention",
    "frames_accepted_after_tombstone_expiry",
    "abandoned_frames_in_saturation_case",
    "tombstones_retained_at_saturation",
    "overcapacity_frame_attempts",
    "tombstone_bound_violations",
    "full_frame_replay_attempts",
    "replayed_frames_admitted",
    "cross_session_replay_attempts",
    "cross_session_replays_admitted",
    "concurrent_cross_session_assembly_cases",
    "duplicate_activation_deliveries",
    "expired_deadline_frame_attempts",
    "expired_frames_admitted",
    "post_abandonment_replay_cases",
    "abandoned_frame_replays_admitted",
    "immediate_replay_cases",
    "late_window_replay_cases",
    "late_window_replay_delays_ms",
    "relabeled_envelope_replay_cases",
    "relabeled_chunks_admitted",
    "unbound_stage_frames_admitted",
    "global_cap_breaches",
    "non_exact_frames_admitted",
    "non_finite_frames_admitted",
    "chain_frame_mismatches_delivered",
    "boundary_tick_replays_admitted",
    "cache_slots_held_by_dead_entries",
    "attempts_without_typed_rejection",
    "verdict_mismatches",
    "stale_upstream_attempts",
    "stale_upstream_kv_served",
    "stale_upstream_entries_before_revocation",
    "revoke_messages_dropped_or_delayed",
    "inflight_split_request_race_cases",
    "outputs_after_upstream_invalidation",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "stale_worker_frame_race_cases",
    "stale_placement_frames_admitted",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
    "control_trips",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 6] = [
    "smuggling_probe_attempts",
    "stale_upstream_attempts",
    "inflight_split_request_race_cases",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "control_trips",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// The controls (naive baselines) that must each trip.
pub const CONTROLS: [&str; 5] = [
    "naive_downstream_cache_serves_stale",
    "chain_keyed_assembler_mixes_frames",
    "denylist_admits_mislabelled_payload",
    "unchunked_large_frame_refused_by_podwire",
    "tombstone_free_assembler_admits_replay",
];

/// Hard counters that mirror a per-cell stale hit as a top-level integer, for the mutation
/// checker (it reads top-level integers only).
pub fn cell_counter(cell: &str) -> String {
    format!("stale_hits_in_{cell}")
}

/// The attempt counters every kind must reach at least `MIN_ATTEMPTS` on.
const ATTEMPTS: [&str; 40] = [
    "mislabelled_kv_payload_attempts",
    "non_allowlisted_type_attempts",
    "kv_sized_allowlisted_smuggling_attempts",
    "activation_control_cases",
    "shape_validation_rejections",
    "positions_length_rejection_attempts",
    "unsigned_chunk_attempts",
    "chain_mismatch_attempts",
    "unbound_stage_attempts",
    "global_cap_attempts",
    "non_exact_upstream_attempts",
    "non_finite_activation_attempts",
    "chain_frame_mismatch_attempts",
    "boundary_tick_replay_attempts",
    "lost_revoke_full_cache_cases",
    "duplicate_chunk_attempts",
    "out_of_window_attempts",
    "bad_chunk_digest_attempts",
    "mixed_version_frame_attempts",
    "incomplete_frames_over_peer_limit",
    "queue_overflow_attempts",
    "splice_attempts",
    "cross_operation_splice_attempts",
    "digest_inconsistency_attempts",
    "mislabelled_frame_digest_attempts",
    "abandoned_frame_cases",
    "abandoned_chunks_sent",
    "reassembly_attempts_after_abandon",
    "late_chunks_after_abandon_attempts",
    "concurrent_abandonment_cases",
    "tombstone_saturation_cases",
    "new_frames_refused_typed",
    "overcapacity_frame_attempts",
    "full_frame_replay_attempts",
    "cross_session_replay_attempts",
    "concurrent_cross_session_assembly_cases",
    "expired_deadline_frame_attempts",
    "post_abandonment_replay_cases",
    "immediate_replay_cases",
    "late_window_replay_cases",
];

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    rules.push("procedure.split_layer >= 1".to_string());
    rules.push("procedure.layers_above_split >= 1".to_string());
    rules.push(format!(
        "activation_frames_crossed >= {}",
        params::MIN_ACTIVATION_FRAMES_CROSSED
    ));
    rules.push(format!(
        "valid_frames_sent >= {}",
        params::MIN_VALID_FRAMES_SENT
    ));
    rules.push(format!(
        "downstream_reuse_cases >= {}",
        params::MIN_ATTEMPTS
    ));
    rules.push(format!(
        "large_frame_cases >= {}",
        params::MIN_LARGE_FRAME_CASES
    ));
    for name in ATTEMPTS {
        rules.push(format!("{name} >= {}", params::MIN_ATTEMPTS));
    }
    for kind in params::SMUGGLING_PROBE_KINDS {
        rules.push(format!(
            "smuggling_probe_attempts.{kind} >= {}",
            params::MIN_ATTEMPTS
        ));
    }
    rules.push(format!(
        "interleaved_frame_pairs >= {}",
        params::MIN_INTERLEAVED_FRAME_PAIRS
    ));
    rules.push(format!(
        "interleaved_chunk_switches >= {}",
        params::MIN_INTERLEAVED_CHUNK_SWITCHES
    ));
    rules.push("buffered_bytes_before_abandon >= 1".to_string());
    rules.push(format!(
        "tombstones_retained_at_saturation == {}",
        params::TOMBSTONES_RETAINED_AT_SATURATION
    ));
    rules.push(format!(
        "abandoned_frames_in_saturation_case == {}",
        params::ABANDONED_FRAMES_IN_SATURATION_CASE
    ));
    rules.push(format!(
        "frames_accepted_after_tombstone_expiry >= {}",
        params::MIN_FRAMES_ACCEPTED_AFTER_TOMBSTONE_EXPIRY
    ));
    rules.push("late_window_delays_in_range >= 1".to_string());
    rules.push(format!(
        "relabeled_envelope_replay_cases >= {}",
        params::MIN_ATTEMPTS
    ));
    for ty in params::INVALIDATION_TYPES {
        rules.push(format!(
            "stale_upstream_attempts.{ty} >= {}",
            params::MIN_ATTEMPTS
        ));
        rules.push(format!(
            "inflight_split_request_race_cases.{ty} >= {}",
            params::MIN_ATTEMPTS
        ));
    }
    rules.push(format!(
        "stale_upstream_entries_before_revocation >= {}",
        params::MIN_STALE_UPSTREAM_ENTRIES_BEFORE_REVOCATION
    ));
    rules.push(format!(
        "revoke_messages_dropped_or_delayed >= {}",
        params::MIN_REVOKE_MESSAGES_DROPPED_OR_DELAYED
    ));
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        rules.push(format!(
            "invalidation_matrix_cases.{cell} >= {}",
            params::MIN_ATTEMPTS
        ));
    }
    rules.push(format!(
        "stale_worker_frame_race_cases >= {}",
        params::MIN_STALE_WORKER_FRAME_RACE_CASES
    ));
    for control in CONTROLS {
        rules.push(format!(
            "control_trips.{control} >= {}",
            params::MIN_CONTROL_TRIPS
        ));
    }
    rules
}

/// The falsifiers: each rule that does not hold falsifies E007 on that seed.
pub fn falsifiers() -> Vec<String> {
    let mut rules: Vec<String> = Vec::new();
    for name in [
        "bitwise_mismatch_count",
        "large_frame_reassembled_bytes_mismatch_count",
        "large_frame_digest_mismatches",
        "wire_kv_handle_count",
        "wire_kv_page_bytes",
        "kv_pages_installed_from_wire",
        "activation_control_kv_installs",
        "chain_mismatch_stores",
        "non_allowlisted_payloads_admitted",
        "smuggling_probes_admitted",
        "attempts_without_typed_rejection",
        "verdict_mismatches",
        "cross_frame_mixing_count",
        "buffered_bytes_after_release",
        "stale_upstream_kv_served",
        "positions_length_mismatches_admitted",
        "mislabelled_frame_digests_admitted",
        "late_chunks_after_abandon_admitted",
        "outputs_after_upstream_invalidation",
        "tombstone_evictions_before_retention",
        "tombstone_bound_violations",
        "cross_operation_splices_admitted",
        "replayed_frames_admitted",
        "valid_frame_bitwise_mismatches",
        "cross_session_replays_admitted",
        "duplicate_activation_deliveries",
        "expired_frames_admitted",
        "abandoned_frame_replays_admitted",
        "stale_placement_frames_admitted",
        "relabeled_chunks_admitted",
        "unbound_stage_frames_admitted",
        "global_cap_breaches",
        "non_exact_frames_admitted",
        "non_finite_frames_admitted",
        "chain_frame_mismatches_delivered",
        "boundary_tick_replays_admitted",
        "cache_slots_held_by_dead_entries",
        "unsigned_chunks_admitted",
    ] {
        rules.push(format!("{name} <= {}", params::MAX_VIOLATIONS));
    }
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        rules.push(format!(
            "invalidation_matrix_stale_hits.{cell} <= {}",
            params::MAX_VIOLATIONS
        ));
    }
    rules.push("abandoned_chunks_released >= abandoned_chunks_sent".to_string());
    rules.push("frames_with_consistent_whole_frame_digest == valid_frames_sent".to_string());
    rules.push("abandoned_chunk_id_set_digest == released_chunk_id_set_digest".to_string());
    rules.push(
        "valid_sent_frame_id_set_digest == consistent_digest_frame_id_set_digest".to_string(),
    );
    rules
}

/// The counters that say the record does not describe the frozen run (the model, the corpus
/// selection or the procedure differ from the pins, the late replay delay left its window).
/// They are the harness's own, never a finding about the product.
pub const HARNESS_COUNTERS: [&str; 4] = [
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
    "late_window_delays_out_of_range",
];

/// The comparisons of what the harness sent with what the receiver reported. With every case
/// run they are observations; after an aborted case they only say what was never observed.
pub const SENT_VERSUS_OBSERVED: [&str; 4] = [
    "abandoned_chunks_unreleased",
    "frames_without_consistent_digest",
    "abandoned_id_set_mismatch",
    "valid_id_set_mismatch",
];

/// The validity rules: each that does not hold makes the seed INVALID. Never a falsifier (the
/// product is not at fault), never a coverage prerequisite (the seed is not vacuous, it is
/// not the frozen run).
pub fn validity() -> Vec<String> {
    HARNESS_COUNTERS
        .iter()
        .map(|name| format!("{name} <= {}", params::MAX_VIOLATIONS))
        .collect()
}

/// The counters a mutation may be expected to fire: top-level integers the mutation checker
/// can read, namely every `<= 0` falsifier counter, the harness counters and the per-cell
/// mirrors.
pub fn hard_counters() -> Vec<String> {
    let mut names: Vec<String> = falsifiers()
        .iter()
        .filter_map(|rule| {
            let tokens: Vec<&str> = rule.split(' ').collect();
            (tokens[1] == "<=" && !tokens[0].contains('.')).then(|| tokens[0].to_string())
        })
        .collect();
    names.extend(HARNESS_COUNTERS.iter().map(|name| name.to_string()));
    for cell in params::INVALIDATION_COVERAGE_MATRIX_CELLS {
        names.push(cell_counter(cell));
    }
    names.extend(SENT_VERSUS_OBSERVED.iter().map(|name| name.to_string()));
    names
}

/// The counters whose firing is an observed violation by the product: the hard counters less
/// the harness's own and, because an aborted case leaves them unobserved, the comparisons of
/// sent with observed.
pub fn product_counters() -> Vec<String> {
    hard_counters()
        .into_iter()
        .filter(|name| {
            !HARNESS_COUNTERS.contains(&name.as_str())
                && !SENT_VERSUS_OBSERVED.contains(&name.as_str())
        })
        .collect()
}
