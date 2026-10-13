//! The metric list of L002 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies L002).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`. The harness
//! judges its own output with them, and a test pins `criteria.toml` to exactly these lists, in
//! order. Every metric is a JSON integer, and every coverage counter is read from an event the
//! run observed (a failpoint that fired, an image that really is shorter, a snapshot a member
//! really installed, a message the router really dropped), never from configuration.

use super::params;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The failpoints, by the name their hit counter has under `failpoint_hits`.
pub const HIT_POINTS: [&str; 24] = [
    "compaction_before_build",
    "ledger_create_after_write_before_sync",
    "ledger_create_after_sync_before_dir_sync",
    "compaction_after_log_built_before_anchor",
    "anchor_after_scratch_write_before_sync",
    "anchor_after_scratch_sync_before_rename",
    "anchor_after_rename_before_dir_sync",
    "anchor_after_dir_sync",
    "compaction_after_anchor_before_install",
    "ledger_sync_parent",
    "raft_append_after_write_before_sync",
    "raft_atomic_after_write_before_sync",
    "raft_atomic_after_sync_before_rename",
    "raft_state_after_rename_before_dir_sync",
    "raft_snapshot_after_payload",
    "raft_snapshot_after_state",
    "raft_snapshot_after_log_rewrite",
    "raft_compact_after_sync_before_rename",
    "raft_compact_after_rename_before_dir_sync",
    "raft_install_after_payload",
    "raft_install_after_log_reset",
    "raft_install_after_log_sync_before_rename",
    "raft_append_before_write",
    "raft_state_before_write",
];

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 65] = [
    // Coverage, observed.
    "cuts",
    "recoveries_cutover",
    "recoveries_member",
    "scenarios_group",
    "scenarios_checked",
    "failpoint_hits",
    "images_lost_bytes",
    "rename_image_pairs",
    "lie_images",
    "unwind_panics_observed",
    "cuts_over_revocation",
    "snapshots_recorded",
    "snapshots_installed",
    "restarts_between_install_and_resume",
    "injected_io_errors",
    "leader_changes",
    "messages_dropped",
    "promises_checked",
    "promises_vote",
    "tamper_controls",
    "revoke_then_recommit_histories",
    "recoveries_with_acked_revocation",
    "revoked_pairs_denied",
    "subjects_min",
    "events_min",
    "probe_calibrated",
    "probe_dirty_seen",
    "probe_clean_checked",
    // Falsifiers.
    "stale_accepts",
    "acked_revocation_missing",
    "recovered_not_prefix",
    "reorder_or_duplicate_index",
    "unclassified_cutover",
    "unjustified_refusals",
    "orphan_hazard",
    "promise_not_durable",
    "member_regressed",
    "reference_divergence",
    "convergence_bound_exceeded",
    "tampered_snapshot_accepted",
    "unsynced_acks",
    "product_panics",
    "product_hangs",
    "authority_mismatches",
    "reference_mismatches",
    "anchored_lie_accepted",
    // Descriptive: no rule reads them.
    "recoveries",
    "recovered",
    "refusals",
    "b1_recoveries",
    "b1_stale_accepts",
    "rounds_to_converge_max",
    "rounds_to_converge_total",
    "restarts",
    "messages_delivered",
    "messages_duplicated",
    "messages_reordered",
    "runtimes_restored",
    "follow_on_appends",
    "retried_cutovers",
    "raft_stalled",
    "images_damaged",
    "deposition_attempts_total",
    "deposition_attempts_max",
    "deposition_clean_attempt_needed",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names.
pub const HARD_COUNTERS: [&str; 18] = [
    "stale_accepts",
    "acked_revocation_missing",
    "recovered_not_prefix",
    "reorder_or_duplicate_index",
    "unclassified_cutover",
    "unjustified_refusals",
    "orphan_hazard",
    "promise_not_durable",
    "member_regressed",
    "reference_divergence",
    "convergence_bound_exceeded",
    "tampered_snapshot_accepted",
    "unsynced_acks",
    "product_panics",
    "product_hangs",
    "authority_mismatches",
    "reference_mismatches",
    "anchored_lie_accepted",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &["failpoint_hits"],
};

/// The sum of the hard counters of a tally.
pub fn hard_sum(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}

/// The sum of the hard counters of a run: `hard_failures`.
pub fn hard_failures(tally: &Tally) -> i64 {
    hard_sum(tally)
}

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules = vec![
        format!("cuts >= {}", params::MIN_CUTS),
        format!("recoveries_cutover >= {}", params::MIN_RECOVERIES_CUTOVER),
        format!("recoveries_member >= {}", params::MIN_RECOVERIES_MEMBER),
        format!("scenarios_group >= {}", params::MIN_SCENARIOS_GROUP),
        format!("scenarios_checked >= {}", params::MIN_SCENARIOS_CHECKED),
    ];
    rules.extend(HIT_POINTS.iter().enumerate().map(|(i, point)| {
        let minimum = if i < 10 {
            params::MIN_CUTOVER_HITS
        } else {
            params::MIN_MEMBER_HITS
        };
        format!("failpoint_hits.{point} >= {minimum}")
    }));
    rules.extend([
        format!("images_lost_bytes >= {}", params::MIN_IMAGES_LOST_BYTES),
        format!("rename_image_pairs >= {}", params::MIN_RENAME_IMAGE_PAIRS),
        format!("lie_images >= {}", params::MIN_LIE_IMAGES),
        format!(
            "unwind_panics_observed >= {}",
            params::MIN_UNWIND_PANICS_OBSERVED
        ),
        format!(
            "cuts_over_revocation >= {}",
            params::MIN_CUTS_OVER_REVOCATION
        ),
        format!("snapshots_recorded >= {}", params::MIN_SNAPSHOTS_RECORDED),
        format!("snapshots_installed >= {}", params::MIN_SNAPSHOTS_INSTALLED),
        format!(
            "restarts_between_install_and_resume >= {}",
            params::MIN_RESTARTS_BETWEEN_INSTALL_AND_RESUME
        ),
        format!("injected_io_errors >= {}", params::MIN_INJECTED_IO_ERRORS),
        format!("leader_changes >= {}", params::MIN_LEADER_CHANGES),
        format!("messages_dropped >= {}", params::MIN_MESSAGES_DROPPED),
        format!("promises_checked >= {}", params::MIN_PROMISES_CHECKED),
        format!("promises_vote >= {}", params::MIN_PROMISES_VOTE),
        format!("tamper_controls >= {}", params::MIN_TAMPER_CONTROLS),
        format!(
            "revoke_then_recommit_histories >= {}",
            params::MIN_REVOKE_THEN_RECOMMIT
        ),
        format!(
            "recoveries_with_acked_revocation >= {}",
            params::MIN_RECOVERIES_WITH_ACKED_REVOCATION
        ),
        format!(
            "revoked_pairs_denied >= {}",
            params::MIN_REVOKED_PAIRS_DENIED
        ),
        format!("subjects_min >= {}", params::MIN_SUBJECTS),
        format!("events_min >= {}", params::MIN_EVENTS),
        format!("probe_calibrated >= {}", params::MIN_PROBE_CALIBRATED),
        format!("probe_dirty_seen >= {}", params::MIN_PROBE_DIRTY_SEEN),
        format!("probe_clean_checked >= {}", params::MIN_PROBE_CLEAN_CHECKED),
    ]);
    rules
}

/// The falsifiers: each tripped one falsifies L002 on this seed.
pub fn falsifiers() -> Vec<String> {
    HARD_COUNTERS
        .iter()
        .map(|name| format!("{name} <= 0"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_metric_names_are_unique_and_the_hard_counters_are_metrics() {
        let unique: std::collections::BTreeSet<&str> = METRIC_NAMES.iter().copied().collect();
        assert_eq!(unique.len(), METRIC_NAMES.len());
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        assert_eq!(falsifiers().len(), HARD_COUNTERS.len());
    }

    #[test]
    fn the_coverage_rules_name_every_failpoint_once() {
        let rules = coverage();
        for point in HIT_POINTS {
            let prefix = format!("failpoint_hits.{point} >=");
            assert_eq!(
                rules
                    .iter()
                    .filter(|rule| rule.starts_with(&prefix))
                    .count(),
                1,
                "{point}"
            );
        }
    }
}
