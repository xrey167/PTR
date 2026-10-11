//! The metric list of L001-v2 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies L001-v2).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`. The harness
//! judges its own output with them, and a test pins `criteria.toml` to exactly these lists, in
//! order. Every metric is a JSON integer, and every coverage counter is read from an event the
//! run observed (a failpoint that fired, an image that really is shorter, a child that really
//! died of SIGABRT, a message the router really dropped), never from configuration.

use super::params;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The failpoints, by the name their hit counter has under `failpoint_hits`.
pub const HIT_POINTS: [&str; 12] = [
    "ledger_before_record_write",
    "ledger_after_length_before_payload",
    "ledger_after_payload_before_sync",
    "ledger_after_sync_before_memory",
    "anchor_after_scratch_write_before_sync",
    "anchor_after_scratch_sync_before_rename",
    "anchor_after_rename_before_dir_sync",
    "anchor_after_dir_sync",
    "engine_append",
    "engine_write",
    "engine_sync",
    "ledger_sync_parent",
];

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 75] = [
    // Coverage, observed.
    "workloads_file",
    "workloads_ack",
    "workloads_engine",
    "scenarios_raft",
    "scenarios_checked",
    "recoveries_file",
    "recoveries_ack",
    "recoveries_engine",
    "members_replayed",
    "failpoint_hits",
    "images_lost_bytes",
    "rename_image_pairs",
    "lie_images",
    "child_aborts_file",
    "child_aborts_ack",
    "child_aborts_engine",
    "unwind_panics_observed",
    "leader_changes",
    "minority_proposals",
    "deposed_leader_restarts",
    "messages_dropped",
    "messages_duplicated",
    "messages_reordered",
    "revoke_then_recommit_histories",
    "recoveries_with_acked_revocation",
    "revoked_pairs_denied",
    "crash_points_at_revocation",
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
    "unclassified_split",
    "unjustified_refusals",
    "raft_revocation_visible_before_commit",
    "product_panics",
    "product_hangs",
    "unsynced_acks",
    "authority_mismatches",
    "reference_mismatches",
    "anchored_lie_accepted",
    "refusal_share_excess",
    // Descriptive: no rule reads them.
    "refusals_lost_ack",
    "refusals_corrupt_tail",
    "undetected_rollback",
    "engine_lie_stale_accepts",
    "engine_lie_refused",
    "b1_recoveries",
    "b1_stale_accepts",
    "follow_on_appends",
    "raft_stalled",
    "raft_unconverged_members",
    "raft_scenario_errors",
    "restarts",
    "messages_delivered",
    "recoveries",
    "recovered",
    "refusals",
    "append_failures",
    "images_damaged",
    "recoveries_damaged",
    "refusals_damaged",
    "refused_share_damaged_permille",
    "recoveries_shape_zeros",
    "recoveries_shape_gap_then_intact",
    "recoveries_shape_intact_then_zeros",
    "recoveries_shape_mixed",
    "refusals_shape_zeros",
    "refusals_shape_gap_then_intact",
    "refusals_shape_intact_then_zeros",
    "refusals_shape_mixed",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names.
pub const HARD_COUNTERS: [&str; 14] = [
    "stale_accepts",
    "acked_revocation_missing",
    "recovered_not_prefix",
    "reorder_or_duplicate_index",
    "unclassified_split",
    "unjustified_refusals",
    "raft_revocation_visible_before_commit",
    "product_panics",
    "product_hangs",
    "unsynced_acks",
    "authority_mismatches",
    "reference_mismatches",
    "anchored_lie_accepted",
    "refusal_share_excess",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &["failpoint_hits"],
};

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules = vec![
        format!("workloads_file >= {}", params::MIN_WORKLOADS_FILE),
        format!("workloads_ack >= {}", params::MIN_WORKLOADS_ACK),
        format!("workloads_engine >= {}", params::MIN_WORKLOADS_ENGINE),
        format!("scenarios_raft >= {}", params::MIN_SCENARIOS_RAFT),
        format!("scenarios_checked >= {}", params::MIN_SCENARIOS_CHECKED),
        format!("recoveries_file >= {}", params::MIN_RECOVERIES_FILE),
        format!("recoveries_ack >= {}", params::MIN_RECOVERIES_ACK),
        format!("recoveries_engine >= {}", params::MIN_RECOVERIES_ENGINE),
        format!("members_replayed >= {}", params::MIN_MEMBERS_REPLAYED),
    ];
    rules.extend(
        HIT_POINTS
            .iter()
            .map(|point| format!("failpoint_hits.{point} >= {}", params::MIN_FAILPOINT_HITS)),
    );
    rules.extend([
        format!("images_lost_bytes >= {}", params::MIN_IMAGES_LOST_BYTES),
        format!("rename_image_pairs >= {}", params::MIN_RENAME_IMAGE_PAIRS),
        format!("lie_images >= {}", params::MIN_LIE_IMAGES),
        format!("child_aborts_file >= {}", params::MIN_CHILD_ABORTS_FILE),
        format!("child_aborts_ack >= {}", params::MIN_CHILD_ABORTS_ACK),
        format!("child_aborts_engine >= {}", params::MIN_CHILD_ABORTS_ENGINE),
        format!("unwind_panics_observed >= {}", params::MIN_UNWIND_PANICS),
        format!("leader_changes >= {}", params::MIN_LEADER_CHANGES),
        format!("minority_proposals >= {}", params::MIN_MINORITY_PROPOSALS),
        format!(
            "deposed_leader_restarts >= {}",
            params::MIN_DEPOSED_LEADER_RESTARTS
        ),
        format!("messages_dropped >= {}", params::MIN_MESSAGES_DROPPED),
        format!("messages_duplicated >= {}", params::MIN_MESSAGES_DUPLICATED),
        format!("messages_reordered >= {}", params::MIN_MESSAGES_REORDERED),
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
        format!(
            "crash_points_at_revocation >= {}",
            params::MIN_CRASH_POINTS_AT_REVOCATION
        ),
        format!("subjects_min >= {}", params::MIN_SUBJECTS),
        format!("events_min >= {}", params::MIN_EVENTS),
        format!("probe_calibrated >= {}", params::MIN_PROBE_CALIBRATED),
        format!("probe_dirty_seen >= {}", params::MIN_PROBE_DIRTY_SEEN),
        format!("probe_clean_checked >= {}", params::MIN_PROBE_CLEAN_CHECKED),
    ]);
    rules
}

/// The falsifiers: each tripped one falsifies L001-v2 on this seed.
pub fn falsifiers() -> Vec<String> {
    HARD_COUNTERS
        .iter()
        .map(|name| format!("{name} <= 0"))
        .collect()
}

/// The sum of the hard counters other than `reference_mismatches`, which is what a baseline
/// run adds when any of the others rose during it.
pub fn hard_sum(tally: &Tally) -> i64 {
    HARD_COUNTERS
        .iter()
        .filter(|name| **name != "reference_mismatches")
        .map(|name| tally.get(name))
        .sum()
}

/// The sum of the hard counters in `tally`.
pub fn hard_failures(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{judge, Json, Status};

    /// A result in which every rule holds exactly at its threshold.
    fn passing() -> Vec<(String, Json)> {
        let thresholds: Vec<(String, i64)> = coverage()
            .iter()
            .map(|rule| {
                let tokens: Vec<&str> = rule.split(' ').collect();
                (tokens[0].to_string(), tokens[2].parse().unwrap())
            })
            .collect();
        let mut pairs: Vec<(String, Json)> = Vec::new();
        for name in METRIC_NAMES {
            let value = if name == "failpoint_hits" {
                Json::Obj(
                    HIT_POINTS
                        .iter()
                        .map(|point| {
                            let key = format!("failpoint_hits.{point}");
                            let bound = thresholds.iter().find(|(n, _)| *n == key).unwrap().1;
                            (point.to_string(), Json::Int(bound))
                        })
                        .collect(),
                )
            } else if let Some((_, bound)) = thresholds.iter().find(|(n, _)| n == name) {
                Json::Int(*bound)
            } else {
                Json::Int(0)
            };
            pairs.push((name.to_string(), value));
        }
        pairs
    }

    fn failing(pairs: Vec<(String, Json)>, rules: &[String]) -> Vec<String> {
        judge(&Json::Obj(pairs), &[], rules)
            .into_iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule)
            .collect()
    }

    fn set(pairs: &mut [(String, Json)], path: &str, value: i64) {
        let mut keys = path.split('.');
        let first = keys.next().unwrap();
        let slot = &mut pairs.iter_mut().find(|(key, _)| key == first).unwrap().1;
        match (keys.next(), slot) {
            (None, slot) => *slot = Json::Int(value),
            (Some(inner), Json::Obj(members)) => {
                members.iter_mut().find(|(key, _)| key == inner).unwrap().1 = Json::Int(value)
            }
            _ => panic!("{path}"),
        }
    }

    #[test]
    fn a_result_at_every_threshold_holds_every_rule() {
        assert!(failing(passing(), &coverage()).is_empty());
        assert!(failing(passing(), &falsifiers()).is_empty());
    }

    #[test]
    fn each_coverage_rule_fails_alone_when_its_counter_leaves_the_threshold() {
        for rule in coverage() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let bound: i64 = tokens[2].parse().unwrap();
            let mut pairs = passing();
            set(&mut pairs, tokens[0], bound - 1);
            assert_eq!(failing(pairs, &coverage()), vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn each_falsifier_trips_alone_on_a_single_violation() {
        for rule in falsifiers() {
            let name = rule.split(' ').next().unwrap();
            let mut pairs = passing();
            set(&mut pairs, name, 1);
            assert_eq!(failing(pairs, &falsifiers()), vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn a_missing_null_or_mistyped_counter_is_not_a_pass() {
        for name in ["stale_accepts", "product_hangs", "unsynced_acks"] {
            for damage in [
                None,
                Some(Json::Null),
                Some(Json::str("0")),
                Some(Json::Float(0.0)),
            ] {
                let mut pairs = passing();
                match damage {
                    None => pairs.retain(|(key, _)| key != name),
                    Some(value) => pairs.iter_mut().find(|(key, _)| key == name).unwrap().1 = value,
                }
                assert!(!failing(pairs, &falsifiers()).is_empty(), "{name}");
            }
        }
    }

    #[test]
    fn the_hard_counters_are_metrics_and_every_falsifier_is_one() {
        for name in HARD_COUNTERS {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        assert_eq!(falsifiers().len(), HARD_COUNTERS.len());
        let unique: std::collections::BTreeSet<&str> = METRIC_NAMES.iter().copied().collect();
        assert_eq!(unique.len(), METRIC_NAMES.len());
    }

    #[test]
    fn filling_a_tally_prints_every_metric_as_a_value() {
        let mut tally = Tally::new(LIST);
        for name in METRIC_NAMES {
            if name != "failpoint_hits" {
                tally.set(name, 1);
            }
        }
        tally.seed_group("failpoint_hits", &HIT_POINTS);
        let (printed, gaps) = tally.metrics();
        assert!(gaps.is_empty());
        assert_eq!(printed.len(), METRIC_NAMES.len());
        tally.set("stale_accepts", 3);
        assert_eq!(hard_failures(&tally), 16);
        assert_eq!(hard_sum(&tally), 15);
    }
}
