//! The preregistered parameters of L002, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. The `[preregistration]` table of the same
//! file repeats these values and is frozen at the freeze.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/lifecycle/L002-raft-recovery/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/lifecycle/L002-raft-recovery/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "raft-recovery";

/// The experiment's id as the harness prints it.
pub const EXPERIMENT_ID: &str = "L002";

/// The runner's command, as the preregistration states it; `experiment.toml` carries the same.
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery <seed>";

/// The manifest's preflight, as the preregistration states it.
pub const PREFLIGHT: &str =
    "cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery --preflight";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// A workload is 10 to 16 events over 3 or 4 subjects (a compaction needs history to discard).
pub const EVENTS_MIN: usize = 10;
pub const EVENTS_MAX: usize = 16;

/// Arm K: workloads per seed, cutovers per workload, and the retention counts a cutover keeps.
pub const CUTOVER_WORKLOADS: usize = 6;
pub const CUTS_PER_WORKLOAD: usize = 2;
pub const KEEP_RECORDS: [u64; 2] = [1, 2];

/// Arm K: appends above the floor that are crashed, per failpoint, per cutover.
pub const ABOVE_FLOOR_IDX_PER_POINT: usize = 1;

/// Arms S and G: group scenarios per failpoint, in each of the three kinds.
pub const CRASH_SCENARIOS_PER_POINT: usize = 3;
pub const IOERR_SCENARIOS_PER_POINT: usize = 3;
pub const HEAL_SCENARIOS: usize = 3;
pub const TAMPER_SCENARIOS: usize = 3;
pub const IMAGE_SCENARIOS: usize = 3;
/// Captured appends per image scenario that are turned into power-loss images.
pub const IMAGE_CAPTURES_PER_SCENARIO: usize = 6;

/// The liveness bound: leader heartbeat rounds after the last fault until every member holds the
/// leader's history.
pub const CONVERGENCE_ROUNDS: usize = 6;

/// The watchdog of one recovery or in-process sequence, in milliseconds, and how many may fire
/// before the run stops issuing them.
pub const HANG_WATCHDOG_MS: u64 = 60_000;
pub const HANG_LIMIT: i64 = 3;

/// The size of a hole of the `holes` policy.
pub const SECTOR_BYTES: usize = 512;

/// Fault rates of the message-fault phase of a heal scenario, per mille of messages.
pub const FAULT_DROP_PERMILLE: u64 = 120;
pub const FAULT_DUPLICATE_PERMILLE: u64 = 150;
pub const FAULT_REORDER_PERMILLE: u64 = 200;

/// The cutover failpoints of `AcknowledgedLedger::compact` (the anchor's four are the
/// publication of the new floor).
pub const CUTOVER_POINTS: [&str; 9] = [
    "compaction.before_build",
    "ledger.create_after_write_before_sync",
    "ledger.create_after_sync_before_dir_sync",
    "compaction.after_log_built_before_anchor",
    "anchor.after_scratch_write_before_sync",
    "anchor.after_scratch_sync_before_rename",
    "anchor.after_rename_before_dir_sync",
    "anchor.after_dir_sync",
    "compaction.after_anchor_before_install",
];
/// The directory synchronization (counted, not crashed at).
pub const SYNC_PARENT_POINT: &str = "ledger.sync_parent";
/// The log and anchor failpoints of an append above the floor.
pub const ABOVE_FLOOR_POINTS: [&str; 8] = [
    "ledger.before_record_write",
    "ledger.after_length_before_payload",
    "ledger.after_payload_before_sync",
    "ledger.after_sync_before_memory",
    "anchor.after_scratch_write_before_sync",
    "anchor.after_scratch_sync_before_rename",
    "anchor.after_rename_before_dir_sync",
    "anchor.after_dir_sync",
];
/// The crash failpoints of a Raft member's durable state.
pub const MEMBER_POINTS: [&str; 12] = [
    "raft.append.after_write_before_sync",
    "raft.atomic.after_write_before_sync",
    "raft.atomic.after_sync_before_rename",
    "raft.state.after_rename_before_dir_sync",
    "raft.snapshot.after_payload",
    "raft.snapshot.after_state",
    "raft.snapshot.after_log_rewrite",
    "raft.compact.after_sync_before_rename",
    "raft.compact.after_rename_before_dir_sync",
    "raft.install.after_payload",
    "raft.install.after_log_reset",
    "raft.install.after_log_sync_before_rename",
];
/// The failpoints at which an I/O error is injected in place of a write.
pub const ERROR_POINTS: [&str; 2] = ["raft.append.before_write", "raft.state.before_write"];
pub const POWER_LOSS_POLICIES: [&str; 5] = ["drop-all", "all", "prefix", "half", "holes"];
pub const SCENARIO_KINDS: [&str; 5] = ["crash", "io-error", "heal", "tamper", "images"];

/// The name under which a failpoint's hits are counted in `failpoint_hits`.
pub fn hit_name(point: &str) -> String {
    point.replace('.', "_")
}

// Coverage prerequisites (a run that misses one is vacuous). Each is a fraction of what the plan
// of the preregistration (section 3) gives by construction or by a structural lower bound.
pub const MIN_CUTS: i64 = 10;
pub const MIN_RECOVERIES_CUTOVER: i64 = 200;
pub const MIN_RECOVERIES_MEMBER: i64 = 90;
pub const MIN_SCENARIOS_GROUP: i64 = 36;
pub const MIN_SCENARIOS_CHECKED: i64 = 36;
pub const MIN_CUTOVER_HITS: i64 = 10;
pub const MIN_MEMBER_HITS: i64 = 10;
/// Heartbeat rounds the harness waits for a leader to commit a proposal it accepted before it
/// proposes the same event again (a dropped message is sent again by the next heartbeat).
pub const PROPOSAL_SETTLE_ROUNDS: usize = 6;
pub const MIN_IMAGES_LOST_BYTES: i64 = 1000;
pub const MIN_RENAME_IMAGE_PAIRS: i64 = 10;
pub const MIN_LIE_IMAGES: i64 = 10;
pub const MIN_UNWIND_PANICS_OBSERVED: i64 = 150;
pub const MIN_CUTS_OVER_REVOCATION: i64 = 6;
pub const MIN_SNAPSHOTS_RECORDED: i64 = 100;
pub const MIN_SNAPSHOTS_INSTALLED: i64 = 40;
pub const MIN_RESTARTS_BETWEEN_INSTALL_AND_RESUME: i64 = 15;
pub const MIN_INJECTED_IO_ERRORS: i64 = 4;
pub const MIN_LEADER_CHANGES: i64 = 3;
pub const MIN_MESSAGES_DROPPED: i64 = 20;
pub const MIN_PROMISES_CHECKED: i64 = 400;
pub const MIN_PROMISES_VOTE: i64 = 100;
pub const MIN_TAMPER_CONTROLS: i64 = 3;
pub const MIN_REVOKE_THEN_RECOMMIT: i64 = 40;
pub const MIN_RECOVERIES_WITH_ACKED_REVOCATION: i64 = 150;
pub const MIN_REVOKED_PAIRS_DENIED: i64 = 200;
pub const MIN_SUBJECTS: i64 = 3;
pub const MIN_EVENTS: i64 = 10;
pub const MIN_PROBE_CALIBRATED: i64 = 1;
pub const MIN_PROBE_DIRTY_SEEN: i64 = 1;
pub const MIN_PROBE_CLEAN_CHECKED: i64 = 300;

/// What a run covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub cutover_workloads: usize,
    pub crash_per_point: usize,
    pub ioerr_per_point: usize,
    pub heal: usize,
    pub tamper: usize,
    pub images: usize,
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: one case of each.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// This many cutover workloads, and this many scenarios per failpoint (the mutation
    /// checker's second argument); no evidence.
    Custom(usize),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
            Self::Custom(_) => "custom",
        }
    }

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => Shape {
                cutover_workloads: CUTOVER_WORKLOADS,
                crash_per_point: CRASH_SCENARIOS_PER_POINT,
                ioerr_per_point: IOERR_SCENARIOS_PER_POINT,
                heal: HEAL_SCENARIOS,
                tamper: TAMPER_SCENARIOS,
                images: IMAGE_SCENARIOS,
            },
            Self::Reduced => Shape {
                cutover_workloads: 1,
                crash_per_point: 1,
                ioerr_per_point: 1,
                heal: 1,
                tamper: 1,
                images: 1,
            },
            Self::Custom(cases) => Shape {
                cutover_workloads: cases,
                crash_per_point: 1,
                ioerr_per_point: 1,
                heal: 1,
                tamper: 1,
                images: 1,
            },
        }
    }
}

/// One value of the compiled table, comparable with a value of `config.toml`.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Leaf {
    Int(i64),
    Bool(bool),
    Text(String),
    Ints(Vec<i64>),
    Texts(Vec<String>),
}

/// The compiled table, one entry per key of `[pinned_thresholds]`.
#[cfg(test)]
pub fn compiled() -> Vec<(&'static str, Leaf)> {
    fn int(value: impl TryInto<i64>) -> Leaf {
        Leaf::Int(value.try_into().ok().expect("fits"))
    }
    fn texts(values: &[&str]) -> Leaf {
        Leaf::Texts(values.iter().map(|value| (*value).to_string()).collect())
    }
    vec![
        ("seed_aggregation", Leaf::Text(SEED_AGGREGATION.into())),
        (
            "seeds",
            Leaf::Ints(SEEDS.iter().map(|seed| *seed as i64).collect()),
        ),
        ("events_min", int(EVENTS_MIN)),
        ("events_max", int(EVENTS_MAX)),
        ("cutover_workloads", int(CUTOVER_WORKLOADS)),
        ("cuts_per_workload", int(CUTS_PER_WORKLOAD)),
        (
            "keep_records",
            Leaf::Ints(KEEP_RECORDS.iter().map(|keep| *keep as i64).collect()),
        ),
        ("above_floor_idx_per_point", int(ABOVE_FLOOR_IDX_PER_POINT)),
        ("crash_scenarios_per_point", int(CRASH_SCENARIOS_PER_POINT)),
        ("ioerr_scenarios_per_point", int(IOERR_SCENARIOS_PER_POINT)),
        ("heal_scenarios", int(HEAL_SCENARIOS)),
        ("tamper_scenarios", int(TAMPER_SCENARIOS)),
        ("image_scenarios", int(IMAGE_SCENARIOS)),
        (
            "image_captures_per_scenario",
            int(IMAGE_CAPTURES_PER_SCENARIO),
        ),
        ("convergence_rounds", int(CONVERGENCE_ROUNDS)),
        ("proposal_settle_rounds", int(PROPOSAL_SETTLE_ROUNDS as i64)),
        ("hang_watchdog_ms", int(HANG_WATCHDOG_MS)),
        ("hang_limit", int(HANG_LIMIT)),
        ("sector_bytes", int(SECTOR_BYTES)),
        ("fault_drop_permille", int(FAULT_DROP_PERMILLE)),
        ("fault_duplicate_permille", int(FAULT_DUPLICATE_PERMILLE)),
        ("fault_reorder_permille", int(FAULT_REORDER_PERMILLE)),
        ("cutover_points", texts(&CUTOVER_POINTS)),
        ("sync_parent_point", Leaf::Text(SYNC_PARENT_POINT.into())),
        ("above_floor_points", texts(&ABOVE_FLOOR_POINTS)),
        ("member_points", texts(&MEMBER_POINTS)),
        ("error_points", texts(&ERROR_POINTS)),
        ("power_loss_policies", texts(&POWER_LOSS_POLICIES)),
        ("scenario_kinds", texts(&SCENARIO_KINDS)),
        ("min_cuts", int(MIN_CUTS)),
        ("min_recoveries_cutover", int(MIN_RECOVERIES_CUTOVER)),
        ("min_recoveries_member", int(MIN_RECOVERIES_MEMBER)),
        ("min_scenarios_group", int(MIN_SCENARIOS_GROUP)),
        ("min_scenarios_checked", int(MIN_SCENARIOS_CHECKED)),
        ("min_cutover_hits", int(MIN_CUTOVER_HITS)),
        ("min_member_hits", int(MIN_MEMBER_HITS)),
        ("min_images_lost_bytes", int(MIN_IMAGES_LOST_BYTES)),
        ("min_rename_image_pairs", int(MIN_RENAME_IMAGE_PAIRS)),
        ("min_lie_images", int(MIN_LIE_IMAGES)),
        (
            "min_unwind_panics_observed",
            int(MIN_UNWIND_PANICS_OBSERVED),
        ),
        ("min_cuts_over_revocation", int(MIN_CUTS_OVER_REVOCATION)),
        ("min_snapshots_recorded", int(MIN_SNAPSHOTS_RECORDED)),
        ("min_snapshots_installed", int(MIN_SNAPSHOTS_INSTALLED)),
        (
            "min_restarts_between_install_and_resume",
            int(MIN_RESTARTS_BETWEEN_INSTALL_AND_RESUME),
        ),
        ("min_injected_io_errors", int(MIN_INJECTED_IO_ERRORS)),
        ("min_leader_changes", int(MIN_LEADER_CHANGES)),
        ("min_messages_dropped", int(MIN_MESSAGES_DROPPED)),
        ("min_promises_checked", int(MIN_PROMISES_CHECKED)),
        ("min_promises_vote", int(MIN_PROMISES_VOTE)),
        ("min_tamper_controls", int(MIN_TAMPER_CONTROLS)),
        (
            "min_revoke_then_recommit_histories",
            int(MIN_REVOKE_THEN_RECOMMIT),
        ),
        (
            "min_recoveries_with_acked_revocation",
            int(MIN_RECOVERIES_WITH_ACKED_REVOCATION),
        ),
        ("min_revoked_pairs_denied", int(MIN_REVOKED_PAIRS_DENIED)),
        ("min_subjects", int(MIN_SUBJECTS)),
        ("min_events", int(MIN_EVENTS)),
        ("min_probe_calibrated", int(MIN_PROBE_CALIBRATED)),
        ("min_probe_dirty_seen", int(MIN_PROBE_DIRTY_SEEN)),
        ("min_probe_clean_checked", int(MIN_PROBE_CLEAN_CHECKED)),
        ("max_every_falsifier", int(0)),
        ("timing_gated", Leaf::Bool(false)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn leaf_of(value: &toml::Value) -> Leaf {
        match value {
            toml::Value::Integer(number) => Leaf::Int(*number),
            toml::Value::Boolean(flag) => Leaf::Bool(*flag),
            toml::Value::String(text) => Leaf::Text(text.clone()),
            toml::Value::Array(items) if items.iter().all(toml::Value::is_integer) => Leaf::Ints(
                items
                    .iter()
                    .map(|item| item.as_integer().expect("integer"))
                    .collect(),
            ),
            toml::Value::Array(items) => Leaf::Texts(
                items
                    .iter()
                    .map(|item| item.as_str().expect("string").to_string())
                    .collect(),
            ),
            other => panic!("unexpected value {other:?}"),
        }
    }

    /// The keys on which the compiled constants and `config` disagree, one reason each. Keys
    /// that begin with `owner_decision_` or `stated_limit_` are prose and not constants.
    fn drift(config: &str) -> Vec<String> {
        let parsed: toml::Table = config.parse().expect("config.toml parses");
        let file: BTreeMap<String, Leaf> = parsed["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]")
            .iter()
            .filter(|(key, _)| {
                !key.starts_with("owner_decision_") && !key.starts_with("stated_limit_")
            })
            .map(|(key, value)| (key.clone(), leaf_of(value)))
            .collect();
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
        let changed = CONFIG_TOML.replace("events_max = 16\n", "events_max = 17\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["events_max: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("events_max = 16\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(drift(&removed), ["events_max: no key in the file"]);
    }

    #[test]
    fn the_compiled_table_has_no_duplicate_key() {
        let all = compiled();
        let unique: std::collections::BTreeSet<&str> = all.iter().map(|(key, _)| *key).collect();
        assert_eq!(unique.len(), all.len());
    }

    #[test]
    fn the_manifest_seeds_are_the_constants_and_the_scales_are_named() {
        let manifest: toml::Table = EXPERIMENT_TOML.parse().expect("experiment.toml parses");
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|seed| seed.as_integer().expect("integer") as u64)
            .collect();
        assert_eq!(seeds, SEEDS);
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Pinned.shape().cutover_workloads, CUTOVER_WORKLOADS);
        assert_ne!(Scale::Reduced.shape(), Scale::Pinned.shape());
        assert_ne!(Scale::Custom(3).shape(), Scale::Pinned.shape());
        assert_eq!(Scale::Custom(3).name(), "custom");
    }

    #[test]
    fn every_failpoint_has_a_distinct_hit_name_in_the_metric_list() {
        let points = CUTOVER_POINTS
            .iter()
            .chain(std::iter::once(&SYNC_PARENT_POINT))
            .chain(&MEMBER_POINTS)
            .chain(&ERROR_POINTS);
        let names: Vec<String> = points.map(|point| hit_name(point)).collect();
        let listed: Vec<String> = super::super::metrics::HIT_POINTS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(names, listed);
    }
}
