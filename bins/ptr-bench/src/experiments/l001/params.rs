//! The preregistered parameters of L001-v2, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. The `[preregistration]` table
//! of the same file repeats these values and is frozen at the freeze.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str =
    include_str!("../../../../../experiments/lifecycle/L001-v2-revocation-crash/config.toml");

/// The experiment's manifest, read at compile time (the metric names come from it).
pub const EXPERIMENT_TOML: &str =
    include_str!("../../../../../experiments/lifecycle/L001-v2-revocation-crash/experiment.toml");

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "revocation-crash";

/// The experiment's id as the harness prints it.
pub const EXPERIMENT_ID: &str = "L001-v2";

/// The runner's command, as the preregistration states it; `experiment.toml` carries the same.
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench --features ledger-experiments -- revocation-crash <seed>";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// A workload is 6 to 14 events over 3 or 4 subjects.
pub const EVENTS_MIN: usize = 6;
pub const EVENTS_MAX: usize = 14;

/// Workloads per seed in each arm, and Raft scenarios per kind.
pub const WORKLOADS_FILE: usize = 8;
pub const WORKLOADS_ACK: usize = 8;
pub const WORKLOADS_ENGINE: usize = 3;
pub const RAFT_SCENARIOS_PER_KIND: usize = 3;

/// Appends per failpoint that are crashed for real, in process (panic) and in a child (abort).
pub const UNWIND_IDX_PER_POINT: usize = 2;
pub const CHILD_IDX_PER_POINT: usize = 1;
pub const ENGINE_CHILD_IDX_PER_POINT: usize = 2;

/// The watchdog of one recovery or in-process append sequence (generous, so that a loaded
/// machine is not mistaken for a hang), and how many may fire before the run stops issuing them.
pub const HANG_WATCHDOG_MS: u64 = 60_000;
/// The watchdog of one child process, in milliseconds (a child runs a whole workload).
pub const CHILD_WATCHDOG_MS: u64 = 60_000;
pub const HANG_LIMIT: i64 = 3;

/// The size of a hole of the `holes` policy.
pub const SECTOR_BYTES: usize = 512;

/// Fault rates of the message-fault scenario, per mille of messages.
pub const FAULT_DROP_PERMILLE: u64 = 120;
pub const FAULT_DUPLICATE_PERMILLE: u64 = 150;
pub const FAULT_REORDER_PERMILLE: u64 = 200;

pub const LOG_POINTS: [&str; 4] = [
    "ledger.before_record_write",
    "ledger.after_length_before_payload",
    "ledger.after_payload_before_sync",
    "ledger.after_sync_before_memory",
];
pub const ANCHOR_POINTS: [&str; 4] = [
    "anchor.after_scratch_write_before_sync",
    "anchor.after_scratch_sync_before_rename",
    "anchor.after_rename_before_dir_sync",
    "anchor.after_dir_sync",
];
/// The directory synchronization (counted, not crashed at).
pub const SYNC_PARENT_POINT: &str = "ledger.sync_parent";
pub const ENGINE_POINTS: [&str; 3] = [
    "file_pipe_log::append",
    "log_file::write::err",
    "log_fd::sync::err",
];

/// The name under which a failpoint's hits are counted in `failpoint_hits`.
pub fn hit_name(point: &str) -> String {
    match point {
        "file_pipe_log::append" => "engine_append".into(),
        "log_file::write::err" => "engine_write".into(),
        "log_fd::sync::err" => "engine_sync".into(),
        other => other.replace('.', "_"),
    }
}

// Coverage prerequisites (a run that misses one is vacuous).
pub const MIN_WORKLOADS_FILE: i64 = 6;
pub const MIN_WORKLOADS_ACK: i64 = 6;
pub const MIN_WORKLOADS_ENGINE: i64 = 2;
pub const MIN_SCENARIOS_RAFT: i64 = 10;
pub const MIN_RECOVERIES_FILE: i64 = 600;
pub const MIN_RECOVERIES_ACK: i64 = 1000;
pub const MIN_RECOVERIES_ENGINE: i64 = 250;
pub const MIN_MEMBERS_REPLAYED: i64 = 70;
pub const MIN_FAILPOINT_HITS: i64 = 20;
pub const MIN_IMAGES_LOST_BYTES: i64 = 1000;
pub const MIN_RENAME_IMAGE_PAIRS: i64 = 50;
pub const MIN_LIE_IMAGES: i64 = 100;
pub const MIN_CHILD_ABORTS_FILE: i64 = 24;
pub const MIN_CHILD_ABORTS_ACK: i64 = 48;
pub const MIN_CHILD_ABORTS_ENGINE: i64 = 12;
pub const MIN_UNWIND_PANICS: i64 = 144;
pub const MIN_LEADER_CHANGES: i64 = 4;
pub const MIN_MINORITY_PROPOSALS: i64 = 2;
pub const MIN_DEPOSED_LEADER_RESTARTS: i64 = 2;
pub const MIN_MESSAGES_DROPPED: i64 = 20;
pub const MIN_MESSAGES_DUPLICATED: i64 = 20;
pub const MIN_MESSAGES_REORDERED: i64 = 20;
pub const MIN_REVOKE_THEN_RECOMMIT: i64 = 14;
pub const MIN_SUBJECTS: i64 = 3;
pub const MIN_EVENTS: i64 = 6;
pub const MIN_SCENARIOS_CHECKED: i64 = 10;

/// The bound on the share of recoveries of damaged images that the product refuses, per mille.
/// Set from the pilot seeds 1001 to 1003 on the product with the zero-tail trim (shape A: no
/// refusal; shapes B, C and D refused 147, 134 and 151 per mille of all recoveries of damaged
/// images), with a margin of about two. See the preregistration.
pub const MAX_REFUSED_SHARE_DAMAGED_PERMILLE: i64 = 300;
pub const MIN_RECOVERIES_WITH_ACKED_REVOCATION: i64 = 700;
pub const MIN_REVOKED_PAIRS_DENIED: i64 = 900;
pub const MIN_CRASH_POINTS_AT_REVOCATION: i64 = 140;
pub const MIN_PROBE_CALIBRATED: i64 = 1;
pub const MIN_PROBE_DIRTY_SEEN: i64 = 1;
pub const MIN_PROBE_CLEAN_CHECKED: i64 = 400;

/// What a run covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub file: usize,
    pub ack: usize,
    pub engine: usize,
    pub raft_per_kind: usize,
}

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests: one workload per arm.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
    /// This many workloads per arm and one scenario per Raft kind (the mutation checker's
    /// second argument); no evidence.
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
                file: WORKLOADS_FILE,
                ack: WORKLOADS_ACK,
                engine: WORKLOADS_ENGINE,
                raft_per_kind: RAFT_SCENARIOS_PER_KIND,
            },
            Self::Reduced => Shape {
                file: 1,
                ack: 1,
                engine: 1,
                raft_per_kind: 1,
            },
            Self::Custom(cases) => Shape {
                file: cases,
                ack: cases,
                engine: cases.div_ceil(2),
                raft_per_kind: 1,
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
        ("workloads_file", int(WORKLOADS_FILE)),
        ("workloads_ack", int(WORKLOADS_ACK)),
        ("workloads_engine", int(WORKLOADS_ENGINE)),
        ("raft_scenarios_per_kind", int(RAFT_SCENARIOS_PER_KIND)),
        ("unwind_idx_per_point", int(UNWIND_IDX_PER_POINT)),
        ("child_idx_per_point", int(CHILD_IDX_PER_POINT)),
        (
            "engine_child_idx_per_point",
            int(ENGINE_CHILD_IDX_PER_POINT),
        ),
        ("hang_watchdog_ms", int(HANG_WATCHDOG_MS)),
        ("child_watchdog_ms", int(CHILD_WATCHDOG_MS)),
        ("hang_limit", int(HANG_LIMIT)),
        ("sector_bytes", int(SECTOR_BYTES)),
        ("fault_drop_permille", int(FAULT_DROP_PERMILLE)),
        ("fault_duplicate_permille", int(FAULT_DUPLICATE_PERMILLE)),
        ("fault_reorder_permille", int(FAULT_REORDER_PERMILLE)),
        ("log_points", texts(&LOG_POINTS)),
        ("anchor_points", texts(&ANCHOR_POINTS)),
        ("engine_points", texts(&ENGINE_POINTS)),
        ("sync_parent_point", Leaf::Text(SYNC_PARENT_POINT.into())),
        (
            "power_loss_policies",
            texts(&["none", "all", "prefix", "half", "holes"]),
        ),
        (
            "raft_scenario_kinds",
            texts(&[
                "minority-leader",
                "term-change",
                "restarts",
                "message-faults",
            ]),
        ),
        ("min_workloads_file", int(MIN_WORKLOADS_FILE)),
        ("min_workloads_ack", int(MIN_WORKLOADS_ACK)),
        ("min_workloads_engine", int(MIN_WORKLOADS_ENGINE)),
        ("min_scenarios_raft", int(MIN_SCENARIOS_RAFT)),
        ("min_recoveries_file", int(MIN_RECOVERIES_FILE)),
        ("min_recoveries_ack", int(MIN_RECOVERIES_ACK)),
        ("min_recoveries_engine", int(MIN_RECOVERIES_ENGINE)),
        ("min_members_replayed", int(MIN_MEMBERS_REPLAYED)),
        ("min_failpoint_hits", int(MIN_FAILPOINT_HITS)),
        ("min_images_lost_bytes", int(MIN_IMAGES_LOST_BYTES)),
        ("min_rename_image_pairs", int(MIN_RENAME_IMAGE_PAIRS)),
        ("min_lie_images", int(MIN_LIE_IMAGES)),
        ("min_child_aborts_file", int(MIN_CHILD_ABORTS_FILE)),
        ("min_child_aborts_ack", int(MIN_CHILD_ABORTS_ACK)),
        ("min_child_aborts_engine", int(MIN_CHILD_ABORTS_ENGINE)),
        ("min_unwind_panics_observed", int(MIN_UNWIND_PANICS)),
        ("min_leader_changes", int(MIN_LEADER_CHANGES)),
        ("min_minority_proposals", int(MIN_MINORITY_PROPOSALS)),
        (
            "min_deposed_leader_restarts",
            int(MIN_DEPOSED_LEADER_RESTARTS),
        ),
        ("min_messages_dropped", int(MIN_MESSAGES_DROPPED)),
        ("min_messages_duplicated", int(MIN_MESSAGES_DUPLICATED)),
        ("min_messages_reordered", int(MIN_MESSAGES_REORDERED)),
        (
            "min_revoke_then_recommit_histories",
            int(MIN_REVOKE_THEN_RECOMMIT),
        ),
        ("min_subjects", int(MIN_SUBJECTS)),
        ("min_events", int(MIN_EVENTS)),
        ("min_scenarios_checked", int(MIN_SCENARIOS_CHECKED)),
        (
            "min_recoveries_with_acked_revocation",
            int(MIN_RECOVERIES_WITH_ACKED_REVOCATION),
        ),
        ("min_revoked_pairs_denied", int(MIN_REVOKED_PAIRS_DENIED)),
        (
            "min_crash_points_at_revocation",
            int(MIN_CRASH_POINTS_AT_REVOCATION),
        ),
        ("min_probe_calibrated", int(MIN_PROBE_CALIBRATED)),
        ("min_probe_dirty_seen", int(MIN_PROBE_DIRTY_SEEN)),
        ("min_probe_clean_checked", int(MIN_PROBE_CLEAN_CHECKED)),
        (
            "max_refused_share_damaged_permille",
            int(MAX_REFUSED_SHARE_DAMAGED_PERMILLE),
        ),
        ("max_every_falsifier", int(0)),
        ("timing_gated", Leaf::Bool(false)),
        ("engine_lie_gated", Leaf::Bool(false)),
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
        let changed = CONFIG_TOML.replace("events_max = 14\n", "events_max = 15\n");
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["events_max: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("events_max = 14\n", "");
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
        assert_eq!(Scale::Pinned.shape().file, WORKLOADS_FILE);
        assert_ne!(Scale::Reduced.shape(), Scale::Pinned.shape());
        assert_ne!(Scale::Custom(3).shape(), Scale::Pinned.shape());
        assert_eq!(Scale::Custom(3).name(), "custom");
    }

    #[test]
    fn every_failpoint_has_a_distinct_hit_name_in_the_metric_list() {
        let points = LOG_POINTS
            .iter()
            .chain(&ANCHOR_POINTS)
            .chain(&ENGINE_POINTS)
            .chain(std::iter::once(&SYNC_PARENT_POINT));
        let names: Vec<String> = points.map(|point| hit_name(point)).collect();
        let listed: Vec<String> = super::super::metrics::HIT_POINTS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        assert_eq!(names, listed);
    }
}
