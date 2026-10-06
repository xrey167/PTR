//! Counters, JSON output, and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R005).
//!
//! Every metric the manifest names has exactly one place in the output: a number, an object
//! (the per-workload, per-cell and per-type metrics), a string (digests) or `null` together
//! with a reason under `unsupported`. A metric the harness did not fill and did not declare
//! unsupported is a harness gap: it is printed as `null`, listed under `harness_gaps` and
//! fails the run, so nothing is dropped silently.

use super::params;
use std::collections::BTreeMap;

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 94] = [
    "prefill_tokens_saved",
    "wall_time_ms",
    "hit_rate",
    "bitwise_mismatch_count",
    "revoked_fork_count",
    "revoked_fork_attempts",
    "peak_trie_nodes",
    "peak_kv_page_bytes",
    "evictions",
    "active_fork_page_evictions",
    "evictions_with_active_fork_pinned",
    "invalidation_suppressed_arms",
    "restore_arms",
    "post_event_lookups",
    "stale_hits",
    "input_revocations_performed",
    "post_revocation_lookups_matching_tokens",
    "revoked_lineage_entry_hits",
    "same_tokens_different_generation_cases",
    "revoked_pinned_page_attempts",
    "revoked_pinned_page_served_after_revocation",
    "revoked_pinned_page_bytes_reclaimed_after_release",
    "cross_principal_attempts",
    "cross_principal_hits",
    "discarded_overlay_entry_attempts",
    "discarded_overlay_entry_hits",
    "peak_metadata_bytes",
    "unreachable_revoked_page_bytes_unreclaimed",
    "same_tokens_distinct_lineage_entries",
    "cross_lineage_hits",
    "active_fork_revocation_attempts",
    "cancelled_forks_restarted_from_fresh_prefill",
    "outputs_escaped_after_revocation",
    "reused_prefix_tokens",
    "total_prompt_tokens",
    "cross_project_attempts",
    "cross_project_hits",
    "forged_witness_fork_attempts",
    "forged_witness_forks_accepted",
    "cancelled_run_effect_attempts",
    "cancelled_run_effects_admitted",
    "evictions_nodes_only_run",
    "evictions_bytes_only_run",
    "stale_eligible_entries_before_event",
    "post_event_lookups_on_stale_eligible_entries",
    "cancelled_forks",
    "restored_pre_revocation_entries",
    "empty_restart_arm",
    "raw_prefix_reuse_cases",
    "raw_identical_prefix_hits",
    "supersession_active_fork_race_cases",
    "memory_supersession_active_fork_race_cases",
    "memory_revocation_active_fork_race_cases",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "capacity_refused_requests_completed",
    "capacity_fallback_output_mismatches",
    "capacity_refusal_cache_inserts",
    "supersession_cancelled_forks",
    "supersession_forks_restarted_from_fresh_prefill",
    "cache_inserts_after_supersession",
    "outputs_after_supersession_semantic",
    "outputs_after_supersession_memory",
    "cancelled_fork_id_set_digest",
    "restarted_fork_id_set_digest",
    "capacity_refused_request_id_set_digest",
    "capacity_completed_request_id_set_digest",
    "peak_trie_nodes_by_workload",
    "peak_kv_page_bytes_by_workload",
    "peak_metadata_bytes_by_workload",
    "base_model_inflight_race_cases",
    "base_model_inflight_outputs_after_invalidation",
    "base_model_inflight_cache_inserts",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "superseded_lookup_cases",
    "superseded_stale_hits",
    "saturation_demand_cases",
    "capacity_exhausted_refusals",
    "current_layer_kv_transform_change_cases",
    "stale_exact_hits",
    "stale_witness_replay_cases",
    "witness_reuse_static_evidence_passing",
    "stale_or_reused_witness_forks_admitted",
    "model_pin_mismatches",
    "fixed_regime_decode_crossing_cases",
    "fixed_regime_crossing_parity_failures",
    "effective_regime_transition_cases",
    "exact_outputs_across_effective_regime_transition_without_recompute",
    "mismatched_position_cases",
    "exact_hits_with_mismatched_positions",
    "cross_environment_restore_cases",
    "exact_hits_across_environment",
    "producer_environment_fields",
];

/// Metrics printed as an object built from the dotted counters under their name.
pub const GROUPED: [&str; 10] = [
    "peak_trie_nodes_by_workload",
    "peak_kv_page_bytes_by_workload",
    "peak_metadata_bytes_by_workload",
    "base_model_inflight_race_cases",
    "base_model_inflight_outputs_after_invalidation",
    "base_model_inflight_cache_inserts",
    "invalidation_matrix_cases",
    "invalidation_matrix_stale_hits",
    "superseded_lookup_cases",
    "superseded_stale_hits",
];

/// A JSON value, rendered on one line.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn str(text: impl Into<String>) -> Self {
        Self::Str(text.into())
    }

    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Self {
        Self::Obj(
            pairs
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub fn strs<S: AsRef<str>>(items: &[S]) -> Self {
        Self::Arr(items.iter().map(|item| Self::str(item.as_ref())).collect())
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Self::Int(number) => out.push_str(&number.to_string()),
            Self::Float(number) if number.is_finite() => {
                // `{:?}` keeps a decimal point, so a float stays a float.
                out.push_str(&format!("{number:?}"));
            }
            Self::Float(_) => out.push_str("null"),
            Self::Str(text) => out.push_str(&super::super::json::json_string(text)),
            Self::Arr(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Obj(pairs) => {
                out.push('{');
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&super::super::json::json_string(key));
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// The counters of one run. Names are the manifest's metric names; a dotted name is one
/// member of a grouped metric. Anything else is an internal counter, printed under
/// `internal`.
#[derive(Debug, Default)]
pub struct Tally {
    ints: BTreeMap<String, i64>,
    floats: BTreeMap<String, f64>,
    json: BTreeMap<String, Json>,
    unsupported: BTreeMap<String, String>,
}

impl Tally {
    pub fn add(&mut self, name: &str, amount: i64) {
        *self.ints.entry(name.to_string()).or_insert(0) += amount;
    }

    pub fn set(&mut self, name: &str, value: i64) {
        self.ints.insert(name.to_string(), value);
    }

    /// Keep the larger of the stored value and `value` (peaks).
    pub fn peak(&mut self, name: &str, value: i64) {
        let slot = self.ints.entry(name.to_string()).or_insert(0);
        *slot = (*slot).max(value);
    }

    pub fn set_float(&mut self, name: &str, value: f64) {
        self.floats.insert(name.to_string(), value);
    }

    pub fn set_json(&mut self, name: &str, value: Json) {
        self.json.insert(name.to_string(), value);
    }

    /// Declare a metric or case the code under test cannot honestly produce.
    pub fn unsupported(&mut self, name: &str, reason: &str) {
        self.unsupported
            .insert(name.to_string(), reason.to_string());
    }

    pub fn get(&self, name: &str) -> i64 {
        self.ints.get(name).copied().unwrap_or(0)
    }

    pub fn float(&self, name: &str) -> Option<f64> {
        self.floats.get(name).copied()
    }

    /// A JSON value stored under `name`, or `null`.
    pub fn json_or_null(&self, name: &str) -> Json {
        self.json.get(name).cloned().unwrap_or(Json::Null)
    }

    pub fn is_unsupported(&self, name: &str) -> Option<&str> {
        self.unsupported.get(name).map(String::as_str)
    }

    /// The members `name.<member>` of a grouped metric, in name order.
    pub fn group(&self, name: &str) -> Vec<(String, i64)> {
        let prefix = format!("{name}.");
        self.ints
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&prefix)
                    .map(|rest| (rest.to_string(), *value))
            })
            .collect()
    }

    /// Make sure every member of a grouped metric exists (at zero), so a cell that never ran
    /// is printed as a zero rather than missing.
    pub fn seed_group<S: AsRef<str>>(&mut self, name: &str, members: &[S]) {
        for member in members {
            self.ints
                .entry(format!("{name}.{}", member.as_ref()))
                .or_insert(0);
        }
    }

    fn render_metric(&self, name: &str) -> Option<Json> {
        if let Some(value) = self.json.get(name) {
            return Some(value.clone());
        }
        if let Some(value) = self.floats.get(name) {
            return Some(Json::Float(*value));
        }
        if GROUPED.contains(&name) {
            let members = self.group(name);
            if !members.is_empty() {
                return Some(Json::Obj(
                    members
                        .into_iter()
                        .map(|(member, value)| (member, Json::Int(value)))
                        .collect(),
                ));
            }
        }
        self.ints.get(name).map(|value| Json::Int(*value))
    }

    /// The metrics of the manifest as `(name, value)` in manifest order, plus the names the
    /// harness neither filled nor declared unsupported.
    pub fn metrics(&self) -> (Vec<(String, Json)>, Vec<String>) {
        let mut printed = Vec::new();
        let mut gaps = Vec::new();
        for name in METRIC_NAMES {
            if self.unsupported.contains_key(name) {
                printed.push((name.to_string(), Json::Null));
            } else if let Some(value) = self.render_metric(name) {
                printed.push((name.to_string(), value));
            } else {
                printed.push((name.to_string(), Json::Null));
                gaps.push(name.to_string());
            }
        }
        (printed, gaps)
    }

    /// Counters that are no manifest metric, for the reader of a dry run.
    pub fn internal(&self) -> Json {
        let named = |key: &str| {
            let base = key.split('.').next().unwrap_or(key);
            METRIC_NAMES.contains(&base)
        };
        Json::Obj(
            self.ints
                .iter()
                .filter(|(key, _)| !named(key))
                .map(|(key, value)| (key.clone(), Json::Int(*value)))
                .collect(),
        )
    }

    pub fn unsupported_json(&self) -> Json {
        Json::Obj(
            self.unsupported
                .iter()
                .map(|(name, reason)| (name.clone(), Json::str(reason.clone())))
                .collect(),
        )
    }
}

/// Whether a prerequisite or falsifier holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    /// Coverage: the prerequisite was met. Falsifier: the criterion held.
    Pass,
    /// Coverage: unmet (the run is vacuous). Falsifier: tripped (R005 is falsified).
    Fail,
    /// The code under test cannot produce the value, so nothing is judged.
    Unsupported,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One judged criterion.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub rule: String,
    pub observed: Json,
    pub status: Status,
}

impl Check {
    pub fn json(&self) -> Json {
        Json::obj([
            ("name", Json::str(self.name.clone())),
            ("rule", Json::str(self.rule.clone())),
            ("observed", self.observed.clone()),
            ("status", Json::str(self.status.name())),
        ])
    }
}

/// How an integer is compared with its threshold.
#[derive(Clone, Copy)]
enum Op {
    AtLeast,
    AtMost,
}

impl Op {
    fn symbol(self) -> &'static str {
        match self {
            Self::AtLeast => ">=",
            Self::AtMost => "<=",
        }
    }

    fn holds(self, observed: i64, threshold: i64) -> bool {
        match self {
            Self::AtLeast => observed >= threshold,
            Self::AtMost => observed <= threshold,
        }
    }
}

struct Judge<'a> {
    tally: &'a Tally,
    checks: Vec<Check>,
}

impl<'a> Judge<'a> {
    fn new(tally: &'a Tally) -> Self {
        Self {
            tally,
            checks: Vec::new(),
        }
    }

    /// Compare the integer counter `counter` (a metric, or a dotted member of a grouped one)
    /// with `threshold`. `metric` is the manifest metric whose `unsupported` entry excuses it.
    fn int(&mut self, name: &str, metric: &str, counter: &str, op: Op, threshold: i64) {
        let rule = format!("{counter} {} {threshold}", op.symbol());
        if let Some(reason) = self.tally.is_unsupported(metric) {
            self.checks.push(Check {
                name: name.to_string(),
                rule,
                observed: Json::str(format!("unsupported: {reason}")),
                status: Status::Unsupported,
            });
            return;
        }
        let observed = self.tally.get(counter);
        self.checks.push(Check {
            name: name.to_string(),
            rule,
            observed: Json::Int(observed),
            status: if op.holds(observed, threshold) {
                Status::Pass
            } else {
                Status::Fail
            },
        });
    }

    fn at_least(&mut self, metric: &str, threshold: u64) {
        self.int(metric, metric, metric, Op::AtLeast, threshold as i64);
    }

    fn at_most(&mut self, metric: &str, threshold: u64) {
        self.int(metric, metric, metric, Op::AtMost, threshold as i64);
    }

    fn equal_counters(&mut self, name: &str, left: &str, right: &str) {
        let (a, b) = (self.tally.get(left), self.tally.get(right));
        self.checks.push(Check {
            name: name.to_string(),
            rule: format!("{left} == {right}"),
            observed: Json::obj([("left", Json::Int(a)), ("right", Json::Int(b))]),
            status: if a == b { Status::Pass } else { Status::Fail },
        });
    }

    fn at_least_counter(&mut self, name: &str, larger: &str, smaller: &str) {
        let (a, b) = (self.tally.get(larger), self.tally.get(smaller));
        self.checks.push(Check {
            name: name.to_string(),
            rule: format!("{larger} >= {smaller}"),
            observed: Json::obj([("larger", Json::Int(a)), ("smaller", Json::Int(b))]),
            status: if a >= b { Status::Pass } else { Status::Fail },
        });
    }

    fn equal_digests(&mut self, left: &str, right: &str) {
        let digest = |name: &str| match self.tally.json.get(name) {
            Some(Json::Str(text)) => Some(text.clone()),
            _ => None,
        };
        let (a, b) = (digest(left), digest(right));
        let status = match (&a, &b) {
            (Some(a), Some(b)) if a == b => Status::Pass,
            _ => Status::Fail,
        };
        self.checks.push(Check {
            name: format!("{left}=={right}"),
            rule: format!("{left} == {right}"),
            observed: Json::obj([
                ("left", a.map_or(Json::Null, Json::Str)),
                ("right", b.map_or(Json::Null, Json::Str)),
            ]),
            status,
        });
    }
}

/// The five per-member names of {semantic, memory} x {revocation, supersession}.
pub const MEMBERS: [&str; 4] = [
    "semantic_revocation",
    "semantic_supersession",
    "memory_revocation",
    "memory_supersession",
];

/// The coverage prerequisites of the manifest: each unmet one makes the run vacuous.
pub fn coverage(tally: &Tally) -> Vec<Check> {
    let mut judge = Judge::new(tally);
    judge.at_least("revoked_fork_attempts", params::MIN_REVOKED_FORK_ATTEMPTS);
    judge.int(
        "evictions_nodes_only_run",
        "evictions_nodes_only_run",
        "evictions_nodes_only_run",
        Op::AtLeast,
        params::MIN_EVICTIONS_NODES_ONLY_RUN as i64,
    );
    judge.at_least(
        "evictions_bytes_only_run",
        params::MIN_EVICTIONS_BYTES_ONLY_RUN,
    );
    judge.at_least(
        "evictions_with_active_fork_pinned",
        params::MIN_EVICTIONS_WITH_ACTIVE_FORK_PINNED,
    );
    judge.at_least(
        "invalidation_suppressed_arms",
        params::MIN_INVALIDATION_SUPPRESSED_ARMS,
    );
    judge.at_least("restore_arms", params::MIN_RESTORE_ARMS);
    judge.at_least(
        "restored_pre_revocation_entries",
        params::MIN_RESTORED_PRE_REVOCATION_ENTRIES,
    );
    judge.at_least(
        "stale_eligible_entries_before_event",
        params::MIN_STALE_ELIGIBLE_ENTRIES_BEFORE_EVENT,
    );
    judge.equal_counters(
        "post_event_lookups_on_stale_eligible_entries",
        "post_event_lookups_on_stale_eligible_entries",
        "stale_eligible_entries_before_event",
    );
    judge.at_least("post_event_lookups", params::MIN_POST_EVENT_LOOKUPS);
    judge.at_least(
        "input_revocations_performed",
        params::MIN_INPUT_REVOCATIONS_PERFORMED,
    );
    judge.at_least(
        "post_revocation_lookups_matching_tokens",
        params::MIN_POST_REVOCATION_LOOKUPS_MATCHING_TOKENS,
    );
    judge.at_least(
        "same_tokens_different_generation_cases",
        params::MIN_SAME_TOKENS_DIFFERENT_GENERATION_CASES,
    );
    judge.at_least(
        "revoked_pinned_page_attempts",
        params::MIN_REVOKED_PINNED_PAGE_ATTEMPTS,
    );
    judge.at_least(
        "cross_principal_attempts",
        params::MIN_CROSS_PRINCIPAL_ATTEMPTS,
    );
    judge.at_least("cross_project_attempts", params::MIN_CROSS_PROJECT_ATTEMPTS);
    judge.at_least(
        "discarded_overlay_entry_attempts",
        params::MIN_DISCARDED_OVERLAY_ENTRY_ATTEMPTS,
    );
    judge.at_least(
        "same_tokens_distinct_lineage_entries",
        params::MIN_SAME_TOKENS_DISTINCT_LINEAGE_ENTRIES,
    );
    judge.at_least(
        "active_fork_revocation_attempts",
        params::MIN_ACTIVE_FORK_REVOCATION_ATTEMPTS,
    );
    judge.at_least(
        "forged_witness_fork_attempts",
        params::MIN_FORGED_WITNESS_FORK_ATTEMPTS,
    );
    judge.at_least(
        "cancelled_run_effect_attempts",
        params::MIN_CANCELLED_RUN_EFFECT_ATTEMPTS,
    );
    judge.at_least("cancelled_forks", params::MIN_CANCELLED_FORKS);
    judge.at_least(
        "saturation_demand_cases",
        params::MIN_SATURATION_DEMAND_CASES,
    );
    judge.at_least(
        "capacity_exhausted_refusals",
        params::MIN_CAPACITY_EXHAUSTED_REFUSALS,
    );
    judge.at_least("raw_prefix_reuse_cases", params::MIN_RAW_PREFIX_REUSE_CASES);
    judge.at_least(
        "supersession_active_fork_race_cases",
        params::MIN_SUPERSESSION_ACTIVE_FORK_RACE_CASES,
    );
    judge.at_least(
        "memory_supersession_active_fork_race_cases",
        params::MIN_MEMORY_SUPERSESSION_ACTIVE_FORK_RACE_CASES,
    );
    judge.at_least(
        "memory_revocation_active_fork_race_cases",
        params::MIN_MEMORY_REVOCATION_ACTIVE_FORK_RACE_CASES,
    );
    judge.at_most(
        "selection_digest_mismatches",
        params::MAX_SELECTION_DIGEST_MISMATCHES,
    );
    judge.at_most("procedure_mismatches", params::MAX_PROCEDURE_MISMATCHES);
    judge.equal_counters(
        "capacity_refused_requests_completed",
        "capacity_refused_requests_completed",
        "capacity_exhausted_refusals",
    );
    judge.at_least(
        "supersession_cancelled_forks",
        params::MIN_SUPERSESSION_CANCELLED_FORKS,
    );
    for member in MEMBERS {
        judge.int(
            &format!("cancelled_forks.{member}"),
            "cancelled_forks",
            &format!("member_cancelled_forks.{member}"),
            Op::AtLeast,
            params::MIN_CANCELLED_FORKS as i64,
        );
        judge.int(
            &format!("active_fork_race_cases.{member}"),
            "cancelled_forks",
            &format!("member_race_cases.{member}"),
            Op::AtLeast,
            1,
        );
    }
    judge.equal_digests(
        "cancelled_fork_id_set_digest",
        "restarted_fork_id_set_digest",
    );
    judge.equal_digests(
        "capacity_refused_request_id_set_digest",
        "capacity_completed_request_id_set_digest",
    );
    for kind_type in params::MATRIX_TYPES {
        judge.int(
            &format!("base_model_inflight_race_cases.{kind_type}"),
            "base_model_inflight_race_cases",
            &format!("base_model_inflight_race_cases.{kind_type}"),
            Op::AtLeast,
            params::MIN_BASE_MODEL_INFLIGHT_RACE_CASES_PER_TYPE as i64,
        );
    }
    for cell in params::MATRIX_CELLS {
        judge.int(
            &format!("invalidation_matrix_cases.{cell}"),
            "invalidation_matrix_cases",
            &format!("invalidation_matrix_cases.{cell}"),
            Op::AtLeast,
            params::MIN_INVALIDATION_MATRIX_CASES_PER_CELL as i64,
        );
    }
    for [kind, path] in params::SUPERSEDED_LOOKUP_CELLS {
        let cell = format!("{kind}_{path}");
        judge.int(
            &format!("superseded_lookup_cases.{cell}"),
            "superseded_lookup_cases",
            &format!("superseded_lookup_cases.{cell}"),
            Op::AtLeast,
            params::MIN_SUPERSEDED_LOOKUP_CASES_PER_CELL as i64,
        );
    }
    judge.at_least("semantic_runtime_segment_cases", 1);
    judge.at_least("semantic_runtime_segment_live_hits", 1);
    judge.at_least("semantic_lifecycle_identity_cases", 1);
    judge.at_least(
        "current_layer_kv_transform_change_cases",
        params::MIN_CURRENT_LAYER_KV_TRANSFORM_CHANGE_CASES,
    );
    judge.at_least(
        "stale_witness_replay_cases",
        params::MIN_STALE_WITNESS_REPLAY_CASES,
    );
    judge.at_least(
        "witness_reuse_static_evidence_passing",
        params::MIN_WITNESS_REUSE_STATIC_EVIDENCE_PASSING,
    );
    judge.at_most("model_pin_mismatches", params::MAX_MODEL_PIN_MISMATCHES);
    judge.at_least(
        "fixed_regime_decode_crossing_cases",
        params::MIN_FIXED_REGIME_DECODE_CROSSING_CASES,
    );
    judge.at_least(
        "effective_regime_transition_cases",
        params::MIN_EFFECTIVE_REGIME_TRANSITION_CASES,
    );
    judge.at_least(
        "mismatched_position_cases",
        params::MIN_MISMATCHED_POSITION_CASES,
    );
    judge.at_least(
        "cross_environment_restore_cases",
        params::MIN_CROSS_ENVIRONMENT_RESTORE_CASES,
    );
    judge.checks
}

/// The falsifiers of the manifest: each tripped one falsifies R005 on this seed.
pub fn falsifiers(tally: &Tally) -> Vec<Check> {
    let mut judge = Judge::new(tally);
    judge.int(
        "bitwise_mismatch_count",
        "bitwise_mismatch_count",
        "bitwise_mismatch_count",
        Op::AtMost,
        0,
    );
    judge.at_most("revoked_fork_count", params::MAX_REVOKED_FORK_COUNT);
    // hit_rate >= 0.40 on the 50%-shared-prefix workload.
    let hit_rate = tally.float("hit_rate");
    judge.checks.push(Check {
        name: "hit_rate".to_string(),
        rule: format!(">= {}", params::MIN_HIT_RATE_AT_50PCT_SHARED_PREFIX),
        observed: hit_rate.map_or(Json::Null, Json::Float),
        status: match hit_rate {
            Some(rate) if rate >= params::MIN_HIT_RATE_AT_50PCT_SHARED_PREFIX => Status::Pass,
            _ => Status::Fail,
        },
    });
    for workload in params::Workload::ALL {
        let limits = workload.limits();
        let name = workload.name();
        for (metric, limit) in [
            ("peak_trie_nodes_by_workload", limits.max_trie_nodes),
            ("peak_kv_page_bytes_by_workload", limits.max_kv_page_bytes),
            ("peak_metadata_bytes_by_workload", limits.max_metadata_bytes),
        ] {
            judge.int(
                &format!("{metric}.{name}"),
                metric,
                &format!("{metric}.{name}"),
                Op::AtMost,
                limit as i64,
            );
        }
    }
    judge.at_most(
        "active_fork_page_evictions",
        params::MAX_ACTIVE_FORK_PAGE_EVICTIONS,
    );
    judge.at_most("stale_hits", params::MAX_STALE_HITS);
    judge.at_most(
        "revoked_lineage_entry_hits",
        params::MAX_REVOKED_LINEAGE_ENTRY_HITS,
    );
    judge.at_most(
        "revoked_pinned_page_served_after_revocation",
        params::MAX_REVOKED_PINNED_PAGE_SERVED_AFTER_REVOCATION,
    );
    // "all": every byte of every revoked pinned page is reclaimed after the release.
    judge.at_least_counter(
        "revoked_pinned_page_bytes_reclaimed_after_release",
        "revoked_pinned_page_bytes_reclaimed_after_release",
        "revoked_pinned_page_bytes_total",
    );
    judge.at_most("cross_principal_hits", params::MAX_CROSS_PRINCIPAL_HITS);
    judge.at_most("cross_project_hits", params::MAX_CROSS_PROJECT_HITS);
    judge.at_most(
        "discarded_overlay_entry_hits",
        params::MAX_DISCARDED_OVERLAY_ENTRY_HITS,
    );
    judge.int(
        "unreachable_revoked_page_bytes_unreclaimed",
        "unreachable_revoked_page_bytes_unreclaimed",
        "unreachable_revoked_page_bytes_unreclaimed",
        Op::AtMost,
        0,
    );
    judge.at_most("cross_lineage_hits", params::MAX_CROSS_LINEAGE_HITS);
    judge.at_least_counter(
        "cancelled_forks_restarted_from_fresh_prefill",
        "cancelled_forks_restarted_from_fresh_prefill",
        "cancelled_forks",
    );
    judge.at_most(
        "outputs_escaped_after_revocation",
        params::MAX_OUTPUTS_ESCAPED_AFTER_REVOCATION,
    );
    judge.at_most(
        "forged_witness_forks_accepted",
        params::MAX_FORGED_WITNESS_FORKS_ACCEPTED,
    );
    judge.at_most(
        "cancelled_run_effects_admitted",
        params::MAX_CANCELLED_RUN_EFFECTS_ADMITTED,
    );
    judge.at_least(
        "raw_identical_prefix_hits",
        params::MIN_RAW_IDENTICAL_PREFIX_HITS,
    );
    judge.at_most(
        "outputs_after_supersession_semantic",
        params::MAX_OUTPUTS_AFTER_SUPERSESSION_SEMANTIC,
    );
    judge.at_most(
        "outputs_after_supersession_memory",
        params::MAX_OUTPUTS_AFTER_SUPERSESSION_MEMORY,
    );
    judge.at_most(
        "capacity_fallback_output_mismatches",
        params::MAX_CAPACITY_FALLBACK_OUTPUT_MISMATCHES,
    );
    judge.at_most(
        "capacity_refusal_cache_inserts",
        params::MAX_CAPACITY_REFUSAL_CACHE_INSERTS,
    );
    judge.at_least_counter(
        "supersession_forks_restarted_from_fresh_prefill",
        "supersession_forks_restarted_from_fresh_prefill",
        "supersession_cancelled_forks",
    );
    judge.at_most(
        "cache_inserts_after_supersession",
        params::MAX_CACHE_INSERTS_AFTER_SUPERSESSION,
    );
    for kind_type in params::MATRIX_TYPES {
        judge.int(
            &format!("base_model_inflight_outputs_after_invalidation.{kind_type}"),
            "base_model_inflight_outputs_after_invalidation",
            &format!("base_model_inflight_outputs_after_invalidation.{kind_type}"),
            Op::AtMost,
            params::MAX_BASE_MODEL_INFLIGHT_OUTPUTS_AFTER_INVALIDATION as i64,
        );
        judge.int(
            &format!("base_model_inflight_cache_inserts.{kind_type}"),
            "base_model_inflight_cache_inserts",
            &format!("base_model_inflight_cache_inserts.{kind_type}"),
            Op::AtMost,
            params::MAX_BASE_MODEL_INFLIGHT_CACHE_INSERTS as i64,
        );
    }
    for cell in params::MATRIX_CELLS {
        judge.int(
            &format!("invalidation_matrix_stale_hits.{cell}"),
            "invalidation_matrix_stale_hits",
            &format!("invalidation_matrix_stale_hits.{cell}"),
            Op::AtMost,
            params::MAX_INVALIDATION_MATRIX_STALE_HITS_PER_CELL as i64,
        );
    }
    for [kind, path] in params::SUPERSEDED_LOOKUP_CELLS {
        let cell = format!("{kind}_{path}");
        judge.int(
            &format!("superseded_stale_hits.{cell}"),
            "superseded_stale_hits",
            &format!("superseded_stale_hits.{cell}"),
            Op::AtMost,
            params::MAX_SUPERSEDED_STALE_HITS as i64,
        );
    }
    judge.at_most("semantic_runtime_segment_stale_hits", 0);
    judge.at_most("semantic_lifecycle_identity_live_hits", 0);
    judge.at_most("stale_exact_hits", params::MAX_STALE_EXACT_HITS);
    judge.at_most(
        "stale_or_reused_witness_forks_admitted",
        params::MAX_STALE_OR_REUSED_WITNESS_FORKS_ADMITTED,
    );
    judge.at_most(
        "fixed_regime_crossing_parity_failures",
        params::MAX_FIXED_REGIME_CROSSING_PARITY_FAILURES,
    );
    judge.at_most(
        "exact_outputs_across_effective_regime_transition_without_recompute",
        params::MAX_EXACT_OUTPUTS_ACROSS_EFFECTIVE_REGIME_TRANSITION_WITHOUT_RECOMPUTE,
    );
    judge.at_most(
        "exact_hits_with_mismatched_positions",
        params::MAX_EXACT_HITS_WITH_MISMATCHED_POSITIONS,
    );
    judge.at_most(
        "exact_hits_across_environment",
        params::MAX_EXACT_HITS_ACROSS_ENVIRONMENT,
    );
    judge.checks
}

/// How many checks have each status.
pub fn count(checks: &[Check], status: Status) -> usize {
    checks.iter().filter(|check| check.status == status).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_renders_on_one_line_and_escapes() {
        let value = Json::obj([
            ("a", Json::Int(-3)),
            ("b", Json::Float(0.5)),
            ("c", Json::str("x\"y\n")),
            ("d", Json::Arr(vec![Json::Null, Json::Bool(true)])),
            ("e", Json::Float(f64::NAN)),
            ("f", Json::Float(2.0)),
        ]);
        assert_eq!(
            value.render(),
            r#"{"a":-3,"b":0.5,"c":"x\"y\n","d":[null,true],"e":null,"f":2.0}"#
        );
    }

    #[test]
    fn an_unfilled_metric_is_a_gap_and_an_unsupported_one_is_null_with_a_reason() {
        let mut tally = Tally::default();
        tally.set("hit_rate_placeholder", 1);
        tally.unsupported("empty_restart_arm", "because");
        let (printed, gaps) = tally.metrics();
        assert_eq!(printed.len(), METRIC_NAMES.len());
        assert!(!gaps.contains(&"empty_restart_arm".to_string()));
        assert!(gaps.contains(&"hit_rate".to_string()));
        assert_eq!(gaps.len(), METRIC_NAMES.len() - 1);
        assert!(tally.unsupported_json().render().contains("because"));
    }

    #[test]
    fn grouped_metrics_print_as_objects_and_peaks_keep_the_maximum() {
        let mut tally = Tally::default();
        tally.add("invalidation_matrix_cases.a", 2);
        tally.add("invalidation_matrix_cases.b", 1);
        tally.peak("peak_trie_nodes", 4);
        tally.peak("peak_trie_nodes", 3);
        let (printed, _) = tally.metrics();
        let find = |name: &str| {
            printed
                .iter()
                .find(|(key, _)| key == name)
                .unwrap()
                .1
                .render()
        };
        assert_eq!(find("invalidation_matrix_cases"), r#"{"a":2,"b":1}"#);
        assert_eq!(find("peak_trie_nodes"), "4");
    }

    #[test]
    fn a_coverage_prerequisite_at_zero_fails_and_an_unsupported_one_is_not_judged() {
        let mut tally = Tally::default();
        tally.unsupported("witness_reuse_static_evidence_passing", "needs cargo test");
        let checks = coverage(&tally);
        let find = |name: &str| {
            checks
                .iter()
                .find(|check| check.name == name)
                .unwrap()
                .status
        };
        assert_eq!(find("revoked_fork_attempts"), Status::Fail);
        assert_eq!(
            find("witness_reuse_static_evidence_passing"),
            Status::Unsupported
        );
        assert_eq!(find("selection_digest_mismatches"), Status::Pass);
        tally.set("revoked_fork_attempts", 1);
        let checks = coverage(&tally);
        let find = |name: &str| {
            checks
                .iter()
                .find(|check| check.name == name)
                .unwrap()
                .status
        };
        assert_eq!(find("revoked_fork_attempts"), Status::Pass);
    }

    #[test]
    fn a_falsifier_trips_on_a_stale_hit_and_on_a_low_hit_rate() {
        let mut tally = Tally::default();
        tally.set_float("hit_rate", 0.39);
        tally.set("stale_hits", 1);
        let checks = falsifiers(&tally);
        let find = |name: &str| {
            checks
                .iter()
                .find(|check| check.name == name)
                .unwrap()
                .status
        };
        assert_eq!(find("hit_rate"), Status::Fail);
        assert_eq!(find("stale_hits"), Status::Fail);
        assert_eq!(find("cross_principal_hits"), Status::Pass);
        tally.set_float("hit_rate", 0.4);
        tally.set("stale_hits", 0);
        let checks = falsifiers(&tally);
        let find = |name: &str| {
            checks
                .iter()
                .find(|check| check.name == name)
                .unwrap()
                .status
        };
        assert_eq!(find("hit_rate"), Status::Pass);
        assert_eq!(find("stale_hits"), Status::Pass);
    }

    #[test]
    fn every_matrix_cell_has_a_coverage_and_a_falsifier_check() {
        let tally = Tally::default();
        let covered = coverage(&tally);
        let falsified = falsifiers(&tally);
        for cell in params::MATRIX_CELLS {
            assert!(covered
                .iter()
                .any(|check| check.name == format!("invalidation_matrix_cases.{cell}")));
            assert!(falsified
                .iter()
                .any(|check| check.name == format!("invalidation_matrix_stale_hits.{cell}")));
        }
    }
}
