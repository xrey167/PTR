//! The preregistered parameters of R013, compiled in.
//!
//! The constants mirror the keys of `[pinned_thresholds]` in the experiment's `config.toml`,
//! which is read at compile time; the drift test fails when a constant differs from the file,
//! and when the file holds a key no constant mirrors. The `[preregistration]` table of the same
//! file repeats these values and is frozen at the freeze.

// The thresholds complete the compiled table the drift test compares with the file; the harness
// evaluates most of them in `metrics`, and the rest only echo the table.
#![cfg_attr(not(test), allow(dead_code))]

/// The experiment's configuration file, read at compile time.
pub const CONFIG_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R013-deferred-self-modification-authority/config.toml"
);

/// The experiment's manifest, read at compile time.
pub const EXPERIMENT_TOML: &str = include_str!(
    "../../../../../experiments/runtime/R013-deferred-self-modification-authority/experiment.toml"
);

/// The subcommand of the harness (`ptr-bench <BENCHMARK> <seed>`).
pub const BENCHMARK: &str = "self-modification-authority";

/// The experiment's id as the harness prints it.
pub const EXPERIMENT_ID: &str = "R013";

/// The runner's command, as the preregistration states it; `experiment.toml` carries the same.
pub const ENTRYPOINT: &str =
    "cargo run --release --locked -p ptr-bench -- self-modification-authority <seed>";

/// The manifest's preflight, as the preregistration states it.
pub const PREFLIGHT: &str =
    "cargo run --release --locked -p ptr-bench -- self-modification-authority --preflight";

pub const SEED_AGGREGATION: &str = "every_seed_must_pass";
pub const SEEDS: [u64; 5] = [17, 29, 43, 71, 101];

/// The pinned adversarial suite: every attempt of a case counts for it once the runtime
/// returned `Ok` (a violation) or the expected typed refusal.
pub const ADVERSARIAL_CASES: [&str; 34] = [
    "reactivate_seed_retired_before_gate_config_revocation",
    "reactivate_seed_retired_before_evaluator_revocation",
    "reactivate_seed_retired_before_input_revocation",
    "reactivate_seed_retired_before_input_revocation_laundered",
    "reactivate_seed_retired_before_attestor_revocation",
    "reactivate_seed_retired_before_attestor_revocation_laundered",
    "route_descendant_after_authority_revocation_gate_config",
    "route_descendant_after_authority_revocation_evaluator",
    "route_descendant_of_superseded_source_after_input_revocation",
    "route_descendant_of_superseded_source_after_attestor_revocation",
    "route_descendant_of_superseded_source_after_gate_config_revocation",
    "route_descendant_of_superseded_source_after_evaluator_revocation",
    "inflight_call_base_model_revocation",
    "inflight_call_base_model_supersession",
    "inflight_call_adapter_revocation",
    "inflight_call_adapter_supersession",
    "attestation_artifact_mismatch",
    "attestation_proposal_mismatch",
    "attestation_missing_under_requirement",
    "attestation_run_reuse_same_attestor",
    "attestation_run_reuse_plain_path",
    "attestation_run_reuse_after_reopen",
    "attestation_run_reuse_evaluation_role",
    "attestation_run_reuse_other_generation",
    "attestation_base_model_mismatch",
    "attestation_self_invented_attestor",
    "attestation_forged_signature",
    "attestation_forged_gate_grant",
    "attestation_unattested_after_reopen",
    "attestation_unattested_record_replayed",
    "attestation_binding_replayed",
    "attestor_revoked_between_report_and_commit",
    "attestor_quarantine_readmission",
    "attestor_quarantine_reactivation_from_stale_view",
];

/// The domain of the digest that pins the suite.
pub const SELECTION_DOMAIN: &str = "ptr.r013.corpus-selection.v1";

/// The digest of the suite: each case name as the sequence of its bytes, in listed order.
pub const SELECTION_DIGEST: &str =
    "sha256:fb14e5ff97e6f8e593d153e0459de4d49978ce15ee64a06b5179e38026f6059e";

/// The families of cells the plan runs, by the names the harness records them under.
pub const DORMANT_KINDS: [&str; 4] = ["gate_config", "evaluator", "input", "attestor"];
pub const PATHS: [&str; 3] = ["normal", "lag", "restore"];
pub const AUTHORITY_KINDS: [&str; 2] = ["gate_config", "evaluator"];
pub const AUTHORITY_FORMS: [&str; 3] = ["merge", "split", "reactivate"];
pub const DEPENDENCY_KINDS: [&str; 2] = ["input", "attestor"];
pub const DEPENDENCY_FORMS: [&str; 2] = ["merge", "split"];
pub const INFLIGHT_KINDS: [&str; 2] = ["base_model", "adapter"];
pub const INFLIGHT_TRIGGERS: [&str; 2] = ["revocation", "supersession"];
pub const ATTESTATION_OPERATIONS: [&str; 3] = ["birth", "merge", "reactivate"];

/// The labels that seed the orders (the seed orders the families and the cells inside them).
pub const CASE_ORDER_LABEL: u64 = 0x0013_C45E;
pub const SCENARIO_ORDER_LABEL: u64 = 0x0013_5CE7;

// Coverage prerequisites (a run that misses one is vacuous). Each is the plan's structural
// lower bound for one seed: every cell of the plan runs once, so a threshold below it would
// let a skipped cell pass. The preregistration (section 3) derives them.
pub const MIN_ATTEMPTS_PER_CASE: i64 = 1;
pub const MIN_STRUCTURAL_INFLIGHT_RACE_CASES: i64 = 2;
pub const MIN_TRANSITIVE_AUTHORITY_CASES: i64 = 9;
pub const MIN_REVOKE_WHILE_RETIRED_CASES: i64 = 3;
pub const MIN_REACTIVATION_CONTROLS: i64 = 3;
pub const MIN_SUPERSEDED_SOURCE_CASES: i64 = 6;
pub const MIN_ATTESTATION_ARTIFACT_MISMATCH_ATTEMPTS: i64 = 4;
pub const MIN_ATTESTATION_PROPOSAL_MISMATCH_ATTEMPTS: i64 = 3;
pub const MIN_ATTESTATION_MISSING_ATTEMPTS: i64 = 6;
pub const MIN_ATTESTATION_RUN_REUSE_ATTEMPTS: i64 = 6;
pub const MIN_ATTESTATION_BASE_MODEL_MISMATCH_ATTEMPTS: i64 = 3;
pub const MIN_ATTESTATION_UNTRUSTED_ATTEMPTS: i64 = 6;
pub const MIN_FORGED_GATE_GRANT_ATTEMPTS: i64 = 3;
pub const MIN_ATTESTATION_REQUIREMENT_AFTER_REOPEN_ATTEMPTS: i64 = 3;
pub const MIN_ATTESTATION_BINDING_CONTROLS: i64 = 11;
pub const MIN_REVOKED_ATTESTOR_EVIDENCE_ATTEMPTS: i64 = 1;
pub const MIN_REVOKED_ATTESTOR_EVIDENCE_CONTROLS: i64 = 1;
pub const MIN_ATTESTOR_QUARANTINE_READMISSION_ATTEMPTS: i64 = 8;
pub const MIN_ATTESTOR_QUARANTINE_RETRAIN_CONTROLS: i64 = 2;

/// The scale of a run: only the pinned one is the experiment. The unit tests and the mutation
/// checker's `<cases> <seed>` form run the same plan (it is small), flagged as no evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// The same plan, not evidence (unit tests, the mutation checker).
    Custom,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Custom => "custom",
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
        ("adversarial_cases", texts(&ADVERSARIAL_CASES)),
        ("selection_digest", Leaf::Text(SELECTION_DIGEST.into())),
        ("dormant_kinds", texts(&DORMANT_KINDS)),
        ("paths", texts(&PATHS)),
        ("authority_kinds", texts(&AUTHORITY_KINDS)),
        ("authority_forms", texts(&AUTHORITY_FORMS)),
        ("dependency_kinds", texts(&DEPENDENCY_KINDS)),
        ("dependency_forms", texts(&DEPENDENCY_FORMS)),
        ("inflight_kinds", texts(&INFLIGHT_KINDS)),
        ("inflight_triggers", texts(&INFLIGHT_TRIGGERS)),
        ("attestation_operations", texts(&ATTESTATION_OPERATIONS)),
        ("min_attempts_per_case", int(MIN_ATTEMPTS_PER_CASE)),
        (
            "min_structural_inflight_race_cases",
            int(MIN_STRUCTURAL_INFLIGHT_RACE_CASES),
        ),
        (
            "min_transitive_authority_cases",
            int(MIN_TRANSITIVE_AUTHORITY_CASES),
        ),
        (
            "min_revoke_while_retired_cases",
            int(MIN_REVOKE_WHILE_RETIRED_CASES),
        ),
        ("min_reactivation_controls", int(MIN_REACTIVATION_CONTROLS)),
        (
            "min_superseded_source_cases",
            int(MIN_SUPERSEDED_SOURCE_CASES),
        ),
        (
            "min_attestation_artifact_mismatch_attempts",
            int(MIN_ATTESTATION_ARTIFACT_MISMATCH_ATTEMPTS),
        ),
        (
            "min_attestation_proposal_mismatch_attempts",
            int(MIN_ATTESTATION_PROPOSAL_MISMATCH_ATTEMPTS),
        ),
        (
            "min_attestation_missing_attempts",
            int(MIN_ATTESTATION_MISSING_ATTEMPTS),
        ),
        (
            "min_attestation_run_reuse_attempts",
            int(MIN_ATTESTATION_RUN_REUSE_ATTEMPTS),
        ),
        (
            "min_attestation_base_model_mismatch_attempts",
            int(MIN_ATTESTATION_BASE_MODEL_MISMATCH_ATTEMPTS),
        ),
        (
            "min_attestation_untrusted_attestor_attempts",
            int(MIN_ATTESTATION_UNTRUSTED_ATTEMPTS),
        ),
        (
            "min_forged_gate_grant_attempts",
            int(MIN_FORGED_GATE_GRANT_ATTEMPTS),
        ),
        (
            "min_attestation_requirement_after_reopen_attempts",
            int(MIN_ATTESTATION_REQUIREMENT_AFTER_REOPEN_ATTEMPTS),
        ),
        (
            "min_attestation_binding_controls_accepted",
            int(MIN_ATTESTATION_BINDING_CONTROLS),
        ),
        (
            "min_revoked_attestor_evidence_attempts",
            int(MIN_REVOKED_ATTESTOR_EVIDENCE_ATTEMPTS),
        ),
        (
            "min_revoked_attestor_evidence_controls_accepted",
            int(MIN_REVOKED_ATTESTOR_EVIDENCE_CONTROLS),
        ),
        (
            "min_attestor_quarantine_readmission_attempts",
            int(MIN_ATTESTOR_QUARANTINE_READMISSION_ATTEMPTS),
        ),
        (
            "min_attestor_quarantine_retrain_controls_accepted",
            int(MIN_ATTESTOR_QUARANTINE_RETRAIN_CONTROLS),
        ),
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
        let changed = CONFIG_TOML.replace(
            "min_transitive_authority_cases = 9\n",
            "min_transitive_authority_cases = 8\n",
        );
        assert_ne!(changed, CONFIG_TOML);
        assert_eq!(drift(&changed), ["min_transitive_authority_cases: drifted"]);
        let added = CONFIG_TOML.replace(
            "[pinned_thresholds]\n",
            "[pinned_thresholds]\nbrand_new_threshold = 1\n",
        );
        assert_eq!(drift(&added), ["brand_new_threshold: no constant"]);
        let removed = CONFIG_TOML.replace("min_transitive_authority_cases = 9\n", "");
        assert_ne!(removed, CONFIG_TOML);
        assert_eq!(
            drift(&removed),
            ["min_transitive_authority_cases: no key in the file"]
        );
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
        assert_eq!(Scale::Custom.name(), "custom");
    }

    #[test]
    fn the_suite_has_distinct_cases() {
        let unique: std::collections::BTreeSet<&str> = ADVERSARIAL_CASES.iter().copied().collect();
        assert_eq!(unique.len(), ADVERSARIAL_CASES.len());
    }
}
