//! The metric list of R013 and the two judgements of a run: the coverage prerequisites (a run
//! that misses one is vacuous) and the falsifiers (a run that trips one falsifies R013).
//!
//! The rules are strings in the grammar of `criteria.toml` (see `layered_kv::emit`): every
//! "< N" prerequisite of the manifest becomes `>= N`, every "> 0" violation `<= 0`. The harness
//! judges its own output with them, and a test pins `criteria.toml` to exactly these lists, in
//! order. Every metric is a JSON integer or an object of integers (a grouped metric); every
//! coverage counter is read from a case the run observed, never from configuration.

use super::params;
use crate::experiments::layered_kv::emit::{MetricList, Tally};

/// The members of the grouped metric `adversarial_attempts_per_case`, in the suite's order.
pub fn attempt_members() -> Vec<&'static str> {
    params::ADVERSARIAL_CASES.to_vec()
}

/// The metrics of the manifest, in its order (a test compares them with the file).
pub const METRIC_NAMES: [&str; 41] = [
    // Coverage, observed.
    "adversarial_attempts_per_case",
    "structural_inflight_race_cases",
    "transitive_authority_revocation_cases",
    "revoke_while_retired_cases",
    "reactivation_controls_accepted",
    "superseded_source_revocation_cases",
    "attestation_artifact_mismatch_attempts",
    "attestation_proposal_mismatch_attempts",
    "attestation_missing_attempts",
    "attestation_run_reuse_attempts",
    "attestation_base_model_mismatch_attempts",
    "attestation_untrusted_attestor_attempts",
    "forged_gate_grant_attempts",
    "attestation_requirement_after_reopen_attempts",
    "attestation_binding_controls_accepted",
    "revoked_attestor_evidence_attempts",
    "revoked_attestor_evidence_controls_accepted",
    "attestor_quarantine_readmission_attempts",
    "attestor_quarantine_retrain_controls_accepted",
    // Falsifiers.
    "structural_inflight_outputs_after_invalidation",
    "routable_descendants_after_source_authority_revocation",
    "reactivations_from_compromised_seed_admitted",
    "compromised_reactivations_routable",
    "descendants_routable_after_superseded_source_revocation",
    "mismatched_attestation_admissions",
    "unattested_admissions",
    "untrusted_attestor_admissions",
    "self_trusted_attestor_admissions",
    "revoked_attestor_evidence_admissions",
    "attestor_quarantine_readmissions_admitted",
    "inflight_fallback_incomplete",
    "inflight_fallback_output_mismatches",
    "fence_set_mismatches",
    "bystanders_closed",
    "registry_replay_mismatches",
    "procedure_mismatches",
    "wrong_typed_refusals",
    "skipped_cells",
    "product_panics",
    // Descriptive: no rule reads them.
    "reactivation_attempts",
    "attestation_controls_by_operation",
];

/// The counters whose sum is `hard_failures` and that `tests/mutations.toml` names.
pub const HARD_COUNTERS: [&str; 20] = [
    "structural_inflight_outputs_after_invalidation",
    "routable_descendants_after_source_authority_revocation",
    "reactivations_from_compromised_seed_admitted",
    "compromised_reactivations_routable",
    "descendants_routable_after_superseded_source_revocation",
    "mismatched_attestation_admissions",
    "unattested_admissions",
    "untrusted_attestor_admissions",
    "self_trusted_attestor_admissions",
    "revoked_attestor_evidence_admissions",
    "attestor_quarantine_readmissions_admitted",
    "inflight_fallback_incomplete",
    "inflight_fallback_output_mismatches",
    "fence_set_mismatches",
    "bystanders_closed",
    "registry_replay_mismatches",
    "procedure_mismatches",
    "wrong_typed_refusals",
    "skipped_cells",
    "product_panics",
];

/// The grouped metrics (printed as objects of integers).
pub const GROUPED: [&str; 7] = [
    "adversarial_attempts_per_case",
    "structural_inflight_race_cases",
    "transitive_authority_revocation_cases",
    "revoke_while_retired_cases",
    "reactivation_controls_accepted",
    "superseded_source_revocation_cases",
    "attestation_controls_by_operation",
];

pub const LIST: MetricList = MetricList {
    names: &METRIC_NAMES,
    grouped: &GROUPED,
};

/// The sum of the hard counters of a tally: `hard_failures`.
pub fn hard_failures(tally: &Tally) -> i64 {
    HARD_COUNTERS.iter().map(|name| tally.get(name)).sum()
}

/// The members of each grouped coverage metric and the threshold each must reach.
fn grouped_coverage() -> Vec<(&'static str, Vec<&'static str>, i64)> {
    vec![
        (
            "structural_inflight_race_cases",
            params::INFLIGHT_KINDS.to_vec(),
            params::MIN_STRUCTURAL_INFLIGHT_RACE_CASES,
        ),
        (
            "transitive_authority_revocation_cases",
            params::AUTHORITY_KINDS.to_vec(),
            params::MIN_TRANSITIVE_AUTHORITY_CASES,
        ),
        (
            "revoke_while_retired_cases",
            params::DORMANT_KINDS.to_vec(),
            params::MIN_REVOKE_WHILE_RETIRED_CASES,
        ),
        (
            "reactivation_controls_accepted",
            params::DORMANT_KINDS.to_vec(),
            params::MIN_REACTIVATION_CONTROLS,
        ),
        (
            "superseded_source_revocation_cases",
            vec!["input", "attestor", "gate_config", "evaluator"],
            params::MIN_SUPERSEDED_SOURCE_CASES,
        ),
    ]
}

/// The coverage prerequisites: each unmet one makes the run vacuous.
pub fn coverage() -> Vec<String> {
    let mut rules: Vec<String> = params::ADVERSARIAL_CASES
        .iter()
        .map(|case| {
            format!(
                "adversarial_attempts_per_case.{case} >= {}",
                params::MIN_ATTEMPTS_PER_CASE
            )
        })
        .collect();
    for (metric, members, minimum) in grouped_coverage() {
        rules.extend(
            members
                .into_iter()
                .map(|member| format!("{metric}.{member} >= {minimum}")),
        );
    }
    rules.extend([
        format!(
            "attestation_artifact_mismatch_attempts >= {}",
            params::MIN_ATTESTATION_ARTIFACT_MISMATCH_ATTEMPTS
        ),
        format!(
            "attestation_proposal_mismatch_attempts >= {}",
            params::MIN_ATTESTATION_PROPOSAL_MISMATCH_ATTEMPTS
        ),
        format!(
            "attestation_missing_attempts >= {}",
            params::MIN_ATTESTATION_MISSING_ATTEMPTS
        ),
        format!(
            "attestation_run_reuse_attempts >= {}",
            params::MIN_ATTESTATION_RUN_REUSE_ATTEMPTS
        ),
        format!(
            "attestation_base_model_mismatch_attempts >= {}",
            params::MIN_ATTESTATION_BASE_MODEL_MISMATCH_ATTEMPTS
        ),
        format!(
            "attestation_untrusted_attestor_attempts >= {}",
            params::MIN_ATTESTATION_UNTRUSTED_ATTEMPTS
        ),
        format!(
            "forged_gate_grant_attempts >= {}",
            params::MIN_FORGED_GATE_GRANT_ATTEMPTS
        ),
        format!(
            "attestation_requirement_after_reopen_attempts >= {}",
            params::MIN_ATTESTATION_REQUIREMENT_AFTER_REOPEN_ATTEMPTS
        ),
        format!(
            "attestation_binding_controls_accepted >= {}",
            params::MIN_ATTESTATION_BINDING_CONTROLS
        ),
        format!(
            "revoked_attestor_evidence_attempts >= {}",
            params::MIN_REVOKED_ATTESTOR_EVIDENCE_ATTEMPTS
        ),
        format!(
            "revoked_attestor_evidence_controls_accepted >= {}",
            params::MIN_REVOKED_ATTESTOR_EVIDENCE_CONTROLS
        ),
        format!(
            "attestor_quarantine_readmission_attempts >= {}",
            params::MIN_ATTESTOR_QUARANTINE_READMISSION_ATTEMPTS
        ),
        format!(
            "attestor_quarantine_retrain_controls_accepted >= {}",
            params::MIN_ATTESTOR_QUARANTINE_RETRAIN_CONTROLS
        ),
    ]);
    rules
}

/// The falsifiers: each tripped one falsifies R013 on this seed.
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
        for name in GROUPED {
            assert!(METRIC_NAMES.contains(&name), "{name}");
        }
        assert_eq!(falsifiers().len(), HARD_COUNTERS.len());
    }

    #[test]
    fn every_grouped_coverage_metric_is_grouped() {
        for (metric, _, _) in grouped_coverage() {
            assert!(GROUPED.contains(&metric), "{metric}");
        }
    }
}
