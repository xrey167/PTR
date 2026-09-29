//! The two verifiers of the harness's grant, both deterministic.
//!
//! `s003-domain-v1` judges every change, whoever asks for it: no counter of
//! the state a change would publish is negative, and no key it writes, removes
//! or derives lies outside the families the workload uses. `s003-diff-v1`
//! judges host writes only: no counter moves by more than the preregistered
//! jump. It checks nothing on a merge, so it can mask no defect of
//! certification, and it exists to make the state a change applies to
//! observable to a verifier (`before`), which probe P13 relies on.
//!
//! Verifier names may not contain `/`, which a record uses to separate a
//! verifier's name from a finding's code.

use ptr_runtime::ChangeOrigin;
use ptr_runtime::SemanticChange;
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

use super::params;
use super::program::from_semantic;

/// The verifier that judges every change.
pub struct Domain;

/// The verifier that judges host writes.
pub struct Diff;

/// The hard finding of a negative counter, as a record names it.
pub const COUNTER_NEGATIVE: &str = "counter-negative";
/// The hard finding of a key outside the workload's families.
pub const UNKNOWN_FAMILY: &str = "unknown-family";
/// The hard finding of a counter that jumps too far in a host write.
pub const COUNTER_JUMP: &str = "counter-jump";

/// The key prefixes of the workload's semantic state.
const FAMILIES: [&str; 5] = ["item:", "total:", "audit:", "ctr:", "set:"];

fn pass() -> VerificationReport {
    report(Vec::new())
}

fn report(findings: Vec<Finding>) -> VerificationReport {
    let hard = !findings.is_empty();
    VerificationReport {
        status: if hard {
            VerificationStatus::Fail
        } else {
            VerificationStatus::Pass
        },
        level: VerificationLevel::Deterministic,
        score: Probability::new(if hard { 0.0 } else { 1.0 }).expect("a bounded score"),
        findings,
    }
}

fn hard(code: &str, message: String) -> Finding {
    Finding {
        code: code.to_owned(),
        message,
        hard: true,
    }
}

fn in_a_family(key: &str) -> bool {
    FAMILIES.iter().any(|family| key.starts_with(family))
}

impl<'a> Verifier<SemanticChange<'a>> for Domain {
    fn verify(&self, change: &SemanticChange<'a>) -> VerificationReport {
        let mut findings = Vec::new();
        let delta = change.delta();
        let mut stray: Vec<&str> = delta
            .upserts
            .keys()
            .chain(delta.removals.iter())
            .chain(delta.dependencies.keys())
            .map(String::as_str)
            .filter(|key| !in_a_family(key))
            .collect();
        stray.sort_unstable();
        stray.dedup();
        for key in stray {
            findings.push(hard(
                UNKNOWN_FAMILY,
                format!("{key} is in no family of the workload"),
            ));
        }
        let after = change.after();
        let negative: Vec<&str> = after
            .keys()
            .filter(|key| key.starts_with("ctr:"))
            .filter(|key| {
                after
                    .value(key)
                    .and_then(|value| from_semantic(value).as_counter())
                    .is_some_and(|count| count < 0)
            })
            .collect();
        for key in negative {
            findings.push(hard(COUNTER_NEGATIVE, format!("{key} would be negative")));
        }
        report(findings)
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for Domain {
    fn name(&self) -> &'static str {
        params::VERIFIERS[0]
    }
}

impl<'a> Verifier<SemanticChange<'a>> for Diff {
    fn verify(&self, change: &SemanticChange<'a>) -> VerificationReport {
        if !matches!(change.origin(), ChangeOrigin::Host { .. }) {
            return pass();
        }
        let before = change.before();
        let after = change.after();
        let mut findings = Vec::new();
        for key in change
            .delta()
            .upserts
            .keys()
            .filter(|key| key.starts_with("ctr:"))
        {
            let count = |view_value| from_semantic(view_value).as_counter();
            let then = before.value(key).and_then(count);
            let now = after.value(key).and_then(count);
            if let (Some(then), Some(now)) = (then, now) {
                if then.abs_diff(now) > params::HOST_COUNTER_JUMP_MAX as u64 {
                    findings.push(hard(
                        COUNTER_JUMP,
                        format!("{key} moves from {then} to {now}"),
                    ));
                }
            }
        }
        report(findings)
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for Diff {
    fn name(&self) -> &'static str {
        params::VERIFIERS[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::s003::model::{Delta, Val};
    use crate::experiments::s003::program::to_semantic_delta;
    use ptr_config::PtrConfig;
    use ptr_runtime::execution::RequiredVerification;
    use ptr_runtime::{PtrRuntime, RuntimeError, SemanticGrant};
    use ptr_types::PrincipalId;

    fn runtime() -> PtrRuntime {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("a default runtime");
        runtime
            .install_semantic_grant(
                SemanticGrant::new(RequiredVerification::Deterministic)
                    .with_verifier(Domain)
                    .with_verifier(Diff)
                    .allow_host_writes(),
            )
            .expect("the grant installs");
        runtime
    }

    fn write(runtime: &mut PtrRuntime, upserts: &[(&str, Val)]) -> Result<(), RuntimeError> {
        let mut delta = Delta::default();
        for (key, value) in upserts {
            delta.upserts.insert((*key).to_string(), value.clone());
        }
        runtime
            .apply_verified_semantic_delta(
                runtime.revision(),
                to_semantic_delta(&delta),
                &PrincipalId::from("s003-test"),
            )
            .map(|_| ())
    }

    fn rejected(result: Result<(), RuntimeError>) -> Vec<String> {
        match result {
            Err(RuntimeError::SemanticVerificationRejected(refusal)) => refusal.hard_findings,
            other => panic!("expected a verification refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_names_are_ones_a_grant_accepts() {
        assert_eq!(Domain.name(), "s003-domain-v1");
        assert_eq!(Diff.name(), "s003-diff-v1");
        runtime();
    }

    #[test]
    fn a_change_inside_the_families_with_sound_counters_is_admitted() {
        let mut runtime = runtime();
        write(
            &mut runtime,
            &[
                ("item:0:0", Val::text("5")),
                ("ctr:0", Val::counter(50, "s003-test")),
            ],
        )
        .expect("admitted");
    }

    #[test]
    fn a_negative_counter_is_a_hard_finding_of_the_domain_verifier() {
        let mut runtime = runtime();
        let findings = rejected(write(
            &mut runtime,
            &[("ctr:0", Val::counter(-1, "s003-test"))],
        ));
        assert_eq!(
            findings,
            vec!["s003-domain-v1/counter-negative".to_string()]
        );
    }

    #[test]
    fn a_key_outside_the_families_is_a_hard_finding() {
        let mut runtime = runtime();
        let findings = rejected(write(&mut runtime, &[("stray:0", Val::text("1"))]));
        assert_eq!(findings, vec!["s003-domain-v1/unknown-family".to_string()]);
    }

    #[test]
    fn a_host_write_that_jumps_a_counter_too_far_is_a_hard_finding_of_the_diff_verifier() {
        let mut runtime = runtime();
        write(&mut runtime, &[("ctr:0", Val::counter(50, "s003-test"))]).expect("admitted");
        let jump = params::HOST_COUNTER_JUMP_MAX + 1;
        let findings = rejected(write(
            &mut runtime,
            &[("ctr:0", Val::counter(50 + jump, "s003-test"))],
        ));
        assert_eq!(findings, vec!["s003-diff-v1/counter-jump".to_string()]);
        write(
            &mut runtime,
            &[(
                "ctr:0",
                Val::counter(50 + params::HOST_COUNTER_JUMP_MAX, "s003-test"),
            )],
        )
        .expect("the largest allowed jump is admitted");
    }
}
