//! Deterministic revision and generation check.
//!
//! A candidate action or claim names the revision and generation it was computed against
//! (invariants 2 and 3). [`RevisionVerifier`] compares that binding with the current
//! [`SnapshotBinding`] it was built with and emits one hard finding per defect, each with the
//! expected and the actual value in its message:
//!
//! * a binding that is absent: `binding.missing` (one finding per absent half);
//! * a revision or generation older than the current one: `revision.stale` /
//!   `generation.stale`;
//! * a revision or generation newer than the current one, which no honest producer can have
//!   seen: `revision.future` / `generation.future`.
//!
//! Anything but exact equality fails; an exact match passes at `Deterministic` level with
//! score 1. The check works on a [`RevisionClaim`] and directly on an [`ActionIr`] (whose
//! revision and generation are always present).
//!
//! This is a verifier for the fabric. It does not replace the runtime's `authorize_action`,
//! which compares against live state at the moment of the effect; the caller who builds a
//! `RevisionVerifier` is responsible for handing it the current binding (a verifier built from
//! an old binding judges against old state). See [`crate::codes`].

use ptr_types::{ActionIr, Generation, Probability, Revision, VerificationLevel};

use crate::bound::{BoundVerificationReport, BoundVerifier, NamedBoundVerifier, SnapshotBinding};
use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// The revision and generation a candidate says it was computed against. `None` is an absent
/// binding, which is itself a defect.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RevisionClaim {
    pub revision: Option<Revision>,
    pub generation: Option<Generation>,
}

impl RevisionClaim {
    /// A claim carrying both halves.
    pub fn bound(revision: Revision, generation: Generation) -> Self {
        Self {
            revision: Some(revision),
            generation: Some(generation),
        }
    }
}

impl From<&ActionIr> for RevisionClaim {
    fn from(action: &ActionIr) -> Self {
        Self::bound(action.revision, action.generation)
    }
}

/// Compares a claimed binding with the current snapshot. See the module documentation.
#[derive(Clone, Copy, Debug)]
pub struct RevisionVerifier {
    current: SnapshotBinding,
}

impl RevisionVerifier {
    pub fn new(current: SnapshotBinding) -> Self {
        Self { current }
    }

    pub fn current(&self) -> SnapshotBinding {
        self.current
    }

    fn judge(&self, claim: &RevisionClaim) -> VerificationReport {
        let mut findings: Vec<Finding> = Vec::new();
        match claim.revision {
            None => findings.push(finding(
                "binding.missing",
                format!(
                    "no revision bound, expected revision {}",
                    self.current.revision.0
                ),
            )),
            Some(actual) if actual < self.current.revision => findings.push(finding(
                "revision.stale",
                format!(
                    "claimed revision {}, expected {}: the claim predates the current snapshot",
                    actual.0, self.current.revision.0
                ),
            )),
            Some(actual) if actual > self.current.revision => findings.push(finding(
                "revision.future",
                format!(
                    "claimed revision {}, expected {}: the claim is newer than the current snapshot",
                    actual.0, self.current.revision.0
                ),
            )),
            Some(_) => {}
        }
        match claim.generation {
            None => findings.push(finding(
                "binding.missing",
                format!(
                    "no generation bound, expected generation {}",
                    self.current.generation.0
                ),
            )),
            Some(actual) if actual < self.current.generation => findings.push(finding(
                "generation.stale",
                format!(
                    "claimed generation {}, expected {}: the generation is superseded",
                    actual.0, self.current.generation.0
                ),
            )),
            Some(actual) if actual > self.current.generation => findings.push(finding(
                "generation.future",
                format!(
                    "claimed generation {}, expected {}: the generation is newer than the current one",
                    actual.0, self.current.generation.0
                ),
            )),
            Some(_) => {}
        }
        let passed = findings.is_empty();
        VerificationReport {
            status: if passed {
                VerificationStatus::Pass
            } else {
                VerificationStatus::Fail
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(if passed { 1.0 } else { 0.0 }).expect("bounded score"),
            findings,
        }
    }
}

impl Verifier<RevisionClaim> for RevisionVerifier {
    fn verify(&self, claim: &RevisionClaim) -> VerificationReport {
        self.judge(claim)
    }
}

impl Verifier<ActionIr> for RevisionVerifier {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        self.judge(&RevisionClaim::from(action))
    }
}

impl NamedVerifier<RevisionClaim> for RevisionVerifier {
    fn name(&self) -> &'static str {
        "revision"
    }
}

impl NamedVerifier<ActionIr> for RevisionVerifier {
    fn name(&self) -> &'static str {
        "revision"
    }
}

impl BoundVerifier<RevisionClaim> for RevisionVerifier {
    fn verify_at_snapshot(&self, claim: &RevisionClaim) -> BoundVerificationReport {
        BoundVerificationReport {
            report: self.judge(claim),
            snapshot: self.current,
        }
    }
}

impl BoundVerifier<ActionIr> for RevisionVerifier {
    fn verify_at_snapshot(&self, action: &ActionIr) -> BoundVerificationReport {
        BoundVerificationReport {
            report: self.judge(&RevisionClaim::from(action)),
            snapshot: self.current,
        }
    }
}

impl NamedBoundVerifier<RevisionClaim> for RevisionVerifier {
    fn bound_name(&self) -> &'static str {
        "revision"
    }
}

impl NamedBoundVerifier<ActionIr> for RevisionVerifier {
    fn bound_name(&self) -> &'static str {
        "revision"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{CapabilityId, Effect, TypeId};

    fn verifier() -> RevisionVerifier {
        RevisionVerifier::new(SnapshotBinding::new(Revision(5), Generation(2)))
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    fn action(revision: u64, generation: u64) -> ActionIr {
        ActionIr {
            operation: "create".into(),
            target: "note".into(),
            capability: CapabilityId::from("note.create"),
            effect: Effect::Mutation,
            input_type: TypeId::from("note.v1"),
            generation: Generation(generation),
            revision: Revision(revision),
            payload: Vec::new(),
        }
    }

    #[test]
    fn the_current_binding_passes_deterministically() {
        let report = verifier().verify(&RevisionClaim::bound(Revision(5), Generation(2)));
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(report.level, VerificationLevel::Deterministic);
        assert_eq!(report.score.get(), 1.0);
        assert!(report.findings.is_empty());
        assert_eq!(
            verifier().verify(&action(5, 2)).status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn every_defect_has_its_own_hard_code_with_expected_and_actual() {
        let cases = [
            (
                RevisionClaim::bound(Revision(4), Generation(2)),
                "revision.stale",
                "claimed revision 4, expected 5",
            ),
            (
                RevisionClaim::bound(Revision(6), Generation(2)),
                "revision.future",
                "claimed revision 6, expected 5",
            ),
            (
                RevisionClaim::bound(Revision(5), Generation(1)),
                "generation.stale",
                "claimed generation 1, expected 2",
            ),
            (
                RevisionClaim::bound(Revision(5), Generation(3)),
                "generation.future",
                "claimed generation 3, expected 2",
            ),
            (
                RevisionClaim {
                    revision: None,
                    generation: Some(Generation(2)),
                },
                "binding.missing",
                "expected revision 5",
            ),
            (
                RevisionClaim {
                    revision: Some(Revision(5)),
                    generation: None,
                },
                "binding.missing",
                "expected generation 2",
            ),
        ];
        for (claim, code, message) in cases {
            let report = verifier().verify(&claim);
            assert_eq!(report.status, VerificationStatus::Fail, "{code}");
            assert_eq!(report.level, VerificationLevel::Deterministic);
            assert_eq!(report.score.get(), 0.0);
            assert_eq!(codes(&report), [code]);
            assert!(report.findings[0].hard);
            assert!(
                report.findings[0].message.contains(message),
                "{code}: {}",
                report.findings[0].message
            );
        }
    }

    #[test]
    fn revision_and_generation_defects_are_reported_together() {
        let report = verifier().verify(&RevisionClaim::bound(Revision(1), Generation(9)));
        assert_eq!(codes(&report), ["revision.stale", "generation.future"]);
        let report = verifier().verify(&RevisionClaim::default());
        assert_eq!(codes(&report), ["binding.missing", "binding.missing"]);
    }

    #[test]
    fn an_action_is_judged_by_its_own_revision_and_generation() {
        let report = verifier().verify(&action(4, 3));
        assert_eq!(codes(&report), ["revision.stale", "generation.future"]);
        assert_eq!(
            RevisionClaim::from(&action(4, 3)),
            RevisionClaim::bound(Revision(4), Generation(3))
        );
    }

    #[test]
    fn the_bound_report_names_the_current_snapshot() {
        let bound = BoundVerifier::<ActionIr>::verify_at_snapshot(&verifier(), &action(1, 1));
        assert_eq!(
            bound.snapshot,
            SnapshotBinding::new(Revision(5), Generation(2))
        );
        assert_eq!(bound.report.status, VerificationStatus::Fail);
        assert_eq!(verifier().current(), bound.snapshot);
        assert_eq!(NamedVerifier::<ActionIr>::name(&verifier()), "revision");
        assert_eq!(
            NamedBoundVerifier::<RevisionClaim>::bound_name(&verifier()),
            "revision"
        );
    }
}
