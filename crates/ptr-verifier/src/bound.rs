//! Snapshot-bound verdicts.
//!
//! [`VerificationReport`] has no field for the snapshot a verdict was judged against and cannot
//! get one without breaking every struct-literal construction in the workspace. A
//! [`BoundVerificationReport`] wraps the unchanged report together with a [`SnapshotBinding`]
//! (revision and generation), a [`BoundVerifier`] produces one, and a [`BoundVerifierFabric`]
//! runs several of them and demands that they all judged the same snapshot: a verifier bound to
//! another revision or generation is a hard `snapshot.mismatch` finding (invariants 2 and 3), so
//! a verdict computed against an old snapshot cannot be merged into a verdict for the current
//! one.
//!
//! [`AtSnapshot`] lifts any plain [`Verifier`] to a [`BoundVerifier`] by declaring the snapshot
//! the host verified it against; [`crate::FactVerifier`] and [`crate::RevisionVerifier`]
//! implement [`BoundVerifier`] natively.

use ptr_types::{Generation, Probability, Revision, VerificationLevel};

use crate::codes::finding;
use crate::{NamedVerifier, VerificationReport, VerificationStatus, Verifier, VerifierFabric};

/// The immutable snapshot a verdict refers to.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SnapshotBinding {
    pub revision: Revision,
    pub generation: Generation,
}

impl SnapshotBinding {
    pub fn new(revision: Revision, generation: Generation) -> Self {
        Self {
            revision,
            generation,
        }
    }
}

/// A [`VerificationReport`] and the snapshot it was judged against.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundVerificationReport {
    pub report: VerificationReport,
    pub snapshot: SnapshotBinding,
}

/// A verifier whose verdict names the snapshot it was judged against.
///
/// The method is not called `verify_bound` so it never collides with the inherent
/// `FactVerifier::verify_bound`, which stays as it was.
pub trait BoundVerifier<T> {
    fn verify_at_snapshot(&self, candidate: &T) -> BoundVerificationReport;
}

/// A [`BoundVerifier`] that a [`BoundVerifierFabric`] can hold.
pub trait NamedBoundVerifier<T>: BoundVerifier<T> + Send + Sync {
    fn bound_name(&self) -> &'static str;
}

impl<T, V: BoundVerifier<T> + ?Sized> BoundVerifier<T> for &V {
    fn verify_at_snapshot(&self, candidate: &T) -> BoundVerificationReport {
        (**self).verify_at_snapshot(candidate)
    }
}

impl<T, V: BoundVerifier<T> + ?Sized> BoundVerifier<T> for Box<V> {
    fn verify_at_snapshot(&self, candidate: &T) -> BoundVerificationReport {
        (**self).verify_at_snapshot(candidate)
    }
}

/// A plain verifier together with the snapshot the host declares it verified against. The
/// declaration is a host duty (the host takes the snapshot and verifies within one critical
/// section); the wrapper cannot check it.
#[derive(Clone, Debug)]
pub struct AtSnapshot<V> {
    verifier: V,
    snapshot: SnapshotBinding,
}

impl<V> AtSnapshot<V> {
    pub fn new(verifier: V, snapshot: SnapshotBinding) -> Self {
        Self { verifier, snapshot }
    }

    pub fn snapshot(&self) -> SnapshotBinding {
        self.snapshot
    }
}

impl<T, V: Verifier<T>> BoundVerifier<T> for AtSnapshot<V> {
    fn verify_at_snapshot(&self, candidate: &T) -> BoundVerificationReport {
        BoundVerificationReport {
            report: self.verifier.verify(candidate),
            snapshot: self.snapshot,
        }
    }
}

impl<T, V: NamedVerifier<T>> NamedBoundVerifier<T> for AtSnapshot<V> {
    fn bound_name(&self) -> &'static str {
        self.verifier.name()
    }
}

/// Runs bound verifiers against one required snapshot and combines their reports exactly as
/// [`crate::VerifierFabric`] does (weakest level, worst status, any hard finding forces `Fail`).
///
/// Every verifier must have judged the fabric's snapshot: one bound to another revision or
/// generation adds a hard `snapshot.mismatch` finding naming it and both bindings, so the
/// combined report is `Fail`. The combined report is always bound to the fabric's snapshot. An
/// empty fabric is `Unknown` at `Unverified`.
pub struct BoundVerifierFabric<T> {
    snapshot: SnapshotBinding,
    verifiers: Vec<Box<dyn NamedBoundVerifier<T>>>,
}

impl<T> BoundVerifierFabric<T> {
    /// A fabric requiring that every verifier judged `snapshot`.
    pub fn new(snapshot: SnapshotBinding) -> Self {
        Self {
            snapshot,
            verifiers: Vec::new(),
        }
    }

    pub fn push<V>(&mut self, verifier: V)
    where
        V: NamedBoundVerifier<T> + 'static,
    {
        self.verifiers.push(Box::new(verifier));
    }

    pub fn snapshot(&self) -> SnapshotBinding {
        self.snapshot
    }

    pub fn is_empty(&self) -> bool {
        self.verifiers.is_empty()
    }
}

impl<T> BoundVerifier<T> for BoundVerifierFabric<T> {
    fn verify_at_snapshot(&self, candidate: &T) -> BoundVerificationReport {
        let mut reports = Vec::with_capacity(self.verifiers.len() + 1);
        let mut mismatches = Vec::new();
        for verifier in &self.verifiers {
            let bound = verifier.verify_at_snapshot(candidate);
            if bound.snapshot != self.snapshot {
                mismatches.push(finding(
                    "snapshot.mismatch",
                    format!(
                        "verifier '{}' judged revision {} generation {}, the fabric requires \
                         revision {} generation {}",
                        verifier.bound_name(),
                        bound.snapshot.revision.0,
                        bound.snapshot.generation.0,
                        self.snapshot.revision.0,
                        self.snapshot.generation.0
                    ),
                ));
            }
            reports.push(bound.report);
        }
        if !mismatches.is_empty() {
            reports.push(VerificationReport {
                status: VerificationStatus::Fail,
                level: VerificationLevel::Deterministic,
                score: Probability::new(0.0).expect("zero probability"),
                findings: mismatches,
            });
        }
        // Reuse the plain fabric's aggregation unchanged: each finished report becomes a member.
        let mut combined = VerifierFabric::<()>::default();
        for report in reports {
            combined.push(Finished(report));
        }
        BoundVerificationReport {
            report: combined.verify(&()),
            snapshot: self.snapshot,
        }
    }
}

/// An already computed report, as a member of the aggregating fabric.
struct Finished(VerificationReport);

impl Verifier<()> for Finished {
    fn verify(&self, _: &()) -> VerificationReport {
        self.0.clone()
    }
}

impl NamedVerifier<()> for Finished {
    fn name(&self) -> &'static str {
        "finished"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Finding;

    struct Fixed(VerificationStatus, bool);
    impl Verifier<u32> for Fixed {
        fn verify(&self, _: &u32) -> VerificationReport {
            VerificationReport {
                status: self.0,
                level: VerificationLevel::SampleVerified,
                score: Probability::new(0.5).unwrap(),
                findings: if self.1 {
                    vec![Finding {
                        code: "hard".into(),
                        message: "m".into(),
                        hard: true,
                    }]
                } else {
                    vec![]
                },
            }
        }
    }
    impl NamedVerifier<u32> for Fixed {
        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    fn at(revision: u64, generation: u64) -> SnapshotBinding {
        SnapshotBinding::new(Revision(revision), Generation(generation))
    }

    #[test]
    fn at_snapshot_reports_the_declared_binding_and_the_unchanged_report() {
        let bound = AtSnapshot::new(Fixed(VerificationStatus::Pass, false), at(3, 1));
        let result = bound.verify_at_snapshot(&0);
        assert_eq!(result.snapshot, at(3, 1));
        assert_eq!(
            result.report,
            Fixed(VerificationStatus::Pass, false).verify(&0)
        );
        assert_eq!(bound.snapshot(), at(3, 1));
        assert_eq!(NamedBoundVerifier::<u32>::bound_name(&bound), "fixed");
        // References and boxes are bound verifiers too.
        let by_ref: &AtSnapshot<Fixed> = &bound;
        assert_eq!(
            BoundVerifier::<u32>::verify_at_snapshot(&by_ref, &0).snapshot,
            at(3, 1)
        );
        let boxed: Box<dyn BoundVerifier<u32>> = Box::new(bound);
        assert_eq!(boxed.verify_at_snapshot(&0).snapshot, at(3, 1));
    }

    #[test]
    fn agreeing_verifiers_combine_like_the_plain_fabric() {
        let mut fabric = BoundVerifierFabric::<u32>::new(at(3, 1));
        fabric.push(AtSnapshot::new(
            Fixed(VerificationStatus::Pass, false),
            at(3, 1),
        ));
        fabric.push(AtSnapshot::new(
            Fixed(VerificationStatus::Unknown, false),
            at(3, 1),
        ));
        let result = fabric.verify_at_snapshot(&0);
        assert_eq!(result.snapshot, at(3, 1));
        assert_eq!(result.report.status, VerificationStatus::Unknown);
        assert_eq!(result.report.level, VerificationLevel::SampleVerified);
        assert!(result.report.findings.is_empty());
    }

    #[test]
    fn a_verifier_at_another_snapshot_is_a_hard_mismatch() {
        for other in [at(2, 1), at(3, 0), at(4, 1)] {
            let mut fabric = BoundVerifierFabric::<u32>::new(at(3, 1));
            fabric.push(AtSnapshot::new(
                Fixed(VerificationStatus::Pass, false),
                at(3, 1),
            ));
            fabric.push(AtSnapshot::new(
                Fixed(VerificationStatus::Pass, false),
                other,
            ));
            let result = fabric.verify_at_snapshot(&0);
            assert_eq!(result.report.status, VerificationStatus::Fail, "{other:?}");
            assert_eq!(result.snapshot, at(3, 1));
            let mismatch: Vec<_> = result
                .report
                .findings
                .iter()
                .filter(|f| f.code == "snapshot.mismatch")
                .collect();
            assert_eq!(mismatch.len(), 1);
            assert!(mismatch[0].hard);
            assert!(mismatch[0].message.contains("'fixed'"));
            assert!(mismatch[0]
                .message
                .contains(&format!("revision {}", other.revision.0)));
        }
    }

    #[test]
    fn empty_bound_fabric_is_unknown_and_unverified() {
        let fabric = BoundVerifierFabric::<u32>::new(at(1, 1));
        assert!(fabric.is_empty());
        assert_eq!(fabric.snapshot(), at(1, 1));
        let result = fabric.verify_at_snapshot(&0);
        assert_eq!(result.report.status, VerificationStatus::Unknown);
        assert_eq!(result.report.level, VerificationLevel::Unverified);
        assert_eq!(result.snapshot, at(1, 1));
    }
}
