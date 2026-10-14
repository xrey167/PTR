use sha2::{Digest, Sha256};

use crate::error::LineageError;
use crate::forgetting::{BoundGateReport, GateViolation};
use crate::lineage::{AdapterId, AdapterRecord};

/// The digest of the frozen evaluation splits a promotion was measured on.
///
/// A promotion is only comparable with the serving model if both were scored on the same
/// splits, so the digest of those splits travels with the gate report: the trusted evaluator
/// puts it in the report's `eval_manifest_digest`, and the runtime refuses a promotion whose
/// lock is another one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SplitLock {
    sha256: [u8; 32],
}

impl SplitLock {
    /// A lock over the digest an evaluator computed for its frozen splits.
    pub fn from_digest(sha256: [u8; 32]) -> Self {
        Self { sha256 }
    }

    /// The lock of the exact bytes of a canonical split manifest.
    pub fn of_manifest(manifest: &[u8]) -> Self {
        Self {
            sha256: Sha256::digest(manifest).into(),
        }
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }
}

const PROMOTION_OPERATION_DOMAIN: &[u8] = b"ptr.lineage.promotion-operation.v1";

/// The operation id a trusted evaluator puts in the [`crate::GateSubject`] of a promotion of
/// `adapter` in `project`. It binds the project into the digested subject: a report evaluated
/// for one project's adapter has another operation id than the same adapter name in another
/// project, so [`PromotionProposal::new`] refuses it there.
pub fn promotion_operation_id(project: &str, adapter: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PROMOTION_OPERATION_DOMAIN);
    for part in [project, adapter] {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hasher.finalize().into()
}

/// A request to let one registered adapter serve, built from the adapter's record and the
/// bound report of a passing gate. This crate stops here: committing the proposal to the
/// ledger is `ptr-runtime`'s `promote_adapter`, which also recomputes the artifact digest from
/// the weights and checks the split lock.
///
/// The fields are private so a proposal exists only if [`PromotionProposal::new`] accepted
/// it.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionProposal {
    project: String,
    adapter: AdapterId,
    base_digest: [u8; 32],
    artifact_sha256: [u8; 32],
    report: BoundGateReport,
}

impl PromotionProposal {
    /// Propose `record` for serving in `project` on the strength of `report`.
    ///
    /// `base_digest` is the digest of the base model the record names, as the caller computed
    /// it; the report's subject must name the same one, the record's artifact digest and the
    /// record's adapter, and its operation id must be
    /// [`promotion_operation_id`]`(project, adapter)`. The report must have been produced by
    /// [`crate::ForgettingGate::evaluate_promotion`] in this process.
    ///
    /// # Errors
    /// [`LineageError::MissingBaseline`] when the report was evaluated without a serving
    /// baseline, [`LineageError::GateFailed`] for any other failing report,
    /// [`LineageError::SubjectMismatch`] when the report's adapter, artifact digest or base
    /// model digest differs from the proposed one, and [`LineageError::InvalidParameter`] for
    /// an all-zero digest.
    pub fn new(
        record: &AdapterRecord,
        project: &str,
        base_digest: [u8; 32],
        report: BoundGateReport,
    ) -> Result<Self, LineageError> {
        if !report.promotion_evaluated() {
            return Err(LineageError::BaselineNotChecked {
                id: record.id.to_string(),
            });
        }
        if report
            .report()
            .violations()
            .iter()
            .any(|violation| matches!(violation, GateViolation::MissingBaseline))
        {
            return Err(LineageError::MissingBaseline {
                id: record.id.to_string(),
            });
        }
        if !report.passed() {
            return Err(LineageError::GateFailed {
                id: record.id.to_string(),
            });
        }
        if report.report().adapter() != &record.id {
            return Err(LineageError::SubjectMismatch { field: "adapter" });
        }
        let subject = report.subject();
        if subject.artifact_digest != record.artifact_sha256 {
            return Err(LineageError::SubjectMismatch {
                field: "artifact digest",
            });
        }
        if subject.operation_id != promotion_operation_id(project, &record.id.0) {
            return Err(LineageError::SubjectMismatch {
                field: "operation id (project)",
            });
        }
        if subject.base_model_digest != base_digest {
            return Err(LineageError::SubjectMismatch {
                field: "base model digest",
            });
        }
        if base_digest == [0; 32] || record.artifact_sha256 == [0; 32] {
            return Err(LineageError::InvalidParameter {
                field: "digest",
                message: "must not be all zero",
            });
        }
        Ok(Self {
            project: project.to_owned(),
            adapter: record.id.clone(),
            base_digest,
            artifact_sha256: record.artifact_sha256,
            report,
        })
    }

    /// The project the promotion was evaluated for.
    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn adapter(&self) -> &AdapterId {
        &self.adapter
    }

    pub fn base_digest(&self) -> [u8; 32] {
        self.base_digest
    }

    /// The weights digest the trainer recorded; the runtime recomputes it from the weights.
    pub fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }

    pub fn report(&self) -> &BoundGateReport {
        &self.report
    }

    /// The digest of the bound gate report, the value the ledger event carries.
    pub fn gate_report_digest(&self) -> [u8; 32] {
        self.report.digest()
    }

    /// The split lock the gate was evaluated under.
    pub fn split_lock(&self) -> SplitLock {
        SplitLock::from_digest(self.report.subject().eval_manifest_digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forgetting::{AccuracyMatrix, ForgettingGate, GateSubject, PromotionSuite};
    use crate::lineage::{AdapterStatus, BaseModel, Origin};
    use ptr_types::ArtifactId;
    use std::collections::BTreeSet;

    fn record(artifact: [u8; 32]) -> AdapterRecord {
        AdapterRecord {
            id: AdapterId::from("a1"),
            domain: "code".into(),
            base: BaseModel {
                id: "base".into(),
                revision: "r1".into(),
            },
            origin: Origin::Trained { parent: None },
            rank: 4,
            target_modules: BTreeSet::new(),
            artifact: ArtifactId::from("art"),
            artifact_sha256: artifact,
            data_fingerprint: [9; 32],
            data_manifest: BTreeSet::new(),
            status: AdapterStatus::Candidate,
        }
    }

    fn gate() -> ForgettingGate {
        ForgettingGate {
            max_average_forgetting: 0.1,
            max_task_forgetting: 0.2,
            min_backward_transfer: -0.05,
            max_public_regression: 0.02,
        }
    }

    fn matrix() -> AccuracyMatrix {
        AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.8]]).expect("matrix")
    }

    fn bound(serving: Option<f64>, subject: GateSubject) -> BoundGateReport {
        gate()
            .evaluate_promotion(
                &AdapterId::from("a1"),
                &matrix(),
                PromotionSuite {
                    serving,
                    candidate: 0.7,
                },
            )
            .expect("evaluates")
            .bind(subject)
    }

    fn subject(artifact: [u8; 32], base: [u8; 32]) -> GateSubject {
        GateSubject::new(
            [1; 32],
            promotion_operation_id("proj", "a1"),
            artifact,
            base,
            [5; 32],
        )
    }

    #[test]
    fn a_missing_serving_baseline_is_red_and_never_a_zero_baseline() {
        let report = bound(None, subject([3; 32], [4; 32]));
        assert!(!report.passed());
        assert!(report
            .report()
            .violations()
            .contains(&GateViolation::MissingBaseline));
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], report),
            Err(LineageError::MissingBaseline { id: "a1".into() })
        );
    }

    #[test]
    fn a_passing_report_with_a_baseline_yields_a_proposal_bound_to_its_subject() {
        let report = bound(Some(0.7), subject([3; 32], [4; 32]));
        let proposal = PromotionProposal::new(&record([3; 32]), "proj", [4; 32], report.clone())
            .expect("proposal");
        assert_eq!(proposal.gate_report_digest(), report.digest());
        assert_eq!(proposal.split_lock(), SplitLock::from_digest([5; 32]));
        assert_eq!(proposal.artifact_sha256(), [3; 32]);
    }

    #[test]
    fn a_subject_for_another_artifact_or_base_is_refused() {
        let other_artifact = bound(Some(0.7), subject([8; 32], [4; 32]));
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], other_artifact),
            Err(LineageError::SubjectMismatch {
                field: "artifact digest"
            })
        );
        let other_base = bound(Some(0.7), subject([3; 32], [8; 32]));
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], other_base),
            Err(LineageError::SubjectMismatch {
                field: "base model digest"
            })
        );
    }

    #[test]
    fn a_report_for_another_adapter_is_refused() {
        let mut other = record([3; 32]);
        other.id = AdapterId::from("a2");
        let report = bound(Some(0.7), subject([3; 32], [4; 32]));
        assert_eq!(
            PromotionProposal::new(&other, "proj", [4; 32], report),
            Err(LineageError::SubjectMismatch { field: "adapter" })
        );
    }

    #[test]
    fn a_regressing_candidate_fails_the_gate_not_the_baseline_rule() {
        let report = gate()
            .evaluate_promotion(
                &AdapterId::from("a1"),
                &matrix(),
                PromotionSuite {
                    serving: Some(0.9),
                    candidate: 0.5,
                },
            )
            .expect("evaluates")
            .bind(subject([3; 32], [4; 32]));
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], report),
            Err(LineageError::GateFailed { id: "a1".into() })
        );
    }

    #[test]
    fn a_missing_baseline_round_trips_through_the_canonical_report_codec() {
        let report = bound(None, subject([3; 32], [4; 32]));
        let decoded =
            BoundGateReport::decode_canonical(&report.encode_canonical()).expect("decode");
        assert_eq!(decoded.encode_canonical(), report.encode_canonical());
        assert_eq!(decoded.digest(), report.digest());
    }

    #[test]
    fn a_nonfinite_baseline_is_refused_before_evaluation() {
        for serving in [f64::NAN, f64::INFINITY] {
            assert!(gate()
                .evaluate_promotion(
                    &AdapterId::from("a1"),
                    &matrix(),
                    PromotionSuite {
                        serving: Some(serving),
                        candidate: 0.7,
                    },
                )
                .is_err());
        }
    }

    #[test]
    fn a_passing_report_from_the_plain_evaluate_path_cannot_justify_a_promotion() {
        // The old path takes a serving score that cannot be absent; a placeholder passes.
        let report = gate()
            .evaluate(
                &AdapterId::from("a1"),
                &matrix(),
                crate::PublicSuite {
                    serving: 0.0,
                    candidate: 0.7,
                },
            )
            .expect("evaluates")
            .bind(subject([3; 32], [4; 32]));
        assert!(report.passed());
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], report),
            Err(LineageError::BaselineNotChecked { id: "a1".into() })
        );
    }

    #[test]
    fn a_decoded_report_cannot_justify_a_promotion() {
        let report = bound(Some(0.7), subject([3; 32], [4; 32]));
        let decoded =
            BoundGateReport::decode_canonical(&report.encode_canonical()).expect("decode");
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "proj", [4; 32], decoded),
            Err(LineageError::BaselineNotChecked { id: "a1".into() })
        );
    }

    #[test]
    fn a_report_evaluated_for_another_project_is_refused() {
        let report = bound(Some(0.7), subject([3; 32], [4; 32]));
        assert_eq!(
            PromotionProposal::new(&record([3; 32]), "other-project", [4; 32], report),
            Err(LineageError::SubjectMismatch {
                field: "operation id (project)"
            })
        );
        assert_ne!(
            promotion_operation_id("a", "bc"),
            promotion_operation_id("ab", "c"),
            "the parts are length-prefixed"
        );
    }
}
