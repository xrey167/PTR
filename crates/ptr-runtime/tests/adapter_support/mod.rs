//! Builders shared by the adapter promotion and improvement budget tests. No test lives here.
use ptr_lineage::{
    promotion_operation_id, AccuracyMatrix, AdapterId, AdapterRecord, AdapterStatus, BaseModel,
    ForgettingGate, GateSubject, Origin, PromotionProposal, PromotionSuite, SplitLock,
};
use ptr_types::ArtifactId;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const WEIGHTS: &[u8] = b"adapter weights, rank 4";
pub const SPLITS: &[u8] = b"frozen split manifest v1";

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn base_digest() -> [u8; 32] {
    digest(b"base-model@r1")
}

pub fn record(id: &str, weights: &[u8]) -> AdapterRecord {
    AdapterRecord {
        id: AdapterId::from(id),
        domain: "code".into(),
        base: BaseModel {
            id: "base".into(),
            revision: "r1".into(),
        },
        origin: Origin::Trained { parent: None },
        rank: 4,
        target_modules: BTreeSet::new(),
        artifact: ArtifactId::from("artifact-1"),
        artifact_sha256: digest(weights),
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

/// A proposal for adapter `id` whose gate saw `serving` as the baseline.
pub fn proposal_with_baseline(
    id: &str,
    weights: &[u8],
    serving: Option<f64>,
) -> Result<PromotionProposal, ptr_lineage::LineageError> {
    proposal_for("alpha", id, weights, serving)
}

/// A proposal evaluated for `project`.
pub fn proposal_for(
    project: &str,
    id: &str,
    weights: &[u8],
    serving: Option<f64>,
) -> Result<PromotionProposal, ptr_lineage::LineageError> {
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.8]]).expect("matrix");
    let subject = GateSubject::new(
        [1; 32],
        promotion_operation_id(project, id),
        digest(weights),
        base_digest(),
        SplitLock::of_manifest(SPLITS).sha256(),
    );
    let report = gate()
        .evaluate_promotion(
            &AdapterId::from(id),
            &matrix,
            PromotionSuite {
                serving,
                candidate: 0.7,
            },
        )
        .expect("evaluates")
        .bind(subject);
    PromotionProposal::new(&record(id, weights), project, base_digest(), report)
}

pub fn proposal(id: &str) -> PromotionProposal {
    proposal_with_baseline(id, WEIGHTS, Some(0.7)).expect("a passing proposal")
}
