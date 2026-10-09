//! Stubs for the unit tests of the error paths that the integration tests cannot reach.

use std::collections::BTreeSet;

use ptr_router::{
    EligibilitySnapshot, FindingCode, NonceSequence, RouterConfig, SamplingKey, SelfModel,
    SelfModelLimits, UniformScorer,
};
use ptr_types::VerificationLevel;
use ptr_types::{Digest, Generation, PodId, Probability, ProjectId, RequestId, TypeId};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use sha2::{Digest as _, Sha256};

use super::{
    ProposeCall, RepairCall, RequestBudget, RolePods, Roster, VcwConfig, VcwLoop, VcwParts,
    VcwPayload, VcwRequest,
};
use crate::expert::{StructurePolicy, StructurePolicyConfig};

/// A payload that is its own finding code: empty passes, anything else fails with that code.
#[derive(Clone, Debug)]
pub(super) struct Text(pub(super) String);

impl VcwPayload for Text {
    fn payload_type(&self) -> TypeId {
        TypeId::from("text")
    }

    fn payload_digest(&self) -> Digest {
        Sha256::digest(format!("t:{}", self.0).as_bytes()).into()
    }
}

pub(super) struct CodeVerifier;

impl Verifier<Text> for CodeVerifier {
    fn verify(&self, candidate: &Text) -> VerificationReport {
        if candidate.0.is_empty() {
            VerificationReport {
                status: VerificationStatus::Pass,
                level: VerificationLevel::Deterministic,
                score: Probability::new(1.0).unwrap(),
                findings: Vec::new(),
            }
        } else {
            VerificationReport {
                status: VerificationStatus::Fail,
                level: VerificationLevel::Deterministic,
                score: Probability::new(0.0).unwrap(),
                findings: vec![Finding {
                    code: candidate.0.clone(),
                    message: String::new(),
                    hard: true,
                }],
            }
        }
    }
}

/// Proposes the fixed text and never repairs.
pub(super) struct Proposer(pub(super) &'static str);

impl RolePods<Text> for Proposer {
    fn propose(&mut self, _: ProposeCall<'_>) -> Option<Text> {
        Some(Text(self.0.to_owned()))
    }

    fn repair(&mut self, _: RepairCall<'_, Text>) -> Option<Text> {
        None
    }
}

pub(super) fn view(pods: &[&str]) -> EligibilitySnapshot {
    let mut view = EligibilitySnapshot::new();
    for pod in pods {
        view.set(PodId::from(*pod), Generation(1));
    }
    view
}

pub(super) fn request(id: &str, calls: u32, rounds: u32) -> VcwRequest {
    VcwRequest {
        request_id: RequestId::from(id),
        project: ProjectId::from("proj"),
        request_digest: Sha256::digest(id.as_bytes()).into(),
        required: VerificationLevel::SampleVerified,
        budget: RequestBudget::new(calls, rounds).unwrap(),
    }
}

pub(super) fn make_loop(
    window_requests: u32,
    repairers: &[(&str, &[&str])],
    structure: StructurePolicyConfig,
) -> VcwLoop {
    let mut config = VcwConfig::new("unit", [1; 32]);
    config.window_requests = window_requests;
    VcwLoop::new(VcwParts {
        config,
        roster: Roster {
            proposers: vec![PodId::from("gen")],
            repairers: repairers
                .iter()
                .map(|(p, codes)| {
                    (
                        PodId::from(*p),
                        codes.iter().map(|c| FindingCode::from(*c)).collect(),
                    )
                })
                .collect(),
        },
        router: RouterConfig::default(),
        scorer: UniformScorer,
        key: SamplingKey::from_bytes([5; 32]),
        nonces: NonceSequence::starting_at([4; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(structure).unwrap(),
    })
    .unwrap()
}

pub(super) fn codes(list: &[&str]) -> BTreeSet<FindingCode> {
    list.iter().map(|c| FindingCode::from(*c)).collect()
}
