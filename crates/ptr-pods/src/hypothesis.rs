use ptr_protocol::TypedPayload;
use ptr_types::{Generation, PodId, Probability, ProvenanceRef};
use ptr_verifier::{VerificationReport, VerificationStatus};
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
pub struct PodHypothesis {
    pub branch_id: String,
    pub pod_id: PodId,
    pub generation: Generation,
    pub evidence: Vec<ProvenanceRef>,
    pub output: TypedPayload,
    pub confidence: Probability,
    pub latency: Duration,
    pub verification: VerificationReport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergedPodResult {
    pub output: TypedPayload,
    pub evidence: Vec<ProvenanceRef>,
    pub source_branches: Vec<String>,
    pub confidence: Probability,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HypothesisError {
    Empty,
    Unverified,
    GenerationConflict,
    InvalidConfidence,
}

pub fn merge_hypotheses(
    mut hypotheses: Vec<PodHypothesis>,
) -> Result<MergedPodResult, HypothesisError> {
    if hypotheses.is_empty() {
        return Err(HypothesisError::Empty);
    }
    let generation = hypotheses[0].generation;
    if hypotheses.iter().any(|item| item.generation != generation) {
        return Err(HypothesisError::GenerationConflict);
    }
    hypotheses.retain(|item| item.verification.status == VerificationStatus::Pass);
    if hypotheses.is_empty() {
        return Err(HypothesisError::Unverified);
    }
    hypotheses.sort_by(|left, right| {
        right
            .confidence
            .get()
            .total_cmp(&left.confidence.get())
            .then_with(|| left.latency.cmp(&right.latency))
            .then_with(|| left.branch_id.cmp(&right.branch_id))
    });
    let winner = hypotheses[0].clone();
    let mut evidence = Vec::new();
    for hypothesis in &hypotheses {
        for item in &hypothesis.evidence {
            if !evidence.contains(item) {
                evidence.push(item.clone());
            }
        }
    }
    let confidence = Probability::new(
        hypotheses
            .iter()
            .map(|item| item.confidence.get())
            .sum::<f32>()
            / hypotheses.len() as f32,
    )
    .ok_or(HypothesisError::InvalidConfidence)?;
    Ok(MergedPodResult {
        output: winner.output,
        evidence,
        source_branches: hypotheses.into_iter().map(|item| item.branch_id).collect(),
        confidence,
    })
}
