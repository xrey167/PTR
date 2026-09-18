use ptr_types::{CapsuleId, Generation};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceStage {
    SearchCandidate,
    PossibleEvidence,
    Observed,
    VerifiedKnown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidencePromotionError {
    WrongStage,
    StaleGeneration,
    SourceNotResolved,
    VerificationFailed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub capsule: CapsuleId,
    pub generation: Generation,
    pub score: f32,
    pub backend: String,
    stage: EvidenceStage,
}

impl SearchHit {
    pub fn new(
        capsule: CapsuleId,
        generation: Generation,
        score: f32,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            capsule,
            generation,
            score,
            backend: backend.into(),
            stage: EvidenceStage::SearchCandidate,
        }
    }

    pub fn stage(&self) -> EvidenceStage {
        self.stage
    }

    pub fn mark_possible(&mut self) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::SearchCandidate {
            return Err(EvidencePromotionError::WrongStage);
        }
        self.stage = EvidenceStage::PossibleEvidence;
        Ok(())
    }

    pub fn observe(
        &mut self,
        current_generation: Generation,
        exact_source_resolved: bool,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::PossibleEvidence {
            return Err(EvidencePromotionError::WrongStage);
        }
        if self.generation != current_generation {
            return Err(EvidencePromotionError::StaleGeneration);
        }
        if !exact_source_resolved {
            return Err(EvidencePromotionError::SourceNotResolved);
        }
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    pub fn promote_verified(
        &mut self,
        verification_passed: bool,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage);
        }
        if !verification_passed {
            return Err(EvidencePromotionError::VerificationFailed);
        }
        self.stage = EvidenceStage::VerifiedKnown;
        Ok(())
    }
}

pub trait SearchIndex {
    fn search(&self, query: &str, limit: usize) -> Vec<SearchHit>;
}

pub fn reciprocal_rank_fusion(lists: &[Vec<SearchHit>], k: f32) -> Vec<(CapsuleId, f32)> {
    let mut scores: BTreeMap<CapsuleId, f32> = BTreeMap::new();
    for list in lists {
        for (rank, hit) in list.iter().enumerate() {
            *scores.entry(hit.capsule.clone()).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}
