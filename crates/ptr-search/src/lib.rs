mod fusion;

use ptr_types::{CapsuleId, Generation};

pub use fusion::{
    check_rank_fusion, convex_score_fusion, retain_live, weighted_rank_fusion, FusedHit,
    FusionError, WeightedList, MAX_TOTAL_WEIGHT,
};

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

/// Unweighted reciprocal rank fusion with `k`, returning `(capsule, score)`.
///
/// Delegates to [`weighted_rank_fusion`], so two generations of one capsule
/// stay two entries and a stale generation never adds to the live one's score;
/// a capsule may therefore appear twice. Prefer [`weighted_rank_fusion`],
/// which also keeps the generation and the contributing backends.
///
/// # Errors
/// Refuses what [`weighted_rank_fusion`] refuses with every weight one: a
/// rank constant that is negative or not finite, a list that names a capsule
/// generation twice, and so many lists that their unit weights sum beyond
/// [`MAX_TOTAL_WEIGHT`].
pub fn reciprocal_rank_fusion(
    lists: &[Vec<SearchHit>],
    k: f32,
) -> Result<Vec<(CapsuleId, f32)>, FusionError> {
    let weighted: Vec<WeightedList<'_>> = lists
        .iter()
        .map(|hits| WeightedList { weight: 1.0, hits })
        .collect();
    Ok(weighted_rank_fusion(&weighted, k)?
        .into_iter()
        .map(|fused| (fused.capsule, fused.score))
        .collect())
}
