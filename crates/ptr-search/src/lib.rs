mod fusion;

use std::fmt;

use ptr_types::{CapsuleId, Generation, Validity};

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

/// Why a hit was not moved to the next evidence stage. A refused step leaves
/// the hit at the stage it had.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidencePromotionError {
    /// The step does not follow from the hit's current stage.
    WrongStage,
    /// The lifecycle authority answers that the hit's generation was
    /// superseded by a later one.
    StaleGeneration,
    /// The lifecycle authority answers that the hit's generation was revoked,
    /// although a revocation leaves it the capsule's live generation.
    RevokedGeneration,
    /// The lifecycle authority answers that the hit's generation is disputed.
    DisputedGeneration,
    /// The lifecycle authority knows no such generation: the capsule is
    /// unknown to it or the generation is ahead of the one it holds.
    UnknownGeneration,
    /// The exact source of the hit was not resolved.
    SourceNotResolved,
    /// Verification of the observed hit failed.
    VerificationFailed,
}

impl EvidencePromotionError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::WrongStage => "PTR_SEARCH_WRONG_STAGE",
            Self::StaleGeneration => "PTR_SEARCH_STALE_GENERATION",
            Self::RevokedGeneration => "PTR_SEARCH_REVOKED_GENERATION",
            Self::DisputedGeneration => "PTR_SEARCH_DISPUTED_GENERATION",
            Self::UnknownGeneration => "PTR_SEARCH_UNKNOWN_GENERATION",
            Self::SourceNotResolved => "PTR_SEARCH_SOURCE_NOT_RESOLVED",
            Self::VerificationFailed => "PTR_SEARCH_VERIFICATION_FAILED",
        }
    }
}

impl fmt::Display for EvidencePromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::WrongStage => "the step does not follow from the hit's evidence stage",
            Self::StaleGeneration => "the hit's generation was superseded",
            Self::RevokedGeneration => "the hit's generation was revoked",
            Self::DisputedGeneration => "the hit's generation is disputed",
            Self::UnknownGeneration => "the lifecycle authority knows no such generation",
            Self::SourceNotResolved => "the hit's exact source was not resolved",
            Self::VerificationFailed => "verification of the observed hit failed",
        })
    }
}

impl std::error::Error for EvidencePromotionError {}

/// A retrieved capsule generation and the evidence stage it has reached.
///
/// The capsule and generation are fixed when the hit is made
/// ([`SearchHit::new`]) and read through [`SearchHit::capsule`] and
/// [`SearchHit::generation`]; like the stage, they cannot be assigned. So the
/// generation [`SearchHit::observe`] asked the lifecycle about is the one a
/// later [`SearchHit::promote_verified`] promotes, and a hit observed live
/// cannot be turned into a hit of a revoked or superseded generation:
///
/// ```compile_fail
/// fn swap(mut hit: ptr_search::SearchHit) -> ptr_search::SearchHit {
///     hit.capsule = ptr_types::CapsuleId::from("revoked");
///     hit
/// }
/// ```
///
/// ```compile_fail
/// fn swap(mut hit: ptr_search::SearchHit) -> ptr_search::SearchHit {
///     hit.generation = ptr_types::Generation(1);
///     hit
/// }
/// ```
///
/// `score` and `backend` describe the retrieval, not the evidence, and stay
/// public.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    capsule: CapsuleId,
    generation: Generation,
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

    /// The capsule this hit retrieved, as [`SearchHit::new`] was given it.
    pub fn capsule(&self) -> &CapsuleId {
        &self.capsule
    }

    /// The generation of the capsule this hit retrieved, as
    /// [`SearchHit::new`] was given it.
    pub fn generation(&self) -> Generation {
        self.generation
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

    /// Move a possible-evidence hit to `Observed`: the lifecycle authority
    /// holds its generation live and its exact source was resolved.
    ///
    /// `validity` is asked once about this hit's capsule and generation,
    /// which no caller can change afterwards, and answers as
    /// `PtrRuntime::generation_validity` does (the question [`retain_live`]
    /// asks): `None` for an unknown capsule or a generation ahead of it. Only
    /// `Some(Validity::Live)` passes. Asking for validity rather than taking
    /// the capsule's live generation is what refuses a revoked hit: a
    /// revocation leaves the live generation in place and adds a tombstone,
    /// so a revoked generation still equals the live one. The answer is only
    /// as current as the view `validity` reads; a revocation committed after
    /// it is not seen.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `PossibleEvidence` (then `validity` is
    /// not called); `StaleGeneration` for `Some(Validity::Superseded)`,
    /// `RevokedGeneration` for `Some(Validity::Revoked)`, `DisputedGeneration`
    /// for `Some(Validity::Disputed)` and `UnknownGeneration` for `None`; and
    /// `SourceNotResolved` when the exact source was not resolved.
    pub fn observe<F>(
        &mut self,
        validity: F,
        exact_source_resolved: bool,
    ) -> Result<(), EvidencePromotionError>
    where
        F: FnOnce(&CapsuleId, Generation) -> Option<Validity>,
    {
        if self.stage != EvidenceStage::PossibleEvidence {
            return Err(EvidencePromotionError::WrongStage);
        }
        match validity(&self.capsule, self.generation) {
            Some(Validity::Live) => {}
            Some(Validity::Superseded) => return Err(EvidencePromotionError::StaleGeneration),
            Some(Validity::Revoked) => return Err(EvidencePromotionError::RevokedGeneration),
            Some(Validity::Disputed) => return Err(EvidencePromotionError::DisputedGeneration),
            None => return Err(EvidencePromotionError::UnknownGeneration),
        }
        if !exact_source_resolved {
            return Err(EvidencePromotionError::SourceNotResolved);
        }
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    /// Move an observed hit to `VerifiedKnown` once its verification passed.
    ///
    /// The generation promoted is the one [`SearchHit::observe`] asked the
    /// lifecycle about, since a hit's capsule and generation cannot change.
    /// The lifecycle is not asked again: a revocation or supersession
    /// committed after `observe` answered is not seen here.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `Observed`, and `VerificationFailed`
    /// when verification did not pass.
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

/// Unweighted reciprocal rank fusion with `k`, returning
/// `(capsule, generation, score)`.
///
/// Delegates to [`weighted_rank_fusion`], so two generations of one capsule
/// stay two entries and a stale generation never adds to the live one's score;
/// a capsule may therefore appear twice, each time with the generation its
/// score belongs to. Prefer [`weighted_rank_fusion`], which also keeps the
/// contributing backends.
///
/// # Errors
/// Refuses what [`weighted_rank_fusion`] refuses with every weight one: a
/// rank constant that is negative or not finite, a list that names a capsule
/// generation twice, and so many lists that their unit weights sum beyond
/// [`MAX_TOTAL_WEIGHT`].
pub fn reciprocal_rank_fusion(
    lists: &[Vec<SearchHit>],
    k: f32,
) -> Result<Vec<(CapsuleId, Generation, f32)>, FusionError> {
    let weighted: Vec<WeightedList<'_>> = lists
        .iter()
        .map(|hits| WeightedList { weight: 1.0, hits })
        .collect();
    Ok(weighted_rank_fusion(&weighted, k)?
        .into_iter()
        .map(|fused| (fused.capsule, fused.generation, fused.score))
        .collect())
}
