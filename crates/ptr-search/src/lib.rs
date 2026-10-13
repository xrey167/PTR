mod filter;
mod fusion;
mod resolver;
mod source;

use std::fmt;

use ptr_types::{CapsuleId, Generation, KnowledgeValidity, Timestamp, Validity};

pub use filter::{
    FilterError, SearchAttributes, SearchFilter, MAX_FILTER_PRINCIPALS, MAX_FILTER_TAGS,
    MAX_FILTER_TEXT_BYTES,
};
pub use resolver::{
    AliasError, AliasTable, ResolveError, Resolved, MAX_ALIASES, MAX_ALIAS_WORDS, MAX_QUERY_WORDS,
};

pub use fusion::{
    check_rank_fusion, convex_score_fusion, retain_live, weighted_rank_fusion, FusedHit,
    FusionError, WeightedList, MAX_TOTAL_WEIGHT,
};
pub use source::{
    verify_claim, Claim, ClaimError, ClaimProof, ExactSourceResolver, RawEventSource,
    ResolvedSource, SourceCitation, SourceError, SourceOwners, SourceOwnership,
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
    /// The claim proof is about another source than the one this hit was observed with.
    ClaimProofMismatch,
    /// The resolved source is a raw event of another capsule or another generation than
    /// the one the hit names, however intact the event and however live the hit's own
    /// generation.
    SourceNotOwnedByHit,
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
            Self::ClaimProofMismatch => "PTR_SEARCH_CLAIM_PROOF_MISMATCH",
            Self::SourceNotOwnedByHit => "PTR_SEARCH_SOURCE_NOT_OWNED_BY_HIT",
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
            Self::ClaimProofMismatch => {
                "the claim proof is about another source than the hit was observed with"
            }
            Self::SourceNotOwnedByHit => {
                "the resolved source does not belong to the capsule generation the hit names"
            }
        })
    }
}

impl std::error::Error for EvidencePromotionError {}

/// Why the validity of a hit, as the authority holds it, did not hold at the instant the hit
/// was relied on. Kept apart from [`EvidencePromotionError`] so the stage rules keep their
/// vocabulary: these are about time, not about the order of steps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidityRefusal {
    /// The validity has not started.
    NotYetValid,
    /// The validity has ended; the end itself is outside it.
    Expired,
    /// The authority holds no validity for the hit's capsule generation, or the hit was
    /// observed without one and so cannot be checked again.
    Unknown,
}

impl ValidityRefusal {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotYetValid => "PTR_SEARCH_NOT_YET_VALID",
            Self::Expired => "PTR_SEARCH_EXPIRED",
            Self::Unknown => "PTR_SEARCH_VALIDITY_UNKNOWN",
        }
    }
}

impl fmt::Display for ValidityRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotYetValid => "the hit's validity has not started",
            Self::Expired => "the hit's validity has ended",
            Self::Unknown => "the authority holds no validity for the hit",
        })
    }
}

impl std::error::Error for ValidityRefusal {}

/// Why a time-checked promotion step ([`SearchHit::observe_at`] and its siblings) was
/// refused: for the reasons any step is ([`EvidencePromotionError`]), or because the
/// authority's validity did not hold ([`ValidityRefusal`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimedPromotionError {
    Promotion(EvidencePromotionError),
    Validity(ValidityRefusal),
}

impl TimedPromotionError {
    /// The stable code of the underlying refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Promotion(error) => error.code(),
            Self::Validity(refusal) => refusal.code(),
        }
    }
}

impl From<EvidencePromotionError> for TimedPromotionError {
    fn from(error: EvidencePromotionError) -> Self {
        Self::Promotion(error)
    }
}

impl From<ValidityRefusal> for TimedPromotionError {
    fn from(refusal: ValidityRefusal) -> Self {
        Self::Validity(refusal)
    }
}

impl fmt::Display for TimedPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Promotion(error) => error.fmt(formatter),
            Self::Validity(refusal) => refusal.fmt(formatter),
        }
    }
}

impl std::error::Error for TimedPromotionError {}

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
///
/// # Two ways to promote
///
/// The typed path ([`SearchHit::observe_resolved`], [`SearchHit::promote_verified_with`])
/// takes proofs that only [`ExactSourceResolver`] and [`verify_claim`] can make, so the
/// caller cannot assert a step it did not take. It is the path for anything a retriever
/// returned. Its guarantee is conditional on the roots it is given: a truthful
/// [`RawEventSource`] and [`SourceOwnership`] for the resolver, and a `validity` closure
/// that answers truthfully; none of them is checked here.
///
/// The boolean path ([`SearchHit::observe`], [`SearchHit::promote_verified`]) takes the
/// caller's word that the source was resolved and the verification passed. It is the
/// trusted-caller path, for a caller that has done those steps by other means and for
/// tests; nothing in it checks them, and a hit promoted through it carries no source or
/// claim ([`SearchHit::resolved_source`] and [`SearchHit::verified_claim`] are `None`).
/// Handing it values taken from a retriever's output is the "boolean trust" baseline of
/// experiment Q002, not a safe use. It is a trusted-caller path in the strict sense: a hit
/// reaches `VerifiedKnown` through it with no raw bytes at all.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    capsule: CapsuleId,
    generation: Generation,
    pub score: f32,
    pub backend: String,
    stage: EvidenceStage,
    source: Option<ResolvedSource>,
    claim: Option<Claim>,
    /// The authority's validity as of the observation, when the hit was observed through a
    /// time-checked path; what [`SearchHit::promote_verified_at`] holds the commit instant
    /// against.
    observed_validity: Option<KnowledgeValidity>,
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
            source: None,
            claim: None,
            observed_validity: None,
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

    /// The resolved source this hit was observed with through
    /// [`SearchHit::observe_resolved`]; `None` before that, and for a hit observed through
    /// the boolean [`SearchHit::observe`].
    pub fn resolved_source(&self) -> Option<&ResolvedSource> {
        self.source.as_ref()
    }

    /// The claim this hit was promoted with through [`SearchHit::promote_verified_with`];
    /// `None` before that, and for a hit promoted through the boolean
    /// [`SearchHit::promote_verified`].
    pub fn verified_claim(&self) -> Option<&Claim> {
        self.claim.as_ref()
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
    /// This is the trusted-caller path: `exact_source_resolved` is the caller's word.
    /// For a hit a retriever returned, use [`SearchHit::observe_resolved`], which takes
    /// the proof [`ExactSourceResolver::resolve`] makes instead.
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
        self.check_lifecycle(validity)?;
        if !exact_source_resolved {
            return Err(EvidencePromotionError::SourceNotResolved);
        }
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    /// Move a possible-evidence hit to `Observed` with the proof that its exact source
    /// was resolved.
    ///
    /// `source` can only come from [`ExactSourceResolver::resolve`]. It is kept with the
    /// hit ([`SearchHit::resolved_source`]) and is what [`SearchHit::promote_verified_with`]
    /// later holds a claim proof against. The lifecycle is asked as in
    /// [`SearchHit::observe`], once, about this hit's own capsule and generation, and
    /// before the proof is kept: a refused hit keeps no source. The source must also
    /// belong to the capsule generation the hit names ([`ResolvedSource::is_owned_by`],
    /// answered by the resolver's [`SourceOwnership`]): the retriever's label says which
    /// generation was asked about, and a live label must not launder an event of a
    /// superseded or revoked one.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `PossibleEvidence` (then `validity` is not called);
    /// `StaleGeneration`, `RevokedGeneration`, `DisputedGeneration` or
    /// `UnknownGeneration` as for [`SearchHit::observe`], in that order of precedence; then
    /// `SourceNotOwnedByHit` when the source does not belong to this hit's capsule and
    /// generation (an event nothing admitted belongs to none). A refused step leaves the
    /// hit at its stage.
    pub fn observe_resolved<F>(
        &mut self,
        source: ResolvedSource,
        validity: F,
    ) -> Result<(), EvidencePromotionError>
    where
        F: FnOnce(&CapsuleId, Generation) -> Option<Validity>,
    {
        self.check_lifecycle(validity)?;
        if !source.is_owned_by(&self.capsule, self.generation) {
            return Err(EvidencePromotionError::SourceNotOwnedByHit);
        }
        self.source = Some(source);
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    /// The step shared by both observe paths: the hit is possible evidence and the
    /// lifecycle authority answers live for its own capsule generation.
    fn check_lifecycle<F>(&self, validity: F) -> Result<(), EvidencePromotionError>
    where
        F: FnOnce(&CapsuleId, Generation) -> Option<Validity>,
    {
        if self.stage != EvidenceStage::PossibleEvidence {
            return Err(EvidencePromotionError::WrongStage);
        }
        match validity(&self.capsule, self.generation) {
            Some(Validity::Live) => Ok(()),
            Some(Validity::Superseded) => Err(EvidencePromotionError::StaleGeneration),
            Some(Validity::Revoked) => Err(EvidencePromotionError::RevokedGeneration),
            Some(Validity::Disputed) => Err(EvidencePromotionError::DisputedGeneration),
            None => Err(EvidencePromotionError::UnknownGeneration),
        }
    }

    /// [`SearchHit::observe`] that also holds the hit to the validity the authority has for
    /// it: the lifecycle must answer live **and** `time_validity`, read from the authority
    /// ("rebuilt from the authority", never from the retrieval cache), must hold at `now`.
    ///
    /// The validity is kept with the hit, so the moment of reliance can be checked again by
    /// [`SearchHit::promote_verified_at`]; a hit observed here is refused by the untimed
    /// [`SearchHit::promote_verified`]. `time_validity` is asked once, after the lifecycle.
    ///
    /// # Errors
    /// As [`SearchHit::observe`], then [`ValidityRefusal::Unknown`] when `time_validity`
    /// answers `None`, [`ValidityRefusal::NotYetValid`] or [`ValidityRefusal::Expired`] when
    /// the validity does not hold at `now`, and finally `SourceNotResolved`. A refused step
    /// leaves the hit at its stage.
    pub fn observe_at<F, T>(
        &mut self,
        validity: F,
        time_validity: T,
        now: Timestamp,
        exact_source_resolved: bool,
    ) -> Result<(), TimedPromotionError>
    where
        F: FnOnce(&CapsuleId, Generation) -> Option<Validity>,
        T: FnOnce(&CapsuleId, Generation) -> Option<KnowledgeValidity>,
    {
        self.check_lifecycle(validity)?;
        let held = self.check_time(time_validity, now)?;
        if !exact_source_resolved {
            return Err(EvidencePromotionError::SourceNotResolved.into());
        }
        self.observed_validity = Some(held);
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    /// [`SearchHit::observe_resolved`] that also holds the hit to the validity the authority
    /// has for it, as [`SearchHit::observe_at`] does. The order of refusals is the lifecycle,
    /// then the time, then the source's ownership.
    pub fn observe_resolved_at<F, T>(
        &mut self,
        source: ResolvedSource,
        validity: F,
        time_validity: T,
        now: Timestamp,
    ) -> Result<(), TimedPromotionError>
    where
        F: FnOnce(&CapsuleId, Generation) -> Option<Validity>,
        T: FnOnce(&CapsuleId, Generation) -> Option<KnowledgeValidity>,
    {
        self.check_lifecycle(validity)?;
        let held = self.check_time(time_validity, now)?;
        if !source.is_owned_by(&self.capsule, self.generation) {
            return Err(EvidencePromotionError::SourceNotOwnedByHit.into());
        }
        self.source = Some(source);
        self.observed_validity = Some(held);
        self.stage = EvidenceStage::Observed;
        Ok(())
    }

    /// The authority's validity for this hit, which must hold at `now`.
    fn check_time<T>(
        &self,
        time_validity: T,
        now: Timestamp,
    ) -> Result<KnowledgeValidity, TimedPromotionError>
    where
        T: FnOnce(&CapsuleId, Generation) -> Option<KnowledgeValidity>,
    {
        let held = time_validity(&self.capsule, self.generation).ok_or(ValidityRefusal::Unknown)?;
        Self::check_held(held, now)?;
        Ok(held)
    }

    fn check_held(held: KnowledgeValidity, now: Timestamp) -> Result<(), ValidityRefusal> {
        if held.valid_from.is_some_and(|from| now.0 < from.0) {
            return Err(ValidityRefusal::NotYetValid);
        }
        if held.valid_until.is_some_and(|until| now.0 >= until.0) {
            return Err(ValidityRefusal::Expired);
        }
        Ok(())
    }

    /// The commit-time check of a hit observed through a time-checked path: the validity the
    /// authority had at observation must still hold at `now`, the instant of reliance.
    fn recheck_time(&self, now: Timestamp) -> Result<(), ValidityRefusal> {
        let held = self.observed_validity.ok_or(ValidityRefusal::Unknown)?;
        Self::check_held(held, now)
    }

    /// Move an observed hit to `VerifiedKnown` once its verification passed.
    ///
    /// This is the trusted-caller path: `verification_passed` is the caller's word. For a
    /// hit observed through [`SearchHit::observe_resolved`], use
    /// [`SearchHit::promote_verified_with`].
    ///
    /// The generation promoted is the one [`SearchHit::observe`] asked the
    /// lifecycle about, since a hit's capsule and generation cannot change.
    /// The lifecycle is not asked again: a revocation or supersession
    /// committed after `observe` answered is not seen here.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `Observed` and was observed without a validity: a hit
    /// observed through [`SearchHit::observe_at`] is promoted only through
    /// [`SearchHit::promote_verified_at`], which checks its validity again, so the untimed
    /// step does not follow from its stage. `VerificationFailed` when verification did not
    /// pass.
    pub fn promote_verified(
        &mut self,
        verification_passed: bool,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage);
        }
        if self.observed_validity.is_some() {
            return Err(EvidencePromotionError::WrongStage);
        }
        if !verification_passed {
            return Err(EvidencePromotionError::VerificationFailed);
        }
        self.stage = EvidenceStage::VerifiedKnown;
        Ok(())
    }

    /// [`SearchHit::promote_verified`] for a hit observed through
    /// [`SearchHit::observe_at`]: the validity the authority had at observation is held
    /// against `now`, the instant of reliance, before the hit becomes known. A document that
    /// expired between retrieval and use is refused here, not promoted on the strength of the
    /// moment it was found.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `Observed`; [`ValidityRefusal::Unknown`] when it was
    /// observed without a validity; [`ValidityRefusal::NotYetValid`] or
    /// [`ValidityRefusal::Expired`] when the validity does not hold at `now`;
    /// `VerificationFailed` when verification did not pass. A refused step leaves the hit at
    /// its stage.
    pub fn promote_verified_at(
        &mut self,
        verification_passed: bool,
        now: Timestamp,
    ) -> Result<(), TimedPromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage.into());
        }
        self.recheck_time(now)?;
        if !verification_passed {
            return Err(EvidencePromotionError::VerificationFailed.into());
        }
        self.stage = EvidenceStage::VerifiedKnown;
        Ok(())
    }

    /// Move an observed hit to `VerifiedKnown` with the proof that a claim is what its
    /// resolved source states.
    ///
    /// `proof` can only come from [`verify_claim`]. It must be about the very source
    /// this hit was observed with through [`SearchHit::observe_resolved`]: a proof made
    /// for another source, however valid, is refused. The proof's claim is kept with the
    /// hit ([`SearchHit::verified_claim`]). Like [`SearchHit::promote_verified`] this does
    /// not ask the lifecycle again.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `Observed` and was observed without a validity (a hit
    /// observed through [`SearchHit::observe_resolved_at`] is promoted through
    /// [`SearchHit::promote_verified_with_at`]); `SourceNotResolved` when it was
    /// observed through the boolean [`SearchHit::observe`] and so holds no source; and
    /// `ClaimProofMismatch` when the proof is about another source. A refused step leaves
    /// the hit at its stage.
    pub fn promote_verified_with(
        &mut self,
        proof: ClaimProof,
    ) -> Result<(), EvidencePromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage);
        }
        if self.observed_validity.is_some() {
            return Err(EvidencePromotionError::WrongStage);
        }
        self.promote_with_proof(proof)
    }

    /// [`SearchHit::promote_verified_with`] for a hit observed through
    /// [`SearchHit::observe_resolved_at`]: the validity the authority had at observation is
    /// held against `now` before the claim proof is accepted.
    ///
    /// # Errors
    /// `WrongStage` unless the hit is `Observed`; a [`ValidityRefusal`] as for
    /// [`SearchHit::promote_verified_at`], checked first; then `SourceNotResolved` and
    /// `ClaimProofMismatch` as for [`SearchHit::promote_verified_with`].
    pub fn promote_verified_with_at(
        &mut self,
        proof: ClaimProof,
        now: Timestamp,
    ) -> Result<(), TimedPromotionError> {
        if self.stage != EvidenceStage::Observed {
            return Err(EvidencePromotionError::WrongStage.into());
        }
        self.recheck_time(now)?;
        Ok(self.promote_with_proof(proof)?)
    }

    fn promote_with_proof(&mut self, proof: ClaimProof) -> Result<(), EvidencePromotionError> {
        let Some(source) = self.source.as_ref() else {
            return Err(EvidencePromotionError::SourceNotResolved);
        };
        if proof.source() != source {
            return Err(EvidencePromotionError::ClaimProofMismatch);
        }
        self.claim = Some(proof.claim().clone());
        self.stage = EvidenceStage::VerifiedKnown;
        Ok(())
    }
}

/// A search over an index: the text to look for, how many hits, and what the asker may see.
///
/// The filter is part of the query, not a post-processing step: an index applies it before it
/// limits, so a filtered query returns `limit` hits whenever that many visible documents match.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SearchQuery {
    pub text: String,
    pub limit: usize,
    pub filter: SearchFilter,
}

pub trait SearchIndex {
    fn search(&self, query: &SearchQuery) -> Vec<SearchHit>;
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
