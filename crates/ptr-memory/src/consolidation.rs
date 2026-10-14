//! Deterministic consolidation (episodic -> semantic promotion) and retention policy.
//!
//! The policy is a pure function of the record, the committed revision horizon and explicit
//! inputs. It contains no learned component: a learned score may be passed in
//! [`PromotionContext::learned_score`] but it can never override a verifier contradiction
//! (Invariant 11), derived-only support (Invariant 5) or missing commitment (Invariant 4).

use crate::record::{MemoryRecord, MemoryRecordId, RecordState};
use crate::MemoryClass;
use ptr_types::{Probability, Revision};

/// A deterministic verifier's contradiction of a record. Unresolved contradictions block
/// promotion and demote authoritative records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Contradiction {
    pub record: MemoryRecordId,
    pub verifier: String,
    pub detail: String,
}

/// Explicit inputs of one promotion decision.
#[derive(Clone, Debug, Default)]
pub struct PromotionContext {
    /// All unresolved contradictions; only those naming the record matter.
    pub contradictions: Vec<Contradiction>,
    /// Optional learned confidence. Recorded for callers; never changes a blocking decision.
    pub learned_score: Option<Probability>,
}

/// Why a promotion is postponed. It may succeed later without anything being wrong.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeferReason {
    /// Not enough distinct sources with committed observations.
    InsufficientIndependentSources { have: usize, need: usize },
    /// The record has observations, but none is at or below the committed horizon.
    NoCommittedObservation,
    /// The record has no support at all.
    NoSupport,
}

/// Why a promotion is refused outright.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectReason {
    /// An unresolved verifier contradiction names the record (Invariant 11).
    VerifierContradiction,
    /// Every piece of support is derived or search provenance (Invariant 5).
    OnlyDerivedOrSearchSupport,
    /// Only episodic records are promoted by this policy.
    NotEpisodic,
    /// The record is retired or revoked.
    TerminalState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromotionDecision {
    Promote,
    Defer(DeferReason),
    Reject(RejectReason),
}

/// What retention does with a record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionVerdict {
    Keep,
    /// Authoritative record contradicted by a verifier: back to `NeedsRevalidation`.
    Demote,
    /// Move to `Retired`.
    Forget,
}

/// Thresholds of the consolidation and retention policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConsolidationPolicy {
    /// Distinct sources with committed observations needed to promote (at least 1).
    pub min_independent_sources: usize,
    /// Logical-clock age after which an unpromoted episodic record is forgotten.
    pub episodic_ttl: u64,
}

impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            min_independent_sources: 2,
            episodic_ttl: 1_000,
        }
    }
}

impl ConsolidationPolicy {
    /// Decides whether an episodic `record` is promoted to semantic memory.
    pub fn decide_promotion(
        &self,
        record: &MemoryRecord,
        committed_horizon: Revision,
        context: &PromotionContext,
    ) -> PromotionDecision {
        match record.state {
            RecordState::Retired | RecordState::Revoked => {
                return PromotionDecision::Reject(RejectReason::TerminalState);
            }
            RecordState::Provisional
            | RecordState::Authoritative
            | RecordState::NeedsRevalidation => {}
        }
        if record.class != MemoryClass::Episodic {
            return PromotionDecision::Reject(RejectReason::NotEpisodic);
        }
        // Contradiction first: no score and no amount of support outranks it.
        if context.contradictions.iter().any(|c| c.record == record.id) {
            return PromotionDecision::Reject(RejectReason::VerifierContradiction);
        }
        if record.only_derived_or_search_support() {
            return PromotionDecision::Reject(RejectReason::OnlyDerivedOrSearchSupport);
        }
        if record.supports.is_empty() {
            return PromotionDecision::Defer(DeferReason::NoSupport);
        }
        if !record.has_committed_support(committed_horizon) {
            return PromotionDecision::Defer(DeferReason::NoCommittedObservation);
        }
        let need = self.min_independent_sources.max(1);
        let have = record.independent_committed_sources(committed_horizon);
        if have < need {
            return PromotionDecision::Defer(DeferReason::InsufficientIndependentSources {
                have,
                need,
            });
        }
        PromotionDecision::Promote
    }

    /// Decides demotion/forgetting for `record` at logical time `now`.
    pub fn decide_retention(
        &self,
        record: &MemoryRecord,
        now: u64,
        contradictions: &[Contradiction],
    ) -> RetentionVerdict {
        // A verifier contradiction outranks pinning (Invariant 11); pinning only prevents
        // forgetting.
        if record.pinned
            && !(record.state == RecordState::Authoritative
                && contradictions.iter().any(|c| c.record == record.id))
        {
            return RetentionVerdict::Keep;
        }
        match record.state {
            // Terminal states stay as tombstones so a revocation is never forgotten.
            RecordState::Retired | RecordState::Revoked => RetentionVerdict::Keep,
            RecordState::Authoritative => {
                if contradictions.iter().any(|c| c.record == record.id) {
                    RetentionVerdict::Demote
                } else {
                    RetentionVerdict::Keep
                }
            }
            RecordState::Provisional | RecordState::NeedsRevalidation => {
                let aged = now.saturating_sub(record.updated_at) >= self.episodic_ttl;
                if record.class == MemoryClass::Episodic && aged {
                    RetentionVerdict::Forget
                } else {
                    RetentionVerdict::Keep
                }
            }
        }
    }
}
