//! Which slots a model is allowed to attend to, decided outside the model.
//!
//! Burn A0 currently feeds the lifecycle in as a *learned validity-id hint*: the
//! validity state becomes an embedding and the network is trained to take it into
//! account. That is the wrong shape for this particular fact. A hint is something
//! the model weighs against everything else, so a sufficiently confident pattern
//! can outvote it — and the fact being outvoted here is "this generation was
//! revoked". Global invariant 3 does not admit a probability.
//!
//! A mask is not a hint. It is computed from committed lifecycle state, applied by
//! construction, and there is nothing for the network to learn around. Attention to
//! an excluded slot is not unlikely; it is unrepresentable.
//!
//! The mask only ever narrows. Every operation here either keeps an exclusion or
//! adds one, because a widening step is precisely how a revoked slot would come
//! back into a computation.
use crate::Validity;
use std::fmt;

/// Why a mask operation was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskError {
    /// Two masks describe different numbers of slots, so combining them would
    /// silently align the wrong positions.
    LengthMismatch { left: usize, right: usize },
    /// A slot index lies outside the mask.
    SlotOutOfRange { slot: usize, len: usize },
}

impl MaskError {
    /// Stable diagnostic code for this mask refusal.
    pub fn code(self) -> &'static str {
        match self {
            Self::LengthMismatch { .. } => "PTR_MASK_LENGTH_MISMATCH",
            Self::SlotOutOfRange { .. } => "PTR_MASK_SLOT_OUT_OF_RANGE",
        }
    }
}

impl fmt::Display for MaskError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for MaskError {}

/// Per-slot admission, derived from lifecycle state.
///
/// `true` means the slot may participate. The only [`Validity`] that admits is
/// [`Validity::Live`]; see [`ValidityMask::admits_validity`] for why each of the
/// others does not.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidityMask {
    admitted: Vec<bool>,
}

impl ValidityMask {
    /// Decide admission for each slot from its lifecycle state.
    pub fn from_validities(validities: &[Validity]) -> Self {
        Self {
            admitted: validities
                .iter()
                .map(|validity| Self::admits_validity(*validity))
                .collect(),
        }
    }

    /// A mask admitting every slot, as a starting point for narrowing.
    pub fn admitting_all(len: usize) -> Self {
        Self {
            admitted: vec![true; len],
        }
    }

    /// A mask admitting nothing.
    pub fn admitting_none(len: usize) -> Self {
        Self {
            admitted: vec![false; len],
        }
    }

    /// Whether a lifecycle state admits participation at all.
    ///
    /// Only `Live` does. `Superseded` and `Revoked` are the cases the invariant is
    /// about. `Disputed` is excluded too, and that is the deliberate part: a
    /// disputed value is one the verifier fabric has contradicted, and letting the
    /// model reason from it while the contradiction stands would put a learned
    /// score above a deterministic finding — which global invariant 11 forbids.
    pub fn admits_validity(validity: Validity) -> bool {
        match validity {
            Validity::Live => true,
            Validity::Superseded | Validity::Revoked | Validity::Disputed => false,
        }
    }

    /// Number of slots governed by this mask.
    pub fn len(&self) -> usize {
        self.admitted.len()
    }

    /// Whether the mask governs no slots.
    pub fn is_empty(&self) -> bool {
        self.admitted.is_empty()
    }

    /// Whether a slot may participate. An out-of-range slot never may.
    ///
    /// Deliberately not a `Result`: a caller iterating slots should not be able to
    /// turn a bounds mistake into an admission, and `false` is the safe reading.
    pub fn admits(&self, slot: usize) -> bool {
        self.admitted.get(slot).copied().unwrap_or(false)
    }

    /// Number of slots that may participate.
    pub fn admitted_count(&self) -> usize {
        self.admitted.iter().filter(|admitted| **admitted).count()
    }

    /// The admitted slot indices, for building a gather or an attention bias.
    pub fn admitted_slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.admitted
            .iter()
            .enumerate()
            .filter_map(|(slot, admitted)| admitted.then_some(slot))
    }

    /// Additive bias for an attention score: `0.0` where admitted, negative
    /// infinity where not.
    ///
    /// Negative infinity rather than a large negative constant: a finite penalty is
    /// a strong hint, and a strong hint can still be overcome by a large enough
    /// score. After a softmax, `-inf` contributes exactly zero weight.
    pub fn attention_bias(&self) -> Vec<f32> {
        self.admitted
            .iter()
            .map(|admitted| if *admitted { 0.0 } else { f32::NEG_INFINITY })
            .collect()
    }

    /// Narrow by another mask: a slot survives only if both admit it.
    ///
    /// There is no widening counterpart, by design.
    pub fn narrow(&mut self, other: &Self) -> Result<(), MaskError> {
        if self.len() != other.len() {
            return Err(MaskError::LengthMismatch {
                left: self.len(),
                right: other.len(),
            });
        }
        for (admitted, keep) in self.admitted.iter_mut().zip(&other.admitted) {
            *admitted &= *keep;
        }
        Ok(())
    }

    /// Exclude one slot.
    pub fn exclude(&mut self, slot: usize) -> Result<(), MaskError> {
        let len = self.len();
        *self
            .admitted
            .get_mut(slot)
            .ok_or(MaskError::SlotOutOfRange { slot, len })? = false;
        Ok(())
    }

    /// Whether every slot this mask admits is also admitted by `other`.
    ///
    /// Use it to check that a pipeline step narrowed rather than widened.
    pub fn is_narrowing_of(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .admitted
                .iter()
                .zip(&other.admitted)
                .all(|(mine, theirs)| !*mine || *theirs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_live_admits_and_every_state_is_decided() {
        // Exhaustive: a new Validity variant fails to compile until its admission
        // is decided here rather than defaulting.
        for validity in [
            Validity::Live,
            Validity::Superseded,
            Validity::Revoked,
            Validity::Disputed,
        ] {
            let expected = match validity {
                Validity::Live => true,
                Validity::Superseded | Validity::Revoked | Validity::Disputed => false,
            };
            assert_eq!(
                ValidityMask::admits_validity(validity),
                expected,
                "{validity:?}"
            );
        }
    }

    #[test]
    fn a_revoked_slot_is_unrepresentable_not_merely_unlikely() {
        let mask = ValidityMask::from_validities(&[
            Validity::Live,
            Validity::Revoked,
            Validity::Live,
            Validity::Superseded,
        ]);
        assert_eq!(mask.admitted_count(), 2);
        assert_eq!(mask.admitted_slots().collect::<Vec<_>>(), [0, 2]);

        // A finite penalty could be outweighed by a confident score; negative
        // infinity contributes exactly zero after a softmax.
        let bias = mask.attention_bias();
        assert_eq!(bias[0], 0.0);
        assert_eq!(bias[2], 0.0);
        assert!(bias[1].is_infinite() && bias[1].is_sign_negative());
        assert!(bias[3].is_infinite() && bias[3].is_sign_negative());
        assert_eq!(bias[1].exp(), 0.0);
    }

    #[test]
    fn masks_narrow_and_never_widen() {
        let live = ValidityMask::admitting_all(3);
        let mut mask = live.clone();
        mask.narrow(&ValidityMask::from_validities(&[
            Validity::Live,
            Validity::Revoked,
            Validity::Live,
        ]))
        .unwrap();
        assert_eq!(mask.admitted_slots().collect::<Vec<_>>(), [0, 2]);
        assert!(mask.is_narrowing_of(&live));

        // Narrowing with an all-admitting mask cannot bring slot 1 back.
        mask.narrow(&ValidityMask::admitting_all(3)).unwrap();
        assert_eq!(mask.admitted_slots().collect::<Vec<_>>(), [0, 2]);
        assert!(!live.is_narrowing_of(&mask));

        mask.exclude(0).unwrap();
        assert_eq!(mask.admitted_slots().collect::<Vec<_>>(), [2]);
        assert_eq!(mask.admitted_count(), 1);
    }

    #[test]
    fn a_mismatched_or_out_of_range_operation_is_refused() {
        let mut mask = ValidityMask::admitting_all(2);
        assert_eq!(
            mask.narrow(&ValidityMask::admitting_all(3)).unwrap_err(),
            MaskError::LengthMismatch { left: 2, right: 3 }
        );
        assert_eq!(
            mask.exclude(2).unwrap_err(),
            MaskError::SlotOutOfRange { slot: 2, len: 2 }
        );
        // The refused operations changed nothing.
        assert_eq!(mask.admitted_count(), 2);
        // An out-of-range read is never an admission.
        assert!(!mask.admits(2));
        assert!(!ValidityMask::admitting_none(4).admits(0));
        assert!(!ValidityMask::admitting_all(0).admits(0));
        assert!(ValidityMask::admitting_all(0).is_empty());
    }

    #[test]
    fn mask_errors_carry_stable_codes() {
        assert_eq!(
            MaskError::LengthMismatch { left: 1, right: 2 }.code(),
            "PTR_MASK_LENGTH_MISMATCH"
        );
        assert_eq!(
            MaskError::SlotOutOfRange { slot: 9, len: 1 }.code(),
            "PTR_MASK_SLOT_OUT_OF_RANGE"
        );
    }
}
