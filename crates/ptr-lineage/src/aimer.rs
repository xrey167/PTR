//! AIMER expert-pruning scores (arXiv:2603.18492, "calibration-free expert
//! pruning").
//!
//! Each expert's weights are flattened into one vector `w` of length `N` and
//! scored by `s(w) = ||w||_1 / (sqrt(N) * ||w||_2)`, a Hoyer-like measure of
//! how evenly the weights are spread, in `[1/sqrt(N), 1]`: one nonzero weight
//! scores `1/sqrt(N)`, equal magnitudes score 1. Within a layer the experts
//! with the highest scores are pruned first. No data or activations are used.
//!
//! This module only ranks. A prune decision must still pass the
//! [`ForgettingGate`](crate::ForgettingGate), and the score says nothing about
//! an expert's quality by itself: it is a property of the weights' shape.

use std::fmt::Display;

use crate::error::LineageError;

/// One expert of a layer: its identifier and its flattened weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertWeights<'a, E> {
    pub id: E,
    pub weights: &'a [f64],
}

/// The AIMER score `||w||_1 / (sqrt(N) * ||w||_2)` of one flattened weight
/// vector.
///
/// Both norms are taken of `w / max|w|`, which leaves the ratio unchanged and
/// cannot overflow, so any finite magnitude is scored; entries so small next to
/// the largest that they underflow after scaling add nothing to either norm at
/// f64 precision. The result is clamped to `[1/sqrt(N), 1]` against rounding.
///
/// The score depends only on the multiset of weights, not on their order: the
/// scaled magnitudes are sorted ascending (`f64::total_cmp`) before both norms
/// are accumulated, so two experts holding the same weights in different
/// flattened layouts get bit-identical scores. This costs O(N log N) time and
/// one `Vec<f64>` of N entries per call.
///
/// # Errors
/// `Empty` for no weights, `NonFinite` for NaN or infinity, `ZeroNorm` for an
/// all-zero vector.
pub fn aimer_score(weights: &[f64]) -> Result<f64, LineageError> {
    if weights.is_empty() {
        return Err(LineageError::Empty { field: "weights" });
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(LineageError::NonFinite { field: "weights" });
    }
    let largest = weights
        .iter()
        .fold(0.0_f64, |top, value| top.max(value.abs()));
    if largest == 0.0 {
        return Err(LineageError::ZeroNorm { field: "weights" });
    }
    let mut scaled: Vec<f64> = weights
        .iter()
        .map(|value| (value / largest).abs())
        .collect();
    scaled.sort_by(f64::total_cmp);
    let (mut l1, mut squares) = (0.0_f64, 0.0_f64);
    for value in &scaled {
        l1 += value;
        squares += value * value;
    }
    let root_n = (weights.len() as f64).sqrt();
    let score = l1 / (root_n * squares.sqrt());
    Ok(score.clamp(1.0 / root_n, 1.0))
}

/// Order the experts of one layer for pruning: highest [`aimer_score`] first.
///
/// Experts with equal scores keep their input order (the sort is stable), so
/// the result is deterministic. All experts are experts of one layer and must
/// flatten to the same length `N`; the first expert sets it.
///
/// # Errors
/// `Empty` for no experts, `ShapeMismatch` for an expert whose weight count
/// differs from the first, and `ExpertRejected` naming the first expert whose
/// weights cannot be scored.
pub fn rank_experts_for_pruning<E: Clone + Display>(
    experts: &[ExpertWeights<'_, E>],
) -> Result<Vec<E>, LineageError> {
    let Some(first) = experts.first() else {
        return Err(LineageError::Empty { field: "experts" });
    };
    let mut scored = Vec::with_capacity(experts.len());
    for expert in experts {
        if expert.weights.len() != first.weights.len() {
            return Err(LineageError::ShapeMismatch {
                field: "expert weights",
                expected: first.weights.len(),
                actual: expert.weights.len(),
            });
        }
        let score = aimer_score(expert.weights).map_err(|source| LineageError::ExpertRejected {
            expert: expert.id.to_string(),
            source: Box::new(source),
        })?;
        scored.push((score, expert.id.clone()));
    }
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    Ok(scored.into_iter().map(|(_, id)| id).collect())
}
