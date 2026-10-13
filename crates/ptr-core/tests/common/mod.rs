//! Shared fixtures for the integration tests (no test annotations here).

#![allow(dead_code)]

use ptr_core::semantic_slots::SemanticSlot;
use ptr_core::CoreConfig;
use ptr_types::{EpistemicState, Generation, Probability, SemanticRole, UncertaintyKind, Validity};

/// A tiny fixed configuration used by the golden and parity tests.
pub fn tiny_config() -> CoreConfig {
    CoreConfig {
        d_model: 4,
        heads: 2,
        d_ff: 8,
        max_slots: 8,
        max_latent_steps: 4,
        diff_vocab: 8,
        role_bias: true,
        seed: 7,
    }
}

/// Four fixed slots with distinct roles and hand-picked content.
pub fn fixed_slots() -> Vec<SemanticSlot> {
    let spec = [
        (
            SemanticRole::Goal,
            EpistemicState::Assumed,
            UncertaintyKind::Point,
        ),
        (
            SemanticRole::Claim,
            EpistemicState::Hypothesis,
            UncertaintyKind::Interval,
        ),
        (
            SemanticRole::Evidence,
            EpistemicState::Observed,
            UncertaintyKind::Distribution,
        ),
        (
            SemanticRole::Action,
            EpistemicState::Verified,
            UncertaintyKind::Point,
        ),
    ];
    spec.iter()
        .enumerate()
        .map(|(i, (role, epistemic, uncertainty))| SemanticSlot {
            id: i as u32,
            role: *role,
            epistemic: *epistemic,
            uncertainty: *uncertainty,
            generation: Generation(1),
            validity: Validity::Live,
            confidence: Probability::default(),
            value: (0..4)
                .map(|j| 0.25 * (i as f32) - 0.1 * (j as f32))
                .collect(),
            provenance: Vec::new(),
        })
        .collect()
}
