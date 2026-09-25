//! T4 of the A0 ablation study: in every arm's configuration, nothing about a
//! slot that is not admitted reaches the router logits: not its role, epistemic
//! state, confidence, payload or provenance. The comparison is exact, because the
//! router multiplies an excluded slot's vote by zero rather than by something
//! small. Each case builds one model and changes only the excluded slot, so these
//! tests do not depend on the global generator and may run in parallel.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0, PtrA0Config, PtrSlotMetadata, SlotValues};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
};

const WIDTH: usize = 8;

/// The excluded slot (index 1) is described by these fields; slots 0 and 2 stay fixed.
#[derive(Clone, Copy)]
struct Excluded {
    role: SemanticRole,
    state: EpistemicState,
    confidence: f32,
    payload: &'static str,
    provenance: i64,
}

fn logits(model: &PtrA0, excluded: Excluded, device: &Device) -> Vec<u32> {
    let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, excluded.role, SemanticRole::Claim]];
    let states: [&[EpistemicState]; 1] = [&[
        EpistemicState::Observed,
        excluded.state,
        EpistemicState::Verified,
    ]];
    let vectors: Vec<_> = ["kept-a", excluded.payload, "kept-b"]
        .iter()
        .map(|payload| {
            SlotEncoding::V1
                .encode(&TypeId::from("Entity"), payload.as_bytes(), WIDTH)
                .expect("a small payload")
        })
        .collect();
    let metadata = PtrSlotMetadata {
        epistemic: CodeGrid::new(&Codebook::V1, &states, device).expect("v1 states"),
        provenance_ids: Tensor::<2, Int>::from_data([[0, excluded.provenance, 2]], device),
        confidence: Tensor::<2>::from_data([[0.9, excluded.confidence, 0.7]], device),
        admission: admission_bias(
            &[ValidityMask::from_validities(&[
                Validity::Live,
                Validity::Revoked,
                Validity::Live,
            ])],
            device,
        ),
    };
    let roles = CodeGrid::new(&Codebook::V1, &roles, device).expect("v1 roles");
    let values = SlotValues::new(&[vectors.as_slice()], device).expect("one row");
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 9, 2]], device);
    let logits: Vec<f32> = model
        .forward(tokens, &roles, &values, metadata)
        .router_logits
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 logits");
    logits.into_iter().map(f32::to_bits).collect()
}

fn arm_configs() -> Vec<(&'static str, PtrA0Config)> {
    let full = PtrA0Config::new(16, WIDTH)
        .with_provenance_buckets(4)
        .with_latent_steps(2);
    vec![
        ("full", full.clone()),
        (
            "no-typed-attention",
            full.clone().with_typed_attention(false),
        ),
        (
            "blind-query-k0",
            full.clone().with_typed_query(false).with_latent_steps(0),
        ),
        (
            "blind-query-k0-no-typed-attention",
            full.clone()
                .with_typed_query(false)
                .with_latent_steps(0)
                .with_typed_attention(false),
        ),
        ("latent-0", full.clone().with_latent_steps(0)),
        (
            "latent-linear",
            full.clone().with_latent_nonlinearity(false),
        ),
        ("latent-1", full.clone().with_latent_steps(1)),
        ("latent-4", full.clone().with_latent_steps(4)),
        ("frozen-router", full.with_frozen_router(true)),
    ]
}

#[test]
fn an_excluded_slot_cannot_reach_the_logits_in_any_arm() {
    let device = Device::flex();
    let first = Excluded {
        role: SemanticRole::Evidence,
        state: EpistemicState::Hypothesis,
        confidence: 0.4,
        payload: "excluded-a",
        provenance: 1,
    };
    let variations = [
        Excluded {
            role: SemanticRole::Action,
            ..first
        },
        Excluded {
            state: EpistemicState::Verified,
            ..first
        },
        Excluded {
            confidence: 0.99,
            ..first
        },
        Excluded {
            payload: "excluded-b",
            ..first
        },
        Excluded {
            provenance: 3,
            ..first
        },
    ];
    for (arm, config) in arm_configs() {
        let model = config.init(&device);
        let reference = logits(&model, first, &device);
        for (field, variation) in ["role", "epistemic", "confidence", "payload", "provenance"]
            .iter()
            .zip(variations)
        {
            assert_eq!(
                logits(&model, variation, &device),
                reference,
                "{arm}: the excluded slot's {field} reached the logits"
            );
        }
    }
}

/// The control: the same change to an admitted slot does reach the logits, so
/// the invariance above is the admission's doing and not an insensitive input.
#[test]
fn the_same_change_to_an_admitted_slot_does_reach_the_logits() {
    let device = Device::flex();
    let model = PtrA0Config::new(16, WIDTH)
        .with_provenance_buckets(4)
        .with_latent_steps(2)
        .init(&device);
    let admitted_role = |role| {
        let roles: [&[SemanticRole]; 1] = [&[role, SemanticRole::Evidence, SemanticRole::Claim]];
        let states: [&[EpistemicState]; 1] = [&[
            EpistemicState::Observed,
            EpistemicState::Hypothesis,
            EpistemicState::Verified,
        ]];
        let vectors: Vec<_> = ["kept-a", "excluded-a", "kept-b"]
            .iter()
            .map(|payload| {
                SlotEncoding::V1
                    .encode(&TypeId::from("Entity"), payload.as_bytes(), WIDTH)
                    .expect("a small payload")
            })
            .collect();
        let metadata = PtrSlotMetadata {
            epistemic: CodeGrid::new(&Codebook::V1, &states, &device).expect("v1 states"),
            provenance_ids: Tensor::<2, Int>::from_data([[0, 1, 2]], &device),
            confidence: Tensor::<2>::from_data([[0.9, 0.4, 0.7]], &device),
            admission: admission_bias(
                &[ValidityMask::from_validities(&[
                    Validity::Live,
                    Validity::Revoked,
                    Validity::Live,
                ])],
                &device,
            ),
        };
        let roles = CodeGrid::new(&Codebook::V1, &roles, &device).expect("v1 roles");
        let values = SlotValues::new(&[vectors.as_slice()], &device).expect("one row");
        model
            .forward(
                Tensor::<2, Int>::from_data([[1, 5, 9, 2]], &device),
                &roles,
                &values,
                metadata,
            )
            .router_logits
    };
    let difference: f32 = (admitted_role(SemanticRole::Goal) - admitted_role(SemanticRole::Action))
        .abs()
        .max()
        .into_scalar();
    assert!(
        difference > 1.0e-5,
        "an admitted slot's role must matter: {difference}"
    );
}
