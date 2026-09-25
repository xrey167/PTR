//! T2 of the A0 ablation study: every switch that claims to change the forward
//! pass does change it, starting from the same weights, and the frozen router,
//! which claims to change only training, does not change it at all. (Its effect
//! on training is tested in `src/lib.rs`, where the router's weights are
//! reachable.)
//!
//! This is the only test in the file on purpose: it compares models drawn from
//! one seed, and a parallel test drawing from the global generator in between
//! would make them differ for the wrong reason.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0, PtrA0Config, PtrSlotMetadata, SlotValues};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
};

const WIDTH: usize = 8;
const SEED: u64 = 29;

fn logits(model: &PtrA0, device: &Device) -> Tensor<2> {
    let roles: [&[SemanticRole]; 2] = [
        &[
            SemanticRole::Goal,
            SemanticRole::Evidence,
            SemanticRole::Claim,
        ],
        &[
            SemanticRole::Action,
            SemanticRole::Constraint,
            SemanticRole::Relation,
        ],
    ];
    let states: [&[EpistemicState]; 2] = [
        &[
            EpistemicState::Observed,
            EpistemicState::Hypothesis,
            EpistemicState::Verified,
        ],
        &[
            EpistemicState::Unknown,
            EpistemicState::Inferred,
            EpistemicState::Assumed,
        ],
    ];
    let vectors: Vec<Vec<_>> = (0..2)
        .map(|row| {
            (0..3)
                .map(|slot| {
                    SlotEncoding::V1
                        .encode(
                            &TypeId::from("Entity"),
                            format!("{row}/{slot}").as_bytes(),
                            WIDTH,
                        )
                        .expect("a small payload")
                })
                .collect()
        })
        .collect();
    let rows: Vec<&[_]> = vectors.iter().map(Vec::as_slice).collect();
    let metadata = PtrSlotMetadata {
        epistemic: CodeGrid::new(&Codebook::V1, &states, device).expect("v1 states"),
        provenance_ids: Tensor::<2, Int>::from_data([[0, 1, 2], [2, 1, 0]], device),
        confidence: Tensor::<2>::from_data([[0.9, 0.4, 0.7], [0.2, 1.0, 0.5]], device),
        admission: admission_bias(
            &[
                ValidityMask::from_validities(&[Validity::Live; 3]),
                ValidityMask::from_validities(&[Validity::Live; 3]),
            ],
            device,
        ),
    };
    let roles = CodeGrid::new(&Codebook::V1, &roles, device).expect("v1 roles");
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 9, 2], [7, 3, 3, 8]], device);
    let values = SlotValues::new(&rows, device).expect("a rectangular batch");
    model
        .forward(tokens, &roles, &values, metadata)
        .router_logits
}

fn at_seed(config: PtrA0Config, device: &Device) -> Tensor<2> {
    device.seed(SEED);
    logits(&config.init(device), device)
}

#[test]
fn each_switch_changes_the_forward_pass_it_claims_to_and_no_other() {
    let device = Device::flex();
    let full = PtrA0Config::new(16, WIDTH)
        .with_provenance_buckets(4)
        .with_latent_steps(2);
    let baseline = at_seed(full.clone(), &device);

    for (switch, config) in [
        (
            "typed_attention off",
            full.clone().with_typed_attention(false),
        ),
        ("typed_query off", full.clone().with_typed_query(false)),
        (
            "latent_nonlinearity off",
            full.clone().with_latent_nonlinearity(false),
        ),
        ("latent_steps 0", full.clone().with_latent_steps(0)),
    ] {
        let difference: f32 = (at_seed(config, &device) - baseline.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            difference > 1.0e-5,
            "{switch} must change the logits at init: {difference}"
        );
    }

    let frozen: f32 = (at_seed(full.with_frozen_router(true), &device) - baseline)
        .abs()
        .max()
        .into_scalar();
    assert_eq!(
        frozen, 0.0,
        "a frozen router must not change the forward pass"
    );

    // With zero refinement steps, neither form of the refinement update runs.
    // Check the interaction with both query and attention modes using the same
    // weights, within this single test so the shared RNG cannot race another test.
    for typed_attention in [false, true] {
        for typed_query in [false, true] {
            let zero_step = PtrA0Config::new(16, WIDTH)
                .with_provenance_buckets(4)
                .with_latent_steps(0)
                .with_typed_attention(typed_attention)
                .with_typed_query(typed_query);
            let nonlinear = at_seed(zero_step.clone(), &device);
            let linear = at_seed(zero_step.with_latent_nonlinearity(false), &device);
            let difference: f32 = (nonlinear - linear).abs().max().into_scalar();
            assert_eq!(
                difference, 0.0,
                "zero steps must ignore the nonlinearity switch: attention={typed_attention}, query={typed_query}"
            );
        }
    }
}
