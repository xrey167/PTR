//! T6 of the A0 ablation study: every arm's configuration, built from the same
//! seed, starts from exactly the same parameters. Each arm's weights are loaded
//! into the default architecture and must then reproduce the default model's raw
//! states, slot states and router logits exactly. Those three outputs read every
//! one of A0's modules, the dead raw->slot branch included (it shapes `raw`).
//! Saved bytes are not compared, because Burn records a per-instance id with
//! every parameter.
//!
//! This is the only test in the file on purpose: a parallel test drawing from the
//! global generator between a seed and an init would make arms differ for the
//! wrong reason.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{
    admission_bias, load, save, CodeGrid, PtrA0, PtrA0Config, PtrA0Output, PtrSlotMetadata,
    SlotValues,
};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
};

const WIDTH: usize = 8;
const SEED: u64 = 43;

fn run(model: &PtrA0, device: &Device) -> PtrA0Output {
    let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Evidence]];
    let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Hypothesis]];
    let vectors: Vec<_> = (0..2)
        .map(|slot| {
            SlotEncoding::V1
                .encode(
                    &TypeId::from("Entity"),
                    format!("slot {slot}").as_bytes(),
                    WIDTH,
                )
                .expect("a small payload")
        })
        .collect();
    let metadata = PtrSlotMetadata {
        epistemic: CodeGrid::new(&Codebook::V1, &states, device).expect("v1 states"),
        provenance_ids: Tensor::<2, Int>::from_data([[0, 1]], device),
        confidence: Tensor::<2>::from_data([[0.9, 0.4]], device),
        admission: admission_bias(
            &[ValidityMask::from_validities(&[Validity::Live; 2])],
            device,
        ),
    };
    let roles = CodeGrid::new(&Codebook::V1, &roles, device).expect("v1 roles");
    let values = SlotValues::new(&[vectors.as_slice()], device).expect("one row");
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 9, 12]], device);
    model.forward(tokens, &roles, &values, metadata)
}

#[test]
fn every_arm_of_one_seed_starts_from_the_same_parameters() {
    let device = Device::flex();
    let full = PtrA0Config::new(16, WIDTH)
        .with_provenance_buckets(4)
        .with_latent_steps(2);

    device.seed(SEED);
    let reference = run(&full.init(&device), &device);

    for (arm, config) in [
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
        ("frozen-router", full.clone().with_frozen_router(true)),
    ] {
        device.seed(SEED);
        let model = config.init(&device);
        // Read it the way this arm reads, before comparing: that is what used to
        // shift the draws when parameters were drawn lazily.
        let _ = run(&model, &device);
        let as_default =
            load(&save(&model).expect("serializable"), &full, &device).expect("the same shape");
        let output = run(&as_default, &device);
        for (name, got, want) in [
            ("raw", output.raw.clone(), reference.raw.clone()),
            ("slots", output.slots.clone(), reference.slots.clone()),
        ] {
            let difference: f32 = (got - want).abs().max().into_scalar();
            assert_eq!(difference, 0.0, "{arm}: {name} differ from the default arm");
        }
        let difference: f32 = (output.router_logits - reference.router_logits.clone())
            .abs()
            .max()
            .into_scalar();
        assert_eq!(
            difference, 0.0,
            "{arm}: router logits differ from the default arm"
        );
    }
}
