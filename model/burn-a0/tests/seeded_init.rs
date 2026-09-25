//! One seed, one set of initial weights, whatever the model reads first.
//!
//! Burn draws a parameter the first time it is read, from one global generator.
//! When `init` left that to the forward pass, an arm that skipped a module (typed
//! attention switched off skips `metadata_bias`) drew every later parameter from a
//! shifted stream, so the two arms of one seed started from different weights, and
//! an ablation compares exactly those two arms. With `init` drawing in declaration
//! order this test passes; with the draw left lazy it fails on `metadata_bias`.
//!
//! This is the only test in the file on purpose: tests in one binary run on
//! parallel threads, and any of them initializing a model would draw from the same
//! global generator between this test's seed and its init.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{
    admission_bias, load, save, CodeGrid, PtrA0, PtrA0Config, PtrSlotMetadata, SlotValues,
};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, TypeId, Validity, ValidityMask,
};

const WIDTH: usize = 8;

fn config() -> PtrA0Config {
    PtrA0Config::new(16, WIDTH)
        .with_provenance_buckets(4)
        .with_latent_steps(1)
}

fn router_logits(model: &PtrA0, device: &Device) -> Tensor<2> {
    let roles: [&[SemanticRole]; 1] = [&[SemanticRole::Goal, SemanticRole::Evidence]];
    let states: [&[EpistemicState]; 1] = [&[EpistemicState::Observed, EpistemicState::Hypothesis]];
    let vectors: Vec<_> = (0..2)
        .map(|slot| {
            SlotEncoding::V1
                .encode(
                    &TypeId::from("Fact"),
                    format!("slot {slot}").as_bytes(),
                    WIDTH,
                )
                .expect("a small payload")
        })
        .collect();
    let values = SlotValues::new(&[vectors.as_slice()], device).expect("one row");
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
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 9]], device);
    model
        .forward(tokens, &roles, &values, metadata)
        .router_logits
}

fn largest_difference(left: Tensor<2>, right: Tensor<2>) -> f32 {
    (left - right).abs().max().into_scalar()
}

#[test]
fn both_arms_of_one_seed_start_from_the_same_weights_whatever_they_read_first() {
    let device = Device::flex();

    device.seed(11);
    let off = config().with_typed_attention(false).init(&device);
    // Reading in the order the ablated forward pass reads, before the other arm
    // exists, is what used to shift the other arm's draws.
    let _ = router_logits(&off, &device);

    device.seed(11);
    let on = config().init(&device);
    let expected = router_logits(&on, &device);

    // `off`'s weights under `on`'s architecture. Saved bytes cannot be compared
    // directly, because Burn records a per-instance id with every parameter.
    let restored =
        load(&save(&off).expect("serializable"), &config(), &device).expect("same shape");
    let same = largest_difference(router_logits(&restored, &device), expected.clone());
    assert_eq!(
        same, 0.0,
        "the arms of seed 11 must start from identical weights"
    );

    // The control: another seed must not already agree, or the comparison above
    // could not tell weights apart.
    device.seed(12);
    let other = config().init(&device);
    let different = largest_difference(router_logits(&other, &device), expected);
    assert!(
        different > 1.0e-4,
        "seed 12 should differ from seed 11: {different}"
    );
}
