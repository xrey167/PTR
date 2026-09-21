//! Lifecycle validity must be enforced, not weighed.
//!
//! The tests do not inspect attention weights. They assert the property that
//! matters instead: an excluded slot's contents cannot change anything the model
//! produces. That holds for any weights the network could have, which is the
//! claim — a test that only checked one initialisation would not be one.

use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, PtrA0, PtrA0Config, PtrSlotMetadata};
use ptr_types::{Validity, ValidityMask};

const SLOTS: usize = 3;
const D_MODEL: usize = 8;

fn model(device: &Device, seed: u64) -> PtrA0 {
    device.seed(seed);
    PtrA0Config::new(32, 4, D_MODEL, 3)
        .with_metadata_sizes(4, 8)
        .with_latent_steps(1)
        .init(device)
}

fn metadata(device: &Device, validities: &[Validity]) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic_ids: Tensor::<2, Int>::zeros([1, SLOTS], device),
        provenance_ids: Tensor::<2, Int>::zeros([1, SLOTS], device),
        confidence: Tensor::<2>::ones([1, SLOTS], device),
        admission: admission_bias(&[ValidityMask::from_validities(validities)], device),
    }
}

/// Slot values where `slot` carries `fill` and the others carry 1.0.
fn slot_values(device: &Device, slot: usize, fill: f32) -> Tensor<3> {
    let mut data = vec![1.0_f32; SLOTS * D_MODEL];
    for column in 0..D_MODEL {
        data[slot * D_MODEL + column] = fill;
    }
    Tensor::<1>::from_data(data.as_slice(), device).reshape([1, SLOTS, D_MODEL])
}

fn run(
    model: &PtrA0,
    device: &Device,
    validities: &[Validity],
    slot: usize,
    fill: f32,
) -> (Vec<f32>, Vec<f32>) {
    let output = model.forward(
        Tensor::<2, Int>::from_data([[1, 2]], device),
        Tensor::<2, Int>::zeros([1, SLOTS], device),
        slot_values(device, slot, fill),
        metadata(device, validities),
    );
    (
        output.raw.into_data().try_into_vec::<f32>().unwrap(),
        output
            .router_logits
            .into_data()
            .try_into_vec::<f32>()
            .unwrap(),
    )
}

#[test]
fn an_excluded_slot_cannot_change_anything_the_model_produces() {
    let device = Device::flex();
    // Several initialisations, because the claim is about the construction and
    // not about one lucky set of weights.
    for seed in [1_u64, 7, 4242] {
        let model = model(&device, seed);
        let validities = [Validity::Live, Validity::Revoked, Validity::Live];

        let (raw, router) = run(&model, &device, &validities, 1, 0.0);
        for fill in [1.0_f32, -50.0, 1_000.0] {
            let (other_raw, other_router) = run(&model, &device, &validities, 1, fill);
            assert_eq!(raw, other_raw, "seed {seed}: raw changed for fill {fill}");
            assert_eq!(
                router, other_router,
                "seed {seed}: router changed for fill {fill}"
            );
        }

        // The control: an admitted slot's contents *must* matter, or the test
        // above would pass on a model that ignores every slot.
        //
        // The router is the observable here, not `raw`. Scores scale with slot
        // values, so a large change saturates the raw-side softmax onto whichever
        // slot wins; when that is not the slot being varied, `raw` stays constant
        // for a reason that has nothing to do with admission. The router averages
        // over admitted slots and responds to every one of them.
        let (_, baseline_router) = run(&model, &device, &validities, 0, 1.0);
        let (_, changed_router) = run(&model, &device, &validities, 0, 1_000.0);
        assert_ne!(
            baseline_router, changed_router,
            "seed {seed}: an admitted slot must reach the router"
        );
    }
}

#[test]
fn every_inadmissible_lifecycle_state_is_excluded_and_live_is_not() {
    let device = Device::flex();
    let model = model(&device, 11);
    for state in [Validity::Revoked, Validity::Superseded, Validity::Disputed] {
        let validities = [Validity::Live, state, Validity::Live];
        let (raw, router) = run(&model, &device, &validities, 1, 0.0);
        let (other_raw, other_router) = run(&model, &device, &validities, 1, 900.0);
        assert_eq!(raw, other_raw, "{state:?} must be excluded");
        assert_eq!(router, other_router, "{state:?} must be excluded");
    }

    // The control, on the router for the reason given above: the same slot with
    // Live in its place must reach the output.
    let live = [Validity::Live, Validity::Live, Validity::Live];
    let (_, router) = run(&model, &device, &live, 1, 0.0);
    let (_, other_router) = run(&model, &device, &live, 1, 900.0);
    assert_ne!(router, other_router, "a Live slot must still matter");
}

#[test]
fn a_batch_row_admitting_nothing_stays_finite() {
    // Every score in the row is negative infinity, which is the one input that
    // turns a softmax into NaN and poisons the whole batch.
    let device = Device::flex();
    let model = model(&device, 5);
    let none = [Validity::Revoked; SLOTS];
    let (raw, router) = run(&model, &device, &none, 0, 1.0);
    assert!(
        raw.iter().all(|value| value.is_finite()),
        "raw went non-finite"
    );
    assert!(
        router.iter().all(|value| value.is_finite()),
        "router went non-finite: {router:?}"
    );
    // With nothing admissible there is no typed state to consult, so the router
    // says nothing rather than something arbitrary.
    assert!(router.iter().all(|value| *value == 0.0), "{router:?}");
}

#[test]
fn admission_bias_is_zero_and_negative_infinity_in_slot_order() {
    let device = Device::flex();
    let bias: Vec<f32> = admission_bias(
        &[ValidityMask::from_validities(&[
            Validity::Live,
            Validity::Revoked,
            Validity::Live,
        ])],
        &device,
    )
    .into_data()
    .try_into_vec::<f32>()
    .unwrap();
    assert_eq!(bias[0], 0.0);
    assert!(bias[1].is_infinite() && bias[1] < 0.0);
    assert_eq!(bias[2], 0.0);
}
