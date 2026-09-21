use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, PtrA0Config, PtrSlotMetadata};
use ptr_types::{Validity, ValidityMask};

fn metadata(device: &Device) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic_ids: Tensor::<2, Int>::from_data([[0, 1, 2, 3], [3, 2, 1, 0]], device),
        provenance_ids: Tensor::<2, Int>::from_data([[1, 2, 3, 4], [4, 3, 2, 1]], device),
        confidence: Tensor::<2>::from_data([[1.0, 0.8, 0.5, 0.2], [0.2, 0.5, 0.8, 1.0]], device),
        admission: admission_bias(
            &[
                ValidityMask::from_validities(&[Validity::Live; 4]),
                ValidityMask::from_validities(&[Validity::Live; 4]),
            ],
            device,
        ),
    }
}

#[test]
fn forward_preserves_raw_and_slot_shapes() {
    let device = Device::flex();
    let model = PtrA0Config::new(64, 8, 16, 5)
        .with_metadata_sizes(4, 8)
        .with_latent_steps(2)
        .init(&device);

    let tokens = Tensor::<2, Int>::from_data([[1, 2, 3], [3, 2, 1]], &device);
    let slot_types = Tensor::<2, Int>::from_data([[0, 1, 2, 3], [3, 2, 1, 0]], &device);
    let slots = Tensor::<3>::zeros([2, 4, 16], &device);

    let output = model.forward(tokens, slot_types, slots, metadata(&device));
    assert_eq!(output.raw.dims(), [2, 3, 16]);
    assert_eq!(output.slots.dims(), [2, 4, 16]);
    assert_eq!(output.router_logits.dims(), [2, 5]);
}

#[test]
fn typed_metadata_and_latent_router_path_support_autodiff() {
    let device = Device::flex().autodiff();
    let model = PtrA0Config::new(32, 4, 8, 3)
        .with_metadata_sizes(4, 8)
        .with_latent_steps(2)
        .init(&device);

    let tokens = Tensor::<2, Int>::from_data([[1, 2], [2, 1]], &device);
    let slot_types = Tensor::<2, Int>::from_data([[0, 1, 2, 3], [3, 2, 1, 0]], &device);
    let slots = Tensor::<3>::zeros([2, 4, 8], &device);

    let output = model.forward(tokens, slot_types, slots, metadata(&device));
    let loss = output.router_logits.sum();
    let _gradients = loss.backward();
}
