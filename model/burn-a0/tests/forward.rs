use burn::{
    backend::{Autodiff, NdArray},
    prelude::*,
    tensor::Int,
};
use ptr_burn_a0::PtrA0Config;

type InferenceBackend = NdArray<f32>;
type TrainingBackend = Autodiff<NdArray<f32>>;

#[test]
fn forward_preserves_raw_and_slot_shapes() {
    let device = Default::default();
    let model = PtrA0Config::new(64, 8, 16, 5).init::<InferenceBackend>(&device);

    let tokens = Tensor::<InferenceBackend, 2, Int>::from_data([[1, 2, 3], [3, 2, 1]], &device);
    let slot_types =
        Tensor::<InferenceBackend, 2, Int>::from_data([[0, 1, 2, 3], [3, 2, 1, 0]], &device);
    let slots = Tensor::<InferenceBackend, 3>::zeros([2, 4, 16], &device);

    let output = model.forward(tokens, slot_types, slots);
    assert_eq!(output.raw.dims(), [2, 3, 16]);
    assert_eq!(output.slots.dims(), [2, 4, 16]);
    assert_eq!(output.router_logits.dims(), [2, 5]);
}

#[test]
fn router_path_supports_autodiff() {
    let device = Default::default();
    let model = PtrA0Config::new(32, 4, 8, 3).init::<TrainingBackend>(&device);

    let tokens = Tensor::<TrainingBackend, 2, Int>::from_data([[1, 2]], &device);
    let slot_types = Tensor::<TrainingBackend, 2, Int>::from_data([[0, 1]], &device);
    let slots = Tensor::<TrainingBackend, 3>::zeros([1, 2, 8], &device);

    let output = model.forward(tokens, slot_types, slots);
    let loss = output.router_logits.sum();
    let _gradients = loss.backward();
}
