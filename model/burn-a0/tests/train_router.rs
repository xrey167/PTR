use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::Int,
};
use ptr_burn_a0::{PtrA0Config, PtrSlotMetadata};

fn batch(
    device: &Device,
) -> (
    Tensor<2, Int>,
    Tensor<2, Int>,
    Tensor<3>,
    PtrSlotMetadata,
    Tensor<1, Int>,
) {
    let tokens = Tensor::<2, Int>::from_data([[1, 1], [1, 1], [1, 1], [1, 1]], device);
    let slot_types = Tensor::<2, Int>::from_data([[0, 4], [1, 4], [2, 4], [3, 4]], device);
    let slots = Tensor::<3>::zeros([4, 2, 12], device);
    let metadata = PtrSlotMetadata {
        epistemic_ids: Tensor::<2, Int>::from_data([[0, 0], [0, 0], [0, 0], [0, 0]], device),
        validity_ids: Tensor::<2, Int>::from_data([[0, 0], [0, 0], [0, 0], [0, 0]], device),
        provenance_ids: Tensor::<2, Int>::from_data([[0, 0], [0, 0], [0, 0], [0, 0]], device),
        confidence: Tensor::<2>::ones([4, 2], device),
    };
    let labels = Tensor::<1, Int>::from_data([0, 1, 2, 3], device);
    (tokens, slot_types, slots, metadata, labels)
}

#[test]
fn tiny_router_task_is_trainable() {
    let device = Device::flex().autodiff();
    device.seed(17);
    let config = PtrA0Config::new(16, 8, 12, 4)
        .with_metadata_sizes(4, 4, 8)
        .with_latent_steps(1);
    let mut model = config.init(&device);
    let mut optimizer = AdamConfig::new().init();

    let (tokens, slot_types, slots, metadata, labels) = batch(&device);
    let initial_logits = model
        .forward(
            tokens.clone(),
            slot_types.clone(),
            slots.clone(),
            PtrSlotMetadata {
                epistemic_ids: metadata.epistemic_ids.clone(),
                validity_ids: metadata.validity_ids.clone(),
                provenance_ids: metadata.provenance_ids.clone(),
                confidence: metadata.confidence.clone(),
            },
        )
        .router_logits;
    let initial = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(initial_logits, labels.clone())
        .into_scalar();

    for _ in 0..120 {
        let output = model.forward(
            tokens.clone(),
            slot_types.clone(),
            slots.clone(),
            PtrSlotMetadata {
                epistemic_ids: metadata.epistemic_ids.clone(),
                validity_ids: metadata.validity_ids.clone(),
                provenance_ids: metadata.provenance_ids.clone(),
                confidence: metadata.confidence.clone(),
            },
        );
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output.router_logits, labels.clone());
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }

    let final_logits = model
        .forward(tokens, slot_types, slots, metadata)
        .router_logits;
    let final_loss = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(final_logits, labels)
        .into_scalar();

    assert!(
        final_loss < initial,
        "expected synthetic router loss to decrease: initial={initial:?} final={final_loss:?}"
    );
}
