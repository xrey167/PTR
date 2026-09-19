use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Int,
};
use ptr_burn_a0::{PtrA0, PtrA0Config, PtrSlotMetadata};

struct Batch {
    tokens: Tensor<2, Int>,
    slot_types: Tensor<2, Int>,
    slots: Tensor<3>,
    metadata: PtrSlotMetadata,
    labels: Tensor<1, Int>,
}

fn batch(device: &Device, typed: bool) -> Batch {
    let tokens = Tensor::<2, Int>::from_data([[1, 1], [1, 1], [1, 1], [1, 1]], device);
    let slot_types = if typed {
        Tensor::<2, Int>::from_data([[0, 4], [1, 4], [2, 4], [3, 4]], device)
    } else {
        Tensor::<2, Int>::from_data([[0, 4], [0, 4], [0, 4], [0, 4]], device)
    };
    let slots = Tensor::<3>::zeros([4, 2, 12], device);
    let metadata = PtrSlotMetadata {
        epistemic_ids: Tensor::<2, Int>::zeros([4, 2], device),
        validity_ids: Tensor::<2, Int>::zeros([4, 2], device),
        provenance_ids: Tensor::<2, Int>::zeros([4, 2], device),
        confidence: Tensor::<2>::ones([4, 2], device),
    };
    let labels = Tensor::<1, Int>::from_data([0, 1, 2, 3], device);
    Batch {
        tokens,
        slot_types,
        slots,
        metadata,
        labels,
    }
}

fn clone_metadata(metadata: &PtrSlotMetadata) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic_ids: metadata.epistemic_ids.clone(),
        validity_ids: metadata.validity_ids.clone(),
        provenance_ids: metadata.provenance_ids.clone(),
        confidence: metadata.confidence.clone(),
    }
}

fn loss(model: &PtrA0, batch: &Batch, device: &Device) -> Tensor<1> {
    let logits = model
        .forward(
            batch.tokens.clone(),
            batch.slot_types.clone(),
            batch.slots.clone(),
            clone_metadata(&batch.metadata),
        )
        .router_logits;
    CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, batch.labels.clone())
}

fn accuracy(model: &PtrA0, batch: &Batch) -> f32 {
    let logits = model
        .forward(
            batch.tokens.clone(),
            batch.slot_types.clone(),
            batch.slots.clone(),
            clone_metadata(&batch.metadata),
        )
        .router_logits;
    let predictions = logits.argmax(1).squeeze::<1>();
    let correct = predictions
        .equal(batch.labels.clone())
        .int()
        .sum()
        .into_scalar::<i64>();
    correct as f32 / 4.0
}

fn train(typed: bool, seed: u64, steps: usize) -> (f32, f32, f32) {
    let device = Device::flex().autodiff();
    device.seed(seed);
    let config = PtrA0Config::new(16, 8, 12, 4)
        .with_metadata_sizes(4, 4, 8)
        .with_latent_steps(1);
    let mut model = config.init(&device);
    let mut optimizer = AdamConfig::new().init();
    let batch = batch(&device, typed);

    let initial = loss(&model, &batch, &device).into_scalar::<f32>();

    for _ in 0..steps {
        let step_loss = loss(&model, &batch, &device);
        let gradients = GradientsParams::from_grads(step_loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }

    let final_loss = loss(&model, &batch, &device).into_scalar::<f32>();
    let final_accuracy = accuracy(&model, &batch);
    (initial, final_loss, final_accuracy)
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(17);
    let steps = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(160);

    let (typed_initial, typed_final, typed_accuracy) = train(true, seed, steps);
    let (ablated_initial, ablated_final, ablated_accuracy) = train(false, seed, steps);

    println!(
        r#"{{"pilot":"M001-mechanism","seed":{},"steps":{},"typed_initial_loss":{},"typed_final_loss":{},"typed_accuracy":{},"ablated_initial_loss":{},"ablated_final_loss":{},"ablated_accuracy":{},"accuracy_delta":{}}}"#,
        seed,
        steps,
        typed_initial,
        typed_final,
        typed_accuracy,
        ablated_initial,
        ablated_final,
        ablated_accuracy,
        typed_accuracy - ablated_accuracy
    );
}
