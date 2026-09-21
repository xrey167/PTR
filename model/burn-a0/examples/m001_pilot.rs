use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Int,
};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0, PtrA0Config, PtrSlotMetadata};
use ptr_types::{Codebook, EpistemicState, SemanticRole, Validity, ValidityMask};

// The typed arm gives each row a distinct first role; the ablated arm gives them
// all the same one. That difference is the mechanism under test, and it is now
// expressed in codebook members rather than in bare integers.
const TYPED: [&[SemanticRole]; 4] = [
    &[SemanticRole::Goal, SemanticRole::Resource],
    &[SemanticRole::Constraint, SemanticRole::Resource],
    &[SemanticRole::Claim, SemanticRole::Resource],
    &[SemanticRole::Evidence, SemanticRole::Resource],
];

const ABLATED: [&[SemanticRole]; 4] = [
    &[SemanticRole::Goal, SemanticRole::Resource],
    &[SemanticRole::Goal, SemanticRole::Resource],
    &[SemanticRole::Goal, SemanticRole::Resource],
    &[SemanticRole::Goal, SemanticRole::Resource],
];

struct Batch {
    tokens: Tensor<2, Int>,
    slot_types: CodeGrid<SemanticRole>,
    slots: Tensor<3>,
    metadata: PtrSlotMetadata,
    labels: Tensor<1, Int>,
}

fn batch(device: &Device, typed: bool) -> Batch {
    let tokens = Tensor::<2, Int>::from_data([[1, 1], [1, 1], [1, 1], [1, 1]], device);
    let roles: &[&[SemanticRole]] = if typed { &TYPED } else { &ABLATED };
    let slot_types =
        CodeGrid::new(&Codebook::V1, roles, device).expect("every role is assigned in v1");
    let slots = Tensor::<3>::zeros([4, 2, 12], device);
    let metadata = PtrSlotMetadata {
        epistemic: CodeGrid::new(
            &Codebook::V1,
            &[&[EpistemicState::Unknown, EpistemicState::Unknown][..]; 4],
            device,
        )
        .expect("every epistemic state is assigned in v1"),
        provenance_ids: Tensor::<2, Int>::zeros([4, 2], device),
        confidence: Tensor::<2>::ones([4, 2], device),
        admission: admission_bias(
            &core::array::from_fn::<_, 4, _>(|_| {
                ValidityMask::from_validities(&[Validity::Live; 2])
            }),
            device,
        ),
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
        epistemic: metadata.epistemic.clone(),
        provenance_ids: metadata.provenance_ids.clone(),
        confidence: metadata.confidence.clone(),
        admission: metadata.admission.clone(),
    }
}

fn loss(model: &PtrA0, batch: &Batch, device: &Device) -> Tensor<1> {
    let logits = model
        .forward(
            batch.tokens.clone(),
            &batch.slot_types,
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
            &batch.slot_types,
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
    let config = PtrA0Config::new(16, 12)
        .with_provenance_buckets(8)
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
