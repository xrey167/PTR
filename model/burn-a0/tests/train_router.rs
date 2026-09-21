use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Int,
};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata};
use ptr_types::{Codebook, EpistemicState, SemanticRole, Validity, ValidityMask};

// One row per target operator. The distinguishing feature is the first slot's
// semantic role, so the task is "route by type", which is the mechanism this
// model exists to test.
const ROLES: [&[SemanticRole]; 4] = [
    &[SemanticRole::Goal, SemanticRole::Resource],
    &[SemanticRole::Constraint, SemanticRole::Resource],
    &[SemanticRole::Claim, SemanticRole::Resource],
    &[SemanticRole::Evidence, SemanticRole::Resource],
];

struct TrainingBatch {
    tokens: Tensor<2, Int>,
    slot_types: CodeGrid<SemanticRole>,
    slots: Tensor<3>,
    metadata: PtrSlotMetadata,
    labels: Tensor<1, Int>,
}

fn metadata(device: &Device) -> PtrSlotMetadata {
    PtrSlotMetadata {
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
    }
}

fn batch(device: &Device) -> TrainingBatch {
    TrainingBatch {
        tokens: Tensor::<2, Int>::from_data([[1, 1], [1, 1], [1, 1], [1, 1]], device),
        slot_types: CodeGrid::new(&Codebook::V1, &ROLES, device)
            .expect("every role is assigned in v1"),
        slots: Tensor::<3>::zeros([4, 2, 12], device),
        metadata: metadata(device),
        labels: Tensor::<1, Int>::from_data([0, 1, 2, 3], device),
    }
}

#[test]
fn tiny_router_task_is_trainable() {
    let device = Device::flex().autodiff();
    device.seed(17);
    let config = PtrA0Config::new(16, 12)
        .with_provenance_buckets(8)
        .with_latent_steps(1);
    let mut model = config.init(&device);
    let mut optimizer = AdamConfig::new().init();

    let TrainingBatch {
        tokens,
        slot_types,
        slots,
        metadata,
        labels,
    } = batch(&device);
    let initial_logits = model
        .forward(
            tokens.clone(),
            &slot_types,
            slots.clone(),
            clone_metadata(&metadata),
        )
        .router_logits;
    let initial = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(initial_logits, labels.clone())
        .into_scalar::<f32>();

    for _ in 0..120 {
        let output = model.forward(
            tokens.clone(),
            &slot_types,
            slots.clone(),
            clone_metadata(&metadata),
        );
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output.router_logits, labels.clone());
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }

    let final_logits = model
        .forward(tokens, &slot_types, slots, metadata)
        .router_logits;
    let final_loss = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(final_logits, labels)
        .into_scalar::<f32>();

    assert!(
        final_loss < initial,
        "expected synthetic router loss to decrease: initial={initial:?} final={final_loss:?}"
    );
}

fn clone_metadata(metadata: &PtrSlotMetadata) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic: metadata.epistemic.clone(),
        provenance_ids: metadata.provenance_ids.clone(),
        confidence: metadata.confidence.clone(),
        admission: metadata.admission.clone(),
    }
}
