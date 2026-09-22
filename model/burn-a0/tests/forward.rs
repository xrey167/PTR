use burn::{prelude::*, tensor::Int};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrA0Config, PtrSlotMetadata, SlotValues};
use ptr_types::{
    Codebook, EpistemicState, ReasoningOperator, SemanticRole, SlotEncoding, TypeId, Validity,
    ValidityMask,
};

/// Committed payloads for a batch of two rows of four slots.
///
/// Every slot carries a distinct payload, so a shape assertion cannot pass because
/// the values happened to be uniform.
fn slot_values(width: usize, device: &Device) -> SlotValues {
    let encoding = SlotEncoding::V1;
    let vectors: Vec<Vec<_>> = (0..2)
        .map(|row| {
            (0..4)
                .map(|slot| {
                    encoding
                        .encode(
                            &TypeId::from("Document"),
                            format!("row {row} slot {slot}").as_bytes(),
                            width,
                        )
                        .expect("a small payload")
                })
                .collect()
        })
        .collect();
    let rows: Vec<&[_]> = vectors.iter().map(Vec::as_slice).collect();
    SlotValues::new(&rows, device).expect("a rectangular batch")
}

const ROLES: [&[SemanticRole]; 2] = [
    &[
        SemanticRole::Goal,
        SemanticRole::Constraint,
        SemanticRole::Claim,
        SemanticRole::Evidence,
    ],
    &[
        SemanticRole::Evidence,
        SemanticRole::Claim,
        SemanticRole::Constraint,
        SemanticRole::Goal,
    ],
];

const STATES: [&[EpistemicState]; 2] = [
    &[
        EpistemicState::Unknown,
        EpistemicState::Assumed,
        EpistemicState::Hypothesis,
        EpistemicState::Observed,
    ],
    &[
        EpistemicState::Observed,
        EpistemicState::Hypothesis,
        EpistemicState::Assumed,
        EpistemicState::Unknown,
    ],
];

fn slot_types(device: &Device) -> CodeGrid<SemanticRole> {
    CodeGrid::new(&Codebook::V1, &ROLES, device).expect("every role is assigned in v1")
}

fn metadata(device: &Device) -> PtrSlotMetadata {
    PtrSlotMetadata {
        epistemic: CodeGrid::new(&Codebook::V1, &STATES, device)
            .expect("every epistemic state is assigned in v1"),
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
    let model = PtrA0Config::new(64, 16)
        .with_provenance_buckets(8)
        .with_latent_steps(2)
        .init(&device);

    let tokens = Tensor::<2, Int>::from_data([[1, 2, 3], [3, 2, 1]], &device);
    let slots = slot_values(16, &device);

    let output = model.forward(tokens, &slot_types(&device), &slots, metadata(&device));
    assert_eq!(output.raw.dims(), [2, 3, 16]);
    assert_eq!(output.slots.dims(), [2, 4, 16]);
    // The router's width is the operator family's cardinality, not a number the
    // caller picked: one logit per operator the codebook defines.
    let operators = usize::from(Codebook::V1.cardinality_of::<ReasoningOperator>());
    assert_eq!(operators, 11, "v1 assigns eleven reasoning operators");
    assert_eq!(output.router_logits.dims(), [2, operators]);
}

#[test]
fn typed_metadata_and_latent_router_path_support_autodiff() {
    let device = Device::flex().autodiff();
    let model = PtrA0Config::new(32, 8)
        .with_provenance_buckets(8)
        .with_latent_steps(2)
        .init(&device);

    let tokens = Tensor::<2, Int>::from_data([[1, 2], [2, 1]], &device);
    let slots = slot_values(8, &device);

    let output = model.forward(tokens, &slot_types(&device), &slots, metadata(&device));
    let loss = output.router_logits.sum();
    let _gradients = loss.backward();
}
