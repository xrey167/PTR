//! Building A0 inputs for a batch of examples, for each kind of arm, through the
//! crate's own constructors (CodeGrid, SlotValues, admission_bias).

use crate::arms::Batch;
use crate::data::{Example, SLOTS};
use burn::{prelude::*, tensor::Int, tensor::TensorData};
use ptr_burn_a0::{admission_bias, CodeGrid, PtrSlotMetadata, SlotValues};
use ptr_types::{
    Codebook, EpistemicState, SemanticRole, SlotEncoding, SlotVector, TypeId, Validity,
    ValidityMask,
};

/// Payload vectors, encoded once: one per entity id and one per slot index.
pub struct Payloads {
    entities: Vec<SlotVector>,
    indices: Vec<SlotVector>,
}

impl Payloads {
    pub fn new(width: usize) -> Self {
        let encode = |kind: &str, bytes: String| {
            SlotEncoding::V1
                .encode(&TypeId::from(kind), bytes.as_bytes(), width)
                .expect("a short payload encodes")
        };
        Self {
            // Entity ids are 0..256 in-distribution and 256..512 in ood_payload.
            entities: (0..512)
                .map(|e| encode("Entity", format!("entity-{e}")))
                .collect(),
            indices: (0..SLOTS)
                .map(|i| encode("SlotIndex", format!("slot-{i}")))
                .collect(),
        }
    }
}

pub struct Inputs {
    pub tokens: Tensor<2, Int>,
    pub roles: CodeGrid<SemanticRole>,
    pub values: SlotValues,
    pub metadata: PtrSlotMetadata,
}

pub fn build(examples: &[&Example], kind: Batch, payloads: &Payloads, device: &Device) -> Inputs {
    let batch = examples.len();
    let length = examples[0].tokens.len();
    let mut tokens = Vec::with_capacity(batch * length);
    let mut roles: Vec<Vec<SemanticRole>> = Vec::with_capacity(batch);
    let mut states: Vec<Vec<EpistemicState>> = Vec::with_capacity(batch);
    let mut values: Vec<Vec<SlotVector>> = Vec::with_capacity(batch);
    let mut confidence = Vec::with_capacity(batch * SLOTS);
    let mut provenance = Vec::with_capacity(batch * SLOTS);
    let mut masks = Vec::with_capacity(batch);
    for example in examples {
        assert_eq!(
            example.tokens.len(),
            length,
            "one split, one sequence length"
        );
        match kind {
            Batch::RawBlind => tokens.extend(std::iter::repeat_n(0_i64, length)),
            _ => tokens.extend_from_slice(&example.tokens),
        }
        let content_free = matches!(kind, Batch::ContentFree { .. });
        roles.push(
            example
                .facts
                .iter()
                .map(|fact| {
                    if content_free {
                        SemanticRole::Goal
                    } else {
                        fact.role
                    }
                })
                .collect(),
        );
        states.push(
            example
                .facts
                .iter()
                .map(|fact| {
                    if content_free {
                        EpistemicState::Unknown
                    } else {
                        fact.epistemic
                    }
                })
                .collect(),
        );
        values.push(
            example
                .facts
                .iter()
                .enumerate()
                .map(|(i, fact)| {
                    if content_free {
                        payloads.indices[i].clone()
                    } else {
                        payloads.entities[fact.entity as usize].clone()
                    }
                })
                .collect(),
        );
        for (i, fact) in example.facts.iter().enumerate() {
            confidence.push(if content_free { 0.0 } else { fact.confidence });
            provenance.push(i as i64);
        }
        masks.push(match kind {
            Batch::ContentFree { masked: false } => ValidityMask::admitting_all(SLOTS),
            _ => {
                let validities: Vec<Validity> =
                    example.facts.iter().map(|fact| fact.validity).collect();
                ValidityMask::from_validities(&validities)
            }
        });
    }
    let role_rows: Vec<&[SemanticRole]> = roles.iter().map(Vec::as_slice).collect();
    let state_rows: Vec<&[EpistemicState]> = states.iter().map(Vec::as_slice).collect();
    let value_rows: Vec<&[SlotVector]> = values.iter().map(Vec::as_slice).collect();
    Inputs {
        tokens: Tensor::<2, Int>::from_data(TensorData::new(tokens, [batch, length]), device),
        roles: CodeGrid::new(&Codebook::V1, &role_rows, device).expect("v1 roles"),
        values: SlotValues::new(&value_rows, device).expect("a rectangular batch"),
        metadata: PtrSlotMetadata {
            epistemic: CodeGrid::new(&Codebook::V1, &state_rows, device).expect("v1 states"),
            provenance_ids: Tensor::<2, Int>::from_data(
                TensorData::new(provenance, [batch, SLOTS]),
                device,
            ),
            confidence: Tensor::<2>::from_data(TensorData::new(confidence, [batch, SLOTS]), device),
            admission: admission_bias(&masks, device),
        },
    }
}

pub fn labels(examples: &[&Example], device: &Device) -> Tensor<1, Int> {
    let labels: Vec<i64> = examples
        .iter()
        .map(|example| example.label as i64)
        .collect();
    Tensor::<1, Int>::from_data(TensorData::new(labels, [examples.len()]), device)
}
