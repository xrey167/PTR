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
    /// Encode all 512 entity IDs and six slot indices at the model's input width.
    /// Panics if `width` is outside the range supported by `SlotEncoding::V1`.
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

/// Build model inputs in example order using the selected arm's batch variant.
///
/// Content-free slots carry their index, fixed role/state, and zero confidence;
/// their admission mask is retained only when requested. Raw-blind batches replace
/// tokens with PAD and keep typed slots. Admission-aware variants also remove raw
/// attribute tokens belonging to non-live facts, including shuffled tokens.
///
/// # Panics
///
/// Panics for an empty batch, unequal token lengths, malformed slot shapes, or
/// an entity ID outside the precomputed payload table when typed payloads are used.
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
        tokens.extend(admitted_tokens(example, kind));
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

/// Return gold operator codes on `device` in the supplied example order.
pub fn labels(examples: &[&Example], device: &Device) -> Tensor<1, Int> {
    let labels: Vec<i64> = examples
        .iter()
        .map(|example| example.label as i64)
        .collect();
    Tensor::<1, Int>::from_data(TensorData::new(labels, [examples.len()]), device)
}

/// Attribute token identities encode the slot, independent of sequence position.
/// Keep this mapping synchronized with operator-routing v1's generator.
fn admitted_tokens(example: &Example, kind: Batch) -> Vec<i64> {
    example
        .tokens
        .iter()
        .map(|&token| {
            if kind == Batch::RawBlind {
                return 0;
            }
            if kind != (Batch::ContentFree { masked: false }) && (1..=144).contains(&token) {
                let slot = ((token - 1) / 24) as usize;
                if example.facts[slot].validity != Validity::Live {
                    return 0;
                }
            }
            token
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Fact;

    #[test]
    fn admission_removes_non_live_attributes_from_shuffled_raw_tokens() {
        let mut example = Example {
            label: 0,
            tokens: vec![145, 25, 1, 48, 24, 149, 152, 0],
            facts: vec![
                Fact {
                    role: SemanticRole::Goal,
                    epistemic: EpistemicState::Unknown,
                    confidence: 0.5,
                    bucket: 1,
                    validity: Validity::Live,
                    entity: 0,
                };
                SLOTS
            ],
        };
        example.facts[1].validity = crate::data::tables()
            .validities
            .into_iter()
            .find(|v| *v != Validity::Live)
            .unwrap();
        for kind in [Batch::Typed, Batch::ContentFree { masked: true }] {
            let original = admitted_tokens(&example, kind);
            assert_eq!(original, vec![145, 0, 1, 0, 24, 149, 152, 0]);
            let mut changed = example.clone();
            changed.tokens[1] = 30;
            changed.tokens[3] = 40;
            assert_eq!(admitted_tokens(&changed, kind), original);
            changed.tokens[2] = 2;
            assert_ne!(admitted_tokens(&changed, kind), original);
        }
        assert_eq!(
            admitted_tokens(&example, Batch::ContentFree { masked: false }),
            example.tokens
        );
        assert_eq!(
            admitted_tokens(&example, Batch::RawBlind),
            vec![0; example.tokens.len()]
        );
    }
}
