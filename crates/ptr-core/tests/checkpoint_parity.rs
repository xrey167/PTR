//! Save/load numerical parity and rejection paths through the public API.

mod common;

use common::{fixed_slots, tiny_config};
use ptr_core::checkpoint::{load_checkpoint, save_checkpoint, CheckpointIoError, MODEL_NAME};
use ptr_core::latent_reasoner::{ReasoningBudget, RecurrenceOptions};
use ptr_core::training::{train_router_head, ToyRouterOptions};
use ptr_core::CoreModel;
use ptr_types::{CheckpointError, CheckpointHeader};

fn opts() -> RecurrenceOptions {
    RecurrenceOptions {
        budget: ReasoningBudget::Medium,
        epsilon: 0.01,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn forward_after_save_load_is_bit_identical() {
    let model = CoreModel::new(tiny_config()).unwrap();
    let loaded = load_checkpoint(&save_checkpoint(&model), &tiny_config()).unwrap();
    let (a, b) = (
        model.forward(&fixed_slots(), &opts()).unwrap(),
        loaded.forward(&fixed_slots(), &opts()).unwrap(),
    );
    assert_eq!(bits(a.states.as_slice()), bits(b.states.as_slice()));
    assert_eq!(bits(&a.pooled), bits(&b.pooled));
    assert_eq!(a.traces, b.traces);
    assert_eq!(
        bits(&model.router_probabilities(&a.pooled).unwrap()),
        bits(&loaded.router_probabilities(&b.pooled).unwrap())
    );
    assert_eq!(
        model.verify_probability(&a.pooled).unwrap().get().to_bits(),
        loaded
            .verify_probability(&b.pooled)
            .unwrap()
            .get()
            .to_bits()
    );
    // PTR-Diff logits are covered by the same parameters.
    let canvas = [model.mask_token(), 1, model.mask_token(), 2];
    assert_eq!(
        model.denoise(&canvas, 2).unwrap(),
        loaded.denoise(&canvas, 2).unwrap()
    );
}

#[test]
fn trained_router_head_survives_a_roundtrip() {
    let mut model = CoreModel::new(tiny_config()).unwrap();
    train_router_head(
        &mut model,
        &ToyRouterOptions {
            examples: 16,
            epochs: 5,
            ..ToyRouterOptions::default()
        },
    )
    .unwrap();
    let loaded = load_checkpoint(&save_checkpoint(&model), &tiny_config()).unwrap();
    let pooled = model.forward(&fixed_slots(), &opts()).unwrap().pooled;
    assert_eq!(
        bits(&model.router_probabilities(&pooled).unwrap()),
        bits(&loaded.router_probabilities(&pooled).unwrap())
    );
    // A model with a different seed would have produced different probabilities.
    let fresh = CoreModel::new(tiny_config()).unwrap();
    assert_ne!(
        bits(&fresh.router_probabilities(&pooled).unwrap()),
        bits(&loaded.router_probabilities(&pooled).unwrap())
    );
}

#[test]
fn checkpoint_bytes_are_deterministic_and_start_with_the_ptr_types_header() {
    let model = CoreModel::new(tiny_config()).unwrap();
    let a = save_checkpoint(&model);
    assert_eq!(a, save_checkpoint(&model));
    assert_eq!(&a[..8], b"PTRCKPT\x00");
    let (header, payload) = CheckpointHeader::read(&a).unwrap();
    assert_eq!(header.model, MODEL_NAME);
    // First payload field: parameter count, little endian.
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    assert_eq!(n, model.named_params().len());
}

#[test]
fn wrong_header_model_shape_and_checksum_are_rejected() {
    let model = CoreModel::new(tiny_config()).unwrap();
    let bytes = save_checkpoint(&model);

    assert_eq!(
        load_checkpoint(&[0_u8; 32], &tiny_config()).err(),
        Some(CheckpointIoError::Header(CheckpointError::NotACheckpoint))
    );

    let (header, payload) = CheckpointHeader::read(&bytes).unwrap();
    let mut renamed = header.clone();
    renamed.model = "ptr-a0".into();
    assert!(matches!(
        load_checkpoint(&renamed.write(payload), &tiny_config()),
        Err(CheckpointIoError::WrongModel { .. })
    ));

    let mut other = tiny_config();
    other.d_ff = 16;
    assert_eq!(
        load_checkpoint(&bytes, &other).err(),
        Some(CheckpointIoError::ArchitectureMismatch)
    );
    let mut forged = header.clone();
    forged.architecture = other.architecture_bytes();
    assert!(matches!(
        load_checkpoint(&forged.write(payload), &other),
        Err(CheckpointIoError::ShapeMismatch { .. })
    ));

    let mut flipped = payload.to_vec();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x40;
    assert!(matches!(
        load_checkpoint(&header.write(&flipped), &tiny_config()),
        Err(CheckpointIoError::ChecksumMismatch { .. })
    ));
}
