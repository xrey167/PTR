//! A checkpoint saved through the bridge is bound as it is saved.
//!
//! Issue #23's C9: binding used to be a separate step (`ptrctl seal`), so a
//! checkpoint could sit on disk unbound. `save_bound_checkpoint` writes the
//! checkpoint, the sealed binding and the retained anchor as one operation through
//! `PtrRuntime::seal_checkpoint`, the step `ptrctl seal` runs too. The verifier these
//! tests use is the runtime's own: the anchor read back from disk with
//! `parse_retained_anchor`, the sealed bytes opened with `NeuralState::open`, and
//! the state admitted with `PtrRuntime::admit`.
mod common;

use burn::prelude::*;
use common::{config, forward, logit_bits, model, world};
use ptr_a0_bridge::{save_bound_checkpoint, BridgeError, CheckpointPaths, ReadSet};
use ptr_burn_a0::load;
use ptr_runtime::neural::{parse_retained_anchor, NeuralState};
use ptr_types::{CodebookVersion, Generation, ProvenanceRef};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ptr-a0-bridge-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn provenance() -> [ProvenanceRef; 1] {
    [ProvenanceRef {
        source: "trainer:a0-bridge".into(),
        note: Some("sealed as it was saved".to_owned()),
    }]
}

#[test]
fn a_bound_save_is_sealed_admitted_and_names_exactly_what_the_model_read() {
    let device = Device::flex();
    let model = model(&device);
    let runtime = world(b"alpha", b"x", true);
    let pass = forward(&runtime, &model, &device);
    let temp = Temp::new("ok");
    let (checkpoint, sealed, anchor) = (
        temp.at("model.ckpt"),
        temp.at("model.sealed"),
        temp.at("model.anchor.toml"),
    );

    let returned = save_bound_checkpoint(
        &runtime,
        &model,
        &pass.read,
        &provenance(),
        &CheckpointPaths {
            checkpoint: &checkpoint,
            sealed: &sealed,
            anchor: &anchor,
        },
    )
    .expect("a model trained on committed state is saved bound to it");
    assert_eq!(
        temp.names(),
        ["model.anchor.toml", "model.ckpt", "model.sealed"]
    );

    let retained = parse_retained_anchor(&std::fs::read_to_string(&anchor).unwrap())
        .expect("the retained anchor parses");
    assert_eq!(retained.anchor, returned);
    let state = NeuralState::open(&std::fs::read(&sealed).unwrap(), retained.anchor)
        .expect("the seal reopens under the retained anchor");
    runtime
        .admit(&state)
        .expect("and the runtime that it was bound against admits it");

    let binding = state.binding();
    assert_eq!(binding.codebook, CodebookVersion::V1);
    assert_eq!(
        binding.semantic_inputs.keys().collect::<Vec<_>>(),
        ["evidence", "goal"],
        "the binding names the admitted slots' keys, recorded by the bridge as it fed them"
    );
    assert_eq!(
        binding.generations.iter().collect::<Vec<_>>(),
        [(&"capsule:a".to_owned(), &Generation(1))],
        "and the generation they were admitted under; the revoked one is not named"
    );
    assert_eq!(binding.provenance, provenance());

    // The checkpoint on disk is the model: it loads and computes what it computed.
    let reloaded = load(&std::fs::read(&checkpoint).unwrap(), &config(), &device)
        .expect("the saved checkpoint loads");
    assert_eq!(
        logit_bits(&forward(&runtime, &reloaded, &device)),
        logit_bits(&pass)
    );
}

/// The control for the test above: the same verifier refuses what it should, so
/// its acceptance there is evidence. A sealed artifact with one byte changed, the
/// bare checkpoint offered as a sealed state, and the intact seal after the
/// generation it was bound under is revoked.
#[test]
fn a_tampered_or_unsealed_checkpoint_is_refused_and_a_revocation_denies_the_seal() {
    let device = Device::flex();
    let model = model(&device);
    let mut runtime = world(b"alpha", b"x", true);
    let read = forward(&runtime, &model, &device).read;
    let temp = Temp::new("control");
    let (checkpoint, sealed, anchor) = (
        temp.at("model.ckpt"),
        temp.at("model.sealed"),
        temp.at("model.anchor.toml"),
    );
    save_bound_checkpoint(
        &runtime,
        &model,
        &read,
        &provenance(),
        &CheckpointPaths {
            checkpoint: &checkpoint,
            sealed: &sealed,
            anchor: &anchor,
        },
    )
    .unwrap();
    let retained = parse_retained_anchor(&std::fs::read_to_string(&anchor).unwrap()).unwrap();
    let bytes = std::fs::read(&sealed).unwrap();

    let mut tampered = bytes.clone();
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0x01;
    assert!(
        NeuralState::open(&tampered, retained.anchor).is_err(),
        "one changed byte must be refused under the retained anchor"
    );
    assert!(
        NeuralState::open(&std::fs::read(&checkpoint).unwrap(), retained.anchor).is_err(),
        "a checkpoint that was never sealed is not a sealed state"
    );

    let state = NeuralState::open(&bytes, retained.anchor).expect("the intact seal opens");
    runtime
        .admit(&state)
        .expect("admitted before the revocation");
    runtime
        .commit(ptr_ledger::LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(
        runtime.admit(&state).is_err(),
        "a checkpoint bound under a generation that is now revoked must be denied"
    );
}

#[test]
fn a_save_after_the_runtime_moved_is_refused_and_writes_nothing() {
    let device = Device::flex();
    let model = model(&device);
    let mut runtime = world(b"alpha", b"x", true);
    let read = forward(&runtime, &model, &device).read;
    // Anything committed after the reads: the binding would now pin a position the
    // model never read at.
    runtime
        .commit(ptr_ledger::LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:c".into(),
            generation: Generation(1),
        })
        .unwrap();

    let temp = Temp::new("moved");
    let error = save_bound_checkpoint(
        &runtime,
        &model,
        &read,
        &provenance(),
        &CheckpointPaths {
            checkpoint: &temp.at("model.ckpt"),
            sealed: &temp.at("model.sealed"),
            anchor: &temp.at("model.anchor.toml"),
        },
    )
    .expect_err("the runtime moved since the reads");
    assert!(
        matches!(error, BridgeError::PositionMoved { .. }),
        "{error:?}"
    );
    assert!(
        temp.names().is_empty(),
        "no unbound checkpoint is left behind"
    );

    // The control: a read set taken now saves.
    let now = ReadSet::at(&runtime).unwrap();
    save_bound_checkpoint(
        &runtime,
        &model,
        &now,
        &provenance(),
        &CheckpointPaths {
            checkpoint: &temp.at("model.ckpt"),
            sealed: &temp.at("model.sealed"),
            anchor: &temp.at("model.anchor.toml"),
        },
    )
    .expect("a read set at the current position saves");
}

#[test]
fn read_sets_from_different_positions_do_not_merge() {
    let device = Device::flex();
    let model = model(&device);
    let mut runtime = world(b"alpha", b"x", true);
    let mut first = forward(&runtime, &model, &device).read;
    let same = forward(&runtime, &model, &device).read;
    first.merge(&same).expect("one position merges");
    runtime
        .commit(ptr_ledger::LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:c".into(),
            generation: Generation(1),
        })
        .unwrap();
    let later = forward(&runtime, &model, &device).read;
    assert!(matches!(
        first.merge(&later),
        Err(BridgeError::MixedPositions { .. })
    ));
}
