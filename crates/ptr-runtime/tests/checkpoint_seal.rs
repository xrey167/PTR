//! Saving, binding and sealing a checkpoint as one operation.
//!
//! `seal_checkpoint` is the step `ptrctl seal` and a trainer that seals as it saves
//! (`ptr-a0-bridge`) share, so these tests drive it directly against the same real
//! `ptr-burn-a0` fixture `checkpoint_binding.rs` binds. What is under test is the
//! *set* of files: a sealed save is the checkpoint, its sealed artifact and the
//! separately retained anchor, and either all of them arrive or none does.
#[path = "common/semantic.rs"]
mod semantic_common;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::neural::{
    parse_retained_anchor, NeuralState, SealCheckpointError, SealTargets, StateDeclaration,
};
use ptr_runtime::PtrRuntime;
use ptr_semdb::SemanticDelta;
use ptr_types::{CodeFamily, CodebookVersion, Generation};
use semantic_common::{granted, host_write_now};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

const FIXTURE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ptr-a0-v1.ckpt"
));

/// The families A0 embeds; see `checkpoint_binding.rs` for why they are named here.
const EMBEDDED: [CodeFamily; 3] = [
    CodeFamily::SemanticRole,
    CodeFamily::EpistemicState,
    CodeFamily::ReasoningOperator,
];

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ptr-runtime-seal-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Every name in the directory, so a test can assert that nothing — not even a
    /// scratch file — was left behind.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .expect("list temp dir")
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

fn runtime() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    granted(&mut runtime);
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("plan".into(), "original plan".into());
    host_write_now(&mut runtime, delta).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
}

fn declaration() -> StateDeclaration {
    StateDeclaration::at(CodebookVersion::V1)
        .reading("plan")
        .under("capsule:a")
}

fn targets(temp: &Temp) -> (PathBuf, PathBuf, PathBuf) {
    (
        temp.at("model.ckpt"),
        temp.at("model.sealed"),
        temp.at("model.anchor.toml"),
    )
}

#[test]
fn a_sealed_save_writes_checkpoint_seal_and_anchor_and_reopens_under_the_retained_anchor() {
    let temp = Temp::new("ok");
    let runtime = runtime();
    let (checkpoint, sealed, anchor) = targets(&temp);

    let returned = runtime
        .seal_checkpoint(
            FIXTURE,
            &declaration(),
            &EMBEDDED,
            &SealTargets {
                checkpoint: Some(&checkpoint),
                sealed: &sealed,
                anchor: &anchor,
            },
        )
        .expect("a checkpoint whose declaration resolves is saved, bound and sealed");

    assert_eq!(
        std::fs::read(&checkpoint).unwrap(),
        FIXTURE,
        "the checkpoint is saved byte for byte, not re-encoded"
    );
    assert_eq!(
        temp.names(),
        ["model.anchor.toml", "model.ckpt", "model.sealed"],
        "exactly the three files, and no scratch file beside them"
    );

    // Read the anchor back from disk the way a loader would, not from the return
    // value: the file is what a catalog retains.
    let retained = parse_retained_anchor(&std::fs::read_to_string(&anchor).unwrap())
        .expect("the retained anchor parses");
    assert_eq!(retained.anchor, returned);
    assert_eq!(retained.sealed, sealed.display().to_string());

    let state = NeuralState::open(&std::fs::read(&sealed).unwrap(), retained.anchor)
        .expect("the sealed artifact reopens under its retained anchor");
    runtime
        .admit(&state)
        .expect("and is admitted by the runtime that sealed it");
}

/// The control for the test above: a verifier that accepted any bytes would pass
/// it too. The same sealed file with one byte changed is refused under the same
/// retained anchor.
#[test]
fn a_sealed_save_with_one_byte_changed_is_refused_under_its_anchor() {
    let temp = Temp::new("tamper");
    let runtime = runtime();
    let (checkpoint, sealed, anchor) = targets(&temp);
    runtime
        .seal_checkpoint(
            FIXTURE,
            &declaration(),
            &EMBEDDED,
            &SealTargets {
                checkpoint: Some(&checkpoint),
                sealed: &sealed,
                anchor: &anchor,
            },
        )
        .unwrap();
    let retained = parse_retained_anchor(&std::fs::read_to_string(&anchor).unwrap()).unwrap();
    let mut bytes = std::fs::read(&sealed).unwrap();
    NeuralState::open(&bytes, retained.anchor).expect("intact, it reopens");

    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x01;
    let error = NeuralState::open(&bytes, retained.anchor).expect_err("tampered");
    assert!(
        format!("{error:?}").contains("AnchorMismatch"),
        "refused by the anchor's digest, got {error:?}"
    );
}

#[test]
fn a_refused_binding_writes_nothing_not_even_the_checkpoint() {
    let temp = Temp::new("refused");
    let runtime = runtime();
    let (checkpoint, sealed, anchor) = targets(&temp);

    let error = runtime
        .seal_checkpoint(
            FIXTURE,
            &StateDeclaration::at(CodebookVersion::V1).reading("a key nothing committed"),
            &EMBEDDED,
            &SealTargets {
                checkpoint: Some(&checkpoint),
                sealed: &sealed,
                anchor: &anchor,
            },
        )
        .expect_err("a declaration naming an uncommitted key cannot be bound");
    assert!(matches!(error, SealCheckpointError::Bind(_)), "{error:?}");
    assert_eq!(error.code(), "PTR_SEAL_UNBINDABLE");

    // An unbound checkpoint on disk is the state of affairs this step exists to
    // end, so a refusal must not produce one either.
    assert!(
        temp.names().is_empty(),
        "nothing may be written, got {:?}",
        temp.names()
    );
}

#[test]
fn a_save_whose_anchor_cannot_be_written_removes_the_checkpoint_and_seal_it_wrote() {
    let temp = Temp::new("anchor-fails");
    let runtime = runtime();
    let (checkpoint, sealed, anchor) = targets(&temp);
    // A non-empty directory where the anchor should go: binding, sealing and the
    // first two writes succeed, and installing the anchor cannot.
    std::fs::create_dir(&anchor).unwrap();
    std::fs::write(anchor.join("occupied"), b"x").unwrap();

    let error = runtime
        .seal_checkpoint(
            FIXTURE,
            &declaration(),
            &EMBEDDED,
            &SealTargets {
                checkpoint: Some(&checkpoint),
                sealed: &sealed,
                anchor: &anchor,
            },
        )
        .expect_err("the anchor could not be installed");
    match &error {
        SealCheckpointError::Write { path, .. } => assert_eq!(path, &anchor),
        other => panic!("expected a write failure at the anchor, got {other:?}"),
    }
    assert_eq!(
        temp.names(),
        ["model.anchor.toml"],
        "the checkpoint and seal written before the anchor are removed, and no scratch file is left"
    );
}
