//! Sealing a checkpoint as part of saving it.
//!
//! The artifact under test is the same real `ptr-burn-a0` checkpoint the runtime's
//! own binding tests use — header from the shared kernel, weights from burn. The
//! two workspaces do not build together, so a committed artifact is the only way
//! this path is exercised against something a trainer actually produced rather
//! than against bytes a test assembled.
//!
//! What these tests are for is the *joining*: that saving and binding are one
//! operation with one outcome, and that the failure of the second does not leave
//! the first looking successful.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::neural::{NeuralAnchor, NeuralState};
use ptr_runtime::PtrRuntime;
use ptr_semdb::SemanticDelta;
use ptr_types::{CheckpointHeader, CodebookVersion, CommitIndex, Generation};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

const CHECKPOINT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/ptr-runtime/tests/fixtures/ptr-a0-v1.ckpt"
));

/// A declaration naming exactly what the journal below commits.
const DECLARATION: &str = r#"
codebook = 1
reading = ["plan"]
under = ["capsule:a"]
requires = ["semantic_role", "epistemic_state", "reasoning_operator"]

[[provenance]]
source = "trainer:a0"
note = "sealed as part of saving"
"#;

/// The retained anchor, read back the way a catalog would read it.
#[derive(Deserialize)]
struct AnchorFile {
    codebook: u32,
    journal_index: u64,
    journal_digest: String,
    digest: String,
}

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ptrctl-seal-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("temp dir");
        Self(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A journal committing the one semantic value and the one generation the
/// declaration names.
fn journal(temp: &Temp) -> PathBuf {
    let path = temp.at("log");
    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).expect("open journal");
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("plan".into(), "original plan".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .expect("commit the semantic value the trainer read");
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:a".into(),
            generation: Generation(1),
        })
        .expect("commit the generation the trainer trained under");
    drop(runtime);
    path
}

fn write(path: &Path, text: &str) {
    std::fs::write(path, text).expect("write fixture");
}

struct Run {
    status: i32,
    stderr: String,
}

fn seal(temp: &Temp, journal: &Path, declaration: &Path) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_ptrctl"))
        .args(["seal", "--journal"])
        .arg(journal)
        .arg("--checkpoint")
        .arg(temp.at("model.ckpt"))
        .arg("--declaration")
        .arg(declaration)
        .arg("--out")
        .arg(temp.at("model.sealed"))
        .arg("--anchor")
        .arg(temp.at("model.anchor.toml"))
        .output()
        .expect("run ptrctl seal");
    Run {
        status: output.status.code().expect("an exit code"),
        stderr: String::from_utf8(output.stderr).expect("utf-8 stderr"),
    }
}

/// Everything a successful seal needs on disk, with the checkpoint already saved.
fn saved(temp: &Temp, declaration_text: &str) -> (PathBuf, PathBuf) {
    write(&temp.at("model.ckpt"), "");
    std::fs::write(temp.at("model.ckpt"), CHECKPOINT).expect("save the checkpoint");
    let declaration = temp.at("declaration.toml");
    write(&declaration, declaration_text);
    (journal(temp), declaration)
}

fn anchor_of(temp: &Temp) -> NeuralAnchor {
    let text = std::fs::read_to_string(temp.at("model.anchor.toml")).expect("read anchor");
    let file: AnchorFile = toml::from_str(&text).expect("the anchor parses");
    let mut journal_digest = [0u8; 32];
    let mut digest = [0u8; 32];
    unhex(&file.journal_digest, &mut journal_digest);
    unhex(&file.digest, &mut digest);
    NeuralAnchor {
        journal: ptr_ledger::integrity::LogAnchor {
            index: CommitIndex(file.journal_index),
            digest: journal_digest,
        },
        codebook: CodebookVersion(file.codebook),
        digest,
    }
}

fn unhex(text: &str, into: &mut [u8; 32]) {
    assert_eq!(text.len(), 64, "a digest is 32 bytes");
    for (slot, pair) in into.iter_mut().zip(text.as_bytes().chunks(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).expect("hex");
    }
}

#[test]
fn a_saved_checkpoint_is_sealed_against_the_journal_and_reopens_under_its_anchor() {
    let temp = Temp::new("ok");
    let (journal_path, declaration) = saved(&temp, DECLARATION);

    let run = seal(&temp, &journal_path, &declaration);
    assert_eq!(run.status, 0, "sealing failed: {}", run.stderr);

    let sealed = std::fs::read(temp.at("model.sealed")).expect("the sealed artifact is written");
    let anchor = anchor_of(&temp);

    // The anchor is evidence only because it was retained separately; reopening
    // proves the sealed bytes are the ones it describes.
    let state = NeuralState::open(&sealed, anchor).expect("the seal reopens under its anchor");

    let (_, payload) = CheckpointHeader::read(CHECKPOINT).expect("the fixture is well formed");
    assert_eq!(
        state.payload_len(),
        payload.len(),
        "sealing must retain the weights it was given, not a summary of them"
    );
    assert_eq!(
        state.binding().codebook,
        CodebookVersion(1),
        "the binding records the assignment the artifact was trained under"
    );
    assert!(
        state.binding().semantic_inputs.contains_key("plan"),
        "the declared semantic input must be resolved against committed state"
    );
    assert_eq!(
        state.binding().generations.get("capsule:a"),
        Some(&Generation(1)),
        "the generation that was live is what the binding pins"
    );
}

#[test]
fn a_sealed_checkpoint_is_admitted_by_the_runtime_that_sealed_it() {
    let temp = Temp::new("admit");
    let (journal_path, declaration) = saved(&temp, DECLARATION);
    assert_eq!(seal(&temp, &journal_path, &declaration).status, 0);

    let sealed = std::fs::read(temp.at("model.sealed")).unwrap();
    let state = NeuralState::open(&sealed, anchor_of(&temp)).expect("reopens");
    let runtime = PtrRuntime::open_durable(PtrConfig::default(), &journal_path).expect("reopen");
    runtime
        .admit(&state)
        .expect("a state sealed against this history is admitted by it");
}

/// The control. The test above would pass just as well against a binding that
/// recorded nothing, so the observable has to be shown to be responsive: move the
/// generation the checkpoint was bound under, and the same sealed bytes are
/// refused by the same runtime.
#[test]
fn moving_the_generation_it_was_bound_under_makes_the_same_seal_inadmissible() {
    let temp = Temp::new("control");
    let (journal_path, declaration) = saved(&temp, DECLARATION);
    assert_eq!(seal(&temp, &journal_path, &declaration).status, 0);

    let sealed = std::fs::read(temp.at("model.sealed")).unwrap();
    let state = NeuralState::open(&sealed, anchor_of(&temp)).expect("reopens");

    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &journal_path).expect("open");
    runtime
        .admit(&state)
        .expect("admitted before the generation moves");
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:a".into(),
            generation: Generation(2),
        })
        .expect("move the generation");

    runtime
        .admit(&state)
        .expect_err("a checkpoint bound under generation 1 must not be admitted under 2");
}

#[test]
fn a_declaration_naming_a_value_the_journal_does_not_have_writes_nothing() {
    let temp = Temp::new("absent");
    let (journal_path, declaration) = saved(
        &temp,
        r#"
codebook = 1
reading = ["a key nothing ever committed"]
under = ["capsule:a"]
"#,
    );

    let run = seal(&temp, &journal_path, &declaration);
    assert_eq!(run.status, 1, "an unresolvable declaration must be refused");

    // The point of the refusal is what is *not* on disk. A sealed file beside the
    // checkpoint would read as a successful binding to anything that found it.
    assert!(
        !temp.at("model.sealed").exists(),
        "a refused binding must not leave a sealed artifact behind"
    );
    assert!(
        !temp.at("model.anchor.toml").exists(),
        "a refused binding must not leave an anchor behind"
    );
}

#[test]
fn a_declaration_disagreeing_with_the_artifact_about_the_codebook_is_refused() {
    let temp = Temp::new("codebook");
    let (journal_path, declaration) = saved(
        &temp,
        r#"
codebook = 2
reading = ["plan"]
under = ["capsule:a"]
"#,
    );

    let run = seal(&temp, &journal_path, &declaration);
    assert_eq!(
        run.status, 1,
        "one of the two is wrong about what every code means"
    );
    assert!(
        !temp.at("model.sealed").exists(),
        "nothing is written when the assignment is in dispute"
    );
}

#[test]
fn a_declaration_naming_a_family_this_build_does_not_define_is_refused() {
    let temp = Temp::new("family");
    let (journal_path, declaration) = saved(
        &temp,
        r#"
codebook = 1
reading = ["plan"]
under = ["capsule:a"]
requires = ["not_a_family"]
"#,
    );

    let run = seal(&temp, &journal_path, &declaration);
    assert_eq!(run.status, 1);
    assert!(
        run.stderr.contains("unknown code family"),
        "the refusal must name what it did not understand, got: {}",
        run.stderr
    );
    assert!(!temp.at("model.sealed").exists());
}
