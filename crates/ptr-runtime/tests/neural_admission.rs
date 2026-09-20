//! Admission for opaque model/KV/checkpoint state: does an admitted state
//! reproduce exactly the same inference after a restart, and is every way of
//! getting a stale state back in refused?
//!
//! The positive direction has to be exact — the same bytes, the same events — or
//! "the cache still works" is unproven. The negative direction has to cover every
//! route back in, because a cache that denies on revocation but not on an edit is
//! not a boundary.
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::LedgerEvent;
use ptr_model_api::{InferenceBackend, ModelError, ModelEvent, ModelRequest};
use ptr_runtime::execution::{
    ActionExecutor, ActionScope, ExecutionGrant, RequiredVerification, VerifiedDispatch,
};
use ptr_runtime::neural::{
    Denial, NeuralAnchor, NeuralError, NeuralState, NeuralStateCache, StateDeclaration,
};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    CapabilityId, CodebookVersion, CommitIndex, Effect, Generation, Probability, ProjectId,
    ProvenanceRef, Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-neural-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn log(&self) -> PathBuf {
        self.0.join("log")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Committed history a neural state can be bound to: two semantic values and two
/// capsule generations, one of which later moves.
fn commit_history(runtime: &mut PtrRuntime) {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("plan".into(), "original plan".into());
    delta.upserts.insert("unrelated".into(), "keep".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:b".into(),
            generation: Generation(1),
        })
        .unwrap();
}

fn fixture() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    commit_history(&mut runtime);
    runtime
}

fn declaration() -> StateDeclaration {
    StateDeclaration::at(CodebookVersion::V1)
        .reading("plan")
        .under("capsule:a")
        .produced_by(ProvenanceRef {
            source: "trainer:a0".into(),
            note: Some("round 1".to_owned()),
        })
}

/// Stand-in for warm neural state. Structured rather than uniform so a single-bit
/// change is a real change everywhere in it.
fn payload() -> Vec<u8> {
    (0..96u32).map(|index| (index * 31 % 251) as u8).collect()
}

fn state(runtime: &PtrRuntime) -> NeuralState {
    NeuralState::new(runtime.bind_state(&declaration()).unwrap(), payload())
}

/// A deterministic function of admitted state and prompt.
///
/// It is not a model and not evidence about model quality; it exists so that
/// "the same state produces the same inference" is an assertion about bytes rather
/// than a claim.
struct CachedBackend(Vec<u8>);

impl InferenceBackend for CachedBackend {
    fn name(&self) -> &'static str {
        "cached-reference"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for byte in self.0.iter().chain(request.raw_text.as_bytes()) {
            digest ^= u64::from(*byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(vec![
            ModelEvent::Token(format!("{digest:016x}")),
            ModelEvent::ConfidenceUpdated {
                slot: 0,
                confidence: Probability::new(((digest % 1000) as f32) / 1000.0).unwrap(),
            },
            ModelEvent::Finished,
        ])
    }
}

/// One framing violation: how to break a sealed artifact, and what opening the
/// result must refuse with.
type Violation = (NeuralError, Box<dyn Fn(Vec<u8>) -> Vec<u8>>);

fn reseal(mut bytes: Vec<u8>, anchor: NeuralAnchor) -> (Vec<u8>, NeuralAnchor) {
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    (bytes, NeuralAnchor { digest, ..anchor })
}

#[test]
fn an_admitted_state_reproduces_the_same_inference_after_a_restart() {
    let tmp = Temp::new();
    let (before, sealed, anchor, binding) = {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
        commit_history(&mut runtime);
        let held = state(&runtime);
        let sealed = held.seal().unwrap();
        let admitted = runtime.admit(&held).unwrap();
        // The admitted handle borrows the state, not the runtime, so a caller can
        // hold the payload and still drive the runtime that cleared it.
        let backend = CachedBackend(admitted.payload().to_vec());
        let before = runtime
            .run_model_once("r1".into(), "what is the plan", &backend)
            .unwrap();
        (
            before,
            sealed.bytes().to_vec(),
            sealed.anchor(),
            held.binding().clone(),
        )
    };

    let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    let reopened = NeuralState::open(&sealed, anchor).unwrap();
    // Exact state: the binding and the opaque payload survive byte-for-byte.
    assert_eq!(reopened.binding(), &binding);
    let admitted = runtime.admit(&reopened).unwrap();
    assert_eq!(admitted.payload(), payload().as_slice());
    assert_eq!(admitted.binding().revision, Revision(1));
    assert_eq!(admitted.binding().journal.index, CommitIndex(3));

    let backend = CachedBackend(admitted.payload().to_vec());
    let after = runtime
        .run_model_once("r1".into(), "what is the plan", &backend)
        .unwrap();
    assert_eq!(before, after);
    // The state was bound below the runtime's current position — the model run
    // before the restart committed a record — and is admitted anyway, because the
    // prefix it names still agrees.
    assert!(runtime.committed_events().len() > 3);
}

#[test]
fn a_revoked_generation_cannot_be_readmitted_after_a_restart() {
    let tmp = Temp::new();
    let (sealed, anchor) = {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
        commit_history(&mut runtime);
        let held = state(&runtime);
        assert!(runtime.admit(&held).is_ok());
        runtime
            .commit(LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(1),
            })
            .unwrap();
        let sealed = held.seal().unwrap();
        (sealed.bytes().to_vec(), sealed.anchor())
    };

    let runtime = PtrRuntime::open_durable(PtrConfig::default(), tmp.log()).unwrap();
    // The artifact is intact and matches its retained anchor: the refusal below is
    // about admission, not corruption.
    let reopened = NeuralState::open(&sealed, anchor).unwrap();
    assert_eq!(
        runtime.admit(&reopened).map(|_| ()),
        Err(Denial::RevokedGeneration {
            target: "capsule:a".to_owned(),
            generation: Generation(1),
        })
    );
    // Rebinding is refused too, so the state cannot be laundered into a fresh
    // binding at the current position.
    assert_eq!(
        runtime.bind_state(&declaration()),
        Err(RuntimeError::Neural(NeuralError::Unbindable(
            Denial::RevokedGeneration {
                target: "capsule:a".to_owned(),
                generation: Generation(1),
            }
        )))
    );
}

#[test]
fn revocation_denial_precedes_every_other_binding_mismatch() {
    let mut runtime = fixture();
    let mut binding = runtime.bind_state(&declaration()).unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();

    // All of these fields are independently invalid. Revocation must still be
    // the verdict because a tombstone is authoritative even when the rest of the
    // artifact cannot be interpreted or its history cannot be verified.
    binding.codebook = CodebookVersion(99);
    binding.codebook_fingerprint = [0; 32];
    binding.journal = LogAnchor {
        index: CommitIndex(u64::MAX),
        digest: [0xff; 32],
    };
    binding.revision = Revision(u64::MAX);
    binding.semantic_inputs.insert("missing".into(), [0; 32]);

    assert_eq!(
        runtime.admission(&binding),
        Err(Denial::RevokedGeneration {
            target: "capsule:a".to_owned(),
            generation: Generation(1),
        })
    );
}

#[test]
fn an_edited_input_denies_while_an_unrelated_commit_does_not() {
    let mut runtime = fixture();
    let held = state(&runtime);

    let mut delta = SemanticDelta::default();
    delta.upserts.insert("unrelated".into(), "changed".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    // Dependency-precise: a commit that touches nothing the state read leaves it
    // admissible. Denying here instead would make the whole mechanism useless.
    assert!(runtime.admit(&held).is_ok());

    let mut delta = SemanticDelta::default();
    delta.upserts.insert("plan".into(), "revised plan".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    assert_eq!(
        runtime.admit(&held).map(|_| ()),
        Err(Denial::EditedInput {
            key: "plan".to_owned()
        })
    );
}

#[test]
fn a_removed_input_denies() {
    let mut runtime = fixture();
    let held = state(&runtime);
    let mut delta = SemanticDelta::default();
    delta.removals.insert("plan".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    // Removal commits a record; the value stops being current, and a state that
    // consumed it stops being admissible.
    assert_eq!(
        runtime.admit(&held).map(|_| ()),
        Err(Denial::MissingInput {
            key: "plan".to_owned()
        })
    );
}

#[test]
fn a_superseded_generation_denies_although_the_live_one_is_newer() {
    let mut runtime = fixture();
    let held = state(&runtime);
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: "capsule:a".into(),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    // Equality, not "at least": state computed under generation 1 describes
    // generation 1, and generation 2 is a different thing rather than a newer
    // view of the same one.
    assert_eq!(
        runtime.admit(&held).map(|_| ()),
        Err(Denial::StaleGeneration {
            target: "capsule:a".to_owned(),
            bound: Generation(1),
            current: Generation(2),
        })
    );
}

#[test]
fn a_cache_decides_admission_at_every_lookup_not_at_insertion() {
    let mut runtime = fixture();
    let mut cache = NeuralStateCache::default();
    cache.insert("warm", state(&runtime));
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.admit(&runtime, "warm").unwrap().payload(),
        payload().as_slice()
    );
    assert!(cache.denied(&runtime).is_empty());
    assert_eq!(
        cache.admit(&runtime, "cold").map(|_| ()),
        Err(Denial::Absent {
            key: "cold".to_owned()
        })
    );

    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();

    // Nothing touched the cache, and the entry is still there — but the lookup is
    // refused, because the decision is taken now and not when it was stored.
    assert_eq!(cache.len(), 1);
    assert!(cache.binding("warm").is_some());
    let expected = Denial::RevokedGeneration {
        target: "capsule:a".to_owned(),
        generation: Generation(1),
    };
    assert_eq!(
        cache.admit(&runtime, "warm").map(|_| ()),
        Err(expected.clone())
    );
    assert_eq!(
        cache.denied(&runtime),
        vec![("warm".to_owned(), expected.clone())]
    );
    assert_eq!(
        cache.evict_denied(&runtime),
        vec![("warm".to_owned(), expected)]
    );
    assert!(cache.is_empty());
}

#[test]
fn a_foreign_history_with_the_same_counters_is_denied() {
    let mine = fixture();
    let held = state(&mine);

    let mut theirs = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("plan".into(), "original plan".into());
    delta.upserts.insert("unrelated".into(), "other".into());
    theirs
        .apply_semantic_delta(theirs.revision(), delta)
        .unwrap();
    theirs
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    theirs
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:b".into(),
            generation: Generation(1),
        })
        .unwrap();

    // Same commit position, same revision, same live generations, same bound
    // input value — and still a different history.
    assert_eq!(theirs.revision(), mine.revision());
    assert_eq!(
        theirs.committed_events().len(),
        mine.committed_events().len()
    );
    assert_eq!(theirs.live_generation("capsule:a"), Generation(1).into());
    assert_eq!(
        theirs.admit(&held).map(|_| ()),
        Err(Denial::ForeignHistory { at: CommitIndex(3) })
    );
}

#[test]
fn a_position_this_runtime_cannot_check_is_denied() {
    let runtime = fixture();
    let held = state(&runtime);
    let events = runtime.committed_events().to_vec();

    let behind = PtrRuntime::replay(PtrConfig::default(), &events[..1]).unwrap();
    assert_eq!(
        behind.admit(&held).map(|_| ()),
        Err(Denial::UnverifiablePosition {
            bound: CommitIndex(3),
            from: CommitIndex(0),
            through: CommitIndex(1),
        })
    );
}

#[test]
fn a_compacted_restore_holds_no_chain_base_and_admits_nothing() {
    let runtime = fixture();
    let binding = runtime.bind_state(&declaration()).unwrap();
    let snapshot = runtime.export_compacted_snapshot().unwrap();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();

    // The restore kept committed semantic and lifecycle state, so every content
    // check would pass — and admission is still refused, because the records that
    // would prove the bound position are gone and no chain base replaced them.
    assert_eq!(restored.revision(), runtime.revision());
    assert_eq!(restored.live_generation("capsule:a"), Generation(1).into());
    assert_eq!(restored.admission(&binding), Err(Denial::UnanchoredHistory));
}

#[test]
fn a_recorded_revision_that_contradicts_its_position_is_denied() {
    let runtime = fixture();
    let mut binding = runtime.bind_state(&declaration()).unwrap();
    assert_eq!(runtime.admission(&binding), Ok(()));
    binding.revision = Revision(99);
    assert_eq!(
        runtime.admission(&binding),
        Err(Denial::RevisionMismatch {
            bound: Revision(99),
            current: Revision(1),
        })
    );
}

#[test]
fn a_codebook_the_build_lacks_or_no_longer_matches_is_denied() {
    let runtime = fixture();
    let valid = runtime.bind_state(&declaration()).unwrap();

    let mut unknown = valid.clone();
    unknown.codebook = CodebookVersion(9);
    assert_eq!(
        runtime.admission(&unknown),
        Err(Denial::UnknownCodebookVersion {
            version: CodebookVersion(9)
        })
    );

    // Same version, different assignment: the tables behind V1 were edited after
    // this state was produced, so its integer identities mean something else now.
    let mut changed = valid.clone();
    changed.codebook_fingerprint = [0; 32];
    assert_eq!(
        runtime.admission(&changed),
        Err(Denial::CodebookAssignmentChanged {
            version: CodebookVersion::V1
        })
    );

    let mut declared = declaration();
    declared.codebook = CodebookVersion(9);
    assert_eq!(
        runtime.bind_state(&declared),
        Err(RuntimeError::Neural(NeuralError::UnsupportedVersion))
    );
}

#[test]
fn a_binding_cannot_name_state_the_runtime_has_no_facts_for() {
    let runtime = fixture();
    let valid = runtime.bind_state(&declaration()).unwrap();

    let mut unknown = valid.clone();
    unknown
        .generations
        .insert("capsule:absent".to_owned(), Generation(1));
    assert_eq!(
        runtime.admission(&unknown),
        Err(Denial::UnknownTarget {
            target: "capsule:absent".to_owned()
        })
    );

    assert_eq!(
        runtime.bind_state(&declaration().under("capsule:absent")),
        Err(RuntimeError::Neural(NeuralError::UndeclarableTarget {
            target: "capsule:absent".to_owned()
        }))
    );
    assert_eq!(
        runtime.bind_state(&declaration().reading("no-such-key")),
        Err(RuntimeError::Neural(NeuralError::UndeclarableInput {
            key: "no-such-key".to_owned()
        }))
    );
}

struct RefusingVerifier;
impl Verifier<ActionIr> for RefusingVerifier {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

struct AmbiguousExecutor;
impl ActionExecutor for AmbiguousExecutor {
    fn execute(&self, _: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
        Err("outcome uncertain".into())
    }
}

#[test]
fn a_fenced_runtime_admits_nothing_and_binds_nothing() {
    let mut runtime = fixture();
    let held = state(&runtime);
    let binding = held.binding().clone();

    let action = ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: runtime.revision(),
        payload: b"payload".to_vec(),
    };
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;

    let session = runtime
        .register_execution_session(
            "alice",
            vec![ExecutionGrant::new(
                ActionScope {
                    project: ProjectId::from("p"),
                    target: action.target.clone(),
                    operation: action.operation.clone(),
                    capability: action.capability.clone(),
                    input_type: action.input_type.clone(),
                    effect: action.effect,
                },
                RequiredVerification::Deterministic,
                RefusingVerifier,
                AmbiguousExecutor,
            )],
            Duration::from_secs(60),
        )
        .unwrap();
    let permit = runtime
        .prepare_execution(
            &session,
            &ProjectId::from("p"),
            &action,
            Duration::from_secs(60),
        )
        .unwrap();
    assert!(runtime.execute_prepared(&session, permit).is_err());

    // An ambiguous outcome means this runtime cannot vouch for its own history, so
    // it can neither clear a state nor produce a new binding.
    assert_eq!(runtime.admission(&binding), Err(Denial::Fenced));
    assert_eq!(runtime.admit(&held).map(|_| ()), Err(Denial::Fenced));
    assert_eq!(
        runtime.bind_state(&declaration()),
        Err(RuntimeError::ExecutionFenced)
    );
}

#[test]
fn every_single_bit_mutation_of_a_sealed_state_is_rejected() {
    let runtime = fixture();
    let sealed = state(&runtime).seal().unwrap();
    let anchor = sealed.anchor();
    assert!(NeuralState::open(sealed.bytes(), anchor).is_ok());

    for index in 0..sealed.bytes().len() {
        for bit in 0..8u32 {
            let mut bytes = sealed.bytes().to_vec();
            bytes[index] ^= 1 << bit;
            assert!(
                NeuralState::open(&bytes, anchor).is_err(),
                "byte {index} bit {bit}"
            );
        }
    }
}

#[test]
fn framing_violations_are_rejected_after_the_digest_is_resealed() {
    let runtime = fixture();
    let sealed = state(&runtime).seal().unwrap();
    let anchor = sealed.anchor();
    let original = sealed.bytes().to_vec();

    let cases: Vec<Violation> = vec![
        (
            NeuralError::UnsupportedVersion,
            Box::new(|mut bytes: Vec<u8>| {
                bytes[7] = b'2';
                bytes
            }),
        ),
        (
            NeuralError::ReservedField,
            Box::new(|mut bytes: Vec<u8>| {
                bytes[24] = 1;
                bytes
            }),
        ),
        (
            NeuralError::LengthMismatch,
            Box::new(|mut bytes: Vec<u8>| {
                let binding_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
                bytes[8..16].copy_from_slice(&(binding_len - 1).to_le_bytes());
                bytes
            }),
        ),
        (
            NeuralError::LengthMismatch,
            Box::new(|mut bytes: Vec<u8>| {
                bytes.remove(40);
                bytes
            }),
        ),
        (
            NeuralError::LengthMismatch,
            Box::new(|mut bytes: Vec<u8>| {
                bytes.insert(40, 0);
                bytes
            }),
        ),
        (
            // The binding section carries its own magic, so a body that is framed
            // correctly but is not a binding is still refused.
            NeuralError::UnsupportedVersion,
            Box::new(|mut bytes: Vec<u8>| {
                bytes[39] = b'9';
                bytes
            }),
        ),
    ];

    for (index, (expected, mutate)) in cases.into_iter().enumerate() {
        let (bytes, anchor) = reseal(mutate(original.clone()), anchor);
        assert_eq!(
            NeuralState::open(&bytes, anchor),
            Err(RuntimeError::Neural(expected)),
            "case {index}"
        );
    }

    // Too short to hold a header and a digest at all.
    assert_eq!(
        NeuralState::open(&original[..16], anchor),
        Err(RuntimeError::Neural(NeuralError::SizeLimit))
    );
}
