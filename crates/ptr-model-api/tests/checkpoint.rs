use ptr_model_api::{
    CheckpointError, CheckpointHandle, CheckpointableBackend, ModelEvent, ModelObservation,
    ModelResumeRequest, ReferenceEchoBackend, ScenarioBackend, CHECKPOINT_VERSION,
};
use ptr_types::{RequestId, Revision, SemanticContext, TypeId};

fn context(rev: u64) -> SemanticContext {
    SemanticContext {
        revision: Revision(rev),
        entries: vec![],
    }
}

fn resume_request() -> ModelResumeRequest {
    ModelResumeRequest {
        request_id: RequestId::from("req-1"),
        semantic_context: context(5),
        round: 2,
        observation: ModelObservation {
            revision: Revision(5),
            source: "tool".into(),
            type_id: TypeId::from("ptr.t.v1"),
            payload: vec![1, 2, 3],
        },
    }
}

#[test]
fn export_import_resume_round_trip_for_both_backends() {
    let request = resume_request();
    let echo = ReferenceEchoBackend;
    let handle = echo.export_checkpoint(&request).unwrap();
    assert_eq!(handle.backend_id(), "reference-echo");
    assert_eq!(handle.version(), CHECKPOINT_VERSION);
    let wire = handle.to_bytes();
    let parsed = CheckpointHandle::from_bytes(&wire).unwrap();
    assert_eq!(
        echo.resume_from_checkpoint(&parsed, context(5)).unwrap(),
        vec![ModelEvent::Finished]
    );

    let scenario = ScenarioBackend;
    let handle = scenario.export_checkpoint(&request).unwrap();
    assert_eq!(handle.backend_id(), "scenario");
    assert_eq!(
        scenario
            .resume_from_checkpoint(&handle, context(5))
            .unwrap(),
        vec![ModelEvent::Finished]
    );
}

#[test]
fn wrong_backend_is_rejected() {
    let handle = ReferenceEchoBackend
        .export_checkpoint(&resume_request())
        .unwrap();
    assert_eq!(
        ScenarioBackend.resume_from_checkpoint(&handle, context(5)),
        Err(CheckpointError::WrongBackend {
            expected: "scenario".into(),
            found: "reference-echo".into()
        })
    );
}

#[test]
fn state_tamper_is_a_digest_mismatch() {
    let handle = ReferenceEchoBackend
        .export_checkpoint(&resume_request())
        .unwrap();
    let mut wire = handle.to_bytes();
    // flip a byte inside the sealed state (just before the trailing 32-byte digest)
    let idx = wire.len() - 33;
    wire[idx] ^= 0xFF;
    let tampered = CheckpointHandle::from_bytes(&wire).unwrap();
    assert_eq!(
        ReferenceEchoBackend.resume_from_checkpoint(&tampered, context(5)),
        Err(CheckpointError::DigestMismatch)
    );
}

#[test]
fn digest_tamper_is_a_digest_mismatch() {
    let handle = ReferenceEchoBackend
        .export_checkpoint(&resume_request())
        .unwrap();
    let mut wire = handle.to_bytes();
    let last = wire.len() - 1;
    wire[last] ^= 1;
    let tampered = CheckpointHandle::from_bytes(&wire).unwrap();
    assert_eq!(
        tampered.verify("reference-echo"),
        Err(CheckpointError::DigestMismatch)
    );
}

#[test]
fn version_tamper_is_a_version_error() {
    let handle = ReferenceEchoBackend
        .export_checkpoint(&resume_request())
        .unwrap();
    let mut wire = handle.to_bytes();
    wire[4] = 9; // version field, little endian, right after the magic
    let tampered = CheckpointHandle::from_bytes(&wire).unwrap();
    assert_eq!(
        tampered.verify("reference-echo"),
        Err(CheckpointError::UnsupportedVersion {
            found: 9,
            supported: CHECKPOINT_VERSION
        })
    );
}

#[test]
fn backend_id_tamper_in_bytes_is_detected() {
    let handle = CheckpointHandle::seal("abc", vec![1]).unwrap();
    let mut wire = handle.to_bytes();
    wire[12] = b'x'; // first id byte: 4 magic + 4 version + 4 id length
    let tampered = CheckpointHandle::from_bytes(&wire).unwrap();
    assert!(matches!(
        tampered.verify("abc"),
        Err(CheckpointError::WrongBackend { .. })
    ));
    assert_eq!(tampered.verify("xbc"), Err(CheckpointError::DigestMismatch));
}

#[test]
fn malformed_bytes_are_typed_errors() {
    let wire = CheckpointHandle::seal("abc", vec![1, 2])
        .unwrap()
        .to_bytes();
    assert!(matches!(
        CheckpointHandle::from_bytes(&wire[..wire.len() - 1]),
        Err(CheckpointError::Malformed(_))
    ));
    let mut extra = wire.clone();
    extra.push(0);
    assert_eq!(
        CheckpointHandle::from_bytes(&extra),
        Err(CheckpointError::Malformed("trailing bytes"))
    );
    let mut bad_magic = wire;
    bad_magic[0] = b'X';
    assert_eq!(
        CheckpointHandle::from_bytes(&bad_magic),
        Err(CheckpointError::Malformed("bad magic"))
    );
    assert!(matches!(
        CheckpointHandle::from_bytes(&[]),
        Err(CheckpointError::Malformed(_))
    ));
}

#[test]
fn resume_with_a_different_revision_is_refused() {
    let handle = ReferenceEchoBackend
        .export_checkpoint(&resume_request())
        .unwrap();
    assert_eq!(
        ReferenceEchoBackend.resume_from_checkpoint(&handle, context(6)),
        Err(CheckpointError::RevisionMismatch {
            recorded: Revision(5),
            supplied: Revision(6)
        })
    );
}

#[test]
fn well_sealed_but_garbage_state_is_malformed() {
    let handle = CheckpointHandle::seal("reference-echo", vec![1, 2, 3]).unwrap();
    assert!(matches!(
        ReferenceEchoBackend.resume_from_checkpoint(&handle, context(1)),
        Err(CheckpointError::Malformed(_))
    ));
}
