//! The semantic grant: installed once and required for every host write,
//! never needed by ingress, and not history, so a rebuilt runtime has none.
//! `commit` takes no semantic record at all.
use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_ledger::{Attestation, CommittedEvent, LedgerEvent, SemanticOrigin};
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::{pod_output_key, request_raw_key};
use ptr_runtime::{PtrRuntime, RuntimeError, SemanticGrant, SemanticGrantInfo};
use ptr_semdb::{is_ingress_key, SemanticDelta, SemanticValue};
use ptr_types::{PodId, PrincipalId, ProjectId, RequestId, Revision, VerificationLevel};
use ptr_verifier::{VerificationReport, Verifier};

#[path = "common/semantic.rs"]
mod semantic_common;
use semantic_common::{
    finding, granted, host_write, operator, pass, scripted_runtime, AcceptAll, FnVerifier,
    ACCEPT_ALL,
};

fn delta(key: &str, value: &str) -> SemanticDelta {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    delta
}

fn origin_of(committed: &CommittedEvent) -> SemanticOrigin {
    match &committed.event {
        LedgerEvent::SemanticDeltaCommitted { origin, .. } => origin.clone(),
        other => panic!("{other:?}"),
    }
}

/// A verifier name that lives as long as the grant needs it to.
fn leaked(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

#[test]
fn a_second_grant_is_refused() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(runtime.semantic_grant(), None);
    granted(&mut runtime);
    let second =
        SemanticGrant::new(RequiredVerification::FullSemantic)
            .with_verifier(FnVerifier::new("other", |_| {
                pass(VerificationLevel::FullSemantic)
            }));
    assert_eq!(
        runtime.install_semantic_grant(second),
        Err(RuntimeError::SemanticGrantInstalled)
    );
    // The first grant stays.
    assert_eq!(
        runtime.semantic_grant(),
        Some(SemanticGrantInfo {
            required: RequiredVerification::Deterministic,
            verifiers: vec![ACCEPT_ALL.into()],
            host_writes: true,
        })
    );
}

#[test]
fn a_grant_with_no_verifier_duplicate_names_or_invalid_text_is_refused() {
    let named =
        |name: &'static str| FnVerifier::new(name, |_| pass(VerificationLevel::Deterministic));
    let invalid_name = "a verifier name is not an identifier of at most 64 bytes without '/'";
    let too_many = (0..=ptr_runtime::merge::MAX_SEMANTIC_VERIFIERS).fold(
        SemanticGrant::new(RequiredVerification::Deterministic),
        |grant, n| grant.with_verifier(named(leaked(format!("v{n}")))),
    );
    let cases = [
        (
            SemanticGrant::new(RequiredVerification::Deterministic),
            "a grant needs at least one verifier",
        ),
        (
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(named("same"))
                .with_verifier(named("same")),
            "two verifiers share a name",
        ),
        (
            too_many,
            "a grant has more verifiers than a record can name",
        ),
    ]
    .into_iter()
    .chain(
        [
            "",
            " padded",
            "has/slash",
            "line\nbreak",
            leaked("n".repeat(ptr_runtime::merge::MAX_VERIFIER_NAME + 1)),
        ]
        .into_iter()
        .map(|name| {
            (
                SemanticGrant::new(RequiredVerification::Deterministic).with_verifier(named(name)),
                invalid_name,
            )
        }),
    );
    for (grant, reason) in cases {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        assert_eq!(
            runtime.install_semantic_grant(grant),
            Err(RuntimeError::InvalidSemanticGrant { reason })
        );
        // A refused grant installs nothing, and a valid one still installs.
        assert_eq!(runtime.semantic_grant(), None);
        granted(&mut runtime);
    }
    // Exactly the bound installs.
    let full = (0..ptr_runtime::merge::MAX_SEMANTIC_VERIFIERS).fold(
        SemanticGrant::new(RequiredVerification::Deterministic),
        |grant, n| grant.with_verifier(named(leaked(format!("w{n}")))),
    );
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.install_semantic_grant(full).unwrap();
    assert_eq!(
        runtime.semantic_grant().unwrap().verifiers.len(),
        ptr_runtime::merge::MAX_SEMANTIC_VERIFIERS
    );
}

#[test]
fn no_grant_refuses_host_writes() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        host_write(&mut runtime, Revision(0), delta("price", "1")),
        Err(RuntimeError::NoSemanticGrant)
    );
    assert_eq!(
        runtime.apply_certified_semantic_delta(
            Revision(0),
            delta("price", "1"),
            &Default::default(),
            &operator(),
        ),
        Err(RuntimeError::NoSemanticGrant)
    );
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn host_writes_are_refused_unless_the_grant_enables_them() {
    // A grant without host writes admits none.
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic).with_verifier(AcceptAll),
        )
        .unwrap();
    assert!(!runtime.semantic_grant().unwrap().host_writes);
    assert_eq!(
        host_write(&mut runtime, Revision(0), delta("price", "1")),
        Err(RuntimeError::HostWritesNotGranted)
    );
    assert!(runtime.committed_events().is_empty());

    // With host writes on, a principal that is not provenance text, and an
    // ingress key written, removed or derived, are each refused before the
    // verifier is asked.
    let (mut runtime, verifier) = scripted_runtime(
        RequiredVerification::Deterministic,
        pass(VerificationLevel::Deterministic),
    );
    for principal in [
        String::new(),
        " padded".into(),
        "p".repeat(ptr_runtime::merge::MAX_PROVENANCE_TEXT + 1),
    ] {
        assert_eq!(
            runtime.apply_verified_semantic_delta(
                Revision(0),
                delta("price", "1"),
                &PrincipalId(principal),
            ),
            Err(RuntimeError::InvalidProvenanceText { field: "principal" })
        );
    }
    let raw = request_raw_key(&RequestId::from("r1"));
    let output = pod_output_key(&RequestId::from("r1"), &PodId::from("echo"));
    let mut removal = SemanticDelta::default();
    removal.removals.insert(output.clone());
    let mut derived = delta(&raw, "forged");
    derived.upserts.clear();
    derived
        .dependencies
        .insert(raw.clone(), ["price".to_owned()].into());
    for (written, key) in [
        (delta(&raw, "forged"), &raw),
        (removal, &output),
        (derived, &raw),
    ] {
        assert_eq!(
            host_write(&mut runtime, Revision(0), written),
            Err(RuntimeError::ReservedSemanticNamespace { key: key.clone() })
        );
    }
    assert_eq!(verifier.calls(), 0);
    assert!(runtime.committed_events().is_empty());

    // An ingress key may be a dependency input: a derived key computed from a
    // request's raw text is a host write like any other.
    runtime.ingest_text(RequestId::from("r1"), "text").unwrap();
    let mut summary = delta("summary:r1", "short");
    summary
        .dependencies
        .insert("summary:r1".into(), [raw.clone()].into());
    let revision = runtime.revision();
    host_write(&mut runtime, revision, summary).unwrap();
    assert_eq!(
        runtime.snapshot().inputs("summary:r1").collect::<Vec<_>>(),
        [raw.as_str()]
    );
}

#[test]
fn generic_commit_refuses_semantic_records() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    granted(&mut runtime);
    let raw = request_raw_key(&RequestId::from("r1"));
    let encoded = delta(&raw, "text").encode().unwrap();
    for origin in [
        SemanticOrigin::Legacy,
        SemanticOrigin::Request {
            request: "r1".into(),
        },
        SemanticOrigin::Host {
            principal: operator().0,
            verification: Attestation {
                required: VerificationLevel::Deterministic,
                level: VerificationLevel::Deterministic,
                verifiers: vec![ACCEPT_ALL.into()],
                findings: Vec::new(),
            },
        },
    ] {
        assert_eq!(
            runtime.commit(LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(0),
                revision: Revision(1),
                encoded_delta: encoded.clone(),
                origin,
            }),
            Err(RuntimeError::SemanticRecordOutsideSemanticPath)
        );
    }
    assert!(runtime.committed_events().is_empty());
    assert_eq!(runtime.revision(), Revision(0));
}

#[test]
fn ingest_text_writes_exactly_its_own_key_as_a_request_record() {
    // Ingress needs no grant: raw input is not something an interpreter may
    // refuse.
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let request = RequestId::from("r1");
    let raw = request_raw_key(&request);
    assert!(is_ingress_key(&raw));
    assert!(is_ingress_key(&pod_output_key(
        &request,
        &PodId::from("echo")
    )));
    assert_eq!(
        runtime.ingest_text(request, "the text").unwrap(),
        Revision(1)
    );

    let committed = &runtime.committed_events()[0];
    assert_eq!(
        origin_of(committed),
        SemanticOrigin::Request {
            request: "r1".into()
        }
    );
    let LedgerEvent::SemanticDeltaCommitted { encoded_delta, .. } = &committed.event else {
        unreachable!("an ingested request is a semantic record");
    };
    let recorded = SemanticDelta::decode(encoded_delta).unwrap();
    assert_eq!(recorded, delta(&raw, "the text"));
    assert_eq!(
        runtime
            .materialized_state()
            .values
            .get(ptr_state::ATTESTED_MARKER),
        Some(&"1".to_owned())
    );
    // Replay accepts what ingress wrote.
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(replayed.snapshot().get(&raw), Some("the text"));
}

struct Asks;
impl InferenceBackend for Asks {
    fn name(&self) -> &'static str {
        "semantic-grant-fixture"
    }
    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: "echo".into(),
            input_type: "bytes".into(),
            payload: vec![7, 8, 9],
        }])
    }
}

impl ResumableInferenceBackend for Asks {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

struct Echo(PodManifest);
impl DynPod for Echo {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

struct Reports(VerificationReport);
impl Verifier<TypedPayload> for Reports {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        self.0.clone()
    }
}

fn echo_pods() -> PodRegistry {
    let mut pods = PodRegistry::default();
    pods.register(Arc::new(Echo(PodManifest {
        project: ProjectId::from("p"),
        id: "echo".into(),
        capabilities: vec!["echo".into()],
        accepts: vec!["bytes".into()],
        produces: vec!["bytes".into()],
        effects: vec![ptr_types::Effect::Pure],
        protocol_version: 1,
    })));
    pods
}

/// The Pod loops that promote a Pod's output.
#[derive(Clone, Copy, Debug)]
enum PodLoop {
    Once,
    Resumable,
}

/// Run request `r1` through `pod_loop` with the echo Pod, whose output the
/// Pod verifier reports as `report`.
fn run_pod_loop(
    runtime: &mut PtrRuntime,
    pod_loop: PodLoop,
    report: VerificationReport,
) -> Result<(), RuntimeError> {
    let (request, project) = (RequestId::from("r1"), ProjectId::from("p"));
    match pod_loop {
        PodLoop::Once => runtime
            .run_model_with_pods(
                request,
                &project,
                "text",
                &Asks,
                &echo_pods(),
                &Reports(report),
            )
            .map(|_| ()),
        PodLoop::Resumable => runtime
            .run_resumable_with_pods(
                request,
                &project,
                "text",
                &Asks,
                &echo_pods(),
                &Reports(report),
                2,
            )
            .map(|_| ()),
    }
}

#[test]
fn a_pod_output_with_a_hard_finding_is_not_promoted() {
    let output_key = pod_output_key(&RequestId::from("r1"), &PodId::from("echo"));
    for pod_loop in [PodLoop::Once, PodLoop::Resumable] {
        // Pass with a hard finding: refused, and nothing but the request's
        // text is recorded.
        let mut hard = pass(VerificationLevel::SampleVerified);
        hard.findings.push(finding("unsafe", true));
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        assert_eq!(
            run_pod_loop(&mut runtime, pod_loop, hard),
            Err(RuntimeError::PodVerificationFailed { pod: "echo".into() }),
            "{pod_loop:?}"
        );
        assert_eq!(runtime.committed_events().len(), 1, "{pod_loop:?}");
        assert_eq!(
            runtime.snapshot().payload(&output_key),
            None,
            "{pod_loop:?}"
        );

        // A soft finding does not refuse it; it is promoted at the level the
        // verifier reported, which no minimum gates.
        let mut soft = pass(VerificationLevel::SampleVerified);
        soft.findings.push(finding("style", false));
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        run_pod_loop(&mut runtime, pod_loop, soft).unwrap();
        let promoted = runtime
            .committed_events()
            .iter()
            .rev()
            .find(|committed| matches!(committed.event, LedgerEvent::SemanticDeltaCommitted { .. }))
            .unwrap();
        assert_eq!(
            origin_of(promoted),
            SemanticOrigin::PodOutput {
                request: "r1".into(),
                pod: "echo".into(),
                level: VerificationLevel::SampleVerified,
            },
            "{pod_loop:?}"
        );
        let stored = runtime.snapshot().payload(&output_key).unwrap().clone();
        assert_eq!(stored.bytes, [7, 8, 9]);
        assert_eq!(stored.source, "echo");
        assert!(matches!(
            runtime.snapshot().value(&output_key),
            Some(SemanticValue::Payload(_))
        ));
        // Replay accepts it.
        assert!(PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).is_ok());
    }
}

#[test]
fn a_runtime_rebuilt_by_replay_writes_nothing_until_a_grant_is_installed() {
    let temp = std::env::temp_dir().join(format!(
        "ptr-semantic-grant-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&temp).unwrap();
    let log = temp.join("log");
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &log).unwrap();
        granted(&mut runtime);
        host_write(&mut runtime, Revision(0), delta("price", "10")).unwrap();
    }
    let history = PtrRuntime::open_durable(PtrConfig::default(), &log)
        .unwrap()
        .committed_events()
        .to_vec();
    let rebuilt = [
        PtrRuntime::replay(PtrConfig::default(), &history).unwrap(),
        PtrRuntime::open_durable(PtrConfig::default(), &log).unwrap(),
    ];
    for mut runtime in rebuilt {
        // The history replays, and names the verifier that admitted it, but
        // the grant itself was never history.
        assert_eq!(runtime.snapshot().get("price"), Some("10"));
        assert_eq!(runtime.semantic_grant(), None);
        let revision = runtime.revision();
        assert_eq!(
            host_write(&mut runtime, revision, delta("price", "11")),
            Err(RuntimeError::NoSemanticGrant)
        );
        granted(&mut runtime);
        host_write(&mut runtime, revision, delta("price", "11")).unwrap();
        assert_eq!(runtime.snapshot().get("price"), Some("11"));
    }
    let _ = std::fs::remove_dir_all(&temp);
}
