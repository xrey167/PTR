//! Resolving a stable pod identity through its committed head.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{
    pod_lifecycle_target, AdvancePodHead, InMemoryPodDirectory, PodDirectory, PtrRuntime,
};
use ptr_types::{
    Generation, PodAddress, PodEndpoint, PodIdentity, PodRevisionAddress, PodTransport,
    ResolveError, RouteConstraints,
};

fn endpoint(generation: u64) -> PodEndpoint {
    PodEndpoint {
        pod: PodRevisionAddress {
            address: PodAddress::new("project".into(), "namespace".into(), "pod".into()).unwrap(),
            semantic_revision: [generation as u8 + 3; 32],
            generation: Generation(generation),
        },
        artifact_id: format!("artifact-{generation}").as_str().into(),
        artifact_active: true,
        node_id: "node-a".into(),
        device_id: Some("cuda0".into()),
        region: "eu-central".into(),
        zone: "eu-central-a".into(),
        peer_id: "peer-a".into(),
        transport: PodTransport::IrohQuic,
        endpoint_epoch: 2,
        fencing_token: 2,
        available_vram_bytes: 1 << 30,
        healthy: true,
        load_bps: 100,
        capabilities: vec!["infer".into()],
        accepts: vec!["tokens".into()],
        produces: vec!["answer".into()],
        mesh: None,
    }
}

fn constraints() -> RouteConstraints {
    RouteConstraints {
        capability: Some("infer".into()),
        input_type: Some("tokens".into()),
        ..RouteConstraints::default()
    }
}

fn identity() -> PodIdentity {
    PodIdentity::from("pod:stable")
}

#[test]
fn an_identity_without_a_head_does_not_resolve() {
    let mut directory = InMemoryPodDirectory::default();
    directory.register(endpoint(1)).unwrap();
    let runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(matches!(
        directory.resolve_head(&runtime, &identity(), &constraints()),
        Err(ResolveError::NotFound)
    ));
}

#[test]
fn an_identity_resolves_to_the_endpoint_of_its_current_head() {
    let mut directory = InMemoryPodDirectory::default();
    let (first, second) = (endpoint(1), endpoint(2));
    directory.register(first.clone()).unwrap();
    directory.register(second.clone()).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .advance_pod_head(AdvancePodHead {
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    let route = directory
        .resolve_head(&runtime, &identity(), &constraints())
        .unwrap();
    assert_eq!(route.destination, first.pod);
    assert_eq!(route.endpoint, first);

    runtime
        .advance_pod_head(AdvancePodHead {
            identity: identity(),
            expected: Some(first.pod.clone()),
            new: second.pod.clone(),
        })
        .unwrap();
    let route = directory
        .resolve_head(&runtime, &identity(), &constraints())
        .unwrap();
    assert_eq!(route.destination, second.pod);
}

#[test]
fn a_revoked_head_does_not_resolve_and_no_older_revision_answers() {
    let mut directory = InMemoryPodDirectory::default();
    let (first, second) = (endpoint(1), endpoint(2));
    directory.register(first.clone()).unwrap();
    directory.register(second.clone()).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .advance_pod_head(AdvancePodHead {
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    runtime
        .advance_pod_head(AdvancePodHead {
            identity: identity(),
            expected: Some(first.pod.clone()),
            new: second.pod.clone(),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&second.pod),
            generation: Generation(2),
        })
        .unwrap();
    assert!(matches!(
        directory.resolve_head(&runtime, &identity(), &constraints()),
        Err(ResolveError::NotFound)
    ));
}
