//! Resolving a stable pod identity through its committed head.
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{
    pod_lifecycle_target, AdvancePodHead, InMemoryPodDirectory, PodDirectory, PtrRuntime,
};
use ptr_types::{
    CapsuleId, Generation, PodAddress, PodEndpoint, PodIdentity, PodRevisionAddress, PodTransport,
    ProjectId, ResolveError, RouteConstraints,
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

fn project() -> ProjectId {
    ProjectId::from("project")
}

/// Makes `revision`'s generation the live one for its pod in the lifecycle.
fn make_live(runtime: &mut PtrRuntime, revision: &PodRevisionAddress) {
    let target = pod_lifecycle_target(revision);
    let event = match runtime.live_generation(&target) {
        None => LedgerEvent::CapsuleCommitted {
            project: "project".into(),
            capsule: CapsuleId::from(target.as_str()),
            generation: revision.generation,
        },
        Some(old) => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(target.as_str()),
            old,
            new: revision.generation,
        },
    };
    runtime.commit(event).unwrap();
}

#[test]
fn an_identity_without_a_head_does_not_resolve() {
    let mut directory = InMemoryPodDirectory::default();
    directory.register(endpoint(1)).unwrap();
    let runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(matches!(
        directory.resolve_head(&runtime, &project(), &identity(), &constraints()),
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
    make_live(&mut runtime, &first.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    let route = directory
        .resolve_head(&runtime, &project(), &identity(), &constraints())
        .unwrap();
    assert_eq!(route.destination, first.pod);
    assert_eq!(route.endpoint, first);

    make_live(&mut runtime, &second.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
            identity: identity(),
            expected: Some(first.pod.clone()),
            new: second.pod.clone(),
        })
        .unwrap();
    let route = directory
        .resolve_head(&runtime, &project(), &identity(), &constraints())
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
    make_live(&mut runtime, &first.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    make_live(&mut runtime, &second.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
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
        directory.resolve_head(&runtime, &project(), &identity(), &constraints()),
        Err(ResolveError::NotFound)
    ));
}

#[test]
fn a_head_on_a_superseded_generation_does_not_resolve() {
    let mut directory = InMemoryPodDirectory::default();
    let (first, second) = (endpoint(1), endpoint(2));
    directory.register(first.clone()).unwrap();
    directory.register(second).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    make_live(&mut runtime, &first.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    assert!(directory
        .resolve_head(&runtime, &project(), &identity(), &constraints())
        .is_ok());
    // The lifecycle moves on to generation 2; the head still names 1 and no longer answers, and
    // the endpoint of generation 1 is not offered in its place.
    make_live(&mut runtime, &endpoint(2).pod);
    assert!(matches!(
        directory.resolve_head(&runtime, &project(), &identity(), &constraints()),
        Err(ResolveError::NotFound)
    ));
}

#[test]
fn an_identity_does_not_resolve_through_another_projects_head() {
    let mut directory = InMemoryPodDirectory::default();
    let first = endpoint(1);
    directory.register(first.clone()).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    make_live(&mut runtime, &first.pod);
    runtime
        .advance_pod_head(AdvancePodHead {
            project: project(),
            identity: identity(),
            expected: None,
            new: first.pod.clone(),
        })
        .unwrap();
    assert!(directory
        .resolve_head(&runtime, &project(), &identity(), &constraints())
        .is_ok());
    // The same identity in another project has no head, whatever this project's head is.
    assert!(matches!(
        directory.resolve_head(
            &runtime,
            &ProjectId::from("other"),
            &identity(),
            &constraints()
        ),
        Err(ResolveError::NotFound)
    ));
}
