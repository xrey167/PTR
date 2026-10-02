use ptr_runtime::{InMemoryPodDirectory, PodDirectory};
use ptr_types::{PodEndpoint, PodRevisionAddress, PodTransport, RouteConstraints};

fn endpoint(epoch: u64) -> PodEndpoint {
    PodEndpoint {
        pod: PodRevisionAddress {
            address: ptr_types::PodAddress::new("project".into(), "namespace".into(), "pod".into())
                .unwrap(),
            semantic_revision: [4; 32],
            generation: ptr_types::Generation(1),
        },
        artifact_id: "artifact-1".into(),
        artifact_active: true,
        node_id: "node-a".into(),
        device_id: Some("cuda0".into()),
        region: "eu-central".into(),
        zone: "eu-central-a".into(),
        peer_id: "peer-a".into(),
        transport: PodTransport::IrohQuic,
        endpoint_epoch: epoch,
        fencing_token: epoch as u128,
        available_vram_bytes: 24 * 1024 * 1024 * 1024,
        healthy: true,
        load_bps: 100,
        capabilities: vec!["infer".into()],
        accepts: vec!["tokens".into()],
        produces: vec!["answer".into()],
        mesh: None,
    }
}

#[test]
fn directory_resolves_logical_address_to_current_endpoint() {
    let endpoint = endpoint(2);
    let address = endpoint.pod.clone();
    let mut directory = InMemoryPodDirectory::default();
    directory.register(endpoint.clone()).unwrap();
    let route = directory
        .resolve(
            &address,
            &RouteConstraints {
                capability: Some("infer".into()),
                input_type: Some("tokens".into()),
                artifact_id: Some("artifact-1".into()),
                output_type: Some("answer".into()),
                region: Some("eu-central".into()),
                zone: None,
                required_vram_bytes: Some(1024),
                min_epoch: Some(2),
            },
        )
        .unwrap();
    assert_eq!(route.endpoint, endpoint);
    assert_eq!(route.placement_epoch, 2);
    assert_eq!(route.fencing_token, 2);
}

#[test]
fn directory_rejects_stale_epoch_and_wrong_region() {
    let endpoint = endpoint(2);
    let address = endpoint.pod.clone();
    let mut directory = InMemoryPodDirectory::default();
    directory.register(endpoint).unwrap();
    let stale = directory.resolve(
        &address,
        &RouteConstraints {
            capability: Some("infer".into()),
            input_type: Some("tokens".into()),
            min_epoch: Some(3),
            ..RouteConstraints::default()
        },
    );
    assert!(matches!(stale, Err(ptr_types::ResolveError::StaleRoute)));
    let wrong_region = directory.resolve(
        &address,
        &RouteConstraints {
            capability: Some("infer".into()),
            input_type: Some("tokens".into()),
            region: Some("us-east".into()),
            ..RouteConstraints::default()
        },
    );
    assert!(matches!(
        wrong_region,
        Err(ptr_types::ResolveError::ConstraintMismatch { field: "region" })
    ));
}

#[test]
fn directory_rejects_stale_or_conflicting_rebinds() {
    let current = endpoint(3);
    let address = current.pod.clone();
    let mut directory = InMemoryPodDirectory::default();
    directory.register(current.clone()).unwrap();

    assert!(matches!(
        directory.register(endpoint(2)),
        Err(ptr_types::ResolveError::StaleRoute)
    ));

    let mut conflict = current.clone();
    conflict.peer_id = "peer-other".into();
    assert!(matches!(
        directory.register(conflict),
        Err(ptr_types::ResolveError::ConflictingEndpoint)
    ));

    let route = directory
        .resolve(
            &address,
            &RouteConstraints {
                capability: Some("infer".into()),
                input_type: Some("tokens".into()),
                ..RouteConstraints::default()
            },
        )
        .unwrap();
    assert_eq!(route.fencing_token, 3);
}
