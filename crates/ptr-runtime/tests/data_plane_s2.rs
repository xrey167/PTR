//! Stage S2 of the Pod data plane (doc 42), simulated on the CPU: the locator-free `InProcess`
//! handle through `HandleTable` admission, and a pipeline split across two cards that has no
//! peer path, whose frames travel by `PinnedHost` handle and are fenced when a card fails.
//!
//! Nothing here touches a GPU, a descriptor or a pinned buffer. The carrier selection, the
//! handle admission, the placement fence and the claim bookkeeping are real; the bytes are a
//! `Vec<u8>` standing in for the buffer.

use ed25519_dalek::{Signer, SigningKey};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{
    content_digest, handle_signing_bytes, Carrier, CarrierLocator, InProcessHandle, TensorHandle,
};
use ptr_runtime::{
    select_carrier, AdmittedHandle, CarrierRefusal, CarrierTable, ChainFault, ConsumeRequest,
    ConsumedHandle, DeviceHealth, DeviceRecord, EndpointCapabilities, EndpointConfinement,
    HandleAdmissionRequest, HandlePolicy, HandleRejection, HandleSource, HandleTable, HostProcess,
    MovedTensor, NodeHealth, NodeRecord, OperationCheck, OperationFault, OperationLiveness,
    PayloadOwnership, PeerProbe, PodPlacementController, ProjectPlacementRequest, PtrRuntime,
    ReclaimReason, RuntimeStageLiveness, SelectionReason, Teardown, TensorLocation,
    TensorMoveRefused, WorkerSession,
};
use ptr_types::{
    ArtifactId, CapsuleId, DeviceId, Generation, NodeId, PodId, ProjectId, Timestamp, TypeId,
};
use std::cell::Cell;
use std::collections::BTreeMap;

const OPERATION: u64 = 7;
const PAYLOAD: &str = "ptr.split.activation.v1";
const PRODUCER_SESSION: WorkerSession = WorkerSession(22);
const CONSUMER_SESSION: WorkerSession = WorkerSession(11);

fn project() -> ProjectId {
    ProjectId::from("p")
}

fn digest_of(name: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = name.len() as u8;
    out[1] = 1;
    out
}

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: digest_of(name),
        fence_epoch: 0,
    }
}

fn signer(stage: &str) -> SigningKey {
    SigningKey::from_bytes(&[stage.len() as u8 + 0x40; 32])
}

/// The authority's record of the operation: the sessions it is bound to. A respawned consumer
/// worker has a new session, and the old one is no longer current.
struct Authority {
    consumer_session: Cell<WorkerSession>,
}

impl OperationLiveness for Authority {
    fn operation_admits(&self, check: OperationCheck<'_>) -> Result<(), OperationFault> {
        let operation = check.parent_operation_id;
        if check.consumer_session != self.consumer_session.get() {
            return Err(OperationFault::ConsumerSessionMismatch { operation });
        }
        Ok(())
    }
}

struct World {
    runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
    authority: Authority,
}

fn host_node(second: DeviceHealth) -> NodeRecord {
    let card = |name: &str, health| DeviceRecord {
        device_id: DeviceId::from(name),
        vram_bytes: 24 << 30,
        used_vram_bytes: 0,
        health,
    };
    NodeRecord {
        node_id: NodeId::from("host"),
        zone: "lan".into(),
        devices: vec![
            card("cuda:0", DeviceHealth::Healthy),
            card("cuda:1", second),
        ],
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
    }
}

fn place(world: &mut World, stage: &str, device: &str) {
    world
        .placement
        .assign_in(ProjectPlacementRequest {
            project: project(),
            pod_id: PodId::from(stage),
            artifact_id: ArtifactId::from(stage),
            generation: Generation(1),
            node_id: NodeId::from("host"),
            device_id: DeviceId::from(device),
            required_vram: 1,
        })
        .expect("the stage is placed");
}

/// Three stages of one pipeline on one host: `stage-a` and `stage-c` on `cuda:0`, `stage-b` on
/// `cuda:1`.
fn world() -> World {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("a runtime");
    let names = ["stage-a", "stage-b", "stage-c", "model"];
    for name in names {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: project(),
                capsule: CapsuleId::from(name),
                generation: Generation(1),
            })
            .expect("the capsule commits");
    }
    let mut placement = PodPlacementController::default();
    placement
        .register_node(host_node(DeviceHealth::Healthy))
        .expect("the host registers");
    let manifests = names
        .into_iter()
        .map(|name| ((name.to_owned(), 1), digest_of(name)))
        .collect();
    let mut world = World {
        runtime,
        placement,
        manifests,
        authority: Authority {
            consumer_session: Cell::new(CONSUMER_SESSION),
        },
    };
    place(&mut world, "stage-a", "cuda:0");
    place(&mut world, "stage-b", "cuda:1");
    place(&mut world, "stage-c", "cuda:0");
    world
}

fn live(world: &World) -> RuntimeStageLiveness<'_> {
    RuntimeStageLiveness {
        runtime: &world.runtime,
        placement: &world.placement,
        manifests: &world.manifests,
    }
}

fn epoch_of(world: &World, stage: &str) -> u64 {
    world
        .placement
        .placement_in(&project(), &PodId::from(stage))
        .expect("the stage is placed")
        .epoch
        .0
}

fn policy() -> HandlePolicy {
    HandlePolicy {
        max_open_handles: 4,
        max_open_per_principal: 4,
        max_tombstones_per_principal: 8,
        max_inflight_bytes_per_principal: 64,
        max_handle_lifetime_ms: 1_000,
        max_clock_skew_ms: 100,
    }
}

/// A table for the stage `consumer`, with `producer` registered as the one stage that may
/// present handles to it.
fn table_for(consumer: &str, principal: &str, producer: &str) -> HandleTable {
    let mut table = HandleTable::new(policy(), consumer).expect("a valid policy");
    let released = table
        .register_producer(
            principal,
            signer(producer).verifying_key(),
            producer,
            PRODUCER_SESSION,
        )
        .expect("registered");
    assert!(released.is_empty());
    table
}

/// The fields of a frame handle from `producer` to `consumer`, unsigned and without a locator.
fn frame(
    world: &World,
    producer: &str,
    consumer: &str,
    nonce: u64,
    deadline_ms: u64,
    tensor: &MovedTensor,
) -> TensorHandle {
    TensorHandle {
        carrier: Carrier::InProcess,
        locator: CarrierLocator::InProcess,
        dtype: BoundaryDtype::F32,
        shape: vec![1, 2],
        byte_len: 8,
        upstream_chain: UpstreamChain {
            stage: element(producer),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: epoch_of(world, producer),
        signed_absolute_deadline_ms: deadline_ms,
        handle_nonce: nonce,
        parent_operation_id: OPERATION,
        consumer: consumer.to_owned(),
        payload_type: TypeId::from(PAYLOAD),
        content_digest: tensor.content_digest(),
        signature: [0; 64],
    }
}

fn sign(producer: &str, mut handle: TensorHandle) -> TensorHandle {
    handle.signature = signer(producer)
        .sign(&handle_signing_bytes(&handle))
        .to_bytes();
    handle
}

fn tensor(fill: u8) -> MovedTensor {
    MovedTensor::new(vec![fill; 8])
}

fn request<'a>(
    world: &'a World,
    principal: &'a str,
    now_ms: u64,
    handle: HandleSource<'a>,
    project: &'a ProjectId,
    live: &'a RuntimeStageLiveness<'a>,
) -> HandleAdmissionRequest<'a> {
    HandleAdmissionRequest {
        now_ms,
        principal,
        expected_operation: OPERATION,
        handle,
        live,
        operations: &world.authority,
        project,
        consumer_session: CONSUMER_SESSION,
        footprint_bytes: 8,
    }
}

fn admit_local(
    table: &mut HandleTable,
    world: &World,
    now_ms: u64,
    handle: &InProcessHandle,
) -> Result<AdmittedHandle, HandleRejection> {
    let project = project();
    let liveness = live(world);
    table.admit(request(
        world,
        "producer-a",
        now_ms,
        HandleSource::InProcess(handle),
        &project,
        &liveness,
    ))
}

fn consume(
    table: &mut HandleTable,
    world: &World,
    now_ms: u64,
    admitted: AdmittedHandle,
) -> Result<ConsumedHandle, HandleRejection> {
    table.consume(ConsumeRequest {
        now_ms,
        admitted,
        live: &live(world),
        operations: &world.authority,
    })
}

/// A sealed local handle from `stage-a` to `stage-b`, and the tensor that moves with it.
fn local(world: &World, nonce: u64, deadline_ms: u64) -> (InProcessHandle, MovedTensor) {
    let moved = tensor(nonce as u8);
    let handle = sign(
        "stage-a",
        frame(world, "stage-a", "stage-b", nonce, deadline_ms, &moved),
    );
    (
        InProcessHandle::new(handle).expect("a well-formed local handle"),
        moved,
    )
}

// ---- the locator-free InProcess handle -------------------------------------------------------

#[test]
fn an_in_process_handle_is_admitted_consumed_and_its_tensor_moves_once() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (handle, moved) = local(&world, 1, 500);
    let admitted = admit_local(&mut table, &world, 10, &handle).expect("admitted");
    assert_eq!(admitted.carrier(), Carrier::InProcess);
    assert_eq!(admitted.handle_digest(), handle.digest());
    assert_eq!(table.open_handles(), 1);
    assert_eq!(table.inflight_bytes("producer-a"), Some(8));

    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    assert_eq!(consumed.locator(), &CarrierLocator::InProcess);
    let bytes = moved.receive(&consumed).expect("the tensor moves");
    assert_eq!(bytes, vec![1; 8]);

    // InProcess is not a shared carrier: finishing the use releases the claim at once.
    let notice = table
        .finish(consumed)
        .expect("finished")
        .expect("one reclaim notice");
    assert_eq!(notice.reason, ReclaimReason::Consumed);
    assert_eq!(notice.carrier, Carrier::InProcess);
    assert_eq!(notice.locator, CarrierLocator::InProcess);
    assert_eq!(table.inflight_bytes("producer-a"), Some(0));
}

#[test]
fn a_second_admission_of_the_same_in_process_handle_is_refused_open_and_after_use() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (handle, moved) = local(&world, 1, 500);
    let admitted = admit_local(&mut table, &world, 10, &handle).expect("admitted");
    assert_eq!(
        admit_local(&mut table, &world, 11, &handle).unwrap_err(),
        HandleRejection::DuplicateOpen
    );
    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    let _ = moved.receive(&consumed).expect("moved");
    assert_eq!(
        admit_local(&mut table, &world, 21, &handle).unwrap_err(),
        HandleRejection::Replayed
    );
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn a_stale_epoch_refuses_the_in_process_handle_at_admission_and_at_use() {
    let mut world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (stale, _stale_tensor) = local(&world, 1, 500);
    let (in_flight, _in_flight_tensor) = local(&world, 2, 500);
    let admitted = admit_local(&mut table, &world, 10, &in_flight).expect("admitted");
    // The producer's stage is placed again: a new epoch, and everything signed before it is old.
    place(&mut world, "stage-a", "cuda:0");
    assert!(matches!(
        admit_local(&mut table, &world, 11, &stale).unwrap_err(),
        HandleRejection::Chain(ChainFault::StalePlacement { .. })
    ));
    assert_eq!(table.slots_held("producer-a"), Some(1));
    assert!(matches!(
        consume(&mut table, &world, 12, admitted).unwrap_err(),
        HandleRejection::Chain(ChainFault::StalePlacement { .. })
    ));
    // The refused use abandons the handle: its nonce cannot come back.
    assert_eq!(
        admit_local(&mut table, &world, 13, &in_flight).unwrap_err(),
        HandleRejection::Abandoned
    );
}

#[test]
fn the_deadline_of_an_in_process_handle_binds_admission_use_and_reclaim() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (reached, _t1) = local(&world, 1, 500);
    assert_eq!(
        admit_local(&mut table, &world, 500, &reached).unwrap_err(),
        HandleRejection::Expired
    );
    let (far, _t2) = local(&world, 2, 5_000);
    assert_eq!(
        admit_local(&mut table, &world, 10, &far).unwrap_err(),
        HandleRejection::DeadlineTooFar
    );
    assert_eq!(table.slots_held("producer-a"), Some(0));

    // Admitted in time, not used by the deadline: refused at use, reclaimed only after the skew.
    // (A table's clock never runs backwards, so this goes through a table of its own.)
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (late, _t3) = local(&world, 3, 500);
    let admitted = admit_local(&mut table, &world, 10, &late).expect("admitted");
    assert_eq!(
        consume(&mut table, &world, 500, admitted).unwrap_err(),
        HandleRejection::Expired
    );
    assert!(table.expire(599).is_empty());
    let notices = table.expire(600);
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].carrier, Carrier::InProcess);
}

#[test]
fn a_tampered_in_process_handle_fails_the_signature() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let moved = tensor(1);
    let mut signed = sign(
        "stage-a",
        frame(&world, "stage-a", "stage-b", 1, 500, &moved),
    );
    // Another nonce under the first signature.
    signed.handle_nonce = 99;
    let handle = InProcessHandle::new(signed).expect("well formed");
    assert_eq!(
        admit_local(&mut table, &world, 10, &handle).unwrap_err(),
        HandleRejection::BadSignature
    );
    // A handle for another stage's audience is refused like a wire handle.
    let elsewhere = InProcessHandle::new(sign(
        "stage-a",
        frame(&world, "stage-a", "stage-c", 2, 500, &moved),
    ))
    .expect("well formed");
    assert!(matches!(
        admit_local(&mut table, &world, 10, &elsewhere).unwrap_err(),
        HandleRejection::WrongConsumer { .. }
    ));
}

#[test]
fn only_the_in_process_source_admits_an_in_process_handle() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (handle, _moved) = local(&world, 1, 500);
    let project = project();
    let liveness = live(&world);
    let as_wire = request(
        &world,
        "producer-a",
        10,
        HandleSource::Decoded(handle.as_tensor_handle()),
        &project,
        &liveness,
    );
    assert!(matches!(
        table.admit(as_wire).unwrap_err(),
        HandleRejection::Malformed(_)
    ));
    assert!(handle.as_tensor_handle().encode().is_err());
    assert_eq!(table.slots_held("producer-a"), Some(0));
}

#[test]
fn a_tensor_that_is_not_the_signed_one_is_refused_and_a_wire_handle_claims_no_moved_tensor() {
    let world = world();
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let (handle, _real) = local(&world, 1, 500);
    let admitted = admit_local(&mut table, &world, 10, &handle).expect("admitted");
    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    // Same length, other bytes: the digest is the commitment.
    assert!(matches!(
        tensor(9).receive(&consumed).unwrap_err(),
        TensorMoveRefused::Content(HandleRejection::Content(_))
    ));
    // Wrong length.
    assert!(matches!(
        MovedTensor::new(vec![1; 4]).receive(&consumed).unwrap_err(),
        TensorMoveRefused::Content(_)
    ));
    let _ = table.finish(consumed);

    // A consumed wire handle is not an in-process claim.
    let mut wire = frame(&world, "stage-a", "stage-b", 2, 500, &tensor(2));
    wire.carrier = Carrier::NvmeFile;
    wire.locator = CarrierLocator::NvmeFile {
        chunk_digest: [3; 32],
    };
    let wire = sign("stage-a", wire);
    let encoded = wire.encode().expect("a wire handle encodes");
    let project = project();
    let liveness = live(&world);
    let admitted = table
        .admit(request(
            &world,
            "producer-a",
            30,
            HandleSource::Encoded(&encoded),
            &project,
            &liveness,
        ))
        .expect("admitted");
    let consumed = consume(&mut table, &world, 31, admitted).expect("consumed");
    assert_eq!(
        tensor(2).receive(&consumed).unwrap_err(),
        TensorMoveRefused::NotInProcess {
            carrier: Carrier::NvmeFile
        }
    );
}

// ---- the pipeline split on the pinned host path --------------------------------------------

fn endpoint(process: u64, device: &str) -> EndpointCapabilities {
    EndpointCapabilities {
        node: NodeId::from("host"),
        process: HostProcess(process),
        confinement: EndpointConfinement::Full,
        tensor_location: TensorLocation::Device(DeviceId::from(device)),
    }
}

/// A frame on the pinned host path: the handle names a pinned region, and the bytes live in a
/// `Vec` that stands in for it.
fn pinned_frame(
    world: &World,
    producer: &str,
    consumer: &str,
    nonce: u64,
    bytes: &[u8; 8],
) -> TensorHandle {
    let mut handle = frame(world, producer, consumer, nonce, 500, &tensor(0));
    handle.carrier = Carrier::PinnedHost;
    handle.locator = CarrierLocator::PinnedHost {
        region: 100 + nonce,
        offset: 0,
    };
    handle.content_digest = content_digest(bytes);
    sign(producer, handle)
}

fn present(
    table: &mut HandleTable,
    world: &World,
    principal: &str,
    now_ms: u64,
    handle: &TensorHandle,
) -> Result<AdmittedHandle, HandleRejection> {
    let encoded = handle.encode().expect("a wire handle encodes");
    let project = project();
    let liveness = live(world);
    table.admit(request(
        world,
        principal,
        now_ms,
        HandleSource::Encoded(&encoded),
        &project,
        &liveness,
    ))
}

#[test]
fn two_cards_without_a_peer_path_carry_a_frame_by_pinned_host_handle() {
    let world = world();
    let node = host_node(DeviceHealth::Healthy);
    let mut carriers = CarrierTable::register(&node).expect("the host registers");
    // The probe ran at registration and found no peer access.
    for (from, to) in [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")] {
        carriers
            .record_probe(
                &DeviceId::from(from),
                &DeviceId::from(to),
                PeerProbe::NoPeer,
            )
            .expect("recorded");
    }
    let selection = select_carrier(
        &endpoint(1, "cuda:0"),
        &endpoint(2, "cuda:1"),
        PayloadOwnership::MovedUnique,
        &carriers,
    )
    .expect("a carrier");
    assert_eq!(selection.carrier, Carrier::PinnedHost);
    assert_eq!(selection.reason, SelectionReason::PeerUnavailable);

    let frame_bytes = [5u8; 8];
    let sent = pinned_frame(&world, "stage-a", "stage-b", 1, &frame_bytes);
    assert_eq!(sent.carrier, selection.carrier);
    let mut table = table_for("stage-b", "producer-a", "stage-a");
    let admitted = present(&mut table, &world, "producer-a", 10, &sent).expect("admitted");
    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    assert_eq!(consumed.locator(), &sent.locator);
    // The consumer copies the pinned region into its own buffer on its card and verifies it.
    consumed
        .verify_content(&frame_bytes)
        .expect("the copy matches the commitment");

    // Pinned host is shared: the use frees nothing, and only the verified teardown of both
    // worker domains returns the region.
    assert_eq!(table.finish(consumed), Ok(None));
    assert_eq!(table.inflight_bytes("producer-a"), Some(8));
    let notices = table.release_after_teardown(Teardown {
        principal: "producer-a",
        producer_session: PRODUCER_SESSION,
        consumer_session: CONSUMER_SESSION,
    });
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].reason, ReclaimReason::TornDown);
    assert_eq!(notices[0].carrier, Carrier::PinnedHost);
}

#[test]
fn a_frame_is_fenced_when_the_second_card_fails() {
    let mut world = world();
    let mut carriers = CarrierTable::register(&host_node(DeviceHealth::Healthy)).expect("host");
    carriers
        .record_probe(
            &DeviceId::from("cuda:0"),
            &DeviceId::from("cuda:1"),
            PeerProbe::NoPeer,
        )
        .expect("recorded");

    // Before the failure: stage-a's frame to stage-b is in flight, and stage-b has produced a
    // frame for stage-c, which has not yet been presented.
    let mut into_b = table_for("stage-b", "producer-a", "stage-a");
    let mut into_c = table_for("stage-c", "producer-b", "stage-b");
    let a_to_b = pinned_frame(&world, "stage-a", "stage-b", 1, &[1; 8]);
    let admitted_a_to_b = present(&mut into_b, &world, "producer-a", 10, &a_to_b).expect("admit");
    let b_to_c = pinned_frame(&world, "stage-b", "stage-c", 1, &[2; 8]);
    let epoch_before = epoch_of(&world, "stage-b");

    // cuda:1 fails: the heartbeat revokes stage-b's placement and moves the epoch, and the
    // carrier table drops every row of the card.
    let failed = host_node(DeviceHealth::Failed);
    world
        .placement
        .heartbeat(&NodeId::from("host"), Timestamp(2), failed.devices.clone())
        .expect("the heartbeat is accepted");
    carriers.reconcile(&failed).expect("reconciled");
    assert!(world
        .placement
        .placement_in(&project(), &PodId::from("stage-b"))
        .is_err());
    assert!(world.placement.epoch().0 > epoch_before);

    // The frame stage-b produced under the old placement is refused by stage-c's table.
    assert!(matches!(
        present(&mut into_c, &world, "producer-b", 30, &b_to_c).unwrap_err(),
        HandleRejection::Chain(ChainFault::StalePlacement { current: None, .. })
    ));
    assert_eq!(into_c.slots_held("producer-b"), Some(0));

    // No carrier is selected towards the failed card, and none is routed around it.
    assert_eq!(
        select_carrier(
            &endpoint(1, "cuda:0"),
            &endpoint(2, "cuda:1"),
            PayloadOwnership::MovedUnique,
            &carriers,
        ),
        Err(CarrierRefusal::DeviceUnavailable(DeviceId::from("cuda:1")))
    );

    // The supervisor respawns the consumer worker under a new session: the frame admitted
    // before the failure can no longer be used, and its pinned region comes back only through
    // the verified teardown of the old pair.
    world.authority.consumer_session.set(WorkerSession(12));
    assert!(matches!(
        consume(&mut into_b, &world, 40, admitted_a_to_b).unwrap_err(),
        HandleRejection::Operation(OperationFault::ConsumerSessionMismatch { .. })
    ));
    assert!(into_b.expire(10_000).is_empty());
    let notices = into_b.release_after_teardown(Teardown {
        principal: "producer-a",
        producer_session: PRODUCER_SESSION,
        consumer_session: CONSUMER_SESSION,
    });
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].reason, ReclaimReason::TornDown);
    assert_eq!(into_b.inflight_bytes("producer-a"), Some(0));
}
