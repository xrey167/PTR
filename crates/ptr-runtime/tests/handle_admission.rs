//! Handle admission against a real ledger and a real placement controller: a tensor handle is
//! admitted like an activation frame, revalidated again before the consumer may touch the
//! memory, single use, bounded, and reclaimed by the producer exactly once.

use ed25519_dalek::{Signer, SigningKey};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{content_digest, handle_signing_bytes, Carrier, CarrierLocator, TensorHandle};
use ptr_runtime::{
    AdmittedHandle, ChainFault, ConsumeRequest, ConsumedHandle, DeviceHealth, DeviceRecord,
    HandleAdmissionRequest, HandlePolicy, HandleRejection, HandleSource, HandleTable, NodeHealth,
    NodeRecord, OperationCheck, OperationFault, OperationLiveness, PodPlacementController,
    PtrRuntime, ReclaimReason, RuntimeStageLiveness, WorkerSession,
};
use ptr_types::{
    ArtifactId, CapsuleId, DeviceId, Generation, NodeId, PodId, ProjectId, Timestamp, TypeId,
    Validity,
};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// The authority's operation record, faked: operation to the inputs it read; an input in `dead`
/// is no longer live at the committed anchor.
#[derive(Default)]
struct Operations {
    inputs: RefCell<BTreeMap<u64, Vec<String>>>,
    dead: RefCell<Vec<String>>,
    expected: RefCell<BTreeMap<u64, (UpstreamChain, Vec<TypeId>)>>,
    /// Operations recorded as Approximate or tainted.
    inexact: RefCell<Vec<u64>>,
    /// The project, consumer session and producer session each operation is bound to.
    bound: RefCell<BTreeMap<u64, (ProjectId, WorkerSession, WorkerSession)>>,
    /// The absolute deadline of each operation, minted by the authority when it started.
    deadline: RefCell<BTreeMap<u64, u64>>,
    /// Every check the authority was asked, in order: project, both sessions, handle deadline.
    asked: RefCell<Vec<(ProjectId, WorkerSession, WorkerSession, u64)>>,
}

impl OperationLiveness for Operations {
    fn operation_admits(&self, check: OperationCheck<'_>) -> Result<(), OperationFault> {
        let operation = check.parent_operation_id;
        self.asked.borrow_mut().push((
            check.project.clone(),
            check.consumer_session,
            check.producer_session,
            check.signed_absolute_deadline_ms,
        ));
        let inputs = self.inputs.borrow();
        let read = inputs
            .get(&operation)
            .ok_or(OperationFault::UnknownOperation { operation })?;
        if let Some(deadline) = self.deadline.borrow().get(&operation) {
            if check.signed_absolute_deadline_ms > *deadline {
                return Err(OperationFault::DeadlineBeyondOperation { operation });
            }
        }
        if let Some((project, consumer, producer)) = self.bound.borrow().get(&operation) {
            if project != check.project {
                return Err(OperationFault::WrongProject { operation });
            }
            if *consumer != check.consumer_session {
                return Err(OperationFault::ConsumerSessionMismatch { operation });
            }
            if *producer != check.producer_session {
                return Err(OperationFault::ProducerSessionMismatch { operation });
            }
        }
        if self.inexact.borrow().contains(&operation) {
            return Err(OperationFault::NotExact { operation });
        }
        let dead = self.dead.borrow();
        if let Some(input) = read.iter().find(|input| dead.contains(input)) {
            return Err(OperationFault::InputNotLive {
                operation,
                input: input.clone(),
            });
        }
        if let Some((chain, accepted)) = self.expected.borrow().get(&operation) {
            if chain != check.upstream_chain {
                return Err(OperationFault::ChainMismatch { operation });
            }
            if !accepted.contains(check.payload_type) {
                return Err(OperationFault::PayloadTypeNotAccepted {
                    operation,
                    payload_type: check.payload_type.clone(),
                });
            }
        }
        Ok(())
    }
}

impl Operations {
    /// The authority's record of `operation`: the inputs it read, the chain it expects and the
    /// payload types it accepts.
    fn know(&self, operation: u64, inputs: &[&str], chain: UpstreamChain, accepted: &[&str]) {
        self.bound
            .borrow_mut()
            .insert(operation, (project(), CONSUMER_SESSION, PRODUCER_SESSION));
        self.inputs
            .borrow_mut()
            .insert(operation, inputs.iter().map(|i| (*i).to_owned()).collect());
        self.expected.borrow_mut().insert(
            operation,
            (chain, accepted.iter().map(|t| TypeId::from(*t)).collect()),
        );
    }
}

/// Every operation is known and all its inputs are live.
struct AllLive;

impl OperationLiveness for AllLive {
    fn operation_admits(&self, _check: OperationCheck<'_>) -> Result<(), OperationFault> {
        Ok(())
    }
}

const PRINCIPAL: &str = "producer";
const CONSUMER: &str = "down";
const OPERATION: u64 = 31;
const PAYLOAD: &str = "ptr.split.activation.v1";
const CONSUMER_SESSION: WorkerSession = WorkerSession(11);
const PRODUCER_SESSION: WorkerSession = WorkerSession(22);

fn project() -> ProjectId {
    ProjectId::from("p")
}

fn digest(name: &str, generation: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = name.len() as u8;
    out[1] = generation as u8;
    out
}

struct World {
    runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
}

fn commit(runtime: &mut PtrRuntime, name: &str, generation: u64) {
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from(name),
            generation: Generation(generation),
        })
        .expect("the capsule commits");
}

fn node(name: &str) -> NodeRecord {
    NodeRecord {
        node_id: NodeId::from(name),
        zone: "z".into(),
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
        devices: vec![DeviceRecord {
            device_id: DeviceId::from("cpu"),
            vram_bytes: 10,
            used_vram_bytes: 0,
            health: DeviceHealth::Healthy,
        }],
    }
}

fn place(placement: &mut PodPlacementController, node: &str) {
    placement
        .assign(
            PodId::from("stage"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from(node),
            DeviceId::from("cpu"),
            1,
        )
        .expect("the stage is placed");
}

fn world() -> World {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("a runtime");
    commit(&mut runtime, "stage", 1);
    commit(&mut runtime, "model", 1);
    let mut placement = PodPlacementController::default();
    placement.register_node(node("n1")).expect("node n1");
    placement.register_node(node("n2")).expect("node n2");
    place(&mut placement, "n1");
    let manifests = [("stage", 1), ("model", 1)]
        .into_iter()
        .map(|(name, generation)| ((name.to_owned(), generation), digest(name, generation)))
        .collect();
    World {
        runtime,
        placement,
        manifests,
    }
}

fn live(world: &World) -> RuntimeStageLiveness<'_> {
    RuntimeStageLiveness {
        runtime: &world.runtime,
        placement: &world.placement,
        manifests: &world.manifests,
    }
}

fn epoch(world: &World) -> u64 {
    world
        .placement
        .placement(&PodId::from("stage"))
        .expect("the stage is placed")
        .epoch
        .0
}

fn re_place(world: &mut World) {
    world
        .placement
        .mark_node_failed(&NodeId::from("n1"))
        .expect("n1 fails");
    place(&mut world.placement, "n2");
}

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: digest(name, 1),
        fence_epoch: 0,
    }
}

fn signer() -> SigningKey {
    SigningKey::from_bytes(&[0xC3; 32])
}

fn policy() -> HandlePolicy {
    HandlePolicy {
        max_open_handles: 3,
        max_open_per_principal: 3,
        max_tombstones_per_principal: 6,
        max_inflight_bytes_per_principal: 24,
        max_handle_lifetime_ms: 1_000,
        max_clock_skew_ms: 100,
    }
}

fn table() -> HandleTable {
    let mut table = HandleTable::new(policy(), CONSUMER).expect("a valid policy");
    let released = table.register_producer(
        PRINCIPAL,
        signer().verifying_key(),
        "stage",
        PRODUCER_SESSION,
    );
    assert!(released.is_empty());
    table
}

/// A handle on a carrier that is not shared (`NvmeFile`): released on `Consumed`, on producer
/// replacement and at the deadline plus the skew.
fn handle(world: &World, nonce: u64, deadline_ms: u64) -> TensorHandle {
    let mut handle = TensorHandle {
        carrier: Carrier::NvmeFile,
        locator: CarrierLocator::NvmeFile {
            chunk_digest: [100 + nonce as u8; 32],
        },
        dtype: BoundaryDtype::F32,
        shape: vec![1, 2],
        byte_len: 8,
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: epoch(world),
        signed_absolute_deadline_ms: deadline_ms,
        handle_nonce: nonce,
        parent_operation_id: OPERATION,
        consumer: CONSUMER.to_owned(),
        payload_type: TypeId::from(PAYLOAD),
        content_digest: content_digest(&[nonce as u8; 8]),
        signature: [0; 64],
    };
    handle.signature = signer().sign(&handle_signing_bytes(&handle)).to_bytes();
    handle
}

/// The same handle on a shared carrier (`HostShm`): released only after verified teardown.
fn shared_handle(world: &World, nonce: u64, deadline_ms: u64) -> TensorHandle {
    let mut handle = handle(world, nonce, deadline_ms);
    handle.carrier = Carrier::HostShm;
    handle.locator = CarrierLocator::HostShm {
        fd_slot: 100 + nonce as u32,
        offset: 0,
    };
    handle.signature = signer().sign(&handle_signing_bytes(&handle)).to_bytes();
    handle
}

fn admit(
    table: &mut HandleTable,
    world: &World,
    now_ms: u64,
    handle: &TensorHandle,
) -> Result<AdmittedHandle, HandleRejection> {
    let encoded = handle.encode().expect("the test handle encodes");
    table.admit(HandleAdmissionRequest {
        now_ms,
        principal: PRINCIPAL,
        expected_operation: OPERATION,
        project: &project(),
        consumer_session: CONSUMER_SESSION,
        operations: &AllLive,
        handle: HandleSource::Encoded(&encoded),
        live: &live(world),
    })
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
        operations: &AllLive,
    })
}

#[test]
fn an_admitted_shared_handle_is_consumed_once_and_released_only_after_teardown() {
    let world = world();
    let mut table = table();
    let sent = shared_handle(&world, 1, 500);
    let admitted = admit(&mut table, &world, 10, &sent).expect("admitted");
    assert_eq!(admitted.shape(), &[1, 2]);
    assert_eq!(table.open_handles(), 1);

    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    assert_eq!(consumed.locator(), &sent.locator);
    // `HostShm` is shared: `Consumed` frees nothing (doc 42 rule 3b).
    assert!(consumed.reclaim().is_none());
    assert_eq!(table.open_handles(), 0);
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(8));
    // Neither does the deadline plus the skew.
    assert!(table.expire(10_000).is_empty());
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(8));
    // The supervisor has reaped both worker domains: the buffer is released once.
    let notices = table.release_after_teardown(PRINCIPAL);
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].reason, ReclaimReason::TornDown);
    assert_eq!(notices[0].handle_nonce, 1);
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    assert!(table.release_after_teardown(PRINCIPAL).is_empty());
}

#[test]
fn a_consumed_handle_is_refused_as_a_replay() {
    let world = world();
    let mut table = table();
    let sent = handle(&world, 1, 500);
    let admitted = admit(&mut table, &world, 10, &sent).expect("admitted");
    let _ = consume(&mut table, &world, 20, admitted).expect("consumed");
    assert_eq!(
        admit(&mut table, &world, 30, &sent).unwrap_err(),
        HandleRejection::Replayed
    );
}

#[test]
fn a_handle_of_a_stale_placement_epoch_is_refused_at_admission() {
    let mut world = world();
    let mut table = table();
    let old = handle(&world, 1, 500);
    re_place(&mut world);
    assert!(matches!(
        admit(&mut table, &world, 10, &old).unwrap_err(),
        HandleRejection::Chain(ChainFault::StalePlacement { .. })
    ));
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    // The producer's new worker, at the new epoch, is admitted.
    let fresh = handle(&world, 2, 500);
    assert!(admit(&mut table, &world, 10, &fresh).is_ok());
}

#[test]
fn an_epoch_that_moved_between_admission_and_use_refuses_the_consume() {
    let mut world = world();
    let mut table = table();
    let sent = handle(&world, 1, 500);
    let admitted = admit(&mut table, &world, 10, &sent).expect("admitted");
    re_place(&mut world);
    assert!(matches!(
        consume(&mut table, &world, 20, admitted).unwrap_err(),
        HandleRejection::Chain(ChainFault::StalePlacement { .. })
    ));
    // Never consumable again, and the buffer comes back at the deadline plus the skew.
    assert_eq!(
        admit(&mut table, &world, 30, &sent).unwrap_err(),
        HandleRejection::Abandoned
    );
    let reclaimed = table.expire(600);
    assert_eq!(reclaimed.len(), 1);
    assert_eq!(reclaimed[0].reason, ReclaimReason::Abandoned);
}

#[test]
fn a_generation_revoked_between_admission_and_use_refuses_the_consume() {
    let mut world = world();
    let mut table = table();
    let admitted = admit(&mut table, &world, 10, &handle(&world, 1, 500)).expect("admitted");
    world
        .runtime
        .commit(LedgerEvent::Revoked {
            subject: "model".to_owned(),
            generation: Generation(1),
        })
        .expect("the revocation commits");
    assert!(matches!(
        consume(&mut table, &world, 20, admitted).unwrap_err(),
        HandleRejection::Chain(ChainFault::NotLive {
            standing: Some(Validity::Revoked),
            ..
        })
    ));
}

#[test]
fn an_unused_handle_yields_one_reclaim_notice_at_its_deadline_plus_skew() {
    let world = world();
    let mut table = table();
    let _admitted = admit(&mut table, &world, 10, &handle(&world, 1, 500)).expect("admitted");
    assert!(table.expire(599).is_empty());
    let notices = table.expire(600);
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].reason, ReclaimReason::Expired);
    assert_eq!(notices[0].handle_nonce, 1);
    assert_eq!(notices[0].carrier, Carrier::NvmeFile);
    assert_eq!(
        notices[0].locator,
        CarrierLocator::NvmeFile {
            chunk_digest: [101; 32]
        }
    );
    assert!(table.expire(601).is_empty());
    assert!(table.expire(10_000).is_empty());
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
}

#[test]
fn registering_the_producer_under_another_key_abandons_its_open_handles() {
    let world = world();
    let mut table = table();
    let open = admit(&mut table, &world, 10, &handle(&world, 1, 500)).expect("admitted");
    let replacement = SigningKey::from_bytes(&[0xD4; 32]).verifying_key();
    let released = table.register_producer(PRINCIPAL, replacement, "stage", PRODUCER_SESSION);
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].reason, ReclaimReason::ProducerReplaced);
    assert_eq!(table.open_handles(), 0);
    assert_eq!(
        consume(&mut table, &world, 20, open).unwrap_err(),
        HandleRejection::Abandoned
    );
    // The old key signs no more handles for this principal.
    assert_eq!(
        admit(&mut table, &world, 20, &handle(&world, 2, 500)).unwrap_err(),
        HandleRejection::BadSignature
    );
}

#[test]
fn the_table_refuses_what_it_has_no_room_for_and_frees_room_on_expiry() {
    let world = world();
    let mut table = table();
    let mut held = Vec::new();
    for nonce in 1..=3 {
        held.push(admit(&mut table, &world, 10, &handle(&world, nonce, 500)).expect("room"));
    }
    assert_eq!(
        admit(&mut table, &world, 10, &handle(&world, 4, 500)).unwrap_err(),
        HandleRejection::TooManyOpenHandles { limit: 3 }
    );
    // Consuming one frees an open slot; the table is bounded, not stuck.
    let first = held.remove(0);
    let _ = consume(&mut table, &world, 20, first).expect("consumed");
    assert!(admit(&mut table, &world, 20, &handle(&world, 4, 500)).is_ok());
    // After the deadline plus the skew, expiry reports the open ones and empties the table.
    let notices = table.expire(600);
    assert_eq!(notices.len(), 3);
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn the_byte_budget_saturates_and_is_released_by_consume_abandon_and_expiry() {
    let world = world();
    // Each handle is 8 bytes and the budget 24; raise the open bounds so bytes are what refuses.
    let mut table = HandleTable::new(
        HandlePolicy {
            max_open_handles: 10,
            max_open_per_principal: 10,
            max_tombstones_per_principal: 10,
            ..policy()
        },
        CONSUMER,
    )
    .expect("policy");
    let _ = table.register_producer(
        PRINCIPAL,
        signer().verifying_key(),
        "stage",
        PRODUCER_SESSION,
    );
    let a = admit(&mut table, &world, 10, &handle(&world, 1, 500)).expect("budget");
    let b = admit(&mut table, &world, 10, &handle(&world, 2, 500)).expect("budget");
    let _c = admit(&mut table, &world, 10, &handle(&world, 3, 500)).expect("budget");
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(24));
    assert_eq!(
        admit(&mut table, &world, 10, &handle(&world, 4, 500)).unwrap_err(),
        HandleRejection::InflightBytesExceeded {
            limit: 24,
            in_flight: 24,
            requested: 8
        }
    );
    let _ = consume(&mut table, &world, 20, a).expect("consumed");
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
    // Abandon releases nothing: the claim is held until the reclaim notice at expiry.
    table.abandon(b).expect("abandoned");
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
    assert!(admit(&mut table, &world, 20, &handle(&world, 4, 500)).is_ok());
    assert!(matches!(
        admit(&mut table, &world, 20, &handle(&world, 5, 500)),
        Err(HandleRejection::InflightBytesExceeded { .. })
    ));
    let _ = table.expire(600);
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
}

#[test]
fn a_handle_at_its_deadline_is_reclaimable_only_after_the_skew() {
    let world = world();
    let mut table = table();
    let _held = admit(&mut table, &world, 10, &handle(&world, 1, 500)).expect("admitted");
    assert!(table.expire(500).is_empty());
    assert!(table.expire(599).is_empty());
    assert_eq!(table.expire(600).len(), 1);
}

#[test]
fn the_consumer_verifies_the_mapped_bytes_against_the_commitment() {
    let world = world();
    let mut table = table();
    let admitted = admit(&mut table, &world, 10, &handle(&world, 2, 500)).expect("admitted");
    let consumed = consume(&mut table, &world, 20, admitted).expect("consumed");
    assert_eq!(consumed.verify_content(&[2u8; 8]), Ok(()));
    assert!(matches!(
        consumed.verify_content(&[9u8; 8]),
        Err(HandleRejection::Content(_))
    ));
}

#[test]
fn the_same_handle_under_another_operation_is_refused_and_leaves_no_state() {
    let world = world();
    let mut table = table();
    let sent = handle(&world, 1, 500);
    let encoded = sent.encode().expect("encodes");
    let result = table.admit(HandleAdmissionRequest {
        now_ms: 10,
        principal: PRINCIPAL,
        expected_operation: OPERATION + 1,
        project: &project(),
        consumer_session: CONSUMER_SESSION,
        operations: &AllLive,
        handle: HandleSource::Encoded(&encoded),
        live: &live(&world),
    });
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::WrongOperation {
            expected: OPERATION + 1,
            found: OPERATION
        }
    );
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    assert_eq!(table.open_handles(), 0);
    // Under its own operation the very same bytes are admitted: nothing was burned.
    assert!(admit(&mut table, &world, 10, &sent).is_ok());
}

#[test]
fn a_handle_addressed_to_another_stage_is_refused() {
    let world = world();
    let mut table = HandleTable::new(policy(), "someone-else").expect("policy");
    let _ = table.register_producer(
        PRINCIPAL,
        signer().verifying_key(),
        "stage",
        PRODUCER_SESSION,
    );
    assert_eq!(
        admit(&mut table, &world, 10, &handle(&world, 1, 500)).unwrap_err(),
        HandleRejection::WrongConsumer {
            expected: "someone-else".to_owned(),
            found: CONSUMER.to_owned()
        }
    );
}

#[test]
fn a_deadline_near_the_maximum_is_refused_as_too_far_so_no_tombstone_is_pinned() {
    let world = world();
    let mut table = table();
    for deadline in [u64::MAX, u64::MAX - 100, u64::MAX / 2] {
        assert_eq!(
            admit(&mut table, &world, 10, &handle(&world, 1, deadline)).unwrap_err(),
            HandleRejection::DeadlineTooFar
        );
    }
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
}

#[test]
fn an_operation_whose_input_is_revoked_after_admission_never_releases_the_locator() {
    let world = world();
    let mut table = table();
    let operations = Operations::default();
    operations
        .inputs
        .borrow_mut()
        .insert(OPERATION, vec!["semdb/doc".to_owned()]);
    let sent = handle(&world, 1, 500);
    let encoded = sent.encode().expect("encodes");
    let admitted = table
        .admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OPERATION,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &operations,
            handle: HandleSource::Encoded(&encoded),
            live: &live(&world),
        })
        .expect("admitted");
    operations.dead.borrow_mut().push("semdb/doc".to_owned());
    let result = table.consume(ConsumeRequest {
        now_ms: 20,
        admitted,
        live: &live(&world),
        operations: &operations,
    });
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::Operation(OperationFault::InputNotLive {
            operation: OPERATION,
            input: "semdb/doc".to_owned()
        })
    );
    assert_eq!(
        admit(&mut table, &world, 30, &sent).unwrap_err(),
        HandleRejection::Abandoned
    );
}

#[test]
fn an_unknown_operation_is_refused_at_admission_with_no_state_recorded() {
    let world = world();
    let mut table = table();
    let sent = handle(&world, 1, 500);
    let encoded = sent.encode().expect("encodes");
    let result = table.admit(HandleAdmissionRequest {
        now_ms: 10,
        principal: PRINCIPAL,
        expected_operation: OPERATION,
        project: &project(),
        consumer_session: CONSUMER_SESSION,
        operations: &Operations::default(),
        handle: HandleSource::Encoded(&encoded),
        live: &live(&world),
    });
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::Operation(OperationFault::UnknownOperation {
            operation: OPERATION
        })
    );
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn a_different_chain_or_payload_type_under_a_known_operation_is_refused_and_records_nothing() {
    let world = world();
    let mut table = table();
    let operations = Operations::default();
    let expected = handle(&world, 1, 500).upstream_chain;
    operations.know(OPERATION, &["semdb/doc"], expected, &[PAYLOAD]);
    let request = |h: &TensorHandle| h.encode().expect("encodes");

    let mut other_model = handle(&world, 1, 500);
    other_model.upstream_chain.base_model.generation = 2;
    other_model.signature = signer()
        .sign(&handle_signing_bytes(&other_model))
        .to_bytes();
    let mut other_type = handle(&world, 2, 500);
    other_type.payload_type = TypeId::from("ptr.kv.page");
    other_type.signature = signer().sign(&handle_signing_bytes(&other_type)).to_bytes();
    for (sent, fault) in [
        (
            other_model,
            OperationFault::ChainMismatch {
                operation: OPERATION,
            },
        ),
        (
            other_type,
            OperationFault::PayloadTypeNotAccepted {
                operation: OPERATION,
                payload_type: TypeId::from("ptr.kv.page"),
            },
        ),
    ] {
        let encoded = request(&sent);
        let result = table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OPERATION,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &operations,
            handle: HandleSource::Encoded(&encoded),
            live: &live(&world),
        });
        assert_eq!(result.unwrap_err(), HandleRejection::Operation(fault));
    }
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn an_approximate_operation_is_refused_at_admission_with_no_state_and_at_consume() {
    let world = world();
    let mut table = table();
    let operations = Operations::default();
    operations.know(
        OPERATION,
        &["semdb/doc"],
        handle(&world, 1, 500).upstream_chain,
        &[PAYLOAD],
    );
    operations.inexact.borrow_mut().push(OPERATION);
    let sent = handle(&world, 1, 500);
    let encoded = sent.encode().expect("encodes");
    let present = |table: &mut HandleTable, operations: &Operations| {
        table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OPERATION,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations,
            handle: HandleSource::Encoded(&encoded),
            live: &live(&world),
        })
    };
    assert_eq!(
        present(&mut table, &operations).unwrap_err(),
        HandleRejection::Operation(OperationFault::NotExact {
            operation: OPERATION
        })
    );
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    operations.inexact.borrow_mut().clear();
    let admitted = present(&mut table, &operations).expect("exact again");
    operations.inexact.borrow_mut().push(OPERATION);
    let result = table.consume(ConsumeRequest {
        now_ms: 20,
        admitted,
        live: &live(&world),
        operations: &operations,
    });
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::Operation(OperationFault::NotExact {
            operation: OPERATION
        })
    );
}

/// A known operation of the standard chain, bound to the standard project and sessions.
fn known_operations(world: &World) -> Operations {
    let operations = Operations::default();
    let expected = handle(world, 1, 500).upstream_chain;
    operations.know(OPERATION, &["semdb/doc"], expected, &[PAYLOAD]);
    operations
}

fn admit_with(
    table: &mut HandleTable,
    world: &World,
    operations: &Operations,
    project: &ProjectId,
    consumer_session: WorkerSession,
    sent: &TensorHandle,
) -> Result<AdmittedHandle, HandleRejection> {
    let encoded = sent.encode().expect("encodes");
    table.admit(HandleAdmissionRequest {
        now_ms: 10,
        principal: PRINCIPAL,
        expected_operation: OPERATION,
        project,
        consumer_session,
        operations,
        handle: HandleSource::Encoded(&encoded),
        live: &live(world),
    })
}

#[test]
fn a_handle_of_an_operation_of_another_project_is_refused_with_no_state_recorded() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    let result = admit_with(
        &mut table,
        &world,
        &operations,
        &ProjectId::from("other"),
        CONSUMER_SESSION,
        &handle(&world, 1, 500),
    );
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::Operation(OperationFault::WrongProject {
            operation: OPERATION
        })
    );
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn a_stale_consumer_session_is_refused_at_admission_and_a_change_before_consume_at_consume() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    let stale = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        WorkerSession(99),
        &handle(&world, 1, 500),
    );
    assert_eq!(
        stale.unwrap_err(),
        HandleRejection::Operation(OperationFault::ConsumerSessionMismatch {
            operation: OPERATION
        })
    );
    assert_eq!(table.slots_held(PRINCIPAL), Some(0));

    let admitted = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 2, 500),
    )
    .expect("admitted");
    // The consumer worker is respawned: the authority's record names the new session.
    operations
        .bound
        .borrow_mut()
        .insert(OPERATION, (project(), WorkerSession(12), PRODUCER_SESSION));
    let result = table.consume(ConsumeRequest {
        now_ms: 20,
        admitted,
        live: &live(&world),
        operations: &operations,
    });
    assert_eq!(
        result.unwrap_err(),
        HandleRejection::Operation(OperationFault::ConsumerSessionMismatch {
            operation: OPERATION
        })
    );
    assert_eq!(table.open_handles(), 0);
}

#[test]
fn a_producer_registered_under_a_new_session_abandons_its_handles_and_the_old_session_is_refused() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    let open = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 1, 500),
    )
    .expect("admitted");
    // The same key and stage under a new session is a respawned producer.
    let released = table.register_producer(
        PRINCIPAL,
        signer().verifying_key(),
        "stage",
        WorkerSession(23),
    );
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].reason, ReclaimReason::ProducerReplaced);
    assert_eq!(released[0].handle_nonce, 1);
    assert_eq!(table.open_handles(), 0);
    assert_eq!(
        consume(&mut table, &world, 20, open).unwrap_err(),
        HandleRejection::Abandoned
    );
    // Registering the same session again changes nothing.
    assert!(table
        .register_producer(
            PRINCIPAL,
            signer().verifying_key(),
            "stage",
            WorkerSession(23)
        )
        .is_empty());
    // The authority still records the old producer session, so a new handle is refused ...
    assert_eq!(
        admit_with(
            &mut table,
            &world,
            &operations,
            &project(),
            CONSUMER_SESSION,
            &handle(&world, 2, 500),
        )
        .unwrap_err(),
        HandleRejection::Operation(OperationFault::ProducerSessionMismatch {
            operation: OPERATION
        })
    );
    // ... and admitted once the record names the respawned producer.
    operations
        .bound
        .borrow_mut()
        .insert(OPERATION, (project(), CONSUMER_SESSION, WorkerSession(23)));
    assert!(admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 2, 500),
    )
    .is_ok());
}

#[test]
fn the_matching_project_and_sessions_are_admitted_and_consumed() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    let admitted = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 1, 500),
    )
    .expect("admitted");
    let consumed = table
        .consume(ConsumeRequest {
            now_ms: 20,
            admitted,
            live: &live(&world),
            operations: &operations,
        })
        .expect("consumed");
    assert_eq!(
        consumed.reclaim().map(|notice| notice.reason),
        Some(ReclaimReason::Consumed)
    );
    assert_eq!(
        ptr_runtime::revalidate_operation(&consumed, &operations, 30),
        Ok(())
    );
}

#[test]
fn a_handle_signed_beyond_the_operation_deadline_is_refused_with_no_state_recorded() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    operations.deadline.borrow_mut().insert(OPERATION, 400);
    let refused = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 1, 500),
    );
    assert!(matches!(
        refused,
        Err(HandleRejection::Operation(
            OperationFault::DeadlineBeyondOperation {
                operation: OPERATION
            }
        ))
    ));
    // Nothing was recorded: the same nonce under a deadline inside the operation's is admitted.
    let admitted = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 1, 400),
    );
    assert!(admitted.is_ok());
}

#[test]
fn admission_consume_and_output_revalidation_ask_the_authority_with_the_same_captured_values() {
    let world = world();
    let mut table = table();
    let operations = known_operations(&world);
    let admitted = admit_with(
        &mut table,
        &world,
        &operations,
        &project(),
        CONSUMER_SESSION,
        &handle(&world, 1, 500),
    )
    .expect("admitted");
    let consumed = table
        .consume(ConsumeRequest {
            now_ms: 20,
            admitted,
            live: &live(&world),
            operations: &operations,
        })
        .expect("consumed");
    ptr_runtime::revalidate_operation(&consumed, &operations, 30).expect("still current");
    let expected = (project(), CONSUMER_SESSION, PRODUCER_SESSION, 500);
    assert_eq!(
        operations.asked.borrow().as_slice(),
        &[expected.clone(), expected.clone(), expected]
    );
}
