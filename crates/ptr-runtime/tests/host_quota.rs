//! The host-wide byte quota (doc 42 rule 3b) shared by several handle tables: a producer that
//! serves several consumer stages, several producers that fill one consumer, and the release
//! rules per carrier. Public API only.

use ed25519_dalek::{Signer, SigningKey};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_protocol::activation::{BoundaryDtype, ChainElement, UpstreamChain};
use ptr_protocol::{content_digest, handle_signing_bytes, Carrier, CarrierLocator, TensorHandle};
use ptr_runtime::{
    AdmittedHandle, ConsumeRequest, ConsumedHandle, DeviceHealth, DeviceRecord,
    HandleAdmissionRequest, HandlePolicy, HandleRejection, HandleSource, HandleTable,
    HostQuotaPolicy, HostQuotaPolicyViolation, NodeHealth, NodeRecord, OperationCheck,
    OperationFault, OperationLiveness, PodPlacementController, ProjectPlacementRequest, PtrRuntime,
    QuotaRefusal, ReclaimReason, RuntimeStageLiveness, SharedHostQuota, Teardown, VerifiedTeardown,
    WorkerSession,
};
use ptr_types::{
    ArtifactId, CapsuleId, DeviceId, Generation, NodeId, PodId, ProjectId, Timestamp, TypeId,
};
use std::collections::BTreeMap;

const PAYLOAD: &str = "ptr.split.activation.v1";
const OPERATION: u64 = 31;
const PAGE: u64 = 4096;

struct AllLive;

impl OperationLiveness for AllLive {
    fn operation_admits(&self, _check: OperationCheck<'_>) -> Result<(), OperationFault> {
        Ok(())
    }
}

fn project() -> ProjectId {
    ProjectId::from("p")
}

struct World {
    runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
}

fn digest(name: &str, generation: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = name.len() as u8;
    out[1] = generation as u8;
    out
}

fn world() -> World {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("a runtime");
    for name in ["stage", "model"] {
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
        .register_node(NodeRecord {
            node_id: NodeId::from("n1"),
            zone: "z".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cpu"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .expect("node");
    placement
        .assign_in(ProjectPlacementRequest {
            project: project(),
            pod_id: PodId::from("stage"),
            artifact_id: ArtifactId::from("artifact"),
            generation: Generation(1),
            node_id: NodeId::from("n1"),
            device_id: DeviceId::from("cpu"),
            required_vram: 1,
        })
        .expect("placed");
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

/// Roomy per-table bounds, so only the host ledger can refuse.
fn table_policy() -> HandlePolicy {
    HandlePolicy {
        max_open_handles: 64,
        max_open_per_principal: 64,
        max_tombstones_per_principal: 64,
        max_inflight_bytes_per_principal: u64::MAX,
        max_handle_lifetime_ms: 1_000,
        max_clock_skew_ms: 100,
    }
}

fn quota(producer: u64, consumer: u64, host: u64, sessions: usize) -> SharedHostQuota {
    SharedHostQuota::new(HostQuotaPolicy::new(producer, consumer, host, sessions))
        .expect("a valid policy")
}

/// A table for consumer stage `consumer` with the producers `(principal, session)` registered.
fn table(consumer: &str, quota: &SharedHostQuota, producers: &[(&str, u64)]) -> HandleTable {
    let mut table = HandleTable::with_host_quota(table_policy(), consumer, quota.clone())
        .expect("a valid policy");
    for (principal, session) in producers {
        let released = table
            .register_producer(
                principal,
                signer().verifying_key(),
                "stage",
                WorkerSession(*session),
            )
            .expect("registered");
        assert!(released.is_empty());
    }
    table
}

fn handle(world: &World, consumer: &str, carrier: Carrier, nonce: u64) -> TensorHandle {
    let (carrier, locator) = match carrier {
        Carrier::HostShm => (
            Carrier::HostShm,
            CarrierLocator::HostShm {
                fd_slot: 100 + nonce as u32,
                offset: 0,
            },
        ),
        Carrier::Bytes => (
            Carrier::Bytes,
            CarrierLocator::Bytes {
                inline: vec![nonce as u8; 8],
            },
        ),
        _ => (
            Carrier::NvmeFile,
            CarrierLocator::NvmeFile {
                chunk_digest: [100 + nonce as u8; 32],
            },
        ),
    };
    let epoch = world
        .placement
        .placement_in(&project(), &PodId::from("stage"))
        .expect("placed")
        .epoch
        .0;
    let mut handle = TensorHandle {
        carrier,
        locator,
        dtype: BoundaryDtype::F32,
        shape: vec![1, 2],
        byte_len: 8,
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: epoch,
        signed_absolute_deadline_ms: nonce * 1_000,
        handle_nonce: nonce,
        parent_operation_id: OPERATION,
        consumer: consumer.to_owned(),
        payload_type: TypeId::from(PAYLOAD),
        content_digest: content_digest(&[nonce as u8; 8]),
        signature: [0; 64],
    };
    handle.signature = signer().sign(&handle_signing_bytes(&handle)).to_bytes();
    handle
}

/// One admission: `principal` sends `sent` to the table's consumer worker `consumer_session`,
/// with `footprint` as the allocated footprint.
fn admit_footprint(
    table: &mut HandleTable,
    world: &World,
    principal: &str,
    consumer_session: u64,
    sent: &TensorHandle,
    footprint: u64,
) -> Result<AdmittedHandle, HandleRejection> {
    let encoded = sent.encode().expect("encodes");
    table.admit(HandleAdmissionRequest {
        // Handle `n` expires at `n * 1000`; admission is 490 ms before that, so nonces ascend
        // with the (monotonic) table clock.
        now_ms: sent.handle_nonce * 1_000 - 490,
        principal,
        expected_operation: OPERATION,
        project: &project(),
        consumer_session: WorkerSession(consumer_session),
        operations: &AllLive,
        handle: HandleSource::Encoded(&encoded),
        footprint_bytes: footprint,
        live: &live(world),
    })
}

/// The footprint of the logical tensor (8 bytes): the ledger rounds it to one page.
fn admit(
    table: &mut HandleTable,
    world: &World,
    principal: &str,
    consumer_session: u64,
    sent: &TensorHandle,
) -> Result<AdmittedHandle, HandleRejection> {
    admit_footprint(table, world, principal, consumer_session, sent, 8)
}

fn consume(
    table: &mut HandleTable,
    world: &World,
    admitted: AdmittedHandle,
) -> Result<ConsumedHandle, HandleRejection> {
    table.consume(ConsumeRequest {
        now_ms: admitted.signed_absolute_deadline_ms() - 400,
        admitted,
        live: &live(world),
        operations: &AllLive,
    })
}

fn is_refused_with(result: Result<AdmittedHandle, HandleRejection>, expected: QuotaRefusal) {
    match result {
        Err(HandleRejection::HostQuota(refusal)) => {
            assert_eq!(
                std::mem::discriminant(&refusal),
                std::mem::discriminant(&expected),
                "{refusal:?}"
            );
        }
        other => panic!("expected {expected:?}, got {other:?}"),
    }
}

const PRODUCER_CAP: QuotaRefusal = QuotaRefusal::ProducerQuota {
    limit: 0,
    held: 0,
    requested: 0,
};
const CONSUMER_CAP: QuotaRefusal = QuotaRefusal::ConsumerQuota {
    limit: 0,
    held: 0,
    requested: 0,
};
const HOST_CAP: QuotaRefusal = QuotaRefusal::HostQuota {
    limit: 0,
    held: 0,
    requested: 0,
};

#[test]
fn one_producer_serving_two_consumer_stages_cannot_exceed_its_producer_cap() {
    let world = world();
    // The producer session may hold two pages; each table alone is far below its own bound.
    let quota = quota(2 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut to_a = table("a", &quota, &[("p", 1)]);
    let mut to_b = table("b", &quota, &[("p", 1)]);
    admit(
        &mut to_a,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("a1");
    admit(
        &mut to_b,
        &world,
        "p",
        102,
        &handle(&world, "b", Carrier::NvmeFile, 2),
    )
    .expect("b1");
    assert_eq!(quota.producer_bytes(WorkerSession(1)), 2 * PAGE);
    is_refused_with(
        admit(
            &mut to_a,
            &world,
            "p",
            101,
            &handle(&world, "a", Carrier::NvmeFile, 3),
        ),
        PRODUCER_CAP,
    );
    is_refused_with(
        admit(
            &mut to_b,
            &world,
            "p",
            102,
            &handle(&world, "b", Carrier::NvmeFile, 4),
        ),
        PRODUCER_CAP,
    );
    assert_eq!(quota.producer_bytes(WorkerSession(1)), 2 * PAGE);
    // The tables themselves hold 8 logical bytes each: they would have admitted.
    assert_eq!(to_a.inflight_bytes("p"), Some(8));
    assert_eq!(to_b.inflight_bytes("p"), Some(8));
}

#[test]
fn several_producers_cannot_fill_one_consumer_session_beyond_its_cap() {
    let world = world();
    let quota = quota(100 * PAGE, 2 * PAGE, 100 * PAGE, 8);
    let producers = [("p1", 1), ("p2", 2), ("p3", 3)];
    let mut table = table("a", &quota, &producers);
    admit(
        &mut table,
        &world,
        "p1",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("1");
    admit(
        &mut table,
        &world,
        "p2",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("2");
    is_refused_with(
        admit(
            &mut table,
            &world,
            "p3",
            101,
            &handle(&world, "a", Carrier::NvmeFile, 3),
        ),
        CONSUMER_CAP,
    );
    assert_eq!(quota.consumer_bytes(WorkerSession(101)), 2 * PAGE);
    // Another consumer session of the same stage has its own cap.
    admit(
        &mut table,
        &world,
        "p3",
        102,
        &handle(&world, "a", Carrier::NvmeFile, 4),
    )
    .expect("4");
}

#[test]
fn several_principals_together_reach_the_host_cap_and_are_refused_typed() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 3 * PAGE, 16);
    let mut to_a = table("a", &quota, &[("p1", 1), ("p2", 2)]);
    let mut to_b = table("b", &quota, &[("p3", 3), ("p4", 4)]);
    admit(
        &mut to_a,
        &world,
        "p1",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("1");
    admit(
        &mut to_a,
        &world,
        "p2",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("2");
    admit(
        &mut to_b,
        &world,
        "p3",
        102,
        &handle(&world, "b", Carrier::NvmeFile, 3),
    )
    .expect("3");
    is_refused_with(
        admit(
            &mut to_b,
            &world,
            "p4",
            102,
            &handle(&world, "b", Carrier::NvmeFile, 4),
        ),
        HOST_CAP,
    );
    let usage = quota.usage();
    assert_eq!(usage.host_bytes, 3 * PAGE);
    assert!(usage.host_bytes <= quota.policy().max_bytes_host);
    assert_eq!(usage.release_faults, 0);
}

#[test]
fn a_shared_host_shm_charge_is_released_only_by_verified_teardown() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    let sent = handle(&world, "a", Carrier::HostShm, 1);
    let admitted = admit(&mut table, &world, "p", 101, &sent).expect("admitted");
    // Counted against both quotas and the host.
    assert_eq!(quota.producer_bytes(WorkerSession(1)), PAGE);
    assert_eq!(quota.consumer_bytes(WorkerSession(101)), PAGE);
    let consumed = consume(&mut table, &world, admitted).expect("consumed");
    // `Consumed` / `finish` does not recredit.
    assert_eq!(table.finish(consumed), Ok(None));
    assert_eq!(quota.host_bytes(), PAGE);
    // Nor does the deadline plus the skew.
    assert!(table.expire(10_000).is_empty());
    assert_eq!(quota.host_bytes(), PAGE);
    // A teardown of another pair does not either.
    let other = table.release_after_teardown(Teardown {
        principal: "p",
        producer_session: WorkerSession(1),
        consumer_session: WorkerSession(999),
    });
    assert!(other.is_empty());
    assert_eq!(quota.host_bytes(), PAGE);
    let notices = table.release_after_teardown(Teardown {
        principal: "p",
        producer_session: WorkerSession(1),
        consumer_session: WorkerSession(101),
    });
    assert_eq!(notices.len(), 1);
    assert_eq!(notices[0].reason, ReclaimReason::TornDown);
    assert_eq!(quota.usage().host_bytes, 0);
    assert_eq!(quota.producer_bytes(WorkerSession(1)), 0);
    assert_eq!(quota.consumer_bytes(WorkerSession(101)), 0);
    // A second teardown releases nothing and never double-credits.
    assert!(table
        .release_after_teardown(Teardown {
            principal: "p",
            producer_session: WorkerSession(1),
            consumer_session: WorkerSession(101),
        })
        .is_empty());
    assert_eq!(quota.usage().release_faults, 0);
}

#[test]
fn an_abandoned_or_expired_shared_handle_keeps_its_charge_until_teardown() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    let abandoned = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::HostShm, 1),
    )
    .expect("admitted");
    table.abandon(abandoned).expect("abandoned");
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::HostShm, 2),
    )
    .expect("admitted");
    assert_eq!(quota.host_bytes(), 2 * PAGE);
    assert!(table.expire(10_000).is_empty());
    assert_eq!(quota.host_bytes(), 2 * PAGE);
    // Replacing or removing the producer releases no shared bytes either.
    assert!(table
        .register_producer("p", signer().verifying_key(), "stage", WorkerSession(2))
        .expect("respawned")
        .is_empty());
    assert!(table.unregister_producer("p").is_empty());
    assert_eq!(quota.host_bytes(), 2 * PAGE);
    let notices = table.release_after_teardown(Teardown {
        principal: "p",
        producer_session: WorkerSession(1),
        consumer_session: WorkerSession(101),
    });
    assert_eq!(notices.len(), 2);
    assert_eq!(quota.usage().host_bytes, 0);
}

#[test]
fn a_late_teardown_of_the_old_pair_leaves_the_respawned_pairs_charge_alone() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::HostShm, 1),
    )
    .expect("old");
    let _ = table
        .register_producer("p", signer().verifying_key(), "stage", WorkerSession(2))
        .expect("respawned");
    admit(
        &mut table,
        &world,
        "p",
        102,
        &handle(&world, "a", Carrier::HostShm, 2),
    )
    .expect("new");
    assert_eq!(quota.host_bytes(), 2 * PAGE);
    let notices = table.release_after_teardown(Teardown {
        principal: "p",
        producer_session: WorkerSession(1),
        consumer_session: WorkerSession(101),
    });
    assert_eq!(notices.len(), 1);
    assert_eq!(quota.host_bytes(), PAGE);
    assert_eq!(quota.pair_bytes(WorkerSession(2), WorkerSession(102)), PAGE);
}

#[test]
fn bytes_and_other_unshared_carriers_are_recredited_at_the_notice_that_frees_the_buffer() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    // `Bytes`: finish releases.
    let admitted = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::Bytes, 1),
    )
    .expect("admitted");
    assert_eq!(quota.host_bytes(), PAGE);
    let consumed = consume(&mut table, &world, admitted).expect("consumed");
    // Consuming releases nothing; only the notice that frees the buffer does.
    assert_eq!(quota.host_bytes(), PAGE);
    let notice = table.finish(consumed).expect("finished").expect("a notice");
    assert_eq!(notice.reason, ReclaimReason::Consumed);
    assert_eq!(quota.host_bytes(), 0);
    // Not finished: the deadline plus the skew frees it.
    let dropped = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("admitted");
    let dropped = consume(&mut table, &world, dropped).expect("consumed");
    assert_eq!(quota.host_bytes(), PAGE);
    drop(dropped);
    assert_eq!(table.expire(2_100).len(), 1);
    assert_eq!(quota.host_bytes(), 0);
    // An abandoned handle is released at expiry, not before.
    let abandoned = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::Bytes, 3),
    )
    .expect("admitted");
    table.abandon(abandoned).expect("abandoned");
    assert_eq!(quota.host_bytes(), PAGE);
    assert_eq!(table.expire(3_100).len(), 1);
    assert_eq!(quota.host_bytes(), 0);
    // Producer removal and replacement release open unshared handles once.
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::Bytes, 4),
    )
    .expect("open");
    assert_eq!(table.unregister_producer("p").len(), 1);
    assert_eq!(quota.host_bytes(), 0);
    assert!(table.expire(4_100).is_empty());
    assert_eq!(quota.usage().release_faults, 0);
    assert_eq!(quota.usage().producer_sessions, 0);
}

#[test]
fn many_small_handles_are_charged_per_page_and_an_overflow_is_refused_typed() {
    let world = world();
    let quota = quota(2 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    // Eight logical bytes each, one page each.
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("1");
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("2");
    is_refused_with(
        admit(
            &mut table,
            &world,
            "p",
            101,
            &handle(&world, "a", Carrier::NvmeFile, 3),
        ),
        PRODUCER_CAP,
    );
    // A footprint just above a page rounds to two.
    let big = quota_footprint(&world, PAGE + 1);
    assert!(matches!(
        big,
        Err(HandleRejection::HostQuota(QuotaRefusal::ProducerQuota {
            requested: 8192,
            ..
        }))
    ));
    // A footprint that cannot be rounded is an overflow, not a wrap.
    let huge = quota_footprint(&world, u64::MAX);
    assert!(matches!(
        huge,
        Err(HandleRejection::HostQuota(QuotaRefusal::Overflow { .. }))
    ));
}

/// A fresh table with room: admit one handle with `footprint` as its allocated footprint.
fn quota_footprint(world: &World, footprint: u64) -> Result<AdmittedHandle, HandleRejection> {
    let quota = quota(2 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    admit(
        &mut table,
        world,
        "p",
        101,
        &handle(world, "a", Carrier::NvmeFile, 1),
    )
    .expect("held");
    admit_footprint(
        &mut table,
        world,
        "p",
        101,
        &handle(world, "a", Carrier::NvmeFile, 2),
        footprint,
    )
}

#[test]
fn a_refused_charge_leaves_table_and_ledger_unchanged_and_the_nonce_reusable() {
    let world = world();
    let quota = quota(PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    let held = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("held");
    let usage = quota.usage();
    let slots = table.slots_held("p");
    let open = table.open_handles();
    let inflight = table.inflight_bytes("p");
    is_refused_with(
        admit(
            &mut table,
            &world,
            "p",
            101,
            &handle(&world, "a", Carrier::NvmeFile, 2),
        ),
        PRODUCER_CAP,
    );
    assert_eq!(quota.usage(), usage);
    assert_eq!(table.slots_held("p"), slots);
    assert_eq!(table.open_handles(), open);
    assert_eq!(table.inflight_bytes("p"), inflight);
    // No tombstone: after room is freed the same nonce is admitted.
    table.abandon(held).expect("abandoned");
    assert_eq!(table.expire(1_100).len(), 1);
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("again");
}

#[test]
fn the_session_map_is_bounded_and_refuses_typed() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 1);
    let mut table = table("a", &quota, &[("p1", 1), ("p2", 2)]);
    admit(
        &mut table,
        &world,
        "p1",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("1");
    let refused = admit(
        &mut table,
        &world,
        "p2",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    );
    assert!(matches!(
        refused,
        Err(HandleRejection::HostQuota(QuotaRefusal::TooManySessions {
            limit: 1
        }))
    ));
    assert_eq!(quota.usage().producer_sessions, 1);
}

#[test]
fn the_policy_refuses_every_zero_bound() {
    let ok = HostQuotaPolicy::new(1, 1, 1, 1);
    assert_eq!(ok.validate(), Ok(()));
    assert_eq!(ok.charge_granularity_bytes, PAGE);
    for (policy, violation) in [
        (
            HostQuotaPolicy {
                max_bytes_per_producer_session: 0,
                ..ok
            },
            HostQuotaPolicyViolation::ZeroProducerBytes,
        ),
        (
            HostQuotaPolicy {
                max_bytes_per_consumer_session: 0,
                ..ok
            },
            HostQuotaPolicyViolation::ZeroConsumerBytes,
        ),
        (
            HostQuotaPolicy {
                max_bytes_host: 0,
                ..ok
            },
            HostQuotaPolicyViolation::ZeroHostBytes,
        ),
        (
            HostQuotaPolicy {
                max_tracked_sessions: 0,
                ..ok
            },
            HostQuotaPolicyViolation::ZeroTrackedSessions,
        ),
        (
            HostQuotaPolicy {
                charge_granularity_bytes: 0,
                ..ok
            },
            HostQuotaPolicyViolation::ZeroGranularity,
        ),
    ] {
        assert_eq!(SharedHostQuota::new(policy).err(), Some(violation));
    }
}

#[test]
fn a_table_without_a_host_ledger_is_unchanged() {
    let world = world();
    let mut table = HandleTable::new(table_policy(), "a").expect("policy");
    let _ = table
        .register_producer("p", signer().verifying_key(), "stage", WorkerSession(1))
        .expect("registered");
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("1");
    assert_eq!(table.inflight_bytes("p"), Some(8));
}

#[test]
fn a_pair_release_recredits_the_orphaned_bytes_and_a_late_table_notice_debits_nothing_else() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    let first = admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 1),
    )
    .expect("first");
    let first = consume(&mut table, &world, first).expect("consumed");
    // The supervisor reaped the pair and releases what it held: exactly the orphaned page.
    let proof = VerifiedTeardown::reaped_and_drained(WorkerSession(1), WorkerSession(101));
    assert_eq!(quota.release_pair(proof), PAGE);
    assert_eq!(quota.host_bytes(), 0);
    // A new charge of the same pair, then the old table notice arrives.
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::NvmeFile, 2),
    )
    .expect("second");
    assert_eq!(quota.pair_bytes(WorkerSession(1), WorkerSession(101)), PAGE);
    let notice = table.finish(first).expect("finished").expect("a notice");
    assert_eq!(notice.reason, ReclaimReason::Consumed);
    // The stale release is a counted no-op: the new charge is intact.
    assert_eq!(quota.pair_bytes(WorkerSession(1), WorkerSession(101)), PAGE);
    assert_eq!(quota.host_bytes(), PAGE);
    assert_eq!(quota.usage().release_faults, 1);
}

#[test]
fn the_supervisor_can_list_tracked_pairs_and_release_those_of_dropped_tables() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 2);
    {
        let mut dropped = table("a", &quota, &[("p", 1)]);
        admit(
            &mut dropped,
            &world,
            "p",
            101,
            &handle(&world, "a", Carrier::HostShm, 1),
        )
        .expect("admitted");
    }
    // The dropped table left its charge: it still occupies the bounded map.
    let pairs = quota.tracked_pairs();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].bytes, PAGE);
    assert!(pairs.len() <= quota.policy().max_tracked_sessions);
    for pair in pairs {
        quota.release_pair(VerifiedTeardown::reaped_and_drained(
            pair.producer_session,
            pair.consumer_session,
        ));
    }
    assert!(quota.tracked_pairs().is_empty());
    assert_eq!(quota.usage().host_bytes, 0);
}

#[test]
fn unregistering_by_session_leaves_a_respawned_producer_alone() {
    let world = world();
    let quota = quota(100 * PAGE, 100 * PAGE, 100 * PAGE, 8);
    let mut table = table("a", &quota, &[("p", 1)]);
    // The producer is respawned under session 2.
    assert!(table
        .register_producer("p", signer().verifying_key(), "stage", WorkerSession(2))
        .expect("respawned")
        .is_empty());
    // Fencing the old session is a no-op: the new one still admits.
    assert!(table
        .unregister_producer_session("p", WorkerSession(1))
        .is_empty());
    admit(
        &mut table,
        &world,
        "p",
        101,
        &handle(&world, "a", Carrier::HostShm, 1),
    )
    .expect("the respawned producer is still registered");
    // Fencing the current session does unregister it.
    let _ = table.unregister_producer_session("p", WorkerSession(2));
    assert_eq!(
        admit(
            &mut table,
            &world,
            "p",
            101,
            &handle(&world, "a", Carrier::HostShm, 2)
        )
        .unwrap_err(),
        HandleRejection::UnknownPrincipal
    );
}
