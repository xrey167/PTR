//! Tier transfer authorization: the publish-time recheck, project-scoped grants and their
//! single-use, bounded replay protection.

use ptr_runtime::{
    ActiveWriteBinding, ManualTierClock, PrefetchOutcome, ReplicaState, TierClock, TierGrant,
    TierGrantLimits, TierResidencyController, TierRuntimeError, TierTransferAuthorization,
    TierTransferReceipt, TOMBSTONE_ENTRY_BYTES,
};
use ptr_storage::{
    ChunkDescriptor, ChunkReceipt, CpuTierBackend, StorageTier, TierBackend, TierBackendId,
    TierCapabilities, TierFuture, TierHealth, TierIntegrityReport, TierObjectDomain,
    TierObjectManifest,
};
use ptr_types::{Digest, Generation, ProjectId, Revision, SessionId};
use std::future::Future;
use std::pin::{pin, Pin};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

type Hook = Arc<dyn Fn() + Send + Sync>;

/// A CPU backend that reports a chosen tier and runs a one-shot hook after the first chunk
/// verification: the moment between the copy and the publish of a transfer into it.
struct TestBackend {
    inner: CpuTierBackend,
    tier: StorageTier,
    hook: Mutex<Option<Hook>>,
}

impl TestBackend {
    fn new(id: &str, tier: StorageTier) -> Arc<Self> {
        Arc::new(Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            tier,
            hook: Mutex::new(None),
        })
    }

    fn after_next_verify(&self, hook: impl Fn() + Send + Sync + 'static) {
        *self.hook.lock().unwrap() = Some(Arc::new(hook));
    }

    fn holds(&self, chunk: &ChunkDescriptor) -> bool {
        block_on(self.inner.get_chunk(chunk)).is_ok()
    }
}

impl TierBackend for TestBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        let mut capabilities = self.inner.capabilities();
        capabilities.tier = self.tier;
        capabilities
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        self.inner.get_chunk(chunk)
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let report = self.inner.verify_chunk(chunk).await?;
            let hook = self.hook.lock().unwrap().take();
            if let Some(hook) = hook {
                hook();
            }
            Ok(report)
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

struct Rig {
    controller: Arc<TierResidencyController>,
    clock: Arc<ManualTierClock>,
    source: Arc<TestBackend>,
    destination: Arc<TestBackend>,
}

const SOURCE: &str = "ram";
const DESTINATION: &str = "gpu";

fn rig_with_limits(limits: TierGrantLimits) -> Rig {
    let clock = Arc::new(ManualTierClock::new(1_000));
    let controller = Arc::new(
        TierResidencyController::default()
            .with_grants(clock.clone() as Arc<dyn TierClock>, limits)
            .unwrap(),
    );
    let source = TestBackend::new(SOURCE, StorageTier::Cpu);
    let destination = TestBackend::new(DESTINATION, StorageTier::Gpu);
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination.clone())).unwrap();
    Rig {
        controller,
        clock,
        source,
        destination,
    }
}

fn rig() -> Rig {
    rig_with_limits(TierGrantLimits::default())
}

fn manifest(logical_id: &str, bytes: &[u8]) -> TierObjectManifest {
    TierObjectManifest::new(
        TierObjectDomain::ModelWeight,
        logical_id,
        Generation(4),
        Revision(8),
        [9; 32],
        vec![ChunkDescriptor::for_bytes(0, 0, bytes)],
    )
    .unwrap()
}

fn id(name: &str) -> TierBackendId {
    TierBackendId::from(name)
}

impl Rig {
    fn seed(&self, logical_id: &str, bytes: &[u8], project: Option<&str>) -> TierObjectManifest {
        let manifest = manifest(logical_id, bytes);
        block_on(
            self.source
                .put_chunk(manifest.chunks[0].clone(), bytes.to_vec()),
        )
        .unwrap();
        match project {
            Some(project) => block_on(self.controller.register_project_object(
                ProjectId::from(project),
                manifest.clone(),
                id(SOURCE),
            )),
            None => block_on(
                self.controller
                    .register_object(manifest.clone(), id(SOURCE)),
            ),
        }
        .unwrap();
        manifest
    }

    fn grant(
        &self,
        project: &str,
        object: &TierObjectManifest,
        nonce: u128,
        session: &str,
    ) -> TierGrant {
        TierGrant {
            project: ProjectId::from(project),
            object: object.root_digest,
            source: id(SOURCE),
            destination: id(DESTINATION),
            write_authorization: None,
            receiver: SessionId::from(session),
            nonce,
            deadline_millis: self.clock.now_millis() + 10_000,
        }
    }

    fn destination_state(&self, object: &Digest) -> Option<ReplicaState> {
        self.controller
            .residency(object)
            .unwrap()
            .replicas
            .iter()
            .find(|replica| replica.backend == id(DESTINATION))
            .map(|replica| replica.state)
    }

    fn source_pins(&self, object: &Digest) -> u64 {
        self.controller
            .residency(object)
            .unwrap()
            .replicas
            .iter()
            .find(|replica| replica.backend == id(SOURCE))
            .unwrap()
            .pins
    }
}

fn session(name: &str) -> SessionId {
    SessionId::from(name)
}

// ---- publish-time recheck of the write authorization ----

fn write_binding(epoch: u64) -> ActiveWriteBinding {
    ActiveWriteBinding {
        generation: Generation(4),
        placement_epoch: epoch,
        fencing_token: 100 + u128::from(epoch),
    }
}

fn authorization(epoch: u64) -> TierTransferAuthorization {
    TierTransferAuthorization {
        generation: Generation(4),
        placement_epoch: epoch,
        fencing_token: 100 + u128::from(epoch),
    }
}

#[test]
fn control_an_unrevoked_authorization_publishes_the_destination_replica() {
    let rig = rig();
    let object = rig.seed("weights-a", b"weight-bytes-a", None);
    rig.controller
        .bind_active_write(object.root_digest, write_binding(1))
        .unwrap();
    let receipt = block_on(rig.controller.transfer(
        object.root_digest,
        &id(SOURCE),
        &id(DESTINATION),
        Some(authorization(1)),
    ))
    .unwrap();
    assert_eq!(receipt.chunks, 1);
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Available)
    );
    assert!(rig.destination.holds(&object.chunks[0]));
}

#[test]
fn a_revocation_between_copy_and_publish_leaves_the_destination_unpublished_and_cleaned_up() {
    let rig = rig();
    let object = rig.seed("weights-a", b"weight-bytes-a", None);
    rig.controller
        .bind_active_write(object.root_digest, write_binding(1))
        .unwrap();
    let controller = rig.controller.clone();
    let digest = object.root_digest;
    rig.destination.after_next_verify(move || {
        // The chunk is copied and verified; the grant is revoked before the publish.
        controller
            .bind_active_write(digest, write_binding(2))
            .unwrap();
    });
    let refusal = block_on(rig.controller.transfer(
        object.root_digest,
        &id(SOURCE),
        &id(DESTINATION),
        Some(authorization(1)),
    ))
    .unwrap_err();
    assert_eq!(refusal, TierRuntimeError::AuthorizationRevoked);
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Revoked)
    );
    assert!(!rig.destination.holds(&object.chunks[0]), "chunks removed");
    assert_eq!(rig.source_pins(&object.root_digest), 0, "source pin freed");
    assert_eq!(
        rig.controller
            .residency(&object.root_digest)
            .unwrap()
            .available_replicas(),
        1
    );
    // The refused replica is not readable.
    assert!(block_on(
        rig.controller
            .read_admitted_object(object.root_digest, &id(DESTINATION))
    )
    .is_err());
    // A transfer under the current authorization works afterwards.
    block_on(rig.controller.transfer(
        object.root_digest,
        &id(SOURCE),
        &id(DESTINATION),
        Some(authorization(2)),
    ))
    .unwrap();
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Available)
    );
}

#[test]
fn cleanup_after_a_revocation_keeps_chunks_another_object_still_uses() {
    let rig = rig();
    let shared = rig.seed("shared-1", b"same-bytes", None);
    let revoked = rig.seed("shared-2", b"same-bytes", None);
    block_on(
        rig.controller
            .transfer(shared.root_digest, &id(SOURCE), &id(DESTINATION), None),
    )
    .unwrap();
    rig.controller
        .bind_active_write(revoked.root_digest, write_binding(1))
        .unwrap();
    let controller = rig.controller.clone();
    let digest = revoked.root_digest;
    rig.destination.after_next_verify(move || {
        controller
            .bind_active_write(digest, write_binding(2))
            .unwrap();
    });
    assert_eq!(
        block_on(rig.controller.transfer(
            revoked.root_digest,
            &id(SOURCE),
            &id(DESTINATION),
            Some(authorization(1)),
        )),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    // The identical chunk belongs to the published object too, so it stays.
    assert!(rig.destination.holds(&shared.chunks[0]));
    assert_eq!(
        rig.destination_state(&shared.root_digest),
        Some(ReplicaState::Available)
    );
}

#[test]
fn a_binding_added_during_an_unauthorized_copy_also_refuses_publication() {
    let rig = rig();
    let object = rig.seed("weights-a", b"weight-bytes-a", None);
    let controller = rig.controller.clone();
    let digest = object.root_digest;
    rig.destination.after_next_verify(move || {
        controller
            .bind_active_write(digest, write_binding(1))
            .unwrap();
    });
    assert_eq!(
        block_on(
            rig.controller
                .transfer(object.root_digest, &id(SOURCE), &id(DESTINATION), None)
        ),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    assert!(!rig.destination.holds(&object.chunks[0]));
}

#[test]
fn a_stale_authorization_is_still_refused_before_the_copy() {
    let rig = rig();
    let object = rig.seed("weights-a", b"weight-bytes-a", None);
    rig.controller
        .bind_active_write(object.root_digest, write_binding(2))
        .unwrap();
    assert_eq!(
        block_on(rig.controller.transfer(
            object.root_digest,
            &id(SOURCE),
            &id(DESTINATION),
            Some(authorization(1)),
        )),
        Err(TierRuntimeError::StaleWriteBinding)
    );
    assert!(!rig.destination.holds(&object.chunks[0]), "nothing copied");
}

#[test]
fn a_grant_revoked_during_the_copy_is_refused_and_not_spent() {
    let rig = rig();
    let object = rig.seed("weights-a", b"weight-bytes-a", Some("alpha"));
    rig.controller
        .bind_active_write(object.root_digest, write_binding(1))
        .unwrap();
    let mut grant = rig.grant("alpha", &object, 7, "worker-1");
    grant.write_authorization = Some(authorization(1));
    let controller = rig.controller.clone();
    let digest = object.root_digest;
    rig.destination.after_next_verify(move || {
        controller
            .bind_active_write(digest, write_binding(2))
            .unwrap();
    });
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&grant, &session("worker-1"))
        ),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    assert!(!rig.destination.holds(&object.chunks[0]));
    assert_eq!(
        rig.controller
            .live_grant_tombstones(&session("worker-1"))
            .unwrap(),
        0,
        "a refused publish does not spend the grant"
    );
}

// ---- project scope ----

#[test]
fn a_grant_of_one_project_never_moves_an_object_of_another() {
    let rig = rig();
    let alpha = rig.seed("alpha-weights", b"alpha-bytes", Some("alpha"));
    let beta = rig.seed("beta-weights", b"beta-bytes!", Some("beta"));
    // Project alpha's grant, aimed at project beta's object.
    let crossing = rig.grant("alpha", &beta, 1, "worker-1");
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&crossing, &session("worker-1"))
        ),
        Err(TierRuntimeError::ProjectMismatch)
    );
    assert_eq!(rig.destination_state(&beta.root_digest), None);
    assert!(!rig.destination.holds(&beta.chunks[0]));
    // The prefetch entry point refuses the same way and publishes nothing.
    assert_eq!(
        block_on(
            rig.controller
                .prefetch_with_grant(&crossing, &session("worker-1"))
        ),
        PrefetchOutcome::Skipped(TierRuntimeError::ProjectMismatch)
    );
    assert_eq!(rig.destination_state(&beta.root_digest), None);
    // Each project's own grant works for its own object only.
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &alpha, 2, "worker-1"),
        &session("worker-1"),
    ))
    .unwrap();
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("beta", &beta, 3, "worker-1"),
        &session("worker-1"),
    ))
    .unwrap();
    assert_eq!(
        rig.controller.object_project(&alpha.root_digest).unwrap(),
        Some(ProjectId::from("alpha"))
    );
}

#[test]
fn a_project_object_needs_a_grant_and_an_unscoped_object_takes_no_project_grant() {
    let rig = rig();
    let scoped = rig.seed("scoped", b"scoped-bytes", Some("alpha"));
    let unscoped = rig.seed("unscoped", b"unscoped-byt", None);
    assert_eq!(
        block_on(
            rig.controller
                .transfer(scoped.root_digest, &id(SOURCE), &id(DESTINATION), None)
        ),
        Err(TierRuntimeError::GrantRequired)
    );
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(
            &rig.grant("alpha", &unscoped, 1, "worker-1"),
            &session("worker-1")
        )),
        Err(TierRuntimeError::ProjectMismatch)
    );
    // The old entry point keeps working for the unscoped object.
    block_on(
        rig.controller
            .transfer(unscoped.root_digest, &id(SOURCE), &id(DESTINATION), None),
    )
    .unwrap();
}

#[test]
fn an_object_cannot_be_registered_under_a_second_project_or_unscoped() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let other = block_on(rig.controller.register_project_object(
        ProjectId::from("beta"),
        object.clone(),
        id(DESTINATION),
    ));
    assert_eq!(other, Err(TierRuntimeError::ProjectMismatch));
    let unscoped = block_on(
        rig.controller
            .register_object(object.clone(), id(DESTINATION)),
    );
    assert_eq!(unscoped, Err(TierRuntimeError::ProjectMismatch));
    assert_eq!(
        block_on(rig.controller.register_project_object(
            ProjectId::from(" "),
            object,
            id(DESTINATION)
        )),
        Err(TierRuntimeError::GrantInvalid)
    );
}

// ---- receiver binding, deadline, replay ----

#[test]
fn a_grant_is_bound_to_its_receiver_session() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let grant = rig.grant("alpha", &object, 1, "worker-1");
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&grant, &session("worker-2"))
        ),
        Err(TierRuntimeError::ReceiverSessionMismatch)
    );
    assert_eq!(rig.destination_state(&object.root_digest), None);
    // The refusal did not spend it.
    block_on(
        rig.controller
            .transfer_with_grant(&grant, &session("worker-1")),
    )
    .unwrap();
}

#[test]
fn a_malformed_grant_is_refused() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let good = rig.grant("alpha", &object, 1, "worker-1");
    let mut zero_nonce = good.clone();
    zero_nonce.nonce = 0;
    let mut no_deadline = good.clone();
    no_deadline.deadline_millis = 0;
    let mut same_backend = good.clone();
    same_backend.destination = same_backend.source.clone();
    let mut no_receiver = good.clone();
    no_receiver.receiver = session(" ");
    for bad in [zero_nonce, no_deadline, same_backend, no_receiver] {
        let presenter = bad.receiver.clone();
        assert_eq!(
            block_on(rig.controller.transfer_with_grant(&bad, &presenter)),
            Err(TierRuntimeError::GrantInvalid)
        );
    }
}

#[test]
fn an_expired_grant_is_refused_and_one_expiring_during_the_copy_is_cleaned_up() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let grant = rig.grant("alpha", &object, 1, "worker-1");
    rig.clock.set(grant.deadline_millis + 1);
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&grant, &session("worker-1"))
        ),
        Err(TierRuntimeError::GrantExpired)
    );
    assert_eq!(rig.destination_state(&object.root_digest), None);

    let fresh = rig.grant("alpha", &object, 2, "worker-1");
    let clock = rig.clock.clone();
    let deadline = fresh.deadline_millis;
    rig.destination
        .after_next_verify(move || clock.set(deadline + 1));
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&fresh, &session("worker-1"))
        ),
        Err(TierRuntimeError::GrantExpired)
    );
    assert!(!rig.destination.holds(&object.chunks[0]));
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Revoked)
    );
    assert_eq!(
        rig.controller
            .live_grant_tombstones(&session("worker-1"))
            .unwrap(),
        0
    );
}

#[test]
fn a_grant_is_consumed_by_the_publish_and_cannot_be_replayed() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let grant = rig.grant("alpha", &object, 42, "worker-1");
    let presenter = session("worker-1");
    block_on(rig.controller.transfer_with_grant(&grant, &presenter)).unwrap();
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 1);
    // Replay while the replica is still there.
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&grant, &presenter)),
        Err(TierRuntimeError::GrantReplayed)
    );
    // Replay after the replica was evicted: it must not be published a second time.
    block_on(rig.controller.evict(object.root_digest, &id(DESTINATION))).unwrap();
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&grant, &presenter)),
        Err(TierRuntimeError::GrantReplayed)
    );
    assert!(!rig.destination.holds(&object.chunks[0]));
    // A fresh grant (new nonce) does.
    block_on(
        rig.controller
            .transfer_with_grant(&rig.grant("alpha", &object, 43, "worker-1"), &presenter),
    )
    .unwrap();
}

#[test]
fn the_same_nonce_is_independent_per_receiver_session() {
    let rig = rig();
    let first = rig.seed("weights-1", b"weight-bytes-1", Some("alpha"));
    let second = rig.seed("weights-2", b"weight-bytes-2", Some("alpha"));
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &first, 5, "worker-1"),
        &session("worker-1"),
    ))
    .unwrap();
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &second, 5, "worker-2"),
        &session("worker-2"),
    ))
    .unwrap();
}

#[test]
fn a_tombstone_lives_until_the_deadline_plus_the_skew() {
    let limits = TierGrantLimits {
        skew_millis: 500,
        ..TierGrantLimits::default()
    };
    let rig = rig_with_limits(limits);
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let grant = rig.grant("alpha", &object, 9, "worker-1");
    let presenter = session("worker-1");
    block_on(rig.controller.transfer_with_grant(&grant, &presenter)).unwrap();
    rig.clock.set(grant.deadline_millis + 500);
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 1);
    rig.clock.set(grant.deadline_millis + 501);
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 0);
    // Past the deadline the grant is expired, so it still cannot be used.
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&grant, &presenter)),
        Err(TierRuntimeError::GrantExpired)
    );
}

#[test]
fn two_transfers_racing_on_one_nonce_publish_only_one() {
    let rig = rig();
    let first = rig.seed("weights-1", b"weight-bytes-1", Some("alpha"));
    let second = rig.seed("weights-2", b"weight-bytes-2", Some("alpha"));
    let first_grant = rig.grant("alpha", &first, 77, "worker-1");
    let second_grant = rig.grant("alpha", &second, 77, "worker-1");
    let controller = rig.controller.clone();
    // While the first transfer holds the nonce between copy and publish, a second transfer
    // (another object) presenting the same nonce is refused before it copies anything.
    rig.destination.after_next_verify(move || {
        assert_eq!(
            block_on(controller.transfer_with_grant(&second_grant, &session("worker-1"))),
            Err(TierRuntimeError::GrantReplayed)
        );
    });
    block_on(
        rig.controller
            .transfer_with_grant(&first_grant, &session("worker-1")),
    )
    .unwrap();
    assert_eq!(
        rig.destination_state(&first.root_digest),
        Some(ReplicaState::Available)
    );
    assert_eq!(rig.destination_state(&second.root_digest), None);
    assert!(!rig.destination.holds(&second.chunks[0]));
}

// ---- bounded tombstone table ----

#[test]
fn the_tombstone_table_is_bounded_per_receiver_by_count() {
    let limits = TierGrantLimits {
        skew_millis: 0,
        max_tombstones_per_receiver: 2,
        ..TierGrantLimits::default()
    };
    let rig = rig_with_limits(limits);
    let presenter = session("worker-1");
    let objects: Vec<_> = (0..4)
        .map(|index| {
            rig.seed(
                &format!("weights-{index}"),
                format!("weight-bytes-{index}").as_bytes(),
                Some("alpha"),
            )
        })
        .collect();
    for (nonce, object) in (1u128..).zip(&objects[..2]) {
        block_on(
            rig.controller
                .transfer_with_grant(&rig.grant("alpha", object, nonce, "worker-1"), &presenter),
        )
        .unwrap();
    }
    let third = rig.grant("alpha", &objects[2], 3, "worker-1");
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&third, &presenter)),
        Err(TierRuntimeError::TombstoneSaturated)
    );
    assert_eq!(rig.destination_state(&objects[2].root_digest), None);
    assert_eq!(
        rig.controller.live_grant_tombstones(&presenter).unwrap(),
        2,
        "no live tombstone was evicted to make room"
    );
    // Another receiver has its own budget.
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &objects[3], 3, "worker-2"),
        &session("worker-2"),
    ))
    .unwrap();
    // Once the first two expire, the first receiver can spend again.
    rig.clock.set(rig.clock.now_millis() + 10_001);
    let again = rig.grant("alpha", &objects[2], 4, "worker-1");
    block_on(rig.controller.transfer_with_grant(&again, &presenter)).unwrap();
}

#[test]
fn the_tombstone_table_is_bounded_per_receiver_by_bytes() {
    let limits = TierGrantLimits {
        max_tombstone_bytes_per_receiver: TOMBSTONE_ENTRY_BYTES + TOMBSTONE_ENTRY_BYTES / 2,
        ..TierGrantLimits::default()
    };
    let rig = rig_with_limits(limits);
    let presenter = session("worker-1");
    let first = rig.seed("weights-1", b"weight-bytes-1", Some("alpha"));
    let second = rig.seed("weights-2", b"weight-bytes-2", Some("alpha"));
    block_on(
        rig.controller
            .transfer_with_grant(&rig.grant("alpha", &first, 1, "worker-1"), &presenter),
    )
    .unwrap();
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&rig.grant("alpha", &second, 2, "worker-1"), &presenter,)
        ),
        Err(TierRuntimeError::TombstoneSaturated)
    );
}

#[test]
fn the_number_of_receivers_is_bounded() {
    let limits = TierGrantLimits {
        max_receivers: 1,
        ..TierGrantLimits::default()
    };
    let rig = rig_with_limits(limits);
    let first = rig.seed("weights-1", b"weight-bytes-1", Some("alpha"));
    let second = rig.seed("weights-2", b"weight-bytes-2", Some("alpha"));
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &first, 1, "worker-1"),
        &session("worker-1"),
    ))
    .unwrap();
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(
            &rig.grant("alpha", &second, 1, "worker-2"),
            &session("worker-2"),
        )),
        Err(TierRuntimeError::TombstoneSaturated)
    );
    assert_eq!(rig.destination_state(&second.root_digest), None);
}

#[test]
fn zero_grant_limits_are_refused() {
    for limits in [
        TierGrantLimits {
            max_receivers: 0,
            ..TierGrantLimits::default()
        },
        TierGrantLimits {
            max_tombstones_per_receiver: 0,
            ..TierGrantLimits::default()
        },
        TierGrantLimits {
            max_tombstone_bytes_per_receiver: TOMBSTONE_ENTRY_BYTES - 1,
            ..TierGrantLimits::default()
        },
        TierGrantLimits {
            max_grant_lifetime_millis: 0,
            ..TierGrantLimits::default()
        },
    ] {
        let clock: Arc<dyn TierClock> = Arc::new(ManualTierClock::new(1));
        assert!(matches!(
            TierResidencyController::default().with_grants(clock, limits),
            Err(TierRuntimeError::GrantInvalid)
        ));
    }
}

// ---- calls that join an in-flight transfer or find the replica already available ----

type Outcome = Result<TierTransferReceipt, TierRuntimeError>;
type JoinFuture = Pin<Box<dyn Future<Output = Outcome> + Send>>;

enum Joined {
    Done(Outcome),
    Waiting(JoinFuture),
}

type JoinSlot = Arc<Mutex<Vec<Joined>>>;

/// Starts `grant`'s transfer from inside the leader's copy: polls it once, so it either
/// refuses at once or parks on the in-flight gate, and keeps it in `slot` to be finished
/// after the leader.
fn join_in_flight(
    controller: &Arc<TierResidencyController>,
    grant: TierGrant,
    presenter: &str,
    slot: &JoinSlot,
) -> impl Fn() + Send + Sync + 'static {
    let controller = controller.clone();
    let presenter = session(presenter);
    let slot = slot.clone();
    move || {
        let controller = controller.clone();
        let grant = grant.clone();
        let presenter = presenter.clone();
        let mut future: JoinFuture =
            Box::pin(async move { controller.transfer_with_grant(&grant, &presenter).await });
        let mut cx = Context::from_waker(Waker::noop());
        let entry = match future.as_mut().poll(&mut cx) {
            Poll::Ready(outcome) => Joined::Done(outcome),
            Poll::Pending => Joined::Waiting(future),
        };
        slot.lock().unwrap().push(entry);
    }
}

fn finish_joined(slot: &JoinSlot) -> Outcome {
    let entry = slot.lock().unwrap().pop().expect("a joined call");
    match entry {
        Joined::Done(outcome) => outcome,
        Joined::Waiting(future) => block_on(future),
    }
}

fn new_slot() -> JoinSlot {
    Arc::new(Mutex::new(Vec::new()))
}

#[test]
fn two_distinct_grants_on_one_transfer_are_each_rechecked_and_spent() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let leader = rig.grant("alpha", &object, 1, "worker-1");
    let follower = rig.grant("alpha", &object, 2, "worker-1");
    let slot = new_slot();
    rig.destination.after_next_verify(join_in_flight(
        &rig.controller,
        follower.clone(),
        "worker-1",
        &slot,
    ));
    let presenter = session("worker-1");
    block_on(rig.controller.transfer_with_grant(&leader, &presenter)).unwrap();
    finish_joined(&slot).unwrap();
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 2);
    for grant in [&leader, &follower] {
        assert_eq!(
            block_on(rig.controller.transfer_with_grant(grant, &presenter)),
            Err(TierRuntimeError::GrantReplayed)
        );
    }
}

#[test]
fn the_same_grant_presented_while_it_is_in_flight_is_a_replay() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let grant = rig.grant("alpha", &object, 1, "worker-1");
    let slot = new_slot();
    rig.destination.after_next_verify(join_in_flight(
        &rig.controller,
        grant.clone(),
        "worker-1",
        &slot,
    ));
    let presenter = session("worker-1");
    block_on(rig.controller.transfer_with_grant(&grant, &presenter)).unwrap();
    assert_eq!(finish_joined(&slot), Err(TierRuntimeError::GrantReplayed));
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 1);
}

#[test]
fn a_follower_whose_grant_expires_during_the_wait_gets_the_refusal() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let leader = rig.grant("alpha", &object, 1, "worker-1");
    let mut follower = rig.grant("alpha", &object, 2, "worker-1");
    follower.deadline_millis = rig.clock.now_millis() + 100;
    let slot = new_slot();
    let clock = rig.clock.clone();
    let join = join_in_flight(&rig.controller, follower.clone(), "worker-1", &slot);
    rig.destination.after_next_verify(move || {
        join();
        clock.advance(500);
    });
    let presenter = session("worker-1");
    block_on(rig.controller.transfer_with_grant(&leader, &presenter)).unwrap();
    assert_eq!(finish_joined(&slot), Err(TierRuntimeError::GrantExpired));
    assert_eq!(
        rig.controller.live_grant_tombstones(&presenter).unwrap(),
        1,
        "only the leader's grant was spent"
    );
}

#[test]
fn a_follower_whose_authorization_is_revoked_during_the_wait_gets_the_refusal() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    rig.controller
        .bind_active_write(object.root_digest, write_binding(1))
        .unwrap();
    let mut leader = rig.grant("alpha", &object, 1, "worker-1");
    leader.write_authorization = Some(authorization(1));
    let mut follower = rig.grant("alpha", &object, 2, "worker-1");
    follower.write_authorization = Some(authorization(1));
    let slot = new_slot();
    rig.destination.after_next_verify(join_in_flight(
        &rig.controller,
        follower.clone(),
        "worker-1",
        &slot,
    ));
    let presenter = session("worker-1");
    // The leader publishes; the binding changes before the follower is satisfied.
    block_on(rig.controller.transfer_with_grant(&leader, &presenter)).unwrap();
    rig.controller
        .bind_active_write(object.root_digest, write_binding(2))
        .unwrap();
    assert_eq!(
        finish_joined(&slot),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 1);
}

#[test]
fn when_the_leader_is_refused_a_joined_grant_is_refused_and_unspent() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    rig.controller
        .bind_active_write(object.root_digest, write_binding(1))
        .unwrap();
    let mut leader = rig.grant("alpha", &object, 1, "worker-1");
    leader.write_authorization = Some(authorization(1));
    let mut follower = rig.grant("alpha", &object, 2, "worker-1");
    follower.write_authorization = Some(authorization(1));
    let slot = new_slot();
    let join = join_in_flight(&rig.controller, follower, "worker-1", &slot);
    let controller = rig.controller.clone();
    let digest = object.root_digest;
    rig.destination.after_next_verify(move || {
        join();
        controller
            .bind_active_write(digest, write_binding(2))
            .unwrap();
    });
    let presenter = session("worker-1");
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&leader, &presenter)),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    assert_eq!(
        finish_joined(&slot),
        Err(TierRuntimeError::AuthorizationRevoked)
    );
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 0);
}

#[test]
fn a_grant_that_finds_the_replica_already_available_is_spent() {
    // Decision: the grant is satisfied by the existing replica, so it is rechecked and spent
    // (single use), and a second presentation is a replay.
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let presenter = session("worker-1");
    block_on(
        rig.controller
            .transfer_with_grant(&rig.grant("alpha", &object, 1, "worker-1"), &presenter),
    )
    .unwrap();
    let second = rig.grant("alpha", &object, 2, "worker-1");
    block_on(rig.controller.transfer_with_grant(&second, &presenter)).unwrap();
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 2);
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&second, &presenter)),
        Err(TierRuntimeError::GrantReplayed)
    );
}

#[test]
fn an_already_available_replica_does_not_launder_a_revoked_authorization() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let presenter = session("worker-1");
    block_on(
        rig.controller
            .transfer_with_grant(&rig.grant("alpha", &object, 1, "worker-1"), &presenter),
    )
    .unwrap();
    rig.controller
        .bind_active_write(object.root_digest, write_binding(3))
        .unwrap();
    let mut stale = rig.grant("alpha", &object, 2, "worker-1");
    stale.write_authorization = Some(authorization(1));
    assert_eq!(
        block_on(rig.controller.transfer_with_grant(&stale, &presenter)),
        Err(TierRuntimeError::StaleWriteBinding)
    );
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 1);
}

// ---- grant lifetime ----

#[test]
fn a_grant_deadline_is_bounded_so_no_tombstone_lives_forever() {
    let limits = TierGrantLimits {
        max_grant_lifetime_millis: 5_000,
        ..TierGrantLimits::default()
    };
    let rig = rig_with_limits(limits);
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    let presenter = session("worker-1");
    for deadline in [u64::MAX, rig.clock.now_millis() + 5_001] {
        let mut grant = rig.grant("alpha", &object, 1, "worker-1");
        grant.deadline_millis = deadline;
        assert_eq!(
            block_on(rig.controller.transfer_with_grant(&grant, &presenter)),
            Err(TierRuntimeError::GrantInvalid)
        );
    }
    assert_eq!(rig.destination_state(&object.root_digest), None);
    assert_eq!(rig.controller.live_grant_tombstones(&presenter).unwrap(), 0);
    let mut edge = rig.grant("alpha", &object, 1, "worker-1");
    edge.deadline_millis = rig.clock.now_millis() + 5_000;
    block_on(rig.controller.transfer_with_grant(&edge, &presenter)).unwrap();
}

// ---- a leftover Revoked replica ----

/// A transfer whose grant expires between the copy and the publish, leaving a Revoked replica.
fn leave_a_revoked_replica(rig: &Rig, object: &TierObjectManifest) {
    let mut short = rig.grant("alpha", object, 1, "worker-1");
    short.deadline_millis = rig.clock.now_millis() + 100;
    let clock = rig.clock.clone();
    rig.destination
        .after_next_verify(move || clock.advance(500));
    assert_eq!(
        block_on(
            rig.controller
                .transfer_with_grant(&short, &session("worker-1"))
        ),
        Err(TierRuntimeError::GrantExpired)
    );
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Revoked)
    );
}

#[test]
fn a_revoked_leftover_replica_is_never_resident_and_blocks_nothing() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    leave_a_revoked_replica(&rig, &object);
    let record = rig.controller.residency(&object.root_digest).unwrap();
    assert_eq!(record.available_replicas(), 1);
    // Not pinnable, so never readable or admitted.
    assert!(rig
        .controller
        .pin(object.root_digest, &id(DESTINATION))
        .is_err());
    // Not counted as a surviving replica: the source is still the last one.
    assert_eq!(
        block_on(rig.controller.evict(object.root_digest, &id(SOURCE))),
        Err(TierRuntimeError::LastReplica)
    );
    // It does not hold up draining its backend.
    rig.controller.begin_detach(&id(DESTINATION)).unwrap();
    rig.controller.finish_detach(&id(DESTINATION)).unwrap();
}

#[test]
fn a_revoked_leftover_replica_does_not_block_a_later_transfer() {
    let rig = rig();
    let object = rig.seed("weights", b"weight-bytes", Some("alpha"));
    leave_a_revoked_replica(&rig, &object);
    block_on(rig.controller.transfer_with_grant(
        &rig.grant("alpha", &object, 2, "worker-1"),
        &session("worker-1"),
    ))
    .unwrap();
    assert_eq!(
        rig.destination_state(&object.root_digest),
        Some(ReplicaState::Available)
    );
    assert!(rig.destination.holds(&object.chunks[0]));
}
