use ptr_config::PtrConfig;
use ptr_runtime::{
    BackendLifecycleState, PtrRuntime, ReplicaState, RuntimeError, TierResidencyController,
};
use ptr_storage::{
    ChunkDescriptor, CpuTierBackend, StorageTier, TierBackend, TierBackendId, TierObjectDomain,
    TierObjectManifest,
};
use ptr_types::{Generation, Revision};
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{SystemTime, UNIX_EPOCH};

fn path(name: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-tier-{name}-{nonce}.ledger"))
}

fn object() -> TierObjectManifest {
    TierObjectManifest::new(
        TierObjectDomain::KvSnapshot,
        "kv-state-7",
        Generation(7),
        Revision(3),
        [4; 32],
        vec![ChunkDescriptor::for_bytes(0, 0, b"snapshot")],
    )
    .unwrap()
}

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
fn durable_reopen_reconstructs_backend_object_and_replica_projection() {
    let path = path("reopen");
    let backend = TierBackendId::from("nvme-a");
    let object = object();
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                BackendLifecycleState::Configured,
                Revision(1),
            )
            .unwrap();
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                BackendLifecycleState::HealthChecked,
                Revision(2),
            )
            .unwrap();
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                BackendLifecycleState::Available,
                Revision(3),
            )
            .unwrap();
        runtime.commit_tier_object(&object).unwrap();
        runtime
            .commit_tier_replica_lifecycle(
                object.root_digest,
                backend.clone(),
                StorageTier::Nvme,
                ReplicaState::Preparing,
                object.generation,
                Revision(4),
            )
            .unwrap();
        runtime
            .commit_tier_replica_lifecycle(
                object.root_digest,
                backend.clone(),
                StorageTier::Nvme,
                ReplicaState::Available,
                object.generation,
                Revision(5),
            )
            .unwrap();
    }

    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(
        reopened.tier_journal().backend(&backend).unwrap().state,
        BackendLifecycleState::Available
    );
    assert_eq!(
        reopened.tier_journal().object(&object.root_digest).unwrap(),
        &object
    );
    assert_eq!(
        reopened
            .tier_journal()
            .replica(&object.root_digest, &backend)
            .unwrap()
            .state,
        ReplicaState::Available
    );
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn invalid_backend_transition_is_rejected_before_append() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let result = runtime.commit_tier_backend_lifecycle(
        TierBackendId::from("nvme-a"),
        StorageTier::Nvme,
        BackendLifecycleState::Available,
        Revision(1),
    );
    assert!(matches!(result, Err(RuntimeError::InvalidConfig(_))));
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn replica_requires_known_available_backend_with_matching_tier() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let object = object();
    runtime.commit_tier_object(&object).unwrap();
    assert!(matches!(
        runtime.commit_tier_replica_lifecycle(
            object.root_digest,
            TierBackendId::from("missing"),
            StorageTier::Nvme,
            ReplicaState::Preparing,
            object.generation,
            Revision(3),
        ),
        Err(RuntimeError::InvalidConfig(_))
    ));

    let backend = TierBackendId::from("nvme-a");
    for (index, state) in [
        BackendLifecycleState::Configured,
        BackendLifecycleState::HealthChecked,
        BackendLifecycleState::Available,
    ]
    .into_iter()
    .enumerate()
    {
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                state,
                Revision(index as u64 + 1),
            )
            .unwrap();
    }
    assert!(matches!(
        runtime.commit_tier_replica_lifecycle(
            object.root_digest,
            backend,
            StorageTier::Cpu,
            ReplicaState::Preparing,
            object.generation,
            Revision(4),
        ),
        Err(RuntimeError::InvalidConfig(_))
    ));
}

#[test]
fn journaled_physical_transfer_reopens_as_available_replica() {
    let path = path("physical-transfer");
    let tiers = TierResidencyController::default();
    let source = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-source")));
    let destination = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-destination")));
    let object = object();
    block_on(source.put_chunk(object.chunks[0].clone(), b"snapshot".to_vec())).unwrap();

    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        block_on(runtime.attach_tier_backend_journaled(&tiers, source.clone(), Revision(1)))
            .unwrap();
        block_on(runtime.attach_tier_backend_journaled(&tiers, destination.clone(), Revision(1)))
            .unwrap();
        block_on(runtime.register_tier_object_journaled(
            &tiers,
            object.clone(),
            source.identity(),
            Revision(1),
        ))
        .unwrap();
        block_on(runtime.transfer_tier_replica_journaled(
            &tiers,
            object.root_digest,
            &source.identity(),
            &destination.identity(),
            None,
            Revision(1),
        ))
        .unwrap();
    }

    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(
        reopened
            .tier_journal()
            .replica(&object.root_digest, &destination.identity())
            .unwrap()
            .state,
        ReplicaState::Available
    );
    std::fs::remove_file(path).unwrap();
}

fn available_replica_on_nvme(
    runtime: &mut PtrRuntime,
    backend: &TierBackendId,
    object: &TierObjectManifest,
) {
    for (index, state) in [
        BackendLifecycleState::Configured,
        BackendLifecycleState::HealthChecked,
        BackendLifecycleState::Available,
    ]
    .into_iter()
    .enumerate()
    {
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                state,
                Revision(index as u64 + 1),
            )
            .unwrap();
    }
    runtime.commit_tier_object(object).unwrap();
    for (state, revision) in [(ReplicaState::Preparing, 4), (ReplicaState::Available, 5)] {
        runtime
            .commit_tier_replica_lifecycle(
                object.root_digest,
                backend.clone(),
                StorageTier::Nvme,
                state,
                object.generation,
                Revision(revision),
            )
            .unwrap();
    }
}

#[test]
fn a_replica_on_a_revoked_backend_can_still_be_retired() {
    for retiring in [ReplicaState::Revoked, ReplicaState::Corrupt] {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        let backend = TierBackendId::from("nvme-a");
        let object = object();
        available_replica_on_nvme(&mut runtime, &backend, &object);
        runtime
            .commit_tier_backend_lifecycle(
                backend.clone(),
                StorageTier::Nvme,
                BackendLifecycleState::Revoked,
                Revision(6),
            )
            .unwrap();
        runtime
            .commit_tier_replica_lifecycle(
                object.root_digest,
                backend.clone(),
                StorageTier::Nvme,
                retiring,
                object.generation,
                Revision(7),
            )
            .unwrap();
        assert_eq!(
            runtime
                .tier_journal()
                .replica(&object.root_digest, &backend)
                .unwrap()
                .state,
            retiring
        );
    }
}

#[test]
fn a_revoked_backend_still_takes_no_new_replica_work() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let backend = TierBackendId::from("nvme-a");
    let object = object();
    available_replica_on_nvme(&mut runtime, &backend, &object);
    runtime
        .commit_tier_backend_lifecycle(
            backend.clone(),
            StorageTier::Nvme,
            BackendLifecycleState::Revoked,
            Revision(6),
        )
        .unwrap();
    assert!(matches!(
        runtime.commit_tier_replica_lifecycle(
            object.root_digest,
            backend,
            StorageTier::Nvme,
            ReplicaState::Draining,
            object.generation,
            Revision(7),
        ),
        Err(RuntimeError::InvalidConfig(_))
    ));
}
