//! Prefix retirement and witness-gated forks (doc 36, phase P2, pods side).

use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryKvSnapshotLease, InMemoryPagedKvBackend,
    InMemoryPagedKvCache, KvBackendError, KvContinuationExecutor, KvPageBinding, KvPageMetrics,
    KvPageSize, KvPagedSnapshot, KvRuntimeLeaseBinding, KvSharedPrefix, KvTensorBackend,
    KvTensorDType, KvTensorSchema, KvTensorSnapshot, KvWitnessRejection, KvWitnessState,
    PagedKvTensorBackend, ReferenceCausalAttentionExecutor, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn binding(principal: &str, generation: u64, page: KvPageSize) -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        generation: Generation(generation),
        execution_manifest: [7; 32],
        principal: PrincipalId::from(principal),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: page,
    }
}

fn rows(count: usize, seed: f32) -> TensorRef {
    TensorRef {
        layer: 0,
        shape: vec![count, 4],
        values: (0..count * 4).map(|i| seed + i as f32).collect(),
    }
}

fn published(
    backend: &InMemoryPagedKvBackend,
    page: KvPageSize,
) -> (InMemoryPagedKvCache, KvSharedPrefix) {
    let mut source = backend
        .allocate_bound(schema(), 8, binding("alice", 1, page))
        .unwrap();
    backend
        .append(&mut source, &[rows(4, 1.0)], &[rows(4, 9.0)])
        .unwrap();
    let prefix = backend.publish_prefix(&mut source).unwrap();
    (source, prefix)
}

fn fresh(
    backend: &InMemoryPagedKvBackend,
    principal: &str,
    generation: u64,
    page: KvPageSize,
) -> InMemoryPagedKvCache {
    backend
        .allocate_bound(schema(), 8, binding(principal, generation, page))
        .unwrap()
}

#[test]
fn retire_prefix_removes_the_record_and_is_not_silently_idempotent() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let (_source, prefix) = published(&backend, page);
    backend.retire_prefix(&prefix.key).unwrap();
    assert_eq!(
        backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound)
    );
    let (witness, _) = authority.admit(&prefix);
    let mut target = fresh(&backend, "alice", 1, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::PrefixNotFound)
    );
    assert_eq!(target.sequence_length, 0);
}

#[test]
fn fork_authority_is_handed_out_once() {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::new(2).unwrap());
    assert!(backend.take_fork_authority().is_ok());
    assert!(matches!(
        backend.take_fork_authority(),
        Err(KvBackendError::Backend(_))
    ));
}

#[test]
fn admitted_fork_ignores_the_callers_generation_but_not_its_principal() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let (_source, prefix) = published(&backend, page);

    // The witness carries admission, so the generation is the catalog's concern.
    let (witness, revoker) = authority.admit(&prefix);
    let mut target = fresh(&backend, "alice", 2, page);
    backend
        .fork_prefix_admitted(&mut target, &prefix, witness)
        .unwrap();
    assert_eq!(target.sequence_length, 4);
    assert_eq!(revoker.state(), KvWitnessState::Consumed);

    let (witness, _) = authority.admit(&prefix);
    let mut other = fresh(&backend, "bob", 1, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut other, &prefix, witness),
        Err(KvBackendError::PrefixMismatch)
    );
}

#[test]
fn forged_foreign_mismatched_and_revoked_witnesses_are_refused() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let other_backend = InMemoryPagedKvBackend::new(page);
    let own = backend.take_fork_authority().unwrap();
    let foreign = other_backend.take_fork_authority().unwrap();
    let (_source, prefix) = published(&backend, page);

    let (witness, _) = foreign.admit(&prefix);
    let mut target = fresh(&backend, "alice", 1, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::ForeignBackend
        ))
    );

    // A witness minted for one prefix does not open another.
    let mut second = backend
        .allocate_bound(schema(), 8, binding("alice", 1, page))
        .unwrap();
    backend
        .append(&mut second, &[rows(4, 100.0)], &[rows(4, 200.0)])
        .unwrap();
    let other_prefix = backend.publish_prefix(&mut second).unwrap();
    assert_ne!(other_prefix.key, prefix.key);
    let (witness, _) = own.admit(&other_prefix);
    assert_eq!(
        backend.fork_prefix_admitted(&mut fresh(&backend, "alice", 1, page), &prefix, witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::PrefixMismatch
        ))
    );
}

#[test]
fn revoked_witness_is_refused_and_a_used_one_cannot_be_revoked() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let (_source, prefix) = published(&backend, page);

    let (witness, revoker) = authority.admit(&prefix);
    assert!(revoker.revoke());
    assert!(!revoker.revoke());
    let mut target = fresh(&backend, "alice", 1, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::Revoked
        ))
    );
    assert_eq!(target.sequence_length, 0);

    let (witness, revoker) = authority.admit(&prefix);
    backend
        .fork_prefix_admitted(&mut target, &prefix, witness)
        .unwrap();
    assert!(!revoker.revoke());
    assert_eq!(revoker.state(), KvWitnessState::Consumed);
}

#[test]
fn caller_supplied_prefix_metadata_cannot_widen_the_registered_record() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let (_source, mut prefix) = published(&backend, page);
    prefix.sequence_length = 8;
    let (witness, _) = authority.admit(&prefix);
    let mut target = fresh(&backend, "alice", 1, page);
    backend
        .fork_prefix_admitted(&mut target, &prefix, witness)
        .unwrap();
    assert_eq!(target.sequence_length, 4);
}

#[test]
fn admitted_fork_then_decode_is_bitwise_identical_to_full_prefill() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let executor =
        ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(4).unwrap(), 2)
            .unwrap();
    let first = vec![1.0, 0.0, 0.5, -0.5];
    let second = vec![0.0, 1.0, -0.5, 0.5];
    let third = vec![0.5, 0.5, 1.0, 0.0];

    let mut producer = fresh(&backend, "alice", 1, page);
    executor
        .prefill(
            &backend,
            &mut producer,
            CausalAttentionInput {
                tokens: vec![first.clone(), second.clone()],
            },
        )
        .unwrap();
    let prefix = backend.publish_prefix(&mut producer).unwrap();

    let (witness, _) = authority.admit(&prefix);
    let mut forked = fresh(&backend, "alice", 1, page);
    backend
        .fork_prefix_admitted(&mut forked, &prefix, witness)
        .unwrap();
    let decoded = executor
        .decode(
            &backend,
            &mut forked,
            CausalAttentionInput {
                tokens: vec![third.clone()],
            },
        )
        .unwrap();

    let mut recomputed = fresh(&backend, "alice", 1, page);
    let full = executor
        .prefill(
            &backend,
            &mut recomputed,
            CausalAttentionInput {
                tokens: vec![first, second, third],
            },
        )
        .unwrap();
    let expected = full.hidden_states.last().unwrap();
    let actual = decoded.hidden_states.last().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(forked.sequence_length, recomputed.sequence_length);
}

/// Delegates everything to the in-memory backend but overrides none of the P2 methods, so
/// it exercises the trait defaults a backend without retirement support would get.
struct NoRetireBackend(InMemoryPagedKvBackend);

impl KvTensorBackend for NoRetireBackend {
    type Cache = InMemoryPagedKvCache;
    fn allocate(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<Self::Cache, KvBackendError> {
        self.0.allocate(schema, capacity_tokens)
    }
    fn append(
        &self,
        cache: &mut Self::Cache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), KvBackendError> {
        self.0.append(cache, keys, values)
    }
    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError> {
        self.0.truncate(cache, new_length)
    }
    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError> {
        self.0.snapshot(cache)
    }
    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &DeviceId,
    ) -> Result<Self::Cache, KvBackendError> {
        self.0.restore(snapshot, device)
    }
    fn release(&self, cache: Self::Cache) -> Result<(), KvBackendError> {
        self.0.release(cache)
    }
}

impl PagedKvTensorBackend for NoRetireBackend {
    type SnapshotLease = InMemoryKvSnapshotLease;
    fn allocate_bound(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError> {
        self.0.allocate_bound(schema, capacity_tokens, binding)
    }
    fn bind_runtime_lease(
        &self,
        cache: &mut Self::Cache,
        binding: KvRuntimeLeaseBinding,
    ) -> Result<(), KvBackendError> {
        self.0.bind_runtime_lease(cache, binding)
    }
    fn revoke_runtime_lease(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError> {
        self.0.revoke_runtime_lease(cache)
    }
    fn seal_snapshot(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Self::SnapshotLease, KvBackendError> {
        self.0.seal_snapshot(cache)
    }
    fn snapshot_from_lease(
        &self,
        lease: &Self::SnapshotLease,
    ) -> Result<KvPagedSnapshot, KvBackendError> {
        self.0.snapshot_from_lease(lease)
    }
    fn restore_paged(
        &self,
        snapshot: KvPagedSnapshot,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError> {
        self.0.restore_paged(snapshot, binding)
    }
    fn publish_prefix(&self, cache: &mut Self::Cache) -> Result<KvSharedPrefix, KvBackendError> {
        self.0.publish_prefix(cache)
    }
    fn synchronize(&self, cache: &Self::Cache) -> Result<(), KvBackendError> {
        self.0.synchronize(cache)
    }
    fn page_metrics(&self, cache: &Self::Cache) -> KvPageMetrics {
        self.0.page_metrics(cache)
    }
}

#[test]
fn default_retire_and_admitted_fork_report_unsupported() {
    let page = KvPageSize::new(2).unwrap();
    let inner = InMemoryPagedKvBackend::new(page);
    let authority = inner.take_fork_authority().unwrap();
    let (_source, prefix) = published(&inner, page);
    let backend = NoRetireBackend(inner);
    assert_eq!(
        backend.retire_prefix(&prefix.key),
        Err(KvBackendError::Unsupported("retire_prefix"))
    );
    assert!(matches!(
        backend.take_fork_authority(),
        Err(KvBackendError::Unsupported(_))
    ));
    let (witness, _) = authority.admit(&prefix);
    let mut target = backend
        .allocate_bound(schema(), 8, binding("alice", 1, page))
        .unwrap();
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::Unsupported("fork_prefix_admitted"))
    );
}
