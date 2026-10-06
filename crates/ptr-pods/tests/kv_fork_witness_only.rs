//! A fork needs a single-use witness; no binding-trusting path remains (doc 36 defect 2).

use ptr_pods::{
    InMemoryPagedKvBackend, InMemoryPagedKvCache, KvBackendError, KvPageBinding, KvPageSize,
    KvSharedPrefix, KvTensorBackend, KvTensorDType, KvTensorSchema, KvWitnessRejection,
    KvWitnessState, PagedKvTensorBackend, TensorRef,
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

fn cache(backend: &InMemoryPagedKvBackend, page: KvPageSize) -> InMemoryPagedKvCache {
    backend
        .allocate_bound(
            schema(),
            8,
            KvPageBinding {
                model: ModelVersion::from("model-v1"),
                adapter: AdapterVersion::from("adapter-v1"),
                generation: Generation(1),
                execution_manifest: [7; 32],
                principal: PrincipalId::from("alice"),
                device: DeviceId::from("cpu"),
                dtype: KvTensorDType::F32,
                page_tokens: page,
            },
        )
        .unwrap()
}

fn published(backend: &InMemoryPagedKvBackend, page: KvPageSize) -> KvSharedPrefix {
    let mut source = cache(backend, page);
    let rows = |seed: f32| TensorRef {
        layer: 0,
        shape: vec![4, 4],
        values: (0..16).map(|i| seed + i as f32).collect(),
    };
    backend
        .append(&mut source, &[rows(1.0)], &[rows(9.0)])
        .unwrap();
    backend.publish_prefix(&mut source).unwrap()
}

#[test]
fn a_witness_is_spent_by_its_fork_and_cannot_be_revoked_afterwards() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let prefix = published(&backend, page);
    let (witness, revoker) = authority.admit(&prefix);

    let mut first = cache(&backend, page);
    backend
        .fork_prefix_admitted(&mut first, &prefix, witness)
        .unwrap();
    assert_eq!(first.sequence_length, 4);
    assert_eq!(revoker.state(), KvWitnessState::Consumed);
    assert!(!revoker.revoke());
}

#[test]
fn a_failed_fork_still_spends_the_witness_and_shares_no_pages() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let authority = backend.take_fork_authority().unwrap();
    let prefix = published(&backend, page);
    // A retired prefix: the fork is refused, nothing is shared, the witness is not reusable.
    let (witness, revoker) = authority.admit(&prefix);
    backend.retire_prefix(&prefix.key).unwrap();
    let mut target = cache(&backend, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::PrefixNotFound)
    );
    assert_eq!(target.sequence_length, 0);
    assert_eq!(backend.page_metrics(&target).shared_pages, 0);
    // The witness was refused before consumption here, so it is still pending and the
    // revoker can still cancel it; a pending witness never forks a retired prefix.
    assert_eq!(revoker.state(), KvWitnessState::Pending);
}

/// A witness minted for one backend is refused by another.
#[test]
fn a_foreign_witness_is_refused() {
    let page = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page);
    let other = InMemoryPagedKvBackend::new(page);
    let foreign = other.take_fork_authority().unwrap();
    let prefix = published(&backend, page);
    let (witness, _) = foreign.admit(&prefix);
    let mut target = cache(&backend, page);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::ForeignBackend
        ))
    );
    assert_eq!(target.sequence_length, 0);
}
