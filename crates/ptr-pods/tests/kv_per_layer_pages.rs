//! Pages are stored per layer (doc 36 defect 5): digests cover every layer and a fork's
//! decode appends never disturb the source.

use ptr_pods::{
    InMemoryPagedKvBackend, KvBackendError, KvPageSize, KvTensorBackend, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, ModelVersion};

const LAYERS: usize = 3;

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("m"),
        adapter: AdapterVersion::from("a"),
        device: DeviceId::from("cpu"),
        layer_count: LAYERS,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn rows(layer: usize, tokens: usize, seed: f32) -> TensorRef {
    TensorRef {
        layer,
        shape: vec![tokens, 4],
        values: (0..tokens * 4)
            .map(|i| seed + (layer * 100 + i) as f32)
            .collect(),
    }
}

fn filled(
    backend: &InMemoryPagedKvBackend,
    tokens: usize,
    bump: Option<usize>,
) -> ptr_pods::InMemoryPagedKvCache {
    let mut cache = backend.allocate(schema(), 16).unwrap();
    let mut keys: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, 1.0)).collect();
    let values: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, 9.0)).collect();
    if let Some(layer) = bump {
        keys[layer].values[0] += 1.0;
    }
    backend.append(&mut cache, &keys, &values).unwrap();
    cache
}

#[test]
fn page_digest_is_deterministic_and_changes_with_any_single_layer() {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
    let digest_of = |bump| {
        let mut cache = filled(&backend, 4, bump);
        let lease = backend.seal_snapshot(&mut cache).unwrap();
        backend.snapshot_from_lease(&lease).unwrap().pages[0].digest
    };
    let base = digest_of(None);
    assert_eq!(base, digest_of(None));
    for layer in 0..LAYERS {
        assert_ne!(base, digest_of(Some(layer)), "layer {layer}");
    }
}

#[test]
fn tampering_with_one_layer_of_a_snapshot_is_refused() {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
    let mut cache = filled(&backend, 4, None);
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    let mut snapshot = backend.snapshot_from_lease(&lease).unwrap();
    snapshot.pages[0].layers[1].values.values[3] += 1.0;
    assert_eq!(
        snapshot.validate(),
        Err(KvBackendError::SnapshotDigestMismatch)
    );
}

#[test]
fn fork_then_append_leaves_the_source_bitwise_unchanged() {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
    let mut source = filled(&backend, 6, None);
    let lease = backend.seal_snapshot(&mut source).unwrap();
    let before = backend.snapshot_from_lease(&lease).unwrap();
    let mut fork = backend
        .restore_paged(before.clone(), binding_of(&before))
        .unwrap();
    let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, 1, 5.0)).collect();
    let values: Vec<_> = (0..LAYERS).map(|l| rows(l, 1, 6.0)).collect();
    backend.append(&mut fork, &keys, &values).unwrap();
    let after = backend.snapshot_from_lease(&lease).unwrap();
    assert_eq!(before, after);
    assert_eq!(fork.sequence_length, 7);
}

fn binding_of(snapshot: &ptr_pods::KvPagedSnapshot) -> ptr_pods::KvPageBinding {
    snapshot.binding.clone()
}
