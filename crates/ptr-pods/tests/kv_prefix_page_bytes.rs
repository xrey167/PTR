//! `prefix_page_bytes` reports each unique page of a published prefix with its resident
//! bytes, in the unit of `page_metrics`.

use ptr_pods::{
    InMemoryPagedKvBackend, KvBackendError, KvPageSize, KvTensorBackend, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, ModelVersion};

const LAYERS: usize = 3;
const PAGE_TOKENS: u16 = 4;

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
    seed: f32,
) -> ptr_pods::InMemoryPagedKvCache {
    let mut cache = backend.allocate(schema(), 32).unwrap();
    let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, seed)).collect();
    let values: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, seed + 8.0)).collect();
    backend.append(&mut cache, &keys, &values).unwrap();
    cache
}

fn backend() -> InMemoryPagedKvBackend {
    InMemoryPagedKvBackend::new(KvPageSize::new(PAGE_TOKENS).unwrap())
}

#[test]
fn per_page_bytes_sum_to_the_caches_resident_bytes() {
    let backend = backend();
    let mut cache = filled(&backend, 12, 1.0);
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let charges = backend.prefix_page_bytes(&prefix).unwrap();
    assert_eq!(charges.len(), 3);
    for (index, (digest, bytes)) in charges.iter().enumerate() {
        assert_eq!(*digest, prefix.page_digests[index]);
        // 4 tokens * (2 heads * 2 dim) * 3 layers * K and V * 4 bytes.
        assert_eq!(*bytes, 4 * 4 * 3 * 2 * 4);
    }
    let sum: u64 = charges.iter().map(|(_, bytes)| bytes).sum();
    assert_eq!(sum, backend.page_metrics(&cache).resident_bytes);
}

#[test]
fn a_repeated_page_digest_is_reported_once() {
    let backend = backend();
    // Identical rows in every page: the three page digests may coincide in content but are
    // listed exactly once per distinct digest, in first-occurrence order.
    let mut cache = backend.allocate(schema(), 32).unwrap();
    let page = |seed| -> (Vec<TensorRef>, Vec<TensorRef>) {
        (
            (0..LAYERS).map(|l| rows(l, 4, seed)).collect(),
            (0..LAYERS).map(|l| rows(l, 4, seed + 8.0)).collect(),
        )
    };
    for seed in [1.0, 2.0, 1.0] {
        let (k, v) = page(seed);
        backend.append(&mut cache, &k, &v).unwrap();
    }
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let charges = backend.prefix_page_bytes(&prefix).unwrap();
    let mut unique = Vec::new();
    for digest in &prefix.page_digests {
        if !unique.contains(digest) {
            unique.push(*digest);
        }
    }
    let listed: Vec<_> = charges.iter().map(|(digest, _)| *digest).collect();
    assert_eq!(listed, unique);
}

#[test]
fn unknown_retired_or_forged_prefixes_are_refused() {
    let backend = backend();
    let mut cache = filled(&backend, 8, 1.0);
    let prefix = backend.publish_prefix(&mut cache).unwrap();

    let mut forged = prefix.clone();
    forged.page_digests.reverse();
    assert!(matches!(
        backend.prefix_page_bytes(&forged),
        Err(KvBackendError::PrefixMismatch)
    ));
    let mut shorter = prefix.clone();
    shorter.page_digests.pop();
    assert!(matches!(
        backend.prefix_page_bytes(&shorter),
        Err(KvBackendError::PrefixMismatch)
    ));

    backend.retire_prefix(&prefix.key).unwrap();
    assert!(matches!(
        backend.prefix_page_bytes(&prefix),
        Err(KvBackendError::PrefixNotFound)
    ));
}
