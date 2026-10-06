//! Layer-scoped prefixes: publish, fork from several parts, checked adoption and shared
//! boundary rows, through the public API only (allocation identity is observed with
//! `InMemoryPagedKvCache::shared_leading_rows`).

use ptr_pods::{
    InMemoryPagedKvBackend, InMemoryPagedKvCache, KvAdapterLayering, KvAdoptError, KvAdoptReport,
    KvBackendError, KvForkAuthority, KvForkWitness, KvLayerAdoptRequest, KvLayerForkRequest,
    KvLayerPublishRequest, KvLayerRange, KvPageBinding, KvPageSize, KvSharedPrefix,
    KvTensorBackend, KvTensorDType, KvTensorSchema, KvWitnessRejection, KvWitnessRevoker,
    KvWitnessState, PagedKvTensorBackend, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

const LAYERS: usize = 4;
const WIDTH: usize = 4;
const PAGE: usize = 4;

fn schema(adapter: &str) -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("m"),
        adapter: AdapterVersion::from(adapter),
        device: DeviceId::from("cpu"),
        layer_count: LAYERS,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn binding(adapter: &str) -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("m"),
        adapter: AdapterVersion::from(adapter),
        generation: Generation(1),
        execution_manifest: [7; 32],
        principal: PrincipalId::from("p"),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(PAGE as u16).unwrap(),
    }
}

fn backend() -> InMemoryPagedKvBackend {
    InMemoryPagedKvBackend::new(KvPageSize::new(PAGE as u16).unwrap())
}

/// Rows `from..from + tokens` of layer `layer`; `salt` makes a layer's content differ
/// between "models" (layers below the adapter's first layer use the base salt).
fn rows(layer: usize, from: usize, tokens: usize, salt: f32) -> TensorRef {
    TensorRef {
        layer,
        shape: vec![tokens, WIDTH],
        values: (0..tokens * WIDTH)
            .map(|i| salt + (layer * 1000 + from * WIDTH + i) as f32)
            .collect(),
    }
}

/// A cache of `tokens` tokens whose layer `l` content uses `salt_for(l)`.
fn cache_with(
    backend: &InMemoryPagedKvBackend,
    adapter: &str,
    tokens: usize,
    salt_for: impl Fn(usize) -> f32,
) -> InMemoryPagedKvCache {
    let mut cache = backend
        .allocate_bound(schema(adapter), 32, binding(adapter))
        .unwrap();
    append(backend, &mut cache, 0, tokens, &salt_for);
    cache
}

fn append(
    backend: &InMemoryPagedKvBackend,
    cache: &mut InMemoryPagedKvCache,
    from: usize,
    tokens: usize,
    salt_for: &impl Fn(usize) -> f32,
) {
    let keys: Vec<_> = (0..LAYERS)
        .map(|l| rows(l, from, tokens, salt_for(l)))
        .collect();
    let values: Vec<_> = (0..LAYERS)
        .map(|l| rows(l, from, tokens, salt_for(l) + 0.5))
        .collect();
    backend.append(cache, &keys, &values).unwrap();
}

fn base_salt(_: usize) -> f32 {
    1.0
}

/// Layers below 2 are the base model's; layers 2.. differ (a LoRA starting at layer 2).
fn lora_salt(layer: usize) -> f32 {
    if layer < 2 {
        1.0
    } else {
        50.0
    }
}

fn lora_layering(first: u16) -> Option<KvAdapterLayering> {
    Some(KvAdapterLayering {
        base_adapter: AdapterVersion::from("base"),
        first_adapted_layer: first,
    })
}

fn span(start: u16, end: u16) -> KvLayerRange {
    KvLayerRange::span(start, end).unwrap()
}

fn publish(
    backend: &InMemoryPagedKvBackend,
    cache: &mut InMemoryPagedKvCache,
    layers: KvLayerRange,
    layering: Option<KvAdapterLayering>,
) -> KvSharedPrefix {
    backend
        .publish_prefix_layers(cache, KvLayerPublishRequest { layers, layering })
        .unwrap()
}

fn admit(
    authority: &KvForkAuthority,
    prefix: &KvSharedPrefix,
) -> (KvSharedPrefix, KvForkWitness, KvWitnessRevoker) {
    let (witness, revoker) = authority.admit(prefix);
    (prefix.clone(), witness, revoker)
}

fn empty(backend: &InMemoryPagedKvBackend, adapter: &str) -> InMemoryPagedKvCache {
    backend
        .allocate_bound(schema(adapter), 32, binding(adapter))
        .unwrap()
}

fn digests(cache: &InMemoryPagedKvCache) -> Vec<Option<[u8; 32]>> {
    cache.page_table().iter().map(|page| page.digest).collect()
}

#[test]
fn cross_adapter_fork_shares_exactly_the_base_effective_layers() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut base = cache_with(&backend, "base", 8, base_salt);
    let mut lora = cache_with(&backend, "lora", 8, lora_salt);
    // The base cache publishes layers 0..2 (its own adapter is effective there); the LoRA
    // cache publishes layers 2..4, where its adapter is effective.
    let low = publish(&backend, &mut base, span(0, 2), None);
    let high = publish(&backend, &mut lora, span(2, 4), lora_layering(2));
    assert_eq!(low.key.adapter, AdapterVersion::from("base"));
    assert_eq!(high.key.adapter, AdapterVersion::from("lora"));

    let mut forked = empty(&backend, "lora");
    let (low_prefix, low_witness, _) = admit(&authority, &low);
    let (high_prefix, high_witness, _) = admit(&authority, &high);
    backend
        .fork_prefix_layers_admitted(
            &mut forked,
            KvLayerForkRequest {
                parts: vec![(high_prefix, high_witness), (low_prefix, low_witness)],
                layering: lora_layering(2),
            },
        )
        .unwrap();
    assert_eq!(forked.sequence_length, 8);
    // Bitwise the cache the LoRA stack computed itself: same page digests.
    assert_eq!(digests(&forked), digests(&lora));
    for page in 0..2 {
        for layer in 0..LAYERS {
            let from_base = forked.shared_leading_rows(&base, page, layer);
            let from_lora = forked.shared_leading_rows(&lora, page, layer);
            if layer < 2 {
                assert_eq!((from_base, from_lora), (PAGE, 0), "layer {layer}");
            } else {
                assert_eq!((from_base, from_lora), (0, PAGE), "layer {layer}");
            }
        }
    }
    // The cache is usable: decode continues and the sources are untouched.
    let before = digests(&base);
    append(&backend, &mut forked, 8, 1, &lora_salt);
    assert_eq!(forked.sequence_length, 9);
    assert_eq!(digests(&base), before);
}

#[test]
fn cross_adapter_fork_is_refused_for_layers_whose_effective_adapter_is_not_base() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut base = cache_with(&backend, "base", 8, base_salt);
    let mut lora = cache_with(&backend, "lora", 8, lora_salt);
    let low = publish(&backend, &mut base, span(0, 2), None);
    let high = publish(&backend, &mut lora, span(2, 4), lora_layering(2));
    // A base-published span over layers the LoRA adapts.
    let base_high = publish(&backend, &mut base, span(2, 4), None);
    // A lora-published span over layers 0..2 without layering is lora-effective.
    let lora_low = publish(&backend, &mut lora, span(0, 2), None);
    assert_eq!(lora_low.key.adapter, AdapterVersion::from("lora"));

    let try_fork = |parts: Vec<&KvSharedPrefix>, layering: Option<KvAdapterLayering>| {
        let mut target = empty(&backend, "lora");
        let parts = parts
            .into_iter()
            .map(|prefix| {
                let (prefix, witness, _) = admit(&authority, prefix);
                (prefix, witness)
            })
            .collect();
        let result = backend
            .fork_prefix_layers_admitted(&mut target, KvLayerForkRequest { parts, layering });
        (result, target)
    };

    // No layering: layers 0..2 of a lora cache are lora-effective, the base prefix is not.
    let (result, target) = try_fork(vec![&low, &high], None);
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    assert_eq!(target.sequence_length, 0);
    // Layering whose first adapted layer is 1 makes layer 1 lora-effective.
    let (result, _) = try_fork(vec![&low, &high], lora_layering(1));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    // A base prefix over adapted layers 2..4 is refused.
    let (result, _) = try_fork(vec![&low, &base_high], lora_layering(2));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    // A lora-effective prefix over the base layers 0..2 is refused when layers 0..2 are base.
    let (result, _) = try_fork(vec![&lora_low, &high], lora_layering(2));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    // Parts must cover every layer disjointly.
    let (result, _) = try_fork(vec![&low], lora_layering(2));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    let (result, _) = try_fork(vec![&low, &low, &high], lora_layering(2));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    let (result, _) = try_fork(vec![], lora_layering(2));
    assert_eq!(result, Err(KvBackendError::PrefixMismatch));
    // And the matching request still works afterwards.
    let (result, target) = try_fork(vec![&low, &high], lora_layering(2));
    assert_eq!(result, Ok(()));
    assert_eq!(target.sequence_length, 8);
}

#[test]
fn fork_witnesses_are_checked_and_spent_by_the_fork() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut base = cache_with(&backend, "base", 8, base_salt);
    let low = publish(&backend, &mut base, span(0, 2), None);
    let high = publish(&backend, &mut base, span(2, 4), None);

    // A witness minted for the other part is refused.
    let mut target = empty(&backend, "base");
    let (low_prefix, _, _) = admit(&authority, &low);
    let (high_prefix, high_witness, _) = admit(&authority, &high);
    let (_, low_witness_for_low, _) = admit(&authority, &low);
    let swapped = backend.fork_prefix_layers_admitted(
        &mut target,
        KvLayerForkRequest {
            parts: vec![
                (low_prefix, high_witness),
                (high_prefix, low_witness_for_low),
            ],
            layering: None,
        },
    );
    assert_eq!(
        swapped,
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::PrefixMismatch
        ))
    );

    // A revoked witness is refused.
    let mut target = empty(&backend, "base");
    let (low_prefix, low_witness, low_revoker) = admit(&authority, &low);
    let (high_prefix, high_witness, _) = admit(&authority, &high);
    assert!(low_revoker.revoke());
    let revoked = backend.fork_prefix_layers_admitted(
        &mut target,
        KvLayerForkRequest {
            parts: vec![(low_prefix, low_witness), (high_prefix, high_witness)],
            layering: None,
        },
    );
    assert_eq!(
        revoked,
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::Revoked
        ))
    );

    // A successful fork consumes every witness.
    let mut target = empty(&backend, "base");
    let (low_prefix, low_witness, low_revoker) = admit(&authority, &low);
    let (high_prefix, high_witness, high_revoker) = admit(&authority, &high);
    backend
        .fork_prefix_layers_admitted(
            &mut target,
            KvLayerForkRequest {
                parts: vec![(low_prefix, low_witness), (high_prefix, high_witness)],
                layering: None,
            },
        )
        .unwrap();
    assert_eq!(low_revoker.state(), KvWitnessState::Consumed);
    assert_eq!(high_revoker.state(), KvWitnessState::Consumed);
    // A layered prefix cannot be forked alone through the whole-model path.
    let mut target = empty(&backend, "base");
    let (prefix, witness, _) = admit(&authority, &low);
    assert_eq!(
        backend.fork_prefix_admitted(&mut target, &prefix, witness),
        Err(KvBackendError::PrefixMismatch)
    );
}

#[test]
fn checked_adoption_swaps_in_the_shared_allocation_only_when_every_row_is_equal() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut source = cache_with(&backend, "base", 9, base_salt);
    let prefix = publish(&backend, &mut source, span(0, 2), None);

    // Independently computed, bitwise-equal rows.
    let mut computed = cache_with(&backend, "base", 9, base_salt);
    let before = digests(&computed);
    for page in 0..2 {
        for layer in 0..LAYERS {
            assert_eq!(computed.shared_leading_rows(&source, page, layer), 0);
        }
    }
    let (prefix_a, witness, revoker) = admit(&authority, &prefix);
    let report = backend
        .adopt_prefix_layers_admitted(
            &mut computed,
            KvLayerAdoptRequest {
                prefix: prefix_a,
                witness,
                layering: None,
            },
        )
        .unwrap();
    assert_eq!(
        report,
        KvAdoptReport {
            compared_rows: 2 * 2 * PAGE,
            swapped_layer_pages: 4
        }
    );
    assert_eq!(revoker.state(), KvWitnessState::Consumed);
    for page in 0..2 {
        for layer in 0..LAYERS {
            let shared = computed.shared_leading_rows(&source, page, layer);
            assert_eq!(shared, if layer < 2 { PAGE } else { 0 }, "{page}/{layer}");
        }
    }
    // Content and digests are unchanged by the swap, and the partial tail page is untouched.
    assert_eq!(digests(&computed), before);
    assert_eq!(computed.sequence_length, 9);

    // One differing value in layer 1 of page 1 (a single row): refused with a count, and
    // the cache is not mutated at all.
    let mut other = cache_with(
        &backend,
        "base",
        9,
        |layer| {
            if layer == 1 {
                2.0
            } else {
                1.0
            }
        },
    );
    let other_digests = digests(&other);
    let (prefix_b, witness, revoker) = admit(&authority, &prefix);
    let refused = backend.adopt_prefix_layers_admitted(
        &mut other,
        KvLayerAdoptRequest {
            prefix: prefix_b,
            witness,
            layering: None,
        },
    );
    // Layer 1 differs in every one of its 8 rows; layer 0 matches.
    assert_eq!(
        refused,
        Err(KvAdoptError::Mismatch {
            mismatched_rows: 2 * PAGE,
            compared_rows: 2 * 2 * PAGE,
        })
    );
    assert_eq!(digests(&other), other_digests);
    for page in 0..2 {
        for layer in 0..LAYERS {
            assert_eq!(other.shared_leading_rows(&source, page, layer), 0);
        }
    }
    // The admission is spent even though the adoption was refused.
    assert_eq!(revoker.state(), KvWitnessState::Consumed);
}

#[test]
fn checked_adoption_is_refused_for_wrong_shape_adapter_or_a_single_flipped_bit() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut source = cache_with(&backend, "base", 8, base_salt);
    let prefix = publish(&backend, &mut source, span(0, 2), None);
    let request = |prefix: &KvSharedPrefix, layering| {
        let (prefix, witness, _) = admit(&authority, prefix);
        KvLayerAdoptRequest {
            prefix,
            witness,
            layering,
        }
    };

    // A cache shorter than the prefix.
    let mut short = cache_with(&backend, "base", 4, base_salt);
    assert_eq!(
        backend.adopt_prefix_layers_admitted(&mut short, request(&prefix, None)),
        Err(KvAdoptError::Backend(KvBackendError::PrefixMismatch))
    );
    // A lora cache without layering is not base-effective for layers 0..2.
    let mut lora = cache_with(&backend, "lora", 8, lora_salt);
    assert_eq!(
        backend.adopt_prefix_layers_admitted(&mut lora, request(&prefix, None)),
        Err(KvAdoptError::Backend(KvBackendError::PrefixMismatch))
    );
    // With layering the same adoption is accepted (rows equal below layer 2).
    assert!(backend
        .adopt_prefix_layers_admitted(&mut lora, request(&prefix, lora_layering(2)))
        .is_ok());
    assert_eq!(lora.shared_leading_rows(&source, 0, 0), PAGE);
    assert_eq!(lora.shared_leading_rows(&source, 0, 2), 0);

    // A single flipped bit (-0.0 vs 0.0 compare equal as floats but not bitwise).
    let mut zero = backend
        .allocate_bound(schema("base"), 32, binding("base"))
        .unwrap();
    let mut zero_source = backend
        .allocate_bound(schema("base"), 32, binding("base"))
        .unwrap();
    for (cache, value) in [(&mut zero, -0.0f32), (&mut zero_source, 0.0f32)] {
        let tensor = |layer| TensorRef {
            layer,
            shape: vec![PAGE, WIDTH],
            values: vec![value; PAGE * WIDTH],
        };
        let keys: Vec<_> = (0..LAYERS).map(tensor).collect();
        backend.append(cache, &keys, &keys).unwrap();
    }
    let zero_prefix = publish(&backend, &mut zero_source, span(0, 1), None);
    assert_eq!(
        backend.adopt_prefix_layers_admitted(&mut zero, request(&zero_prefix, None)),
        Err(KvAdoptError::Mismatch {
            mismatched_rows: PAGE,
            compared_rows: PAGE
        })
    );
}

#[test]
fn boundary_page_shares_exactly_the_pre_invocation_rows_after_the_post_rows_are_appended() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    // Page size 4: the invocation is at token 7, so page 1 has 3 pre-invocation rows.
    let invocation = 7usize;
    let pre = invocation % PAGE;
    let mut base = cache_with(&backend, "base", 8, base_salt);
    let prefix = publish(&backend, &mut base, KvLayerRange::Full, None);
    assert_eq!(prefix.key.layers, KvLayerRange::Full);

    // The aLoRA cache starts from the base prefix, drops the rows at and after the
    // invocation and appends rows computed under the adapter.
    let mut alora = empty(&backend, "alora");
    let (prefix, witness, _) = admit(&authority, &prefix);
    backend
        .fork_prefix_layers_admitted(
            &mut alora,
            KvLayerForkRequest {
                parts: vec![(prefix, witness)],
                layering: lora_layering(LAYERS as u16),
            },
        )
        .unwrap();
    assert_eq!(alora.sequence_length, 8);
    backend.truncate(&mut alora, invocation).unwrap();
    let post_salt = |_: usize| 90.0;
    append(&backend, &mut alora, invocation, 1, &post_salt);

    for layer in 0..LAYERS {
        // Page 0 is shared whole, page 1 shares exactly the rows before the invocation.
        assert_eq!(alora.shared_leading_rows(&base, 0, layer), PAGE);
        assert_eq!(
            alora.shared_leading_rows(&base, 1, layer),
            pre,
            "layer {layer}"
        );
    }
    assert_eq!(alora.sequence_length, 8);

    // The digest of the boundary page is that of the same rows built without any sharing.
    let mut plain = empty(&backend, "alora");
    append(&backend, &mut plain, 0, invocation, &base_salt);
    append(&backend, &mut plain, invocation, 1, &post_salt);
    assert_eq!(digests(&alora), digests(&plain));
    assert!(digests(&alora).iter().all(Option::is_some));
    // The base cache is bitwise unchanged.
    let base_reference = cache_with(&backend, "base", 8, base_salt);
    assert_eq!(digests(&base), digests(&base_reference));
    let dense = |cache: &InMemoryPagedKvCache| backend.snapshot(cache).unwrap().layers;
    assert_eq!(dense(&alora)[0].keys.shape, vec![8, WIDTH]);
    assert_eq!(
        dense(&alora)[1].keys.values[..invocation * WIDTH],
        dense(&base)[1].keys.values[..invocation * WIDTH]
    );
    assert_ne!(dense(&alora)[1].keys.values, dense(&base)[1].keys.values);
}

#[test]
fn shared_boundary_prefix_is_also_kept_when_truncating_again() {
    let backend = backend();
    let authority = backend.take_fork_authority().unwrap();
    let mut base = cache_with(&backend, "base", 8, base_salt);
    let prefix = publish(&backend, &mut base, KvLayerRange::Full, None);
    let mut other = empty(&backend, "base");
    let (prefix, witness, _) = admit(&authority, &prefix);
    backend
        .fork_prefix_layers_admitted(
            &mut other,
            KvLayerForkRequest {
                parts: vec![(prefix, witness)],
                layering: None,
            },
        )
        .unwrap();
    backend.truncate(&mut other, 7).unwrap();
    append(&backend, &mut other, 7, 1, &|_| 90.0);
    backend.truncate(&mut other, 5).unwrap();
    for layer in 0..LAYERS {
        assert_eq!(other.shared_leading_rows(&base, 1, layer), 1);
    }
    append(&backend, &mut other, 5, 3, &base_salt);
    // Rows 5..8 re-appended with the base salt: content equals the base cache again.
    assert_eq!(digests(&other), digests(&base));
    for layer in 0..LAYERS {
        assert_eq!(other.shared_leading_rows(&base, 1, layer), 1);
    }
}

#[test]
fn layer_scoped_publish_is_refused_for_bad_ranges_and_layering() {
    let backend = backend();
    let mut cache = cache_with(&backend, "base", 8, base_salt);
    let publish = |layers, layering| {
        backend.publish_prefix_layers(
            &mut cache.clone(),
            KvLayerPublishRequest { layers, layering },
        )
    };
    assert!(KvLayerRange::span(2, 2).is_err());
    assert!(KvLayerRange::span(3, 1).is_err());
    assert_eq!(
        publish(span(2, 5), None),
        Err(KvBackendError::InvalidLayer(5))
    );
    assert_eq!(
        publish(span(0, 2), lora_layering(9)),
        Err(KvBackendError::InvalidLayer(9))
    );
    // A cache without a full page has nothing to publish.
    let mut short = cache_with(&backend, "base", 3, base_salt);
    assert_eq!(
        backend.publish_prefix_layers(
            &mut short,
            KvLayerPublishRequest {
                layers: span(0, 2),
                layering: None
            }
        ),
        Err(KvBackendError::PrefixMismatch)
    );
    // A span covering every layer is the same key as `Full`.
    let full = publish(span(0, LAYERS as u16), None).unwrap();
    assert_eq!(full.key.layers, KvLayerRange::Full);
    let _ = &mut cache;
}

#[test]
fn layered_prefix_digests_depend_on_content_span_and_effective_adapter() {
    let backend = backend();
    let mut a = cache_with(&backend, "base", 8, base_salt);
    let mut b = cache_with(&backend, "base", 8, base_salt);
    let mut c = cache_with(&backend, "base", 8, |l| if l == 1 { 3.0 } else { 1.0 });
    let key = |cache: &mut InMemoryPagedKvCache, layers| {
        publish(&backend, cache, layers, None).key.prefix_digest
    };
    assert_eq!(key(&mut a, span(0, 2)), key(&mut b, span(0, 2)));
    assert_ne!(key(&mut a, span(0, 2)), key(&mut c, span(0, 2)));
    // A layer outside the span does not matter, a different span does.
    assert_eq!(key(&mut a, span(2, 4)), key(&mut c, span(2, 4)));
    assert_ne!(key(&mut a, span(0, 2)), key(&mut a, span(0, 3)));
    // The effective adapter is part of the key.
    let mut lora = cache_with(&backend, "lora", 8, base_salt);
    let base_key = publish(&backend, &mut a, span(0, 2), None).key;
    let lora_key = publish(&backend, &mut lora, span(0, 2), None).key;
    let lora_as_base = publish(&backend, &mut lora, span(0, 2), lora_layering(2)).key;
    assert_ne!(base_key.schema_digest, lora_key.schema_digest);
    assert_eq!(base_key.schema_digest, lora_as_base.schema_digest);
    assert_eq!(base_key.adapter, lora_as_base.adapter);
}
