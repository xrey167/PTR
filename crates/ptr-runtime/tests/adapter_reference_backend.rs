//! The reference KV-reuse backend learns adapter promotions and revocations from committed
//! events: a revoked or superseded adapter generation is refused on a cache miss as well, not
//! only through the pins and entries that were already cached.
use ptr_ledger::LedgerEvent;
use ptr_model_api::{PromptSegment, SegmentedModelRequest, Tokenizer};
use ptr_pods::{Qwen2Adapter, Qwen2AdapterKind, Qwen2AdapterSpec};
use ptr_runtime::{
    BindingError, ReferenceKvError, ReferenceKvLimits, ReferenceKvReuseBackend, ReferenceKvScope,
    StackBinding,
};
use ptr_types::{Generation, PrincipalId, ProjectId, RequestId, Revision, SemanticContext};

const KEY: &str = "ptr-adapter:alpha/coder";

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("alpha"),
    }
}

fn backend_fresh() -> ReferenceKvReuseBackend {
    backend()
}

fn backend() -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap()
}

fn request(backend: &ReferenceKvReuseBackend) -> SegmentedModelRequest {
    let body: String = (0..70)
        .map(|i| char::from(b'a' + ((i * 7 + i / 16 * 5) % 26) as u8))
        .collect();
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        backend.tokenizer().identity().clone(),
        vec![PromptSegment::raw_from_text(backend.tokenizer(), &body).unwrap()],
    )
    .unwrap()
}

/// The adapter of lifecycle key [`KEY`] at `generation`.
fn adapter(backend: &ReferenceKvReuseBackend, generation: u64) -> Qwen2Adapter {
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &Qwen2AdapterSpec::pinned(KEY, generation, Qwen2AdapterKind::Qkv, 2),
        20240607,
    )
    .unwrap()
}

/// A promotion of the adapter of key [`KEY`] at `generation` with the digest of its real
/// weights, as a runtime that hashed the artifact would commit it.
fn promoted(backend: &ReferenceKvReuseBackend, generation: u64) -> LedgerEvent {
    promoted_with(
        generation,
        backend.adapter_artifact_digest(&adapter(backend, generation)),
        backend.base_digest(),
    )
}

fn promoted_with(generation: u64, artifact_sha256: [u8; 32], base: [u8; 32]) -> LedgerEvent {
    LedgerEvent::AdapterPromoted {
        project: ProjectId::from("alpha"),
        adapter: "coder".into(),
        generation: Generation(generation),
        base,
        artifact_sha256,
        gate_report_digest: [3; 32],
    }
}

fn revoked(generation: u64) -> LedgerEvent {
    LedgerEvent::AdapterRevoked {
        project: ProjectId::from("alpha"),
        adapter: "coder".into(),
        generation: Generation(generation),
        reason: "training input revoked".into(),
    }
}

#[test]
fn a_revoked_adapter_generation_is_refused_on_a_cache_miss() {
    let backend = backend();
    let req = request(&backend);
    let binding = StackBinding::with_adapter(adapter(&backend, 2));
    backend
        .apply_committed_event(&promoted(&backend, 2))
        .unwrap();
    // Promoted and live: nothing is cached yet, the run is a cache miss and is served.
    assert!(backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .is_ok());

    backend.apply_committed_event(&revoked(2)).unwrap();
    // Revoked: the backend's own liveness now knows it, so a later call is refused rather than
    // run on the revoked generation, whatever the cache holds.
    let fresh = request(&backend);
    assert!(matches!(
        backend.infer_scoped_bound(&scope(), &fresh, None, &binding),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
}

#[test]
fn a_superseded_adapter_generation_is_refused_and_the_new_one_serves() {
    let backend = backend();
    let req = request(&backend);
    backend
        .apply_committed_event(&promoted(&backend, 2))
        .unwrap();
    backend
        .apply_committed_event(&promoted(&backend, 3))
        .unwrap();
    assert!(matches!(
        backend.infer_scoped_bound(
            &scope(),
            &req,
            None,
            &StackBinding::with_adapter(adapter(&backend, 2))
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
    assert!(backend
        .infer_scoped_bound(
            &scope(),
            &req,
            None,
            &StackBinding::with_adapter(adapter(&backend, 3))
        )
        .is_ok());
}

#[test]
fn only_the_exact_promoted_generation_is_live_not_every_generation_above_it() {
    let backend = backend();
    let req = request(&backend);
    backend
        .apply_committed_event(&promoted(&backend, 2))
        .unwrap();
    for generation in [3, 999] {
        assert!(
            matches!(
                backend.infer_scoped_bound(
                    &scope(),
                    &req,
                    None,
                    &StackBinding::with_adapter(adapter(&backend, generation))
                ),
                Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
            ),
            "generation {generation} was never promoted"
        );
    }
    assert!(backend
        .infer_scoped_bound(
            &scope(),
            &req,
            None,
            &StackBinding::with_adapter(adapter(&backend, 2))
        )
        .is_ok());
}

#[test]
fn an_adapter_key_no_promotion_named_is_refused_at_every_generation() {
    let backend = backend();
    let req = request(&backend);
    let refused = |generation: u64| {
        matches!(
            backend.infer_scoped_bound(
                &scope(),
                &req,
                None,
                &StackBinding::with_adapter(adapter(&backend, generation))
            ),
            Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
        )
    };
    // Nothing promoted yet: no generation of the key is live.
    assert!(refused(1) && refused(2) && refused(999));
    backend
        .apply_committed_event(&promoted(&backend, 1))
        .unwrap();
    // After promote(1) only generation 1 serves.
    assert!(!refused(1));
    assert!(refused(2) && refused(999));
}

#[test]
fn a_host_that_requires_promoted_adapters_refuses_an_adapter_outside_the_namespace() {
    let backend = backend();
    let req = request(&backend);
    let outside = || {
        Qwen2Adapter::seeded(
            backend.stack_config(),
            &Qwen2AdapterSpec::pinned("foo:bar", 1, Qwen2AdapterKind::Qkv, 2),
            20240607,
        )
        .unwrap()
    };
    // Standalone mode: an adapter outside the lifecycle namespace is live unless revoked.
    assert!(backend
        .infer_scoped_bound(&scope(), &req, None, &StackBinding::with_adapter(outside()))
        .is_ok());
    // Once the host requires promoted adapters, no promotion names it and it is refused.
    backend.require_promoted_adapters().unwrap();
    let fresh = request(&backend);
    assert!(matches!(
        backend.infer_scoped_bound(
            &scope(),
            &fresh,
            None,
            &StackBinding::with_adapter(outside())
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
}

#[test]
fn the_first_adapter_lifecycle_event_arms_the_rule_for_every_adapter_identity() {
    let backend = backend();
    let outside = || {
        Qwen2Adapter::seeded(
            backend.stack_config(),
            &Qwen2AdapterSpec::pinned("foo:bar", 1, Qwen2AdapterKind::Qkv, 2),
            20240607,
        )
        .unwrap()
    };
    // No adapter event yet: standalone mode, the adapter outside the namespace serves.
    let before = request(&backend);
    assert!(backend
        .infer_scoped_bound(
            &scope(),
            &before,
            None,
            &StackBinding::with_adapter(outside())
        )
        .is_ok());
    // One AdapterPromoted for another key arms the rule, with no explicit opt-in.
    backend
        .apply_committed_event(&promoted_with(1, [2; 32], [1; 32]))
        .unwrap();
    let after = request(&backend);
    assert!(matches!(
        backend.infer_scoped_bound(
            &scope(),
            &after,
            None,
            &StackBinding::with_adapter(outside())
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
    // A revocation arms it as well.
    let other = backend_fresh();
    other.apply_committed_event(&revoked(1)).unwrap();
    let probe = request(&other);
    assert!(matches!(
        other.infer_scoped_bound(
            &scope(),
            &probe,
            None,
            &StackBinding::with_adapter(outside())
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
}

/// The adapter of key [`KEY`] at `generation`, seeded with `seed`.
fn seeded(backend: &ReferenceKvReuseBackend, generation: u64, seed: u64) -> Qwen2Adapter {
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &Qwen2AdapterSpec::pinned(KEY, generation, Qwen2AdapterKind::Qkv, 2),
        seed,
    )
    .unwrap()
}

#[test]
fn other_weights_under_a_promoted_identity_and_generation_are_refused() {
    let backend = backend();
    let req = request(&backend);
    let real = seeded(&backend, 2, 20240607);
    // Promote with the real digest of these weights: the adapter is served.
    let digest = backend.adapter_artifact_digest(&real);
    assert_eq!(
        digest,
        {
            use sha2::{Digest, Sha256};
            let bytes: [u8; 32] = Sha256::digest(backend.adapter_artifact_bytes(&real)).into();
            bytes
        },
        "the artifact digest is the SHA-256 of the canonical bytes"
    );
    backend
        .apply_committed_event(&promoted_with(2, digest, backend.base_digest()))
        .unwrap();
    assert!(backend
        .infer_scoped_bound(&scope(), &req, None, &StackBinding::with_adapter(real))
        .is_ok());
    // Re-seeded weights with the same identity and generation are not the promoted adapter.
    let fresh = request(&backend);
    assert!(matches!(
        backend.infer_scoped_bound(
            &scope(),
            &fresh,
            None,
            &StackBinding::with_adapter(seeded(&backend, 2, 7))
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
    // So are foreign weights promoted under a digest that is not theirs.
    let other = backend_fresh();
    other
        .apply_committed_event(&promoted_with(2, [2; 32], other.base_digest()))
        .unwrap();
    let probe = request(&other);
    assert!(matches!(
        other.infer_scoped_bound(
            &scope(),
            &probe,
            None,
            &StackBinding::with_adapter(seeded(&other, 2, 20240607))
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
}

#[test]
fn an_adapter_promoted_for_one_base_is_refused_on_a_backend_with_another() {
    let base_a = backend();
    let base_b =
        ReferenceKvReuseBackend::with_seed(ReferenceKvLimits::default(), scope(), 7).unwrap();
    assert_ne!(base_a.base_digest(), base_b.base_digest());
    // The adapter is promoted for base A, with the real digest of its weights.
    let weights = seeded(&base_a, 2, 20240607);
    let event = promoted_with(
        2,
        base_a.adapter_artifact_digest(&weights),
        base_a.base_digest(),
    );
    base_a.apply_committed_event(&event).unwrap();
    base_b.apply_committed_event(&event).unwrap();
    // On base A it is served.
    let req_a = request(&base_a);
    assert!(base_a
        .infer_scoped_bound(&scope(), &req_a, None, &StackBinding::with_adapter(weights))
        .is_ok());
    // On base B, where the same bytes would load, it is not live: it was promoted for base A.
    let weights_b = seeded(&base_b, 2, 20240607);
    assert_eq!(
        base_b.adapter_artifact_digest(&weights_b),
        base_a.adapter_artifact_digest(&seeded(&base_a, 2, 20240607)),
        "the same adapter bytes, so only the base differs"
    );
    let req_b = request(&base_b);
    assert!(matches!(
        base_b.infer_scoped_bound(
            &scope(),
            &req_b,
            None,
            &StackBinding::with_adapter(weights_b)
        ),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
}
