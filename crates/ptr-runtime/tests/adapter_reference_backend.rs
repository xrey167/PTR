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

fn promoted(generation: u64) -> LedgerEvent {
    LedgerEvent::AdapterPromoted {
        project: ProjectId::from("alpha"),
        adapter: "coder".into(),
        generation: Generation(generation),
        base: [1; 32],
        artifact_sha256: [2; 32],
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
    backend.apply_committed_event(&promoted(2)).unwrap();
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
    backend.apply_committed_event(&promoted(2)).unwrap();
    backend.apply_committed_event(&promoted(3)).unwrap();
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
    backend.apply_committed_event(&promoted(2)).unwrap();
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
