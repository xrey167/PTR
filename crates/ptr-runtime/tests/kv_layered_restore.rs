//! Restoring per-layer catalog entries: a restored entry keeps its layer scope, so with its
//! backend pages present it forks, with them absent it misses cleanly, and revoked at rest it
//! is refused at use. A scope disagreement is counted (`layered_scope_mismatches`), never a
//! silent miss. The v2 compatibility rule is tested byte for byte in `kv_prefix_layers.rs`.
//! Public API only. Not an R009 or R010 result.

use ptr_model_api::{
    KvReusePolicy, PromptSegment, SegmentSource, SegmentedModelRequest, Tokenizer,
};
use ptr_pods::{Qwen2Adapter, Qwen2AdapterKind, Qwen2AdapterSpec};
use ptr_runtime::{
    IndexUpdate, LayerAssembly, ReferenceKvFaults, ReferenceKvLimits, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope, StackBinding,
};
use ptr_types::{Generation, PrincipalId, ProjectId, RequestId, Revision, SemanticContext};
use std::collections::BTreeSet;

const LAYERS: usize = 4;
const SEMANTIC: usize = 64;
const TAIL: usize = 6;

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("p"),
    }
}

fn backend() -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap()
}

fn varied(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 3 + i / 16 * 5) % 26) as u8))
        .collect()
}

fn context() -> SemanticContext {
    SemanticContext {
        revision: Revision(1),
        entries: Vec::new(),
    }
}

/// `SEMANTIC` bytes cited from `doc:one` followed by a raw tail.
fn semantic_request(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    tail_seed: usize,
) -> SegmentedModelRequest {
    let cited = PromptSegment::from_text(
        backend.tokenizer(),
        SegmentSource::Semantic {
            key: "doc:one".to_owned(),
            generation: Generation(1),
        },
        &varied(SEMANTIC, 5),
    )
    .unwrap();
    let tail = PromptSegment::raw_from_text(backend.tokenizer(), &varied(TAIL, tail_seed)).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        context(),
        backend.tokenizer().identity().clone(),
        vec![cited, tail],
    )
    .unwrap()
}

/// The same tokens as one raw segment: another lineage, so it republishes the content-
/// addressed layer pages without using or replacing a restored entry.
fn raw_twin(backend: &ReferenceKvReuseBackend, id: &str) -> SegmentedModelRequest {
    let text = format!("{}{}", varied(SEMANTIC, 5), varied(TAIL, 9));
    let segment = PromptSegment::raw_from_text(backend.tokenizer(), &text).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        context(),
        backend.tokenizer().identity().clone(),
        vec![segment],
    )
    .unwrap()
}

fn lora(backend: &ReferenceKvReuseBackend) -> Qwen2Adapter {
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &Qwen2AdapterSpec::pinned("lora-a", 1, Qwen2AdapterKind::Qkv, 2),
        20240607,
    )
    .unwrap()
}

fn bindings(backend: &ReferenceKvReuseBackend) -> Vec<(&'static str, StackBinding)> {
    vec![
        ("base_layered", StackBinding::base_layered()),
        ("lora_f2", StackBinding::with_adapter(lora(backend))),
    ]
}

fn same_output(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) {
    assert_eq!(a.generated, b.generated);
    assert_eq!(a.logits_digest, b.logits_digest);
    assert_eq!(a.run.events, b.run.events);
}

/// A backend that warmed `binding`, and the snapshot of its catalog.
fn warmed(binding: &StackBinding) -> (Vec<u8>, usize) {
    let source = backend();
    let warm = source
        .infer_scoped_bound(
            &scope(),
            &semantic_request(&source, "warm", 7),
            None,
            binding,
        )
        .unwrap();
    assert_eq!(warm.inserted_chunks, 4);
    let entries = warm.layered.as_ref().unwrap().inserted_layer_entries;
    assert_eq!(entries, 4 * LAYERS);
    let snapshot = source.export_catalog().unwrap();
    assert!(snapshot.starts_with(b"PTRKVL03"));
    (snapshot, entries)
}

#[test]
fn a_restored_per_layer_entry_forks_once_its_pages_exist() {
    for (name, binding) in bindings(&backend()) {
        let (snapshot, entries) = warmed(&binding);
        let restored = backend();
        let report = restored.restore_catalog(&snapshot).unwrap();
        assert_eq!(
            (report.entries, report.skipped_unscoped_layered),
            (entries, 0),
            "{name}"
        );
        // The pages come back by recomputing the same content under another lineage.
        restored
            .infer_scoped_bound(&scope(), &raw_twin(&restored, "pages"), None, &binding)
            .unwrap();

        let probe = semantic_request(&restored, "probe", 11);
        let cold = backend()
            .infer_cold_bound(&scope(), &probe, None, &binding)
            .unwrap();
        let live = restored.own_liveness();
        let mut session = restored
            .begin_session_bound(&scope(), &probe, None, &binding, &live)
            .unwrap();
        assert_eq!(
            session.hit_tokens(),
            64,
            "{name}: the restored entries must fork"
        );
        let out = match session.run_to_end(&live).unwrap() {
            ptr_runtime::KvSessionFinish::Completed(outcome) => *outcome,
            other => panic!("{name}: {other:?}"),
        };
        same_output(&out, &cold);
        let layered = out.layered.clone().unwrap();
        assert_eq!(layered.assembly, LayerAssembly::Fork, "{name}");
        assert_eq!(layered.shared_rows, LAYERS * 64, "{name}");
        let stats = restored.catalog_stats().unwrap();
        assert_eq!(stats.layered_scope_mismatches, 0, "{name}");
        assert_eq!(stats.backend_missing_rejections, 0, "{name}");
    }
}

#[test]
fn restored_entries_without_their_pages_miss_cleanly_and_the_output_is_cold() {
    for (name, binding) in bindings(&backend()) {
        let (snapshot, entries) = warmed(&binding);
        let restored = backend();
        restored.restore_catalog(&snapshot).unwrap();
        let probe = semantic_request(&restored, "probe", 11);
        let cold = backend()
            .infer_cold_bound(&scope(), &probe, None, &binding)
            .unwrap();
        let out = restored
            .infer_scoped_bound(&scope(), &probe, None, &binding)
            .unwrap();
        same_output(&out, &cold);
        assert_eq!(out.hit_tokens, 0, "{name}");
        let stats = restored.catalog_stats().unwrap();
        // A missing page is a clean miss, distinct from a scope bug.
        assert_eq!(stats.layered_scope_mismatches, 0, "{name}");
        assert!(stats.backend_missing_rejections >= 1, "{name}");
        assert!(stats.restored_entries as usize == entries, "{name}");
    }
}

#[test]
fn restored_entries_revoked_at_rest_are_refused_at_use_not_served() {
    for (name, binding) in bindings(&backend()) {
        let (snapshot, _) = warmed(&binding);
        let restored = backend();
        restored.restore_catalog(&snapshot).unwrap();
        restored
            .infer_scoped_bound(&scope(), &raw_twin(&restored, "pages"), None, &binding)
            .unwrap();
        // The revocation reaches the backend's liveness set only: the catalog index stays
        // stale, so only the use-time check can keep the restored entries from forking.
        restored
            .set_faults(ReferenceKvFaults {
                index_update: IndexUpdate::Suppressed,
            })
            .unwrap();
        let revoked: BTreeSet<(String, Generation)> = [("doc:one".to_owned(), Generation(1))]
            .into_iter()
            .collect();
        assert_eq!(restored.invalidate_inputs(&revoked).unwrap(), 0);
        let probe = semantic_request(&restored, "probe", 11);
        let cold = backend()
            .infer_cold_bound(&scope(), &probe, None, &binding)
            .unwrap();
        let out = restored
            .infer_scoped_bound(&scope(), &probe, None, &binding)
            .unwrap();
        same_output(&out, &cold);
        assert_eq!(out.hit_tokens, 0, "{name}");
        let stats = restored.catalog_stats().unwrap();
        assert!(stats.use_time_rejections >= 1, "{name}");
        assert_eq!(stats.layered_scope_mismatches, 0, "{name}");
    }
}

#[test]
fn a_restore_never_serves_under_a_wrong_scope() {
    // The reuse policy of the probes is the default (exact only); a restored per-layer entry
    // is only ever forked by the catalog entry whose key scope the backend accepts.
    let (name, binding) = ("lora_f2", StackBinding::with_adapter(lora(&backend())));
    let (snapshot, _) = warmed(&binding);
    let restored = backend();
    restored.restore_catalog(&snapshot).unwrap();
    restored
        .infer_scoped_bound(&scope(), &raw_twin(&restored, "pages"), None, &binding)
        .unwrap();
    // Probing with the base layered binding must not use the adapted entries of layers 2 and
    // 3, and must not count a scope bug: base entries were never in this snapshot's chains.
    let probe = semantic_request(&restored, "probe", 11);
    let base = StackBinding::base_layered();
    let cold = backend()
        .infer_cold_bound(&scope(), &probe, None, &base)
        .unwrap();
    let out = restored
        .infer_scoped_bound(
            &scope(),
            &probe,
            Some(&ptr_model_api::KvReuseHint {
                requested_prefix_tokens: 0,
                policy: KvReusePolicy::ExactOnly,
            }),
            &base,
        )
        .unwrap();
    same_output(&out, &cold);
    assert_eq!(
        restored.catalog_stats().unwrap().layered_scope_mismatches,
        0,
        "{name}"
    );
    assert!(out.layered.unwrap().shared_layers <= 2, "{name}");
}
