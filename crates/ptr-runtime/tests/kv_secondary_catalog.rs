//! The secondary KV index inside the prefix catalog (R006 WP2): every invalidation path of
//! the catalog reaches it, index lag is covered by use-time liveness, and its runs ride in an
//! optional v4 snapshot section that leaves the earlier snapshot bytes untouched.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, KvReuseMode, PromptSegment, SegmentKind, SegmentSource, SegmentedModelRequest,
    TokenId, Tokenizer,
};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, InMemoryPagedKvCache,
    KvBackendError, KvContinuationExecutor, KvPageBinding, KvPageSize, KvSharedPrefix,
    KvTensorDType, KvTensorSchema, PagedKvTensorBackend, ReferenceCausalAttentionExecutor,
};
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvChainId, KvDependency, KvDependencyKind, KvInsert,
    KvLivenessOracle, KvPrefixCatalog, KvRequestPositions, KvSalt, KvSnapshotError, PtrRuntime,
    SecondaryConfig, SecondaryError, SecondaryInsert, SecondaryInsertOutcome, SecondaryLimits,
    SecondaryLookup, SecondaryPinState, SecondaryRefusalReason, SecondaryResource, SecondaryRows,
    SecondaryRunId, SecondarySectionError, DEFAULT_WINDOW_TOKENS, KV_CHUNK_TOKENS,
};
use ptr_types::{
    AdapterVersion, CapsuleId, DeviceId, Generation, KnowledgeObjectId, ModelVersion, PodId,
    PrincipalId, ProjectId, RequestId, Revision, SemanticContext,
};
use sha2::{Digest as _, Sha256};

const ROW: usize = 4;
const WIDTH: usize = 4;
const CHAIN: KvChainId = KvChainId([5; 32]);
const DIGEST: [u8; 32] = [9; 32];

// ----- fixtures ----------------------------------------------------------------------

struct AllLive;

impl KvLivenessOracle for AllLive {
    fn is_live(&self, _: &KvDependency) -> bool {
        true
    }
}

/// Everything is live except the listed `(identity, generation)` pairs.
struct Dead(BTreeSet<(String, u64)>);

impl Dead {
    fn of(items: &[(&str, u64)]) -> Self {
        Self(items.iter().map(|(id, g)| ((*id).to_owned(), *g)).collect())
    }
}

impl KvLivenessOracle for Dead {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        !self
            .0
            .contains(&(dependency.identity.clone(), dependency.generation.0))
    }
}

fn salt() -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from("alice"),
        &ProjectId::from("p1"),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        segments,
    )
    .unwrap()
}

fn raw(text: &str) -> PromptSegment {
    PromptSegment::raw_from_text(&ByteTokenizer::new(), text).unwrap()
}

fn typed(source: SegmentSource, text: &str) -> PromptSegment {
    PromptSegment::from_text(&ByteTokenizer::new(), source, text).unwrap()
}

fn semantic(key: &str, generation: u64, text: &str) -> PromptSegment {
    typed(
        SegmentSource::Semantic {
            key: key.to_owned(),
            generation: Generation(generation),
        },
        text,
    )
}

fn memory(id: &str, generation: u64, text: &str) -> PromptSegment {
    typed(
        SegmentSource::Memory {
            id: KnowledgeObjectId::from(id),
            generation: Generation(generation),
        },
        text,
    )
}

fn derived(id: &str, generation: u64, text: &str) -> PromptSegment {
    typed(
        SegmentSource::Derived {
            id: ptr_types::EvidenceId::from(id),
            generation: Generation(generation),
        },
        text,
    )
}

/// Rows that are a pure function of the token ids.
fn rows_for(tokens: &[TokenId]) -> SecondaryRows {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for token in tokens {
        for column in 0..ROW {
            keys.push(token.0 as f32 * 0.25 + column as f32 - 3.0);
            values.push(token.0 as f32 * -0.5 + column as f32 * 0.125);
        }
    }
    SecondaryRows::new(keys.into(), values.into(), ROW)
}

fn new_catalog(limits: CatalogLimits) -> (KvPrefixCatalog, InMemoryPagedKvBackend) {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let catalog = KvPrefixCatalog::new(backend.take_fork_authority().unwrap(), limits)
        .unwrap()
        .with_retirer(Arc::new(backend.clone()));
    (catalog, backend)
}

fn catalog() -> KvPrefixCatalog {
    new_catalog(CatalogLimits::default()).0
}

fn insert_of<'a>(
    salt: &'a KvSalt,
    request: &'a SegmentedModelRequest,
    segment: usize,
    extras: Vec<KvDependency>,
) -> SecondaryInsert<'a> {
    SecondaryInsert {
        salt,
        chain: &CHAIN,
        mode: KvReuseMode::Exact,
        request,
        segment,
        rows: rows_for(request.segments()[segment].tokens()),
        extra_dependencies: extras,
    }
}

fn put_with(
    catalog: &mut KvPrefixCatalog,
    request: &SegmentedModelRequest,
    segment: usize,
    extras: Vec<KvDependency>,
) -> SecondaryRunId {
    let salt = salt();
    match catalog
        .secondary_mut()
        .insert(insert_of(&salt, request, segment, extras), &AllLive)
        .unwrap()
    {
        SecondaryInsertOutcome::Inserted(id)
        | SecondaryInsertOutcome::AlreadyPresent(id)
        | SecondaryInsertOutcome::RowsAttached(id) => id,
    }
}

/// A run for the single segment of `request`.
fn put(catalog: &mut KvPrefixCatalog, request: &SegmentedModelRequest) -> SecondaryRunId {
    put_with(catalog, request, 0, Vec::new())
}

fn live(catalog: &KvPrefixCatalog, run: SecondaryRunId) -> bool {
    catalog.secondary().contains_live_run(run)
}

fn expert_identity(project: &str, pod: &str) -> String {
    format!("expert:{project}/{pod}")
}

fn expert(project: &str, pod: &str, generation: u64, epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(
        &ProjectId::from(project),
        &PodId::from(pod),
        Generation(generation),
        DIGEST,
        epoch,
    )
}

fn lookup_all(
    catalog: &mut KvPrefixCatalog,
    request: &SegmentedModelRequest,
    liveness: &dyn KvLivenessOracle,
) -> ptr_runtime::SecondaryLookupOutcome {
    let salt = salt();
    catalog
        .secondary_mut()
        .lookup_positional(
            SecondaryLookup {
                salt: &salt,
                chain: &CHAIN,
                mode: KvReuseMode::Exact,
                request,
                positions: &KvRequestPositions::Canonical,
            },
            liveness,
        )
        .unwrap()
}

// ----- every invalidate_* function reaches the index -----------------------------------

const TEXT_A: &str = "alpha alpha alpha alpha";
const TEXT_B: &str = "bravo bravo bravo bravo";
const TEXT_C: &str = "charlie charlie charlie";

fn pair(g: u64) -> BTreeSet<(String, Generation)> {
    BTreeSet::from([("m".to_owned(), Generation(g))])
}

#[test]
fn invalidate_inputs_kills_exactly_the_named_identity_and_generation() {
    let mut c = catalog();
    let hit = put(&mut c, &request(vec![memory("m", 1, TEXT_A)]));
    let other_generation = put(&mut c, &request(vec![memory("m", 2, TEXT_B)]));
    let other_identity = put(&mut c, &request(vec![memory("n", 1, TEXT_C)]));
    let retired = c.invalidate_inputs(&pair(1));
    assert!(retired.is_empty(), "rows are copies: nothing to retire");
    assert!(!live(&c, hit));
    assert!(live(&c, other_generation));
    assert!(live(&c, other_identity));
    assert_eq!(c.secondary().stats().invalidated, 1);
    assert_eq!(c.secondary().stats().runs, 2);
}

#[test]
fn invalidate_identities_kills_every_generation_of_the_identity() {
    let mut c = catalog();
    let g1 = put(&mut c, &request(vec![memory("m", 1, TEXT_A)]));
    let g2 = put(&mut c, &request(vec![memory("m", 2, TEXT_B)]));
    let other = put(&mut c, &request(vec![memory("n", 1, TEXT_C)]));
    c.invalidate_identities(&BTreeSet::from(["m".to_owned()]));
    assert!(!live(&c, g1) && !live(&c, g2));
    assert!(live(&c, other));
}

#[test]
fn invalidate_kind_identities_leaves_other_kinds_with_the_same_identity() {
    let mut c = catalog();
    let sem = put(&mut c, &request(vec![semantic("x", 1, TEXT_A)]));
    let mem = put(&mut c, &request(vec![memory("x", 1, TEXT_B)]));
    let drv = put(&mut c, &request(vec![derived("x", 1, TEXT_C)]));
    c.invalidate_kind_identities(
        KvDependencyKind::Segment(SegmentKind::Semantic),
        &BTreeSet::from(["x".to_owned()]),
    );
    assert!(!live(&c, sem));
    assert!(live(&c, mem) && live(&c, drv));
}

#[test]
fn invalidate_superseded_kills_only_generations_below_the_live_one() {
    let mut c = catalog();
    let old = put(&mut c, &request(vec![memory("m", 1, TEXT_A)]));
    let current = put(&mut c, &request(vec![memory("m", 2, TEXT_B)]));
    let newer = put(&mut c, &request(vec![memory("m", 3, TEXT_C)]));
    c.invalidate_superseded(&BTreeMap::from([("m".to_owned(), Generation(2))]));
    assert!(!live(&c, old));
    assert!(live(&c, current) && live(&c, newer));
}

#[test]
fn invalidate_kind_superseded_reaches_base_model_and_adapter_extras_by_kind() {
    let mut c = catalog();
    let req = request(vec![raw(TEXT_A)]);
    let base = put_with(
        &mut c,
        &req,
        0,
        vec![KvDependency::base_model(
            "model:base",
            Generation(1),
            DIGEST,
        )],
    );
    let adapter_req = request(vec![raw(TEXT_B)]);
    let adapter = put_with(
        &mut c,
        &adapter_req,
        0,
        vec![KvDependency::adapter("model:base", Generation(1), DIGEST)],
    );
    let live_map = BTreeMap::from([("model:base".to_owned(), Generation(2))]);
    c.invalidate_kind_superseded(KvDependencyKind::Adapter, &live_map);
    assert!(!live(&c, adapter));
    assert!(
        live(&c, base),
        "the base-model run shares the identity but not the kind"
    );
    c.invalidate_kind_superseded(KvDependencyKind::BaseModel, &live_map);
    assert!(!live(&c, base));
}

#[test]
fn invalidate_expert_fences_kills_only_dependencies_below_the_new_epoch() {
    let mut c = catalog();
    let below = put_with(
        &mut c,
        &request(vec![raw(TEXT_A)]),
        0,
        vec![expert("p", "e1", 1, 3)],
    );
    let unfenced = put_with(
        &mut c,
        &request(vec![raw(TEXT_B)]),
        0,
        vec![expert("p", "e1", 1, 0)],
    );
    let at_epoch = put_with(
        &mut c,
        &request(vec![raw(TEXT_C)]),
        0,
        vec![expert("p", "e1", 1, 4)],
    );
    let other_expert = put_with(
        &mut c,
        &request(vec![raw("delta delta delta delta")]),
        0,
        vec![expert("p", "e2", 1, 0)],
    );
    c.invalidate_expert_fences(&BTreeMap::from([(expert_identity("p", "e1"), 4)]));
    assert!(!live(&c, below));
    assert!(
        !live(&c, unfenced),
        "a dependency that recorded no epoch counts as below"
    );
    assert!(live(&c, at_epoch));
    assert!(live(&c, other_expert));
}

#[test]
fn invalidate_all_experts_kills_every_expert_run_and_nothing_else() {
    let mut c = catalog();
    let e1 = put_with(
        &mut c,
        &request(vec![raw(TEXT_A)]),
        0,
        vec![expert("p", "e1", 1, 1)],
    );
    let e2 = put_with(
        &mut c,
        &request(vec![raw(TEXT_B)]),
        0,
        vec![expert("q", "e9", 4, 0)],
    );
    let segment = put(&mut c, &request(vec![memory("m", 1, TEXT_C)]));
    let base = put_with(
        &mut c,
        &request(vec![raw("delta delta delta delta")]),
        0,
        vec![KvDependency::base_model(
            "model:base",
            Generation(1),
            DIGEST,
        )],
    );
    c.invalidate_all_experts();
    assert!(!live(&c, e1) && !live(&c, e2));
    assert!(live(&c, segment) && live(&c, base));
}

#[test]
fn an_invalidation_of_a_run_with_an_unrelated_dependency_set_changes_nothing() {
    let mut c = catalog();
    let run = put(&mut c, &request(vec![memory("m", 1, TEXT_A)]));
    c.invalidate_inputs(&BTreeSet::from([("zzz".to_owned(), Generation(1))]));
    c.invalidate_identities(&BTreeSet::from(["zzz".to_owned()]));
    c.invalidate_superseded(&BTreeMap::from([("zzz".to_owned(), Generation(9))]));
    c.invalidate_expert_fences(&BTreeMap::from([(expert_identity("p", "e"), 9)]));
    assert!(live(&c, run));
    assert_eq!(c.secondary().stats().invalidated, 0);
}

#[test]
fn a_pinned_run_is_revoked_at_once_and_reclaimed_with_its_last_pin() {
    let mut c = catalog();
    let origin = request(vec![memory("m", 1, TEXT_A), raw("tail tail")]);
    put_with(&mut c, &origin, 0, Vec::new());
    let shifted = request(vec![
        raw("lead lead lead"),
        memory("m", 1, TEXT_A),
        raw("tail tail"),
    ]);
    let found = lookup_all(&mut c, &shifted, &AllLive);
    assert_eq!(found.splices.len(), 1);
    let pin = found.splices[0].pin;
    assert_eq!(c.secondary().pin_state(&pin), SecondaryPinState::Live);
    c.invalidate_identities(&BTreeSet::from(["m".to_owned()]));
    assert_eq!(c.secondary().pin_state(&pin), SecondaryPinState::Revoked);
    assert_eq!(c.secondary().stats().tombstoned_pinned_runs, 1);
    assert!(c.secondary_mut().release(pin));
    assert_eq!(c.secondary().stats().runs, 0);
}

// ----- lag: a lost invalidation is covered by use-time liveness ----------------------------

#[test]
fn a_lost_invalidation_is_refused_by_use_time_liveness_and_never_admitted() {
    let mut c = catalog();
    let origin = request(vec![
        memory("m", 1, TEXT_A),
        semantic("k", 1, TEXT_B),
        raw("tail tail"),
    ]);
    let run = put_with(&mut c, &origin, 1, Vec::new());
    let shifted = request(vec![
        raw("lead lead lead"),
        memory("m", 1, TEXT_A),
        semantic("k", 1, TEXT_B),
        raw("tail tail"),
    ]);
    // No invalidate_* call is made: the index never heard of the revocation of `m`.
    assert!(live(&c, run));
    let found = lookup_all(&mut c, &shifted, &Dead::of(&[("m", 1)]));
    assert!(
        found.splices.is_empty(),
        "a dead upstream is never admitted"
    );
    assert_eq!(
        found.refusals_for(SecondaryRefusalReason::OriginalUpstreamDead),
        1
    );
    assert_eq!(c.secondary().stats().refusals.original_upstream_dead, 1);
    assert!(
        !live(&c, run),
        "the dead run is tombstoned at the lookup that found it"
    );
    // The same content under a live oracle is a plain miss now: the run is gone.
    let again = lookup_all(&mut c, &shifted, &AllLive);
    assert!(again.splices.is_empty() && again.refusals.is_empty());
}

#[test]
fn a_lost_invalidation_of_an_extra_dependency_is_a_binding_refusal() {
    let mut c = catalog();
    let origin = request(vec![memory("m", 1, TEXT_A), raw("tail tail")]);
    put_with(
        &mut c,
        &origin,
        0,
        vec![KvDependency::base_model(
            "model:base",
            Generation(1),
            DIGEST,
        )],
    );
    let shifted = request(vec![
        raw("lead lead lead"),
        memory("m", 1, TEXT_A),
        raw("tail tail"),
    ]);
    let found = lookup_all(&mut c, &shifted, &Dead::of(&[("model:base", 1)]));
    assert!(found.splices.is_empty());
    assert_eq!(
        found.refusals_for(SecondaryRefusalReason::BindingDependencyDead),
        1
    );
}

// ----- the committed-event mapping reaches the index ------------------------------------------

fn runtime_with(catalog: KvPrefixCatalog) -> PtrRuntime {
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    runtime.enable_kv_prefix_catalog(catalog).unwrap();
    runtime
}

fn runtime_live(runtime: &PtrRuntime, run: SecondaryRunId) -> bool {
    runtime
        .kv_prefix_catalog()
        .unwrap()
        .secondary()
        .contains_live_run(run)
}

#[test]
fn committed_revocation_and_supersession_reach_the_index_through_the_runtime() {
    let mut c = catalog();
    let revoked = put(&mut c, &request(vec![memory("capsule:a", 1, TEXT_A)]));
    let superseded = put(&mut c, &request(vec![memory("capsule:b", 1, TEXT_B)]));
    let current = put(&mut c, &request(vec![memory("capsule:b", 2, TEXT_C)]));
    let unrelated = put(
        &mut c,
        &request(vec![memory("capsule:z", 1, "zulu zulu zulu zulu")]),
    );
    let mut runtime = runtime_with(c);
    for capsule in ["capsule:a", "capsule:b"] {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p1"),
                capsule: CapsuleId::from(capsule),
                generation: Generation(1),
            })
            .unwrap();
    }
    // Committing generation 1 supersedes nothing: every run is still reachable.
    for run in [revoked, superseded, current, unrelated] {
        assert!(runtime_live(&runtime, run));
    }
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(!runtime_live(&runtime, revoked));
    assert!(runtime_live(&runtime, unrelated));
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:b"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    assert!(!runtime_live(&runtime, superseded));
    assert!(
        runtime_live(&runtime, current),
        "the live generation survives"
    );
    assert!(runtime_live(&runtime, unrelated));
}

fn expert_fence_event() -> LedgerEvent {
    LedgerEvent::ExpertFenceCommitted {
        project: ProjectId::from("p"),
        pod: PodId::from("e1"),
        action: ptr_ledger::ExpertFenceAction::Fence,
        cause: ptr_ledger::ExpertFenceCause::AuthorityQuarantine,
        resume: ptr_ledger::ExpertFenceResume::Serving,
        generation: Generation(1),
        expected_state_version: 0,
        epoch: 4,
        revocation: None,
        evidence: None,
    }
}

#[test]
fn apply_committed_event_is_the_runtime_hooks_mapping_on_a_catalog_the_host_owns() {
    let events = [
        LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        },
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(1),
            new: Generation(2),
        },
    ];
    for event in events {
        let build = || {
            let mut c = catalog();
            let victim = put(&mut c, &request(vec![memory("capsule:a", 1, TEXT_A)]));
            let bystander = put(&mut c, &request(vec![memory("capsule:z", 1, TEXT_C)]));
            (c, victim, bystander)
        };
        // A standalone catalog, through the public wrapper.
        let (mut owned, victim, bystander) = build();
        owned.apply_committed_event(&event);
        assert!(!live(&owned, victim), "{event:?}: the run survived");
        assert!(
            live(&owned, bystander),
            "{event:?}: a bystander was flushed"
        );
        // The runtime's own hook reaches the same state for the same event.
        let (hosted, hosted_victim, hosted_bystander) = build();
        let mut runtime = runtime_with(hosted);
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p1"),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(1),
            })
            .unwrap();
        runtime.commit(event.clone()).unwrap();
        assert_eq!(
            runtime_live(&runtime, hosted_victim),
            live(&owned, victim),
            "{event:?}: the hook and the wrapper disagree"
        );
        assert_eq!(
            runtime_live(&runtime, hosted_bystander),
            live(&owned, bystander),
            "{event:?}"
        );
    }
}

#[test]
fn apply_committed_event_spares_other_experts_and_ignores_events_that_invalidate_nothing() {
    let mut c = catalog();
    let below = put_with(
        &mut c,
        &request(vec![raw(TEXT_A)]),
        0,
        vec![expert("p", "e1", 1, 1)],
    );
    let other = put_with(
        &mut c,
        &request(vec![raw(TEXT_B)]),
        0,
        vec![expert("p", "e2", 1, 0)],
    );
    let plain = put(&mut c, &request(vec![memory("capsule:z", 1, TEXT_C)]));
    // An event that invalidates nothing is a no-op.
    let noop = LedgerEvent::VerifierAttested {
        subject: "capsule:z".into(),
        passed: true,
    };
    assert_eq!(c.apply_committed_event(&noop), 0);
    assert!([below, other, plain].iter().all(|run| live(&c, *run)));
    // A semantic delta event carries no delta: it changes nothing here.
    let semantic_event = LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: Vec::new(),
        origin: ptr_ledger::SemanticOrigin::Legacy,
    };
    assert_eq!(c.apply_committed_event(&semantic_event), 0);
    assert!([below, other, plain].iter().all(|run| live(&c, *run)));
    // A fence of e1 kills the e1 run below the new epoch and nothing else.
    c.apply_committed_event(&expert_fence_event());
    assert!(!live(&c, below));
    assert!(live(&c, other), "another expert is untouched");
    assert!(live(&c, plain), "an unrelated run is untouched");
}

#[test]
fn a_committed_revocation_kills_the_trie_entry_and_the_run_together() {
    let (mut c, backend) = new_catalog(CatalogLimits::default());
    let block: String = "m".repeat(KV_CHUNK_TOKENS);
    let both = request(vec![
        raw(&"a".repeat(KV_CHUNK_TOKENS)),
        memory("capsule:a", 1, &block),
    ]);
    let prefix = publish(&backend, &both.tokens());
    let salt = salt();
    c.insert(
        KvInsert {
            salt: &salt,
            chain: &CHAIN,
            request: &both,
            prefix: prefix.clone(),
            page_bytes: 1000,
            extra_dependencies: Vec::new(),
        },
        &AllLive,
    )
    .unwrap();
    let run = put_with(&mut c, &both, 1, Vec::new());
    assert_eq!(c.stats().entries, 1);
    let mut runtime = runtime_with(c);
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p1"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(!runtime_live(&runtime, run));
    assert_eq!(runtime.kv_prefix_catalog().unwrap().stats().entries, 0);
    assert_eq!(
        backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound),
        "the trie side still retires on the backend"
    );
}

// ----- snapshots -------------------------------------------------------------------------

fn executor() -> ReferenceCausalAttentionExecutor {
    ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(WIDTH).unwrap(), 2)
        .unwrap()
}

fn embed(token: TokenId) -> Vec<f32> {
    (0..WIDTH)
        .map(|i| ((token.0 as f32) + i as f32) / 97.0 - 0.5)
        .collect()
}

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn binding() -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        generation: Generation(1),
        execution_manifest: [3; 32],
        principal: PrincipalId::from("alice"),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::DEFAULT,
    }
}

fn publish(backend: &InMemoryPagedKvBackend, tokens: &[TokenId]) -> KvSharedPrefix {
    let mut cache: InMemoryPagedKvCache = backend.allocate_bound(schema(), 128, binding()).unwrap();
    executor()
        .prefill(
            backend,
            &mut cache,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    backend.publish_prefix(&mut cache).unwrap()
}

/// A catalog with two trie entries and, when asked, secondary runs on a memory and a
/// semantic segment.
fn populated(with_runs: bool) -> (KvPrefixCatalog, InMemoryPagedKvBackend) {
    let (mut c, backend) = new_catalog(CatalogLimits::default());
    let salt = salt();
    for second in ['b', 'c'] {
        let req = request(vec![raw(&format!(
            "{}{}",
            "a".repeat(KV_CHUNK_TOKENS),
            second.to_string().repeat(KV_CHUNK_TOKENS)
        ))]);
        let prefix = publish(&backend, &req.tokens());
        c.insert(
            KvInsert {
                salt: &salt,
                chain: &CHAIN,
                request: &req,
                prefix,
                page_bytes: 1000,
                extra_dependencies: Vec::new(),
            },
            &AllLive,
        )
        .unwrap();
    }
    if with_runs {
        let origin = request(vec![
            semantic("price:sku", 1, TEXT_A),
            memory("capsule:a", 1, TEXT_B),
            raw("tail tail"),
        ]);
        put_with(&mut c, &origin, 0, Vec::new());
        put_with(&mut c, &origin, 1, Vec::new());
    }
    (c, backend)
}

fn checksum(domain: &[u8], body: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(body);
    hasher.finalize().into()
}

#[test]
fn a_catalog_without_runs_exports_the_byte_identical_previous_snapshot() {
    let (plain, _) = populated(false);
    let bytes = plain.export_entries().unwrap();
    assert!(
        bytes.starts_with(b"PTRKVC01"),
        "no section: still the v1 magic"
    );
    let (body, trailer) = bytes.split_at(bytes.len() - 32);
    assert_eq!(
        trailer,
        checksum(b"ptr.kv.catalog-snapshot.v1\0", body),
        "the v1 checksum domain is unchanged"
    );

    // A catalog that saw runs come and go exports the very same bytes.
    let (mut churned, _) = populated(false);
    let run = put(&mut churned, &request(vec![memory("m", 1, TEXT_A)]));
    churned.invalidate_identities(&BTreeSet::from(["m".to_owned()]));
    assert!(!live(&churned, run));
    assert_eq!(churned.export_entries().unwrap(), bytes);

    // And a run whose only holder is a tombstoned, still pinned slot is not exported.
    let (mut pinned, _) = populated(false);
    let origin = request(vec![memory("m", 1, TEXT_A), raw("tail tail")]);
    put_with(&mut pinned, &origin, 0, Vec::new());
    let shifted = request(vec![
        raw("lead lead lead"),
        memory("m", 1, TEXT_A),
        raw("tail tail"),
    ]);
    let found = lookup_all(&mut pinned, &shifted, &AllLive);
    pinned.invalidate_identities(&BTreeSet::from(["m".to_owned()]));
    assert_eq!(pinned.export_entries().unwrap(), bytes);
    pinned.secondary_mut().release(found.splices[0].pin);
}

#[test]
fn a_v1_snapshot_still_restores_with_no_secondary_runs() {
    // A snapshot without a section restores exactly as before and adds no run.
    let (plain, _) = populated(false);
    let bytes = plain.export_entries().unwrap();
    let (mut target, _) = new_catalog(CatalogLimits::default());
    let report = target.restore_entries(&bytes).unwrap();
    assert_eq!((report.entries, report.secondary_runs), (2, 0));
    assert_eq!(target.export_entries().unwrap(), bytes);
}

#[test]
fn a_catalog_with_runs_exports_v4_and_restores_row_less_runs() {
    let (source, _) = populated(true);
    let bytes = source.export_entries().unwrap();
    assert!(bytes.starts_with(b"PTRKVS04"));
    assert_eq!(bytes, source.export_entries().unwrap(), "deterministic");
    let (body, trailer) = bytes.split_at(bytes.len() - 32);
    assert_eq!(trailer, checksum(b"ptr.kv.catalog-snapshot.v4\0", body));

    let (mut target, _) = new_catalog(CatalogLimits::default());
    let report = target.restore_entries(&bytes).unwrap();
    assert_eq!(report.entries, 2);
    assert_eq!(report.secondary_runs, 2);
    assert_eq!(report.skipped_unscoped_layered, 0);
    assert_eq!(target.stats().entries, 2);
    assert_eq!(target.secondary().stats().runs, 2);
    assert_eq!(
        target.secondary().stats().row_bytes,
        0,
        "a snapshot carries no rows"
    );
    assert_eq!(
        target.export_entries().unwrap(),
        bytes,
        "round trip is byte-stable"
    );

    // Restored runs miss until the same content is re-inserted under a live oracle.
    let origin = request(vec![
        semantic("price:sku", 1, TEXT_A),
        memory("capsule:a", 1, TEXT_B),
        raw("tail tail"),
    ]);
    let shifted = request(vec![
        raw("lead lead lead"),
        semantic("price:sku", 1, TEXT_A),
        memory("capsule:a", 1, TEXT_B),
        raw("tail tail"),
    ]);
    let miss = lookup_all(&mut target, &shifted, &AllLive);
    assert!(miss.splices.is_empty() && miss.refusals.is_empty());
    assert_eq!(target.secondary().stats().rows_missing_misses, 2);

    let salt = salt();
    let mut wrong = insert_of(&salt, &origin, 1, Vec::new());
    let mut keys = wrong.rows.keys.to_vec();
    keys[0] = f32::from_bits(keys[0].to_bits() ^ 1);
    wrong.rows.keys = keys.into();
    assert_eq!(
        target.secondary_mut().insert(wrong, &AllLive).unwrap_err(),
        SecondaryError::RowsDigestMismatch
    );
    // The conflicting restored run is tombstoned (and, unpinned, dropped) for good.
    assert_eq!(target.secondary().stats().conflict_tombstones, 1);
    assert_eq!(target.secondary().stats().runs, 1);
    assert_eq!(
        target
            .secondary_mut()
            .insert(
                insert_of(&salt, &origin, 1, Vec::new()),
                &Dead::of(&[("price:sku", 1)])
            )
            .unwrap_err(),
        SecondaryError::DependencyNotLive,
        "rows are never attached to a run whose input died at rest"
    );
    assert!(matches!(
        target
            .secondary_mut()
            .insert(insert_of(&salt, &origin, 0, Vec::new()), &AllLive)
            .unwrap(),
        SecondaryInsertOutcome::RowsAttached(_)
    ));
    // The tombstoned run's slot is free: the correct content is a fresh run.
    assert!(matches!(
        target
            .secondary_mut()
            .insert(insert_of(&salt, &origin, 1, Vec::new()), &AllLive)
            .unwrap(),
        SecondaryInsertOutcome::Inserted(_)
    ));
    let hit = lookup_all(&mut target, &shifted, &AllLive);
    assert_eq!(hit.splices.len(), 2);
    assert_eq!(hit.splices[0].rows, rows_for(origin.segments()[0].tokens()));
    assert_eq!(hit.splices[1].rows, rows_for(origin.segments()[1].tokens()));
    for splice in &hit.splices {
        target.secondary_mut().release(splice.pin);
    }
    assert_eq!(target.export_entries().unwrap(), bytes);
}

#[test]
fn a_v4_snapshot_may_carry_only_runs() {
    let (mut source, _) = new_catalog(CatalogLimits::default());
    put(&mut source, &request(vec![memory("m", 1, TEXT_A)]));
    let bytes = source.export_entries().unwrap();
    assert!(bytes.starts_with(b"PTRKVS04"));
    let (mut target, _) = new_catalog(CatalogLimits::default());
    let report = target.restore_entries(&bytes).unwrap();
    assert_eq!((report.entries, report.secondary_runs), (0, 1));
    assert_eq!(target.stats().entries, 0);
}

#[test]
fn restoring_twice_into_a_catalog_that_holds_entries_is_refused() {
    let (source, _) = populated(true);
    let bytes = source.export_entries().unwrap();
    let (mut target, _) = new_catalog(CatalogLimits::default());
    target.restore_entries(&bytes).unwrap();
    assert_eq!(
        target.restore_entries(&bytes).unwrap_err(),
        KvSnapshotError::NotEmpty
    );
    assert_eq!(target.secondary().stats().runs, 2);
}

/// Re-seal a v4 body with the v4 checksum, so a malformed body reaches the decoder.
fn reseal(mut body: Vec<u8>) -> Vec<u8> {
    let trailer = checksum(b"ptr.kv.catalog-snapshot.v4\0", &body);
    body.extend_from_slice(&trailer);
    body
}

#[test]
fn a_malformed_v4_snapshot_admits_nothing() {
    let (source, _) = populated(true);
    let good = source.export_entries().unwrap();
    let section = source.secondary().export_section().unwrap();
    let body = good[..good.len() - 32].to_vec();

    let restore = |bytes: &[u8]| {
        let (mut target, _) = new_catalog(CatalogLimits::default());
        let (stats, secondary) = (target.stats(), target.secondary().stats());
        let result = target.restore_entries(bytes);
        assert_eq!(target.stats().entries, 0, "nothing stays admitted");
        assert_eq!(target.secondary().stats().runs, 0, "nothing stays admitted");
        assert_eq!(target.stats(), stats, "no catalog counter moved");
        assert_eq!(
            target.secondary().stats(),
            secondary,
            "no index counter moved"
        );
        result
    };

    // Corruption and truncation of the sealed bytes.
    let mut flipped = good.clone();
    flipped[good.len() / 2] ^= 1;
    assert_eq!(restore(&flipped).unwrap_err(), KvSnapshotError::BadChecksum);
    assert!(restore(&good[..good.len() - 1]).is_err());

    // A section whose own magic is wrong: sealed correctly, refused by the section decoder,
    // after the trie entries were admitted, which are then taken back.
    let mut bad_section = body.clone();
    let start = bad_section.len() - section.len();
    bad_section[start] ^= 1;
    assert_eq!(
        restore(&reseal(bad_section)).unwrap_err(),
        KvSnapshotError::Secondary(SecondaryError::Section(SecondarySectionError::BadMagic))
    );

    // An empty section is not canonical (the writer never emits one).
    let mut empty = body[..body.len() - section.len() - 4].to_vec();
    empty.extend_from_slice(&0u32.to_le_bytes());
    assert!(matches!(
        restore(&reseal(empty)).unwrap_err(),
        KvSnapshotError::NonCanonical(_)
    ));

    // A declared length above the section bound.
    let mut huge = body[..body.len() - section.len() - 4].to_vec();
    huge.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        restore(&reseal(huge)).unwrap_err(),
        KvSnapshotError::LimitExceeded(_)
    ));

    // Trailing bytes after the section.
    let mut trailing = body.clone();
    trailing.push(0);
    assert!(matches!(
        restore(&reseal(trailing)).unwrap_err(),
        KvSnapshotError::NonCanonical(_)
    ));

    // A section the target index cannot hold: refused with the bound, trie rolled back.
    let (mut tiny, _) = new_catalog(CatalogLimits::default());
    tiny = tiny
        .with_secondary_config(SecondaryConfig {
            window_tokens: DEFAULT_WINDOW_TOKENS,
            limits: SecondaryLimits {
                max_indexed_windows: 5,
                ..SecondaryLimits::default()
            },
        })
        .unwrap();
    assert_eq!(
        tiny.restore_entries(&good).unwrap_err(),
        KvSnapshotError::Secondary(SecondaryError::CapacityExhausted(
            SecondaryResource::IndexedWindows
        ))
    );
    assert_eq!(tiny.stats().entries, 0);
    assert_eq!(tiny.secondary().stats().runs, 0);
    // The refusal is decided before any trie entry is admitted: every counter, including
    // the index's capacity counter, is as it was.
    let (mut tiny2, _) = new_catalog(CatalogLimits::default());
    tiny2 = tiny2
        .with_secondary_config(SecondaryConfig {
            window_tokens: DEFAULT_WINDOW_TOKENS,
            limits: SecondaryLimits {
                max_indexed_windows: 5,
                ..SecondaryLimits::default()
            },
        })
        .unwrap();
    let (stats, secondary) = (tiny2.stats(), tiny2.secondary().stats());
    assert!(tiny2.restore_entries(&good).is_err());
    assert_eq!(tiny2.stats(), stats);
    assert_eq!(tiny2.secondary().stats(), secondary);
}

#[test]
fn a_conflicting_restored_run_is_tombstoned_and_refused_not_fatal() {
    let (source, _) = populated(true);
    let bytes = source.export_entries().unwrap();
    let (mut target, _) = new_catalog(CatalogLimits::default());
    target.restore_entries(&bytes).unwrap();
    assert_eq!(target.secondary().stats().runs, 2);

    let origin = request(vec![
        semantic("price:sku", 1, TEXT_A),
        memory("capsule:a", 1, TEXT_B),
        raw("tail tail"),
    ]);
    let salt = salt();
    // Rows that do not match the digest the restored run recorded.
    let altered: Vec<TokenId> = origin.segments()[1]
        .tokens()
        .iter()
        .map(|token| TokenId(token.0 ^ 1))
        .collect();
    let mut conflicting = insert_of(&salt, &origin, 1, Vec::new());
    conflicting.rows = rows_for(&altered);
    assert_eq!(
        target
            .secondary_mut()
            .insert(conflicting, &AllLive)
            .unwrap_err(),
        SecondaryError::RowsDigestMismatch
    );
    let stats = target.secondary().stats();
    assert_eq!(stats.conflict_tombstones, 1);
    assert_eq!(stats.runs, 1, "the conflicting run is gone");

    // The slot is free for the correct content, which is a fresh run.
    let fresh = target
        .secondary_mut()
        .insert(insert_of(&salt, &origin, 1, Vec::new()), &AllLive)
        .unwrap();
    assert!(matches!(fresh, SecondaryInsertOutcome::Inserted(_)));
    assert_eq!(target.secondary().stats().runs, 2);
}

// ----- configuration ------------------------------------------------------------------------

#[test]
fn a_catalog_defaults_to_the_default_secondary_config_and_is_otherwise_unchanged() {
    let c = catalog();
    assert_eq!(c.secondary().window_tokens(), DEFAULT_WINDOW_TOKENS);
    assert_eq!(c.secondary().limits(), SecondaryLimits::default());
    assert!(c.secondary().is_empty());
    let stats = c.stats();
    assert_eq!((stats.entries, stats.nodes, stats.hits), (0, 0, 0));
}

#[test]
fn a_secondary_config_is_validated_and_only_applies_to_an_empty_index() {
    let custom = SecondaryConfig {
        window_tokens: 4,
        limits: SecondaryLimits {
            max_runs: 1,
            ..SecondaryLimits::default()
        },
    };
    let mut c = catalog().with_secondary_config(custom).unwrap();
    assert_eq!(c.secondary().window_tokens(), 4);
    put(&mut c, &request(vec![memory("m", 1, TEXT_A)]));
    // The bound of the custom config binds: a second run has no room (the first is not
    // pinned, so it is evicted to make room, which proves the limit is the configured one).
    put(&mut c, &request(vec![memory("n", 1, TEXT_B)]));
    assert_eq!(c.secondary().stats().runs, 1);
    assert_eq!(c.secondary().stats().evictions, 1);

    let invalid = SecondaryConfig {
        window_tokens: 0,
        limits: SecondaryLimits::default(),
    };
    assert_eq!(
        catalog().with_secondary_config(invalid).unwrap_err(),
        KvCatalogError::InvalidLimits
    );
    assert_eq!(
        c.with_secondary_config(SecondaryConfig::default())
            .unwrap_err(),
        KvCatalogError::InvalidLimits,
        "a populated index is never replaced"
    );
}
