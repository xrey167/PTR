//! What the arms share: the scope, the backends at each workload's limits, the session
//! drivers, the comparison with a cold run, and a bare catalog over the reference paged KV
//! backend for the cases the stack backend does not carry (a dependency of another kind,
//! a forged witness, positions).

use super::metrics::Tally;
use super::params::{self, Workload};
use super::workload::{block, hex};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{ByteTokenizer, SegmentedModelRequest, TokenId, Tokenizer};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, InMemoryPagedKvCache,
    KvContinuationExecutor, KvPageBinding, KvPageSize, KvSharedPrefix, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, ReferenceCausalAttentionExecutor,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{
    CatalogLimits, KvCatalogError, KvCatalogHit, KvChainId, KvDependency, KvInsert,
    KvLivenessOracle, KvLookup, KvPrefixCatalog, KvRequestPositions, KvSalt, KvSession,
    KvSessionFinish, KvSessionPhase, KvSessionStep, PtrRuntime, ReferenceKvLimits,
    ReferenceKvOutcome, ReferenceKvReuseBackend, ReferenceKvScope, SemanticChange, SemanticGrant,
};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    AdapterVersion, CapsuleId, DeviceId, Generation, ModelVersion, PrincipalId, Probability,
    ProjectId, VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

/// A failure of the harness itself (not a falsifier): the code under test refused a step the
/// arm needs, or a fixture did not build.
pub type Res<T> = Result<T, String>;

pub fn fail<E: std::fmt::Debug>(context: &str) -> impl FnOnce(E) -> String + '_ {
    move |error| format!("{context}: {error:?}")
}

pub fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from(params::PRINCIPAL),
        project: ProjectId::from(params::PROJECT),
    }
}

pub fn other_principal_scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from(params::OTHER_PRINCIPAL),
        project: ProjectId::from(params::PROJECT),
    }
}

pub fn other_project_scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from(params::PRINCIPAL),
        project: ProjectId::from(params::OTHER_PROJECT),
    }
}

/// The catalog limits of a workload, as pinned.
pub fn catalog_limits(workload: Workload) -> CatalogLimits {
    let limits = workload.limits();
    CatalogLimits {
        max_trie_nodes: limits.max_trie_nodes as usize,
        max_kv_page_bytes: limits.max_kv_page_bytes,
        max_metadata_bytes: limits.max_metadata_bytes,
    }
}

pub fn backend_limits(workload: Workload, max_sessions: usize) -> ReferenceKvLimits {
    ReferenceKvLimits {
        max_prompt_tokens: params::MAX_PROMPT_TOKENS,
        max_new_tokens: params::MAX_NEW_TOKENS,
        max_sessions,
        catalog: catalog_limits(workload),
    }
}

/// A reference backend at `workload`'s limits with the pinned weights.
pub fn backend(workload: Workload, max_sessions: usize) -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::new(backend_limits(workload, max_sessions), scope())
        .map_err(fail("the reference backend"))
}

/// A backend at the main workload's limits, as the small cases use.
pub fn main_backend() -> Res<ReferenceKvReuseBackend> {
    backend(Workload::Main, 8)
}

/// A runtime whose ledger commits `identities` at generation 1, so each is live as the
/// committed lifecycle state sees it (the ledger-anchored oracle).
pub fn runtime_with(identities: &[&str]) -> Res<PtrRuntime> {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(fail("the runtime"))?;
    for identity in identities {
        commit_capsule(&mut runtime, identity, 1)?;
    }
    Ok(runtime)
}

pub fn commit_capsule(runtime: &mut PtrRuntime, identity: &str, generation: u64) -> Res<()> {
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from(params::PROJECT),
            capsule: CapsuleId::from(identity),
            generation: Generation(generation),
        })
        .map(|_| ())
        .map_err(fail("committing a capsule"))
}

/// Commit the revocation of `identity` at `generation` to the ledger.
pub fn commit_revocation(runtime: &mut PtrRuntime, identity: &str, generation: u64) -> Res<()> {
    runtime
        .commit(LedgerEvent::Revoked {
            subject: identity.to_string(),
            generation: Generation(generation),
        })
        .map(|_| ())
        .map_err(fail("committing a revocation"))
}

/// Commit the supersession of generation `old` of `identity` by `new` to the ledger.
pub fn commit_supersession(
    runtime: &mut PtrRuntime,
    identity: &str,
    old: u64,
    new: u64,
) -> Res<()> {
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(identity),
            old: Generation(old),
            new: Generation(new),
        })
        .map(|_| ())
        .map_err(fail("committing a supersession"))
}

/// The verifier of the semantic grant the harness writes semantic state under: it passes
/// every change, because the writes only exist to move the committed state the liveness
/// oracle reads.
pub struct AcceptAll;

impl<'a> Verifier<SemanticChange<'a>> for AcceptAll {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).expect("a bounded score"),
            findings: Vec::new(),
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for AcceptAll {
    fn name(&self) -> &'static str {
        "r005-accept-all"
    }
}

/// The principal the harness's semantic writes are recorded as written by.
fn operator() -> PrincipalId {
    PrincipalId::from("r005-operator")
}

/// A runtime whose committed semantic state holds `key` (written by a host write under a
/// grant that passes every change): the way a semantic input really lives, with no
/// lifecycle generation.
pub fn semantic_runtime(key: &str) -> Res<PtrRuntime> {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(fail("the runtime"))?;
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AcceptAll)
                .allow_host_writes(),
        )
        .map_err(fail("the semantic grant"))?;
    write_semantic(&mut runtime, key, "v1")?;
    Ok(runtime)
}

/// Commit a host write of `value` under `key`: an edit of the semantic input.
pub fn write_semantic(runtime: &mut PtrRuntime, key: &str, value: &str) -> Res<()> {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.to_string(), value.into());
    let expected = runtime.revision();
    runtime
        .apply_verified_semantic_delta(expected, delta, &operator())
        .map(|_| ())
        .map_err(fail("a semantic write"))
}

/// The run of a session to its end, or its cancellation.
pub enum Ran {
    Completed(Box<ReferenceKvOutcome>),
    Cancelled,
}

/// Begin a session for `request` and drive it to its end under `oracle`.
pub fn run(
    backend: &ReferenceKvReuseBackend,
    scope: &ReferenceKvScope,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
) -> Res<Ran> {
    let mut session = backend
        .begin_session(scope, request, None, oracle)
        .map_err(fail("beginning a session"))?;
    match session
        .run_to_end(oracle)
        .map_err(fail("driving a session"))?
    {
        KvSessionFinish::Completed(outcome) => Ok(Ran::Completed(outcome)),
        KvSessionFinish::Cancelled(_) => Ok(Ran::Cancelled),
    }
}

/// Like [`run`], for a request that must complete.
pub fn run_completed(
    backend: &ReferenceKvReuseBackend,
    scope: &ReferenceKvScope,
    request: &SegmentedModelRequest,
    oracle: &dyn KvLivenessOracle,
) -> Res<ReferenceKvOutcome> {
    match run(backend, scope, request, oracle)? {
        Ran::Completed(outcome) => Ok(*outcome),
        Ran::Cancelled => Err("a session was cancelled where none should be".to_string()),
    }
}

/// The cold run of `request` (no lookup, no insert).
pub fn cold(
    backend: &ReferenceKvReuseBackend,
    scope: &ReferenceKvScope,
    request: &SegmentedModelRequest,
) -> Res<ReferenceKvOutcome> {
    backend
        .infer_cold(scope, request)
        .map_err(fail("a cold run"))
}

/// Whether two outcomes are bitwise equal: the generated tokens, the digest of every logits
/// vector they were chosen from, and the events.
pub fn same_result(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) -> bool {
    a.generated == b.generated && a.logits_digest == b.logits_digest && a.run.events == b.run.events
}

/// Count one comparison with the cold run and, when the two differ, one bitwise mismatch.
pub fn compare(
    tally: &mut Tally,
    observed: &ReferenceKvOutcome,
    cold: &ReferenceKvOutcome,
) -> bool {
    tally.add("bitwise_compared", 1);
    let equal = same_result(observed, cold);
    if !equal {
        tally.add("bitwise_mismatch_count", 1);
    }
    equal
}

/// The steps of a session driven by hand: `prefill_chunks` chunks first, then decode steps.
pub struct Steps<'a> {
    pub session: KvSession<'a>,
}

impl Steps<'_> {
    /// Steps a session of `total` prompt tokens takes before `finish`, `hit` of them served
    /// from the catalog: the prefill chunks left, then one decode step per new token.
    pub fn count(total: usize, hit: usize) -> usize {
        (total - hit).div_ceil(params::CHUNK_TOKENS) + params::MAX_NEW_TOKENS
    }

    /// Run one step (a prefill chunk or a decode step, whichever the phase allows). Returns
    /// whether the session was cancelled by it.
    pub fn step(&mut self, oracle: &dyn KvLivenessOracle) -> Res<bool> {
        let step = match self.session.phase() {
            KvSessionPhase::Prefilling => self.session.prefill_chunk(oracle),
            KvSessionPhase::Decoding => self.session.decode_step(oracle),
            other => return Err(format!("no step in phase {other:?}")),
        }
        .map_err(fail("a session step"))?;
        Ok(matches!(step, KvSessionStep::Cancelled(_)))
    }
}

/// A bare catalog over the reference paged KV backend, with pages from the width-4 reference
/// attention executor: the fixture of the cases that need a dependency of another kind, a
/// forged witness or explicit positions. Its pages are not the R005 tiny model's; nothing
/// measured here is a bitwise claim.
pub struct CatLab {
    pub backend: InMemoryPagedKvBackend,
    pub catalog: KvPrefixCatalog,
}

const WIDTH: usize = 4;

/// The chain id every [`CatLab`] entry is stored under unless a case supplies another.
pub const CAT_CHAIN: KvChainId = KvChainId([1; 32]);

pub fn cat_schema() -> KvTensorSchema {
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

fn cat_binding() -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        generation: Generation(1),
        execution_manifest: [3; 32],
        principal: PrincipalId::from(params::PRINCIPAL),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::DEFAULT,
    }
}

pub fn cat_salt() -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from(params::PRINCIPAL),
        &ProjectId::from(params::PROJECT),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

fn embed(token: TokenId) -> Vec<f32> {
    (0..WIDTH)
        .map(|index| ((token.0 as f32) + index as f32) / 97.0 - 0.5)
        .collect()
}

fn executor() -> Res<ReferenceCausalAttentionExecutor> {
    let weights = CausalAttentionWeights::identity(WIDTH).map_err(fail("attention weights"))?;
    ReferenceCausalAttentionExecutor::new(weights, 2).map_err(fail("the attention executor"))
}

impl CatLab {
    pub fn new(limits: CatalogLimits) -> Res<Self> {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
        let authority = backend
            .take_fork_authority()
            .map_err(fail("the fork authority"))?;
        let catalog = KvPrefixCatalog::new(authority, limits)
            .map_err(fail("the catalog"))?
            .with_retirer(Arc::new(backend.clone()));
        Ok(Self { backend, catalog })
    }

    pub fn fresh_cache(&self) -> Res<InMemoryPagedKvCache> {
        self.backend
            .allocate_bound(cat_schema(), 128, cat_binding())
            .map_err(fail("a cache"))
    }

    /// Prefill `tokens` and publish the whole pages: the prefix and its resident bytes.
    pub fn publish(&self, tokens: &[TokenId]) -> Res<(KvSharedPrefix, u64)> {
        let mut cache = self.fresh_cache()?;
        executor()?
            .prefill(
                &self.backend,
                &mut cache,
                CausalAttentionInput {
                    tokens: tokens.iter().copied().map(embed).collect(),
                },
            )
            .map_err(fail("a prefill"))?;
        let prefix = self
            .backend
            .publish_prefix(&mut cache)
            .map_err(fail("publishing a prefix"))?;
        let bytes = self.backend.page_metrics(&cache).resident_bytes;
        Ok((prefix, bytes))
    }

    /// Insert the whole-chunk prefix of `request` with `extra` dependencies.
    pub fn insert(
        &mut self,
        request: &SegmentedModelRequest,
        extra: Vec<KvDependency>,
        oracle: &dyn KvLivenessOracle,
    ) -> Res<Result<KvSharedPrefix, KvCatalogError>> {
        self.insert_under(&CAT_CHAIN, request, extra, oracle)
    }

    pub fn insert_under(
        &mut self,
        chain: &KvChainId,
        request: &SegmentedModelRequest,
        extra: Vec<KvDependency>,
        oracle: &dyn KvLivenessOracle,
    ) -> Res<Result<KvSharedPrefix, KvCatalogError>> {
        let mut tokens = request.tokens();
        tokens.truncate(tokens.len() / params::CHUNK_TOKENS * params::CHUNK_TOKENS);
        let (prefix, page_bytes) = self.publish(&tokens)?;
        let salt = cat_salt();
        Ok(self
            .catalog
            .insert(
                KvInsert {
                    salt: &salt,
                    chain,
                    request,
                    prefix: prefix.clone(),
                    page_bytes,
                    extra_dependencies: extra,
                },
                oracle,
            )
            .map(|()| prefix))
    }

    pub fn lookup(
        &mut self,
        request: &SegmentedModelRequest,
        oracle: &dyn KvLivenessOracle,
    ) -> Option<KvCatalogHit> {
        self.lookup_with(&CAT_CHAIN, &KvRequestPositions::Canonical, request, oracle)
    }

    pub fn lookup_with(
        &mut self,
        chain: &KvChainId,
        positions: &KvRequestPositions,
        request: &SegmentedModelRequest,
        oracle: &dyn KvLivenessOracle,
    ) -> Option<KvCatalogHit> {
        let salt = cat_salt();
        self.catalog.lookup(
            KvLookup {
                salt: &salt,
                chain,
                request,
                positions,
            },
            oracle,
        )
    }
}

/// A liveness oracle a case scripts: every dependency is live except the named
/// `(identity, generation)` pairs, and an expert whose recorded fence epoch is below the
/// epoch now in force for its identity.
#[derive(Default)]
pub struct Scripted {
    dead: std::collections::BTreeSet<(String, u64)>,
    fences: std::collections::BTreeMap<String, u64>,
}

impl Scripted {
    pub fn kill(&mut self, identity: &str, generation: u64) {
        self.dead.insert((identity.to_string(), generation));
    }

    /// Everything of `identity` below `generation` is superseded.
    pub fn supersede(&mut self, identity: &str, generation: u64) {
        for old in 0..generation {
            self.kill(identity, old);
        }
    }

    pub fn fence(&mut self, identity: &str, epoch: u64) {
        self.fences.insert(identity.to_string(), epoch);
    }
}

impl KvLivenessOracle for Scripted {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        if self
            .dead
            .contains(&(dependency.identity.clone(), dependency.generation.0))
        {
            return false;
        }
        match self.fences.get(&dependency.identity) {
            Some(epoch) => {
                dependency.kind != ptr_runtime::KvDependencyKind::Expert
                    || dependency.fence_epoch >= *epoch
            }
            None => true,
        }
    }
}

/// Lower-case hex SHA-256 over the sorted request ids of a set, the identity-set digests.
pub fn id_set_digest<'a>(ids: impl IntoIterator<Item = &'a str>) -> String {
    let mut sorted: Vec<&str> = ids.into_iter().collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut material = Vec::new();
    for id in sorted {
        material.extend_from_slice(&(id.len() as u64).to_le_bytes());
        material.extend_from_slice(id.as_bytes());
    }
    format!("sha256:{}", hex(&ptr_ledger::integrity::sha256(&material)))
}

/// A two-chunk raw request of blocks `first` and `second`, for the catalog-level cases.
pub fn two_blocks(first: u64, second: u64) -> SegmentedModelRequest {
    super::workload::request(
        "cat",
        &[super::workload::Part::raw(
            block(first)
                .into_iter()
                .chain(block(second))
                .collect::<Vec<_>>(),
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_id_set_digest_is_order_and_duplicate_insensitive_and_discriminates() {
        assert_eq!(id_set_digest(["b", "a"]), id_set_digest(["a", "b", "a"]));
        assert_ne!(id_set_digest(["a"]), id_set_digest(["a", "b"]));
        assert_ne!(id_set_digest(["ab"]), id_set_digest(["a", "b"]));
    }

    #[test]
    fn a_scripted_oracle_kills_generations_supersedes_below_and_fences_by_epoch() {
        let mut oracle = Scripted::default();
        let dependency = |generation, epoch| KvDependency {
            kind: ptr_runtime::KvDependencyKind::Expert,
            identity: "expert:p/x".to_string(),
            generation: Generation(generation),
            content_digest: [0; 32],
            fence_epoch: epoch,
        };
        assert!(oracle.is_live(&dependency(1, 3)));
        oracle.fence("expert:p/x", 4);
        assert!(!oracle.is_live(&dependency(1, 3)));
        assert!(oracle.is_live(&dependency(1, 4)));
        oracle.supersede("expert:p/x", 2);
        assert!(!oracle.is_live(&dependency(1, 4)));
        assert!(oracle.is_live(&dependency(2, 4)));
    }

    #[test]
    fn the_catalog_lab_inserts_and_hits_and_refuses_a_dead_dependency() {
        let mut lab = CatLab::new(CatalogLimits::default()).unwrap();
        let request = two_blocks(1, 2);
        let live = Scripted::default();
        lab.insert(&request, Vec::new(), &live).unwrap().unwrap();
        let hit = lab.lookup(&request, &live).expect("a hit");
        assert_eq!(hit.matched_tokens(), 32);
        lab.catalog.release(hit.pin());
        let mut dead = Scripted::default();
        dead.kill("model:base", 1);
        let dependency = KvDependency::base_model("model:base", Generation(1), [7; 32]);
        let refused = lab
            .insert(&two_blocks(3, 4), vec![dependency], &dead)
            .unwrap();
        assert_eq!(refused.unwrap_err(), KvCatalogError::DependencyNotLive);
    }

    #[test]
    fn the_backend_builders_use_the_pinned_limits_and_weights() {
        let backend = main_backend().unwrap();
        let limits = backend.limits();
        assert_eq!(limits.catalog.max_trie_nodes as u64, params::MAX_TRIE_NODES);
        assert_eq!(limits.catalog.max_kv_page_bytes, params::MAX_KV_PAGE_BYTES);
        assert_eq!(
            limits.catalog.max_metadata_bytes,
            params::MAX_METADATA_BYTES
        );
        assert_eq!(limits.max_new_tokens, params::MAX_NEW_TOKENS);
        assert_eq!(limits.max_prompt_tokens as u64, params::PROMPT_TOKENS);
        assert_eq!(
            ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
            params::WEIGHT_SEED,
            "the backend's weight seed is the pinned one"
        );
    }
}
