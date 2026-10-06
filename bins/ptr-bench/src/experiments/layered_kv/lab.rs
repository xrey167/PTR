//! What the arms share: the scope, the backend, the ledger-anchored liveness oracle, the
//! session drivers and the comparison with a cold run.

use super::corpus::hex;
use super::emit::Tally;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{SegmentedModelRequest, Tokenizer};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{
    BindingError, KvDependency, KvLivenessOracle, KvSession, KvSessionFinish, KvSessionPhase,
    KvSessionStep, PtrRuntime, ReferenceKvError, ReferenceKvLimits, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope, SemanticChange, SemanticGrant, StackBinding,
};
use ptr_semdb::SemanticDelta;
use ptr_types::{CapsuleId, Generation, PrincipalId, Probability, ProjectId, VerificationLevel};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// A failure of the harness itself (not a falsifier): the code under test refused a step the
/// arm needs, or a fixture did not build.
pub type Res<T> = Result<T, String>;

pub fn fail<E: std::fmt::Debug>(context: &str) -> impl FnOnce(E) -> String + '_ {
    move |error| format!("{context}: {error:?}")
}

pub const PRINCIPAL: &str = "layered-kv-principal";
pub const PROJECT: &str = "layered-kv-project";

/// The lifecycle identity under which the backend's base-model dependency is recorded
/// (`MODEL_NAME` in `crates/ptr-runtime/src/kv_infer.rs`); a run that cites another name
/// cannot begin, so a wrong value fails loudly.
pub const BASE_MODEL_IDENTITY: &str = "r005-tiny";
pub const BASE_MODEL_GENERATION: u64 = 1;

pub fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from(PRINCIPAL),
        project: ProjectId::from(PROJECT),
    }
}

/// A reference backend at the backend's default limits and the pinned weights.
pub fn backend() -> Res<ReferenceKvReuseBackend> {
    ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope())
        .map_err(fail("the reference backend"))
}

/// The ledger-anchored liveness oracle of a case: a [`PtrRuntime`] whose committed state
/// answers every dependency kind. Capsules (the base model, adapters, memory and derived
/// inputs) are lifecycle targets committed at generation 1; a semantic input is a key of the
/// committed semantic state; an expert is a record the real admission path wrote.
pub struct Authority {
    pub runtime: PtrRuntime,
}

impl KvLivenessOracle for Authority {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        self.runtime.is_live(dependency)
    }
}

impl Authority {
    /// A runtime whose ledger commits the base model and every identity of `capsules` at
    /// generation 1.
    pub fn new(capsules: &[&str]) -> Res<Self> {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(fail("the runtime"))?;
        commit_capsule(&mut runtime, BASE_MODEL_IDENTITY, BASE_MODEL_GENERATION)?;
        for identity in capsules {
            commit_capsule(&mut runtime, identity, 1)?;
        }
        Ok(Self { runtime })
    }

    /// Like [`Self::new`], with the semantic key `key` written to the committed semantic
    /// state by a host write (the way a semantic input really lives: no lifecycle generation).
    pub fn with_semantic(key: &str, capsules: &[&str]) -> Res<Self> {
        let mut authority = Self::new(capsules)?;
        authority
            .runtime
            .install_semantic_grant(
                SemanticGrant::new(RequiredVerification::Deterministic)
                    .with_verifier(AcceptAll)
                    .allow_host_writes(),
            )
            .map_err(fail("the semantic grant"))?;
        write_semantic(&mut authority.runtime, key, "v1")?;
        Ok(authority)
    }
}

pub fn commit_capsule(runtime: &mut PtrRuntime, identity: &str, generation: u64) -> Res<()> {
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from(PROJECT),
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
        "layered-kv-accept-all"
    }
}

/// Commit a host write of `value` under `key`: an edit of the semantic input.
pub fn write_semantic(runtime: &mut PtrRuntime, key: &str, value: &str) -> Res<()> {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.to_string(), value.into());
    let expected = runtime.revision();
    runtime
        .apply_verified_semantic_delta(expected, delta, &PrincipalId::from("layered-kv-operator"))
        .map(|_| ())
        .map_err(fail("a semantic write"))
}

/// What a session run ended in.
pub enum Ran {
    Completed(Box<ReferenceKvOutcome>),
    Cancelled,
}

/// Begin a session for `request` on `binding` and drive it to its end under `oracle`.
pub fn run_bound(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    binding: &StackBinding,
    oracle: &dyn KvLivenessOracle,
) -> Res<Ran> {
    let mut session = backend
        .begin_session_bound(&scope(), request, None, binding, oracle)
        .map_err(fail("beginning a session"))?;
    match session
        .run_to_end(oracle)
        .map_err(fail("driving a session"))?
    {
        KvSessionFinish::Completed(outcome) => Ok(Ran::Completed(outcome)),
        KvSessionFinish::Cancelled(_) => Ok(Ran::Cancelled),
    }
}

/// Like [`run_bound`], for a request that must complete.
pub fn run_completed(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    binding: &StackBinding,
    oracle: &dyn KvLivenessOracle,
) -> Res<ReferenceKvOutcome> {
    match run_bound(backend, request, binding, oracle)? {
        Ran::Completed(outcome) => Ok(*outcome),
        Ran::Cancelled => Err("a session was cancelled where none should be".to_string()),
    }
}

/// The cold run of `request` on `binding` (no lookup, no insert).
pub fn cold(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    binding: &StackBinding,
) -> Res<ReferenceKvOutcome> {
    backend
        .infer_cold_bound(&scope(), request, None, binding)
        .map_err(fail("a cold run"))
}

/// Whether two outcomes are bitwise equal: the generated tokens, the digest of every logits
/// vector they were chosen from, and the events.
pub fn same_result(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) -> bool {
    a.generated == b.generated && a.logits_digest == b.logits_digest && a.run.events == b.run.events
}

/// Count one comparison with the cold run and, when the two differ, one bitwise mismatch
/// (and one mismatching run tagged Exact when the observed run is).
pub fn compare(
    tally: &mut Tally,
    observed: &ReferenceKvOutcome,
    cold: &ReferenceKvOutcome,
) -> bool {
    tally.add("bitwise_compared", 1);
    let equal = same_result(observed, cold);
    if !equal {
        tally.add("bitwise_mismatch_count", 1);
        let exact = observed
            .run
            .kv_report
            .as_ref()
            .is_some_and(|report| report.mode() == ptr_model_api::KvReuseMode::Exact);
        if exact {
            tally.add("mismatching_runs_tagged_exact", 1);
        }
    }
    equal
}

/// Whether beginning a bound session was refused because a dependency of the binding is not
/// live.
pub fn refused_not_live<T>(result: &Result<T, ReferenceKvError>) -> bool {
    matches!(
        result,
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    )
}

/// The steps of a session driven by hand.
pub struct Steps<'a> {
    pub session: KvSession<'a>,
}

impl Steps<'_> {
    /// Steps a session of `total` prompt tokens takes before `finish`, `hit` of them served
    /// from the catalog: the prefill chunks left, then one decode step per new token.
    pub fn count(total: usize, hit: usize, max_new: usize) -> usize {
        (total - hit).div_ceil(super::corpus::CHUNK_TOKENS) + max_new
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

/// The request identities the backends' own bookkeeping saw cancelled and restarted.
#[derive(Debug, Default)]
pub struct Ids {
    pub cancelled: std::collections::BTreeSet<String>,
    pub restarted: std::collections::BTreeSet<String>,
}

impl Ids {
    pub fn absorb(&mut self, backend: &ReferenceKvReuseBackend) -> Res<()> {
        let stats = backend.session_stats().map_err(fail("session stats"))?;
        self.cancelled
            .extend(stats.cancelled_requests.iter().map(|id| id.0.clone()));
        self.restarted
            .extend(stats.restarted_requests.iter().map(|id| id.0.clone()));
        Ok(())
    }

    pub fn cancelled_digest(&self) -> String {
        id_set_digest(self.cancelled.iter().map(String::as_str))
    }

    pub fn restarted_digest(&self) -> String {
        id_set_digest(self.restarted.iter().map(String::as_str))
    }
}

/// The byte tokenizer's identity text, `name@version`.
pub fn tokenizer_text(backend: &ReferenceKvReuseBackend) -> String {
    let identity = backend.tokenizer().identity();
    format!("{}@{}", identity.name(), identity.version())
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
    fn the_authority_knows_the_backends_base_model_identity() {
        let authority = Authority::new(&[]).unwrap();
        let live = KvDependency::base_model(
            BASE_MODEL_IDENTITY,
            Generation(BASE_MODEL_GENERATION),
            [0; 32],
        );
        assert!(authority.is_live(&live));
        let other = KvDependency::base_model("not-the-model", Generation(1), [0; 32]);
        assert!(!authority.is_live(&other));
    }
}
