//! The protocol's decisions, driven without a transport.
//!
//! Everything a connection contributes — which key is at the other end, which key
//! this endpoint is — is supplied here directly, so that what remains under test is
//! the decision itself. `tests/wire.rs` then drives the same decisions over a real
//! authenticated connection.
//!
//! The property most of these are about: **a requester learns nothing about a
//! project it is not in.** Not "learns little", not "learns only the name" —
//! nothing, because the refusal it receives is the same one it would receive for a
//! capability nobody anywhere has ever registered.
//!
//! Every scope here is committed to a host runtime's history and read back by the
//! policy, never installed beside it: the policy has no way to admit or withdraw,
//! and the last test reopens a durable runtime to show the decision is history's.
use ptr_config::PtrConfig;
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_podwire::{answer, PodAccessPolicy, PodScope, RefusalCode};
use ptr_protocol::TypedPayload;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, Effect, NodeId, Probability, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const SUMMARIZE: &str = "Summarize<Document>";
const DOCUMENT: &str = "Document";
const SUMMARY: &str = "Summary";

/// A Pod that answers, so a positive result is a Pod having run rather than a
/// refusal that happened not to fire.
struct Summarizer {
    manifest: PodManifest,
}

impl DynPod for Summarizer {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(TypedPayload {
            type_id: TypeId::from(SUMMARY),
            bytes: input.bytes.iter().take(4).copied().collect(),
        })
    }
}

/// A Pod that refuses whatever it is given.
struct Broken {
    manifest: PodManifest,
}

impl DynPod for Broken {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, _: TypedPayload) -> Result<TypedPayload, String> {
        Err("this pod is broken in a way the requester must not be told about".into())
    }
}

fn manifest(project: &str, id: &str, effect: Effect, protocol_version: u32) -> PodManifest {
    PodManifest {
        project: ProjectId::from(project),
        id: ptr_types::PodId::from(id),
        capabilities: vec![CapabilityId::from(SUMMARIZE)],
        accepts: vec![TypeId::from(DOCUMENT)],
        produces: vec![TypeId::from(SUMMARY)],
        effects: vec![effect],
        protocol_version,
    }
}

fn summarizer(project: &str, id: &str) -> Arc<Summarizer> {
    Arc::new(Summarizer {
        manifest: manifest(project, id, Effect::Pure, 1),
    })
}

struct Pass;
impl Verifier<TypedPayload> for Pass {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        report(VerificationStatus::Pass)
    }
}

struct Reject;
impl Verifier<TypedPayload> for Reject {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        report(VerificationStatus::Fail)
    }
}

fn report(status: VerificationStatus) -> VerificationReport {
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).unwrap(),
        findings: vec![],
    }
}

fn peer(name: &str) -> NodeId {
    NodeId::from(name)
}

/// A scope over one project allowing exactly the one pair used throughout.
fn scope(project: &str) -> PodScope {
    PodScope::new(ProjectId::from(project))
        .allow(CapabilityId::from(SUMMARIZE), TypeId::from(DOCUMENT))
}

fn document() -> TypedPayload {
    TypedPayload {
        type_id: TypeId::from(DOCUMENT),
        bytes: b"a document worth summarizing".to_vec(),
    }
}

/// A host runtime whose history admits exactly these scopes, as the host would
/// have committed them: through the journal, never into a table beside it.
fn runtime_with(scopes: Vec<(&str, PodScope)>) -> Arc<Mutex<PtrRuntime>> {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for (who, scope) in scopes {
        runtime.admit_pod_scope(peer(who), &scope).unwrap();
    }
    Arc::new(Mutex::new(runtime))
}

/// The access policy those committed scopes describe.
fn policy_of(scopes: Vec<(&str, PodScope)>) -> PodAccessPolicy {
    PodAccessPolicy::from_runtime(runtime_with(scopes))
}

/// The request every test makes, so that what differs between them is the setup.
fn ask(
    policy: &PodAccessPolicy,
    registry: &PodRegistry,
    who: &str,
) -> Result<TypedPayload, RefusalCode> {
    answer(
        policy,
        registry,
        &Pass,
        &ptr_net::AuthenticatedPeer::assume_authenticated(peer(who)),
        &CapabilityId::from(SUMMARIZE),
        1,
        document(),
    )
}

#[test]
fn an_admitted_peer_reaches_the_pod_in_its_own_project() {
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let policy = policy_of(vec![("k-alpha", scope("alpha"))]);

    let output = ask(&policy, &registry, "k-alpha").expect("an admitted peer reaches its Pod");
    assert_eq!(output.type_id, TypeId::from(SUMMARY));
    assert_eq!(
        output.bytes, b"a do",
        "the Pod ran, rather than something echoing the request back"
    );
}

#[test]
fn a_peer_the_policy_never_bound_reaches_nothing_and_no_pod_is_consulted() {
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let policy = policy_of(Vec::new());

    assert_eq!(
        ask(&policy, &registry, "k-stranger").expect_err("no entry, no access"),
        RefusalCode::NotAdmitted
    );

    // The control: the very same registry and request, with an entry. Without this
    // the refusal above would also be what an empty registry produces.
    let policy = policy_of(vec![("k-stranger", scope("alpha"))]);
    assert!(ask(&policy, &registry, "k-stranger").is_ok());
}

#[test]
fn a_pod_in_another_project_is_refused_in_exactly_the_same_words_as_one_that_does_not_exist() {
    // The oracle this protocol exists to close. A requester admitted to `alpha`
    // asks for a capability only `beta` serves; a second requester, also in
    // `alpha`, asks for a capability nobody anywhere serves. If those two answers
    // differed, the first requester would have learned that `beta` has a
    // summarizer — from outside `beta`, without ever reaching it.
    let mut registry = PodRegistry::default();
    registry.register(summarizer("beta", "summarizer"));
    let policy = policy_of(vec![("k-alpha", scope("alpha"))]);

    let across_the_boundary = ask(&policy, &registry, "k-alpha").expect_err("not in this project");

    let empty = PodRegistry::default();
    let nowhere_at_all = ask(&policy, &empty, "k-alpha").expect_err("nobody serves this");

    assert_eq!(
        across_the_boundary, nowhere_at_all,
        "a Pod in another project must be indistinguishable from one that does not exist"
    );
    assert_eq!(across_the_boundary, RefusalCode::Unavailable);

    // And the control, without which both refusals above would be consistent with a
    // registry nothing can ever resolve in: the same Pod, reached by a peer the
    // policy places in `beta`.
    let policy = policy_of(vec![("k-beta", scope("beta"))]);
    assert!(
        ask(&policy, &registry, "k-beta").is_ok(),
        "the Pod is reachable from inside its own project"
    );
}

#[test]
fn a_capability_outside_a_peer_s_scope_is_refused_in_the_same_words_too() {
    // The third case folded into `Unavailable`. A distinguishable out-of-scope
    // refusal would turn the scope into a directory: a requester could enumerate
    // which capabilities its host serves by watching which refusal came back.
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let policy = policy_of(vec![(
        "k-narrow",
        PodScope::new(ProjectId::from("alpha"))
            .allow(CapabilityId::from("Translate<Text>"), TypeId::from("Text")),
    )]);

    assert_eq!(
        ask(&policy, &registry, "k-narrow").expect_err("outside this peer's scope"),
        RefusalCode::Unavailable
    );
}

#[test]
fn a_scope_matches_the_exact_pair_and_never_half_of_it() {
    // `resolve` keys on both, so a scope that held only the capability would admit
    // a payload type the host never meant to expose — and one that held only the
    // type would admit every capability that accepts it.
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let policy = policy_of(vec![(
        "k-alpha",
        PodScope::new(ProjectId::from("alpha"))
            .allow(CapabilityId::from(SUMMARIZE), TypeId::from("Spreadsheet")),
    )]);

    // Right capability, wrong type: the scope does not cover it.
    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("the pair is exact"),
        RefusalCode::Unavailable
    );

    // Right type, wrong capability.
    let policy = policy_of(vec![(
        "k-alpha",
        PodScope::new(ProjectId::from("alpha")).allow(
            CapabilityId::from("Translate<Document>"),
            TypeId::from(DOCUMENT),
        ),
    )]);
    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("the pair is exact"),
        RefusalCode::Unavailable
    );
}

#[test]
fn an_effectful_pod_is_sent_to_the_other_protocol_rather_than_run_here() {
    // This wire commits no attempt and can report no uncertainty. A Pod whose
    // effect could half-happen must not be reachable through a channel that has no
    // way to say so.
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Summarizer {
        manifest: manifest("alpha", "summarizer", Effect::Mutation, 1),
    }));
    let policy = policy_of(vec![("k-alpha", scope("alpha"))]);

    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("effects belong on ALPN_EXEC"),
        RefusalCode::RequiresActionBoundary
    );

    // The control: the same Pod, declared Read, does run. So the refusal is about
    // the effect and not about this fixture.
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Summarizer {
        manifest: manifest("alpha", "summarizer", Effect::Read, 1),
    }));
    assert!(ask(&policy, &registry, "k-alpha").is_ok());
}

#[test]
fn a_payload_composed_for_another_protocol_version_is_refused_rather_than_handed_over() {
    // `PodManifest::protocol_version` has been declared since the type existed and,
    // until this wire, nothing anywhere read it: in one process the caller and the
    // Pod are built together. Across a boundary they are not.
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Summarizer {
        manifest: manifest("alpha", "summarizer", Effect::Pure, 2),
    }));
    let policy = policy_of(vec![("k-alpha", scope("alpha"))]);

    let refused = answer(
        &policy,
        &registry,
        &Pass,
        &ptr_net::AuthenticatedPeer::assume_authenticated(peer("k-alpha")),
        &CapabilityId::from(SUMMARIZE),
        1,
        document(),
    )
    .expect_err("composed for version 1, served by version 2");
    assert_eq!(refused, RefusalCode::ProtocolMismatch);

    // The control: the same everything, composed for the version the Pod declares.
    assert!(answer(
        &policy,
        &registry,
        &Pass,
        &ptr_net::AuthenticatedPeer::assume_authenticated(peer("k-alpha")),
        &CapabilityId::from(SUMMARIZE),
        2,
        document(),
    )
    .is_ok());
}

#[test]
fn a_pod_that_fails_and_an_answer_the_host_will_not_vouch_for_are_different_refusals() {
    // Different because they are actionable in opposite directions: one says the
    // input was not usable, the other says the host would not stand behind the
    // output. Collapsing them would leave a requester unable to act on either.
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Broken {
        manifest: manifest("alpha", "broken", Effect::Pure, 1),
    }));
    let policy = policy_of(vec![("k-alpha", scope("alpha"))]);

    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("the Pod itself failed"),
        RefusalCode::PodFailed
    );

    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let refused = answer(
        &policy,
        &registry,
        &Reject,
        &ptr_net::AuthenticatedPeer::assume_authenticated(peer("k-alpha")),
        &CapabilityId::from(SUMMARIZE),
        1,
        document(),
    )
    .expect_err("the host will not vouch for this output");
    assert_eq!(refused, RefusalCode::Unverified);

    // The control for the second: the same Pod and the same request, verified by a
    // verifier that passes. Without it, `Unverified` would also be what a Pod that
    // produced nothing looks like.
    assert!(ask(&policy, &registry, "k-alpha").is_ok());
}

#[test]
fn withdrawing_a_peer_takes_effect_on_its_very_next_request() {
    // The scope is read from committed history at every request rather than
    // remembered, so there is no cached authority to outlive a withdrawal — and the
    // withdrawal is a commit on the runtime, not a call on the policy.
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));
    let runtime = runtime_with(vec![("k-alpha", scope("alpha"))]);
    let policy = PodAccessPolicy::from_runtime(Arc::clone(&runtime));
    assert!(ask(&policy, &registry, "k-alpha").is_ok());

    runtime
        .lock()
        .unwrap()
        .withdraw_pod_scope(&peer("k-alpha"))
        .unwrap();
    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("withdrawn"),
        RefusalCode::NotAdmitted
    );
    assert!(
        matches!(
            runtime.lock().unwrap().withdraw_pod_scope(&peer("k-alpha")),
            Err(RuntimeError::InvalidPodScope { .. })
        ),
        "withdrawing twice is refused: there was nothing to withdraw"
    );

    // The control: admitting it again through the same runtime is seen on the
    // next request by the same policy, with nothing installed on the policy.
    runtime
        .lock()
        .unwrap()
        .admit_pod_scope(peer("k-alpha"), &scope("alpha"))
        .unwrap();
    assert!(ask(&policy, &registry, "k-alpha").is_ok());
}

#[test]
fn a_second_entry_for_one_peer_is_refused_rather_than_replacing_the_first() {
    // A silent replacement is how a later entry widens or narrows an earlier one
    // without anybody deciding to — the rule execution membership already applies,
    // and the same rule here because it is the runtime's one rule, live and on
    // replay, for every Pod-scope record.
    let mut registry = PodRegistry::default();
    registry.register(summarizer("beta", "summarizer"));
    let runtime = runtime_with(vec![("k-alpha", scope("alpha"))]);
    let policy = PodAccessPolicy::from_runtime(Arc::clone(&runtime));

    assert_eq!(
        runtime
            .lock()
            .unwrap()
            .admit_pod_scope(peer("k-alpha"), &scope("beta"))
            .expect_err("one peer, one scope"),
        RuntimeError::InvalidPodScope {
            peer: peer("k-alpha"),
            reason: "the peer already has a Pod scope",
        }
    );

    // And the first entry is the one still in force: the refused second entry did
    // not quietly move this peer into `beta`.
    assert_eq!(
        ask(&policy, &registry, "k-alpha").expect_err("still confined to alpha"),
        RefusalCode::Unavailable
    );
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-podwire-access-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn log(&self) -> PathBuf {
        self.0.join("journal.log")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The reason Pod access is journaled at all. Before, a host that withdrew a
/// peer's scope and restarted would answer that peer again with whatever table it
/// installed next. Now the access decision after a reopen is the one history
/// describes: the withdrawn peer is not admitted, the peer that was not withdrawn
/// still reaches its Pod (the control), and the cross-project refusal is still
/// the one a Pod that does not exist gets.
#[test]
fn the_access_decision_after_a_reopen_is_the_one_history_describes() {
    let temp = Temp::new();
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
        runtime
            .admit_pod_scope(peer("k-alpha"), &scope("alpha"))
            .unwrap();
        runtime
            .admit_pod_scope(peer("k-gone"), &scope("alpha"))
            .unwrap();
        runtime.withdraw_pod_scope(&peer("k-gone")).unwrap();
    }

    let reopened = PtrRuntime::open_durable(PtrConfig::default(), temp.log()).unwrap();
    let policy = PodAccessPolicy::from_runtime(Arc::new(Mutex::new(reopened)));
    let mut registry = PodRegistry::default();
    registry.register(summarizer("alpha", "summarizer"));

    assert_eq!(
        ask(&policy, &registry, "k-gone").expect_err("withdrawn before the restart"),
        RefusalCode::NotAdmitted
    );
    assert!(
        ask(&policy, &registry, "k-alpha").is_ok(),
        "the scope that was not withdrawn survives the restart and answers"
    );

    let mut beta_only = PodRegistry::default();
    beta_only.register(summarizer("beta", "summarizer"));
    assert_eq!(
        ask(&policy, &beta_only, "k-alpha").expect_err("another project"),
        ask(&policy, &PodRegistry::default(), "k-alpha").expect_err("nothing at all"),
        "after a reopen, another project's Pod is still indistinguishable from none"
    );
}
