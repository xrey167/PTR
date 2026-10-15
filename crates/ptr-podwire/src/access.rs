//! Who may reach which Pod, and the whole decision that answers one request.
//!
//! Nothing in this module touches a transport, so the protocol's decisions can be
//! driven directly. What the endpoint adds on top is one connection and two
//! checks that only a connection can make: which key is at the other end, and
//! which key this endpoint is.
//!
//! **The project is policy, not payload.** In one process
//! `run_model_with_pods` is handed the project the request belongs to by the host
//! that already knows it. Across a boundary there is no such host, and a requester
//! that named its own project would be choosing which project's Pods it reaches —
//! which would make `PodRegistry`'s project key, in its own words, *advisory*. So
//! the project is looked up from the authenticated peer, in the same table and for
//! the same reason that the execution wire looks up a principal.
use crate::frame::RefusalCode;
use ptr_pods::PodRegistry;
#[cfg(feature = "podwire-backend")]
use ptr_pods::RegisteredPodBinding;
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, NodeId, ProjectId, TypeId};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::{BTreeMap, BTreeSet};

/// What one peer may reach: one project, and an exact set of `(capability,
/// payload type)` pairs inside it.
///
/// Exact, never a prefix or a wildcard, for the reason `ActionScope` gives in the
/// runtime: a scope that matched by prefix would grow every time somebody
/// registered a Pod with a longer name. The pair is the pair `PodRegistry::resolve`
/// keys on, so a scope cannot be expressed that resolution would not honour.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodScope {
    project: ProjectId,
    allow: BTreeSet<(CapabilityId, TypeId)>,
}

impl PodScope {
    /// A scope over one project that allows nothing yet.
    pub fn new(project: ProjectId) -> Self {
        Self {
            project,
            allow: BTreeSet::new(),
        }
    }

    /// Allow one exact capability and payload type.
    pub fn allow(mut self, capability: CapabilityId, input_type: TypeId) -> Self {
        self.allow.insert((capability, input_type));
        self
    }

    /// The project this scope is confined to.
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    /// Whether this exact pair is allowed.
    pub fn covers(&self, capability: &CapabilityId, input_type: &TypeId) -> bool {
        self.allow
            .contains(&(capability.clone(), input_type.clone()))
    }
}

/// Which peers may reach Pods, and what each may reach.
///
/// Installed by the trusted host. A peer with no entry is not a peer with fewer
/// rights; it reaches nothing at all.
#[derive(Default)]
pub struct PodAccessPolicy {
    entries: BTreeMap<NodeId, PodScope>,
}

/// Why a scope could not be admitted into the policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PolicyError {
    /// This peer already has an entry.
    ///
    /// Refused rather than replaced, for the reason `AdmissionPolicy::admit` gives:
    /// a silent replacement is how a later entry widens or narrows an earlier one
    /// without anybody deciding to.
    DuplicatePeer { peer: NodeId },
}

impl std::fmt::Display for PolicyError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicatePeer { .. } => f.write_str("PTR_PODW_DUPLICATE_PEER"),
        }
    }
}

impl std::error::Error for PolicyError {}

impl PodAccessPolicy {
    /// An empty policy: nobody reaches anything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a peer the transport authenticated to one scope.
    pub fn admit(&mut self, peer: NodeId, scope: PodScope) -> Result<(), PolicyError> {
        if self.entries.contains_key(&peer) {
            return Err(PolicyError::DuplicatePeer { peer });
        }
        self.entries.insert(peer, scope);
        Ok(())
    }

    /// Withdraw a peer. It reaches nothing from the next request onward, because
    /// the scope is read from this table at every request rather than remembered.
    pub fn withdraw(&mut self, peer: &NodeId) -> bool {
        self.entries.remove(peer).is_some()
    }

    /// The scope bound to a peer, if any.
    pub fn scope(&self, peer: &NodeId) -> Option<&PodScope> {
        self.entries.get(peer)
    }
}

/// Resolve and run one request, or refuse it.
///
/// The order below is the protocol, and it is an order rather than a set. Read it
/// as the answer to "what can a requester learn by asking":
///
/// 1. **Admitted?** A peer with no entry is refused before the registry is
///    touched, so an unadmitted peer cannot probe for Pods at all.
/// 2. **In scope?** Checked against the peer's own scope, still before the
///    registry is touched. A capability outside the scope is refused with the same
///    code as one nothing serves, so the scope itself is not a directory.
/// 3. **Resolvable in *this peer's* project?** The project comes from step 1. A
///    Pod in another project is simply not found here — not found *differently*,
///    not found at all — which is what makes the cross-project refusal and the
///    does-not-exist refusal the same fact rather than two facts reported alike.
/// 4. **Pure or Read?** An effectful Pod is `ALPN_EXEC`'s business. This wire
///    commits no attempt and can report no uncertainty, so it must not carry
///    something that could half-happen.
/// 5. **The protocol version the requester composed for?** A payload composed for
///    one version of a Pod is not a payload for another.
/// 6. **The Pod runs, and its output is verified** before it leaves the host — the
///    same verification the in-process loop applies, because an answer the host
///    would not vouch for internally is not one to put on a wire.
///
/// Steps 1 and 2 reach no Pod, so a refusal from either costs the host nothing and
/// tells the requester nothing about what exists.
pub fn answer<V>(
    policy: &PodAccessPolicy,
    registry: &PodRegistry,
    verifier: &V,
    peer: &NodeId,
    capability: &CapabilityId,
    expect_protocol: u32,
    payload: TypedPayload,
) -> Result<TypedPayload, RefusalCode>
where
    V: Verifier<TypedPayload> + ?Sized,
{
    // (1) Admitted, or nothing.
    let scope = policy.scope(peer).ok_or(RefusalCode::NotAdmitted)?;

    // (2) In this peer's own scope. Still no registry lookup: an out-of-scope
    // capability must not be distinguishable from one nothing serves.
    if !scope.covers(capability, &payload.type_id) {
        return Err(RefusalCode::Unavailable);
    }

    // (3) Resolvable inside the project the *policy* named.
    let pod = registry
        .resolve(scope.project(), capability, &payload.type_id)
        .ok_or(RefusalCode::Unavailable)?;
    let manifest = pod.manifest();

    // (4) Pure or Read only. The same rule the in-process loop applies, and for a
    // sharper reason here: there is no attempt behind this request to be fenced by.
    if manifest
        .effects
        .iter()
        .any(|effect| !matches!(effect, Effect::Pure | Effect::Read))
    {
        return Err(RefusalCode::RequiresActionBoundary);
    }

    // (5) Composed for this Pod's protocol, not for another's.
    if manifest.protocol_version != expect_protocol {
        return Err(RefusalCode::ProtocolMismatch);
    }

    // (6) Run, then verify. The Pod's own error message stays here.
    let output = pod.invoke(payload).map_err(|_| RefusalCode::PodFailed)?;
    let report = verifier.verify(&output);
    if report.status != VerificationStatus::Pass
        || report.findings.iter().any(|finding| finding.hard)
    {
        return Err(RefusalCode::Unverified);
    }
    Ok(output)
}

/// Resolve and run the exact Pod captured by an admitted generation-bound
/// binding. Unlike [`answer`], this path never performs a fresh capability-only
/// selection after the binding has been accepted: the registry entry must still be
/// the captured one ([`PodRegistry::resolve_bound`]: same manifest, namespace and
/// generation), so a re-registration after the binding was made is not invoked.
#[cfg(feature = "podwire-backend")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn answer_bound<V>(
    policy: &PodAccessPolicy,
    registry: &PodRegistry,
    verifier: &V,
    peer: &NodeId,
    bound: &RegisteredPodBinding,
    capability: &CapabilityId,
    expect_protocol: u32,
    payload: TypedPayload,
) -> Result<TypedPayload, RefusalCode>
where
    V: Verifier<TypedPayload> + ?Sized,
{
    let scope = policy.scope(peer).ok_or(RefusalCode::NotAdmitted)?;
    if scope.project() != bound.project() || !scope.covers(capability, &payload.type_id) {
        return Err(RefusalCode::Unavailable);
    }
    let pod = registry
        .resolve_bound(bound)
        .ok_or(RefusalCode::Unavailable)?;
    let manifest = pod.manifest();
    if manifest
        .effects
        .iter()
        .any(|effect| !matches!(effect, Effect::Pure | Effect::Read))
    {
        return Err(RefusalCode::RequiresActionBoundary);
    }
    if manifest.protocol_version != expect_protocol {
        return Err(RefusalCode::ProtocolMismatch);
    }
    let output = pod.invoke(payload).map_err(|_| RefusalCode::PodFailed)?;
    let report = verifier.verify(&output);
    if report.status != VerificationStatus::Pass
        || report.findings.iter().any(|finding| finding.hard)
    {
        return Err(RefusalCode::Unverified);
    }
    Ok(output)
}

#[cfg(all(test, feature = "podwire-backend"))]
mod bound_tests {
    use super::*;
    use ptr_pods::{DynPod, PodManifest};
    use ptr_types::{Generation, NamespaceId, PodId, Probability, VerificationLevel};
    use ptr_verifier::VerificationReport;
    use std::sync::Arc;

    struct Echo(PodManifest);
    impl DynPod for Echo {
        fn manifest(&self) -> &PodManifest {
            &self.0
        }
        fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
            Ok(input)
        }
    }

    struct Pass;
    impl Verifier<TypedPayload> for Pass {
        fn verify(&self, _: &TypedPayload) -> VerificationReport {
            VerificationReport {
                status: VerificationStatus::Pass,
                level: VerificationLevel::Deterministic,
                score: Probability::new(1.0).unwrap(),
                findings: vec![],
            }
        }
    }

    fn echo() -> Arc<Echo> {
        Arc::new(Echo(PodManifest {
            project: ProjectId::from("p"),
            id: PodId::from("echo"),
            capabilities: vec![CapabilityId::from("Echo")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        }))
    }

    /// The bound path resolves the captured binding: an entry re-registered with the same
    /// manifest under another generation or namespace after the binding was taken is not
    /// invoked.
    #[test]
    fn a_re_registration_after_binding_is_not_invoked() {
        let peer = NodeId("peer".into());
        let mut policy = PodAccessPolicy::new();
        policy
            .admit(
                peer.clone(),
                PodScope::new(ProjectId::from("p"))
                    .allow(CapabilityId::from("Echo"), TypeId::from("Text")),
            )
            .unwrap();
        let payload = || TypedPayload {
            type_id: TypeId::from("Text"),
            bytes: b"hi".to_vec(),
        };
        let mut registry = PodRegistry::default();
        registry.register_at(NamespaceId::from("ns"), Generation(1), echo());
        let bound = registry
            .bind_registered(&ProjectId::from("p"), &PodId::from("echo"))
            .unwrap();
        let ask = |registry: &PodRegistry| {
            answer_bound(
                &policy,
                registry,
                &Pass,
                &peer,
                &bound,
                &CapabilityId::from("Echo"),
                1,
                payload(),
            )
        };
        assert!(ask(&registry).is_ok());
        registry.register_at(NamespaceId::from("ns"), Generation(2), echo());
        assert_eq!(ask(&registry), Err(RefusalCode::Unavailable));
        registry.register_at(NamespaceId::from("other"), Generation(1), echo());
        assert_eq!(ask(&registry), Err(RefusalCode::Unavailable));
        registry.register_at(NamespaceId::from("ns"), Generation(1), echo());
        assert!(ask(&registry).is_ok());
    }
}
