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
//!
//! **The table is history.** Which scope a peer holds is committed to the host
//! runtime's ledger and read from it at every request ([`PodAccessPolicy`]), so
//! Pod access and execution membership share one durability story.
use crate::frame::RefusalCode;
use ptr_net::AuthenticatedPeer;
#[cfg(feature = "podwire-backend")]
use ptr_pods::PodManifest;
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
pub use ptr_runtime::PodScope;
use ptr_runtime::PtrRuntime;
#[cfg(feature = "podwire-backend")]
use ptr_types::PodId;
use ptr_types::{CapabilityId, Effect, NodeId};
use ptr_verifier::{VerificationStatus, Verifier};
use std::sync::{Arc, Mutex};

/// Which peers may reach Pods, and what each may reach: read from the host
/// runtime's committed history, never installed beside it.
///
/// **One durability story.** Execution membership is journaled (D5 in issue
/// #23): a peer withdrawn before a restart is still withdrawn after it. Pod
/// access used to be a table the host installed in memory, so the same
/// withdrawal of Pod access was forgotten by every restart, and the two
/// policies answered "is this peer still admitted?" differently. Now a scope is
/// committed by `PtrRuntime::admit_pod_scope` and withdrawn by
/// `PtrRuntime::withdraw_pod_scope`, both replayed on every open and carried by
/// compaction, and this type is only the read side: it holds the runtime the
/// host process shares and asks it, at every request, which scope its history
/// binds the peer to.
///
/// **Read, never write.** The runtime handle is private to this module and the
/// only thing done with it is `PtrRuntime::pod_scope`. `PodHost` holds this
/// type and no runtime of its own, so a Pod invoked over the wire still has no
/// path to the host's history: the module that could reach the handle is these
/// lines, and they read.
///
/// A peer with no committed scope is not a peer with fewer rights; it reaches
/// nothing at all.
#[derive(Clone)]
pub struct PodAccessPolicy {
    runtime: Arc<Mutex<PtrRuntime>>,
}

impl PodAccessPolicy {
    /// The access policy the runtime's committed Pod scopes describe.
    ///
    /// Shared rather than owned, for the reason `ExecutionHost::bind_shared`
    /// gives: one host process has one runtime over one ledger, and the scope a
    /// host admits through it is the scope this policy reads on the very next
    /// request.
    pub fn from_runtime(runtime: Arc<Mutex<PtrRuntime>>) -> Self {
        Self { runtime }
    }

    /// The scope committed history binds this peer to now, if any.
    ///
    /// Copied out so the lock is released before any Pod runs: a slow Pod must
    /// not hold the runtime the execution wire and the HTTP API also serve. A
    /// runtime whose lock was poisoned by a panic mid-commit reaches nothing
    /// rather than being read in whatever state the panic left it.
    pub fn scope(&self, peer: &NodeId) -> Option<PodScope> {
        self.runtime.lock().ok()?.pod_scope(peer).cloned()
    }
}

/// Resolve and run one request, or refuse it.
///
/// The order below is the protocol, and it is an order rather than a set. Read it
/// as the answer to "what can a requester learn by asking":
///
/// 1. **Admitted?** A peer with no committed scope is refused before the
///    registry is touched, so an unadmitted peer cannot probe for Pods at all.
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
    peer: &AuthenticatedPeer,
    capability: &CapabilityId,
    expect_protocol: u32,
    payload: TypedPayload,
) -> Result<TypedPayload, RefusalCode>
where
    V: Verifier<TypedPayload> + ?Sized,
{
    // (1) Admitted, or nothing.
    let scope = policy.scope(peer.id()).ok_or(RefusalCode::NotAdmitted)?;

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
/// selection after the binding has been accepted.
#[cfg(feature = "podwire-backend")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn answer_bound<V>(
    policy: &PodAccessPolicy,
    registry: &PodRegistry,
    verifier: &V,
    peer: &AuthenticatedPeer,
    pod_id: &PodId,
    expected_manifest: &PodManifest,
    capability: &CapabilityId,
    expect_protocol: u32,
    payload: TypedPayload,
) -> Result<TypedPayload, RefusalCode>
where
    V: Verifier<TypedPayload> + ?Sized,
{
    let scope = policy.scope(peer.id()).ok_or(RefusalCode::NotAdmitted)?;
    if scope.project() != &expected_manifest.project || !scope.covers(capability, &payload.type_id)
    {
        return Err(RefusalCode::Unavailable);
    }
    let pod = registry
        .get(&expected_manifest.project, pod_id)
        .filter(|pod| pod.manifest() == expected_manifest)
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
