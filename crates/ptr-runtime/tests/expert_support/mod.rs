//! Shared host-side setup for the expert tests: the identity, policy key and principal a grant
//! rests on, and the evidence constants a report must carry.
#![allow(dead_code)]

use ed25519_dalek::SigningKey;
use ptr_runtime::execution::AdmissionPolicy;
use ptr_runtime::expert::{
    ExpertGateRequirement, ExpertGrantBinding, ExpertGrantGeneration, ExpertGrantKind,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{AuthenticationLevel, IdentityContext, NodeId, SessionId, Timestamp};

/// The digest of the base model every test expert is on.
pub const BASE_MODEL_DIGEST: [u8; 32] = [9; 32];
/// The evaluation manifest digest the test hosts expect.
pub const EVAL_MANIFEST: [u8; 32] = [5; 32];
pub const KEY_ID: &str = "policy-key";
/// The transport peer grants are authenticated over.
pub const PEER: &str = "expert-host-peer";
pub const SESSION: &str = "session-1";
/// The host clock reading grants are minted and committed at.
pub const NOW: Timestamp = Timestamp(1_000);
pub const ISSUED_AT: u64 = 100;
pub const EXPIRES_AT: u64 = 10_000;

pub fn identity_for(session: &str, issued_at: u64, expires_at: u64) -> IdentityContext {
    let mut identity = IdentityContext {
        subject: "host".into(),
        issuer: "issuer".into(),
        groups: vec!["operators".into()],
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: SessionId(session.into()),
        issued_at: Timestamp(issued_at),
        expires_at: Timestamp(expires_at),
        identity_digest: [0; 32],
    };
    identity.identity_digest = identity.canonical_digest();
    identity
}

pub fn peer() -> NodeId {
    NodeId::from(PEER)
}

pub fn identity() -> IdentityContext {
    identity_for(SESSION, ISSUED_AT, EXPIRES_AT)
}

/// Install the policy key and admit the identity's principal, as the host does before it mints a
/// grant.
pub fn prime(rt: &mut PtrRuntime) {
    rt.install_policy_key(KEY_ID, SigningKey::from_bytes(&[7; 32]).verifying_key());
    rt.admit_identity_context(&identity(), NOW).unwrap();
    let mut policy = AdmissionPolicy::new();
    policy
        .admit(peer(), "host", std::time::Duration::from_secs(60), Vec::new)
        .unwrap();
    rt.install_admission_policy(policy);
}

/// A binding at the initial grant generation, the one a fresh runtime stands at.
pub fn binding() -> ExpertGrantBinding {
    binding_at(ExpertGrantGeneration::INITIAL)
}

pub fn binding_at(generation: ExpertGrantGeneration) -> ExpertGrantBinding {
    ExpertGrantBinding::new(identity(), peer(), NOW, KEY_ID, generation)
}

/// A binding at the generation `rt` currently stands at for `kind`.
pub fn current_binding(rt: &PtrRuntime, kind: ExpertGrantKind) -> ExpertGrantBinding {
    binding_at(rt.expert_grant_generation(kind))
}

pub fn requirement(gate_epoch: u64, config_digest: [u8; 32]) -> ExpertGateRequirement {
    ExpertGateRequirement {
        gate_epoch,
        config_digest,
        eval_manifest_digest: EVAL_MANIFEST,
    }
}
