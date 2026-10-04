//! A runtime holding the committed facts the bridge tests read, and the A0 model
//! they feed.
//!
//! The world has three slots, all committed: a text goal and a document of
//! evidence, both under `capsule:a` generation 1, which is live; and a withdrawn
//! document under `capsule:b` generation 1, which is revoked unless a test asks
//! otherwise. A test varies one payload, or one revocation, and holds everything
//! else fixed.
#![allow(dead_code)]

use burn::{prelude::*, tensor::Int};
use ptr_a0_bridge::{forward_from_runtime, Lifecycle, RuntimeForward, SlotSpec};
use ptr_burn_a0::{PtrA0, PtrA0Config};
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{PtrRuntime, SemanticChange, SemanticGrant};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{EpistemicState, Generation, Probability, SemanticRole, TypeId, VerificationLevel};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};

pub const D_MODEL: usize = 8;
pub const PROVENANCE_BUCKETS: usize = 4;

/// Admits every change: the fixture's host writes need a grant, and what they
/// write is not under test.
pub struct AcceptAll;

impl<'a> Verifier<SemanticChange<'a>> for AcceptAll {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: Vec::new(),
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for AcceptAll {
    fn name(&self) -> &'static str {
        "test-accept-all"
    }
}

pub fn document(bytes: &[u8]) -> SemanticValue {
    SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from("Document"),
        source: "fixture".to_owned(),
        bytes: bytes.to_vec(),
    })
}

/// The world, with `evidence` as the admitted slot's payload and `withdrawn` as
/// the third slot's, whose target is revoked when `revoke` is set.
pub fn world(evidence: &[u8], withdrawn: &[u8], revoke: bool) -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AcceptAll)
                .allow_host_writes(),
        )
        .unwrap();
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("goal".into(), "ship the gate".into());
    delta.upserts.insert("evidence".into(), document(evidence));
    delta
        .upserts
        .insert("withdrawn".into(), document(withdrawn));
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &"test-operator".into())
        .unwrap();
    for capsule in ["capsule:a", "capsule:b"] {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "p".into(),
                capsule: capsule.into(),
                generation: Generation(1),
            })
            .unwrap();
    }
    if revoke {
        runtime
            .commit(LedgerEvent::Revoked {
                subject: "capsule:b".into(),
                generation: Generation(1),
            })
            .unwrap();
    }
    runtime
}

/// One fixed initialisation; every comparison in a test uses the same weights.
pub fn model(device: &Device) -> PtrA0 {
    device.seed(17);
    config().init(device)
}

pub fn config() -> PtrA0Config {
    PtrA0Config::new(32, D_MODEL)
        .with_provenance_buckets(PROVENANCE_BUCKETS)
        .with_latent_steps(1)
}

pub fn slot<'a>(key: &'a str, target: &'a str, role: SemanticRole) -> SlotSpec<'a> {
    SlotSpec {
        key,
        lifecycle: Lifecycle {
            target,
            generation: Generation(1),
        },
        role,
        epistemic: EpistemicState::Observed,
        provenance_bucket: 1,
        confidence: Probability::new(0.9).unwrap(),
    }
}

/// The world's three slots, in one row.
pub fn row() -> [SlotSpec<'static>; 3] {
    [
        slot("goal", "capsule:a", SemanticRole::Goal),
        slot("evidence", "capsule:a", SemanticRole::Evidence),
        slot("withdrawn", "capsule:b", SemanticRole::Claim),
    ]
}

pub fn forward(runtime: &PtrRuntime, model: &PtrA0, device: &Device) -> RuntimeForward {
    let row = row();
    forward_from_runtime(
        runtime,
        model,
        Tensor::<2, Int>::from_data([[1, 2, 3]], device),
        &[&row],
        device,
    )
    .expect("every slot is committed")
}

/// The router's logits as raw bits, so equality is bitwise and not approximate.
pub fn logit_bits(forward: &RuntimeForward) -> Vec<u32> {
    forward
        .output
        .router_logits
        .clone()
        .into_data()
        .try_into_vec::<f32>()
        .expect("logits are f32")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}
