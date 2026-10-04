//! A trainer that binds its checkpoint as it saves it.
//!
//! Commits a few semantic values and a lifecycle generation to a runtime, trains A0
//! for a handful of steps on payloads read from that runtime through the bridge,
//! and saves the result with `save_bound_checkpoint`: checkpoint, sealed binding and
//! retained anchor, written together or not at all. There is no separate sealing
//! step to forget.
//!
//! It writes `model.ckpt`, `model.sealed` and `model.anchor.toml` into the directory
//! it is given. The committed fixture the training pipeline's `require_seal` test
//! reads is produced by it, from the repository root:
//!
//! ```text
//! cargo +1.99.0 run --manifest-path model/burn-a0/Cargo.toml -p ptr-a0-bridge \
//!     --example bound_checkpoint --locked -- training/tests/fixtures/bound-a0
//! ```
//!
//! The training is a mechanics demonstration on two synthetic rows. It is not
//! evidence about what A0 learns, and nothing here claims it is.
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Int,
};
use ptr_a0_bridge::{
    forward_from_runtime, save_bound_checkpoint, CheckpointPaths, Lifecycle, ReadSet, SlotSpec,
};
use ptr_burn_a0::PtrA0Config;
use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{PtrRuntime, SemanticChange, SemanticGrant};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{
    EpistemicState, Generation, Probability, ProvenanceRef, SemanticRole, TypeId, VerificationLevel,
};
use ptr_verifier::{NamedVerifier, VerificationReport, VerificationStatus, Verifier};
use std::path::PathBuf;

/// The example's host writes need a grant; what they write is the example's own.
struct AcceptAll;

impl<'a> Verifier<SemanticChange<'a>> for AcceptAll {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).expect("one is a probability"),
            findings: Vec::new(),
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for AcceptAll {
    fn name(&self) -> &'static str {
        "example-accept-all"
    }
}

fn runtime() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("a default runtime");
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AcceptAll)
                .allow_host_writes(),
        )
        .expect("the first grant installs");
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("goal".into(), "ship the gate".into());
    delta.upserts.insert(
        "evidence".into(),
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("Document"),
            source: "example".to_owned(),
            bytes: b"the gate's tests pass".to_vec(),
        }),
    );
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &"example-operator".into())
        .expect("the values commit");
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "example".into(),
            capsule: "capsule:a".into(),
            generation: Generation(1),
        })
        .expect("the generation commits");
    runtime
}

fn slot(key: &'static str, role: SemanticRole) -> SlotSpec<'static> {
    SlotSpec {
        key,
        lifecycle: Lifecycle {
            target: "capsule:a",
            generation: Generation(1),
        },
        role,
        epistemic: EpistemicState::Observed,
        provenance_bucket: 0,
        confidence: Probability::new(1.0).expect("one is a probability"),
    }
}

fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: bound_checkpoint <output-directory>"),
    );
    std::fs::create_dir_all(&out).expect("the output directory is creatable");

    let runtime = runtime();
    let device = Device::flex().autodiff();
    device.seed(5);
    let mut model = PtrA0Config::new(8, 4)
        .with_provenance_buckets(4)
        .init(&device);
    let mut optimizer = AdamConfig::new().init();

    let first = [
        slot("goal", SemanticRole::Goal),
        slot("evidence", SemanticRole::Evidence),
    ];
    let second = [
        slot("evidence", SemanticRole::Evidence),
        slot("goal", SemanticRole::Goal),
    ];
    let rows: [&[SlotSpec<'_>]; 2] = [&first, &second];

    // Every pass reads the same runtime position, so their read sets merge into the
    // one the binding names.
    let mut read = ReadSet::at(&runtime).expect("the runtime is not fenced");
    for _ in 0..8 {
        let pass = forward_from_runtime(
            &runtime,
            &model,
            Tensor::<2, Int>::from_data([[1, 2], [2, 1]], &device),
            &rows,
            &device,
        )
        .expect("every slot is committed");
        read.merge(&pass.read).expect("one position");
        let loss = CrossEntropyLossConfig::new().init(&device).forward(
            pass.output.router_logits,
            Tensor::<1, Int>::from_data([0, 1], &device),
        );
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.02, model, gradients);
    }

    let (checkpoint, sealed, anchor) = (
        out.join("model.ckpt"),
        out.join("model.sealed"),
        out.join("model.anchor.toml"),
    );
    let retained = save_bound_checkpoint(
        &runtime,
        &model,
        &read,
        &[ProvenanceRef {
            source: "example:bound_checkpoint".into(),
            note: Some("sealed as it was saved".to_owned()),
        }],
        &CheckpointPaths {
            checkpoint: &checkpoint,
            sealed: &sealed,
            anchor: &anchor,
        },
    )
    .unwrap_or_else(|error| panic!("the save is refused: {error} ({error:?})"));
    println!(
        "{} sealed at journal index {}",
        sealed.display(),
        retained.journal.index.0
    );
}
