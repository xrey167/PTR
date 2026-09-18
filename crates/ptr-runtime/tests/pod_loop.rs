use ptr_config::PtrConfig;
use ptr_model_api::{InferenceBackend, ModelError, ModelEvent, ModelRequest};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{
    CapabilityId, Effect, PodId, Probability, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

struct WantsEcho;
impl InferenceBackend for WantsEcho {
    fn name(&self) -> &'static str {
        "wants-echo"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![
            ModelEvent::PodRequested {
                capability: CapabilityId::from("Echo<Text>"),
                input_type: TypeId::from("Text"),
                payload: b"hello".to_vec(),
            },
            ModelEvent::Finished,
        ])
    }
}

struct Echo {
    manifest: PodManifest,
}
impl DynPod for Echo {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
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

struct Fail;
impl Verifier<TypedPayload> for Fail {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Fail,
            level: VerificationLevel::Deterministic,
            score: Probability::new(0.0).unwrap(),
            findings: vec![],
        }
    }
}

fn registry(effect: Effect) -> PodRegistry {
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Echo {
        manifest: PodManifest {
            id: PodId::from("echo"),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![effect],
            protocol_version: 1,
        },
    }));
    registry
}

#[test]
fn model_to_pod_to_verifier_to_semdb_loop_executes() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let output = runtime
        .run_model_with_pods(
            "r1".into(),
            "echo hello",
            &WantsEcho,
            &registry(Effect::Pure),
            &Pass,
        )
        .unwrap();

    assert_eq!(output.len(), 1);
    assert_eq!(output[0].bytes, b"hello");
    assert_eq!(runtime.revision().0, 2);
    assert!(runtime.events().iter().any(|event| {
        matches!(
            &event.event,
            ptr_events::RuntimeEvent::VerifierResult { passed: true, .. }
        )
    }));
}

#[test]
fn failed_verification_stops_pod_output_promotion() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        runtime.run_model_with_pods(
            "r1".into(),
            "echo hello",
            &WantsEcho,
            &registry(Effect::Pure),
            &Fail,
        ),
        Err(RuntimeError::PodVerificationFailed { pod: "echo".into() })
    );
    assert_eq!(runtime.revision().0, 1);
}

#[test]
fn mutating_pod_cannot_bypass_action_boundary() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        runtime.run_model_with_pods(
            "r1".into(),
            "echo hello",
            &WantsEcho,
            &registry(Effect::Mutation),
            &Pass,
        ),
        Err(RuntimeError::PodEffectRequiresActionBoundary { pod: "echo".into() })
    );
}
