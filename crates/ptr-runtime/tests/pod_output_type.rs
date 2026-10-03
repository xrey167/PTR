mod common;

use ptr_config::PtrConfig;
use ptr_model_api::{InferenceBackend, ModelError, ModelEvent, ModelRequest};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

struct Backend;
impl InferenceBackend for Backend {
    fn name(&self) -> &'static str {
        "type-mismatch"
    }
    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("answer"),
            input_type: TypeId::from("text"),
            payload: b"hello".to_vec(),
        }])
    }
}

struct WrongType {
    manifest: PodManifest,
}
impl DynPod for WrongType {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }
    fn invoke(&self, _: TypedPayload) -> Result<TypedPayload, String> {
        Ok(TypedPayload {
            type_id: TypeId::from("wrong"),
            bytes: b"bad".to_vec(),
        })
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

#[test]
fn runtime_rejects_output_type_outside_manifest() {
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(WrongType {
        manifest: PodManifest {
            project: ProjectId::from("p"),
            id: PodId::from("wrong"),
            capabilities: vec![CapabilityId::from("answer")],
            accepts: vec![TypeId::from("text")],
            produces: vec![TypeId::from("answer")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    }));
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        runtime.run_model_with_pods(
            "r1".into(),
            &ProjectId::from("p"),
            "x",
            &Backend,
            &registry,
            &Pass
        ),
        Err(RuntimeError::PodOutputTypeMismatch {
            pod: "wrong".into(),
            expected: "answer".into(),
            actual: "wrong".into(),
        })
    );
}
