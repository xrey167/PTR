use ptr_config::PtrConfig;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

struct TwoStep;
impl InferenceBackend for TwoStep {
    fn name(&self) -> &'static str {
        "two-step"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("Echo<Text>"),
            input_type: TypeId::from("Text"),
            payload: b"observation".to_vec(),
        }])
    }
}
impl ResumableInferenceBackend for TwoStep {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![
            ModelEvent::Token(format!(
                "round={} revision={} source={} payload={}",
                request.round,
                request.revision.0,
                request.observation.source,
                String::from_utf8_lossy(&request.observation.payload)
            )),
            ModelEvent::Finished,
        ])
    }
}

struct LoopForever;
impl InferenceBackend for LoopForever {
    fn name(&self) -> &'static str {
        "loop-forever"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(request_event())
    }
}
impl ResumableInferenceBackend for LoopForever {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(request_event())
    }
}

fn request_event() -> Vec<ModelEvent> {
    vec![ModelEvent::PodRequested {
        capability: CapabilityId::from("Echo<Text>"),
        input_type: TypeId::from("Text"),
        payload: b"again".to_vec(),
    }]
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

fn registry() -> PodRegistry {
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(Echo {
        manifest: PodManifest {
            project: ProjectId::from("p"),
            id: PodId::from("echo"),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    }));
    registry
}

#[test]
fn verified_observation_advances_revision_and_resumes_model() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let run = runtime
        .run_resumable_with_pods(
            "r1".into(),
            &ProjectId::from("p"),
            "start",
            &TwoStep,
            &registry(),
            &Pass,
            2,
        )
        .unwrap();

    assert_eq!(runtime.revision().0, 2);
    assert_eq!(run.observations.len(), 1);
    assert!(run.model_events.iter().any(|event| {
        matches!(event, ModelEvent::Token(text) if text.contains("round=1 revision=2"))
    }));
    assert!(runtime.events().iter().any(|event| {
        matches!(
            &event.event,
            ptr_events::RuntimeEvent::RequestFinished(id) if id.to_string() == "r1"
        )
    }));
}

#[test]
fn resume_budget_stops_unbounded_tool_loop() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(
        runtime.run_resumable_with_pods(
            "r1".into(),
            &ProjectId::from("p"),
            "start",
            &LoopForever,
            &registry(),
            &Pass,
            1,
        ),
        Err(RuntimeError::ModelResumeLimit { max_rounds: 1 })
    );
}
