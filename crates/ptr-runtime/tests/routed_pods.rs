//! Pod selection through the trusted `route_checked` boundary.

use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    CheckedRouter, NonceSequence, PodRouter, RouteDistribution, RouteError, RouteRequest, Router,
    SamplingKey,
};
use ptr_runtime::{PodSampler, PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

struct Ask(&'static [u8]);
impl InferenceBackend for Ask {
    fn name(&self) -> &'static str {
        "ask"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("Echo<Text>"),
            input_type: TypeId::from("Text"),
            payload: self.0.to_vec(),
        }])
    }
}
impl ResumableInferenceBackend for Ask {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

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

fn registry(project: &str, ids: &[&str]) -> PodRegistry {
    let mut registry = PodRegistry::default();
    for id in ids {
        registry.register(Arc::new(Echo(PodManifest {
            project: ProjectId::from(project),
            id: PodId::from(*id),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        })));
    }
    registry
}

fn sampler(key: u8, instance: u8) -> CheckedRouter<PodRouter> {
    CheckedRouter::new(
        PodRouter,
        SamplingKey::from_bytes([key; 32]),
        NonceSequence::starting_at([instance; 8], 0),
    )
}

fn run(
    runtime: &mut PtrRuntime,
    id: &str,
    project: &str,
    payload: &'static [u8],
    pods: &PodRegistry,
    router: &dyn PodSampler,
) -> Result<(), RuntimeError> {
    runtime
        .run_resumable_with_pods_using_router(
            id.into(),
            &ProjectId::from(project),
            "go",
            &Ask(payload),
            pods,
            router,
            &Pass,
            2,
        )
        .map(|_| ())
}

#[derive(Debug, PartialEq)]
struct Routed {
    pod: String,
    propensity: f64,
    nonce: [u8; 16],
    draw: u64,
    key_id: [u8; 8],
}

fn routed(runtime: &PtrRuntime) -> Vec<Routed> {
    runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::PodRouted {
                pod,
                propensity,
                nonce,
                draw,
                key_id,
            } => Some(Routed {
                pod: pod.clone(),
                propensity: *propensity,
                nonce: *nonce,
                draw: *draw,
                key_id: *key_id,
            }),
            _ => None,
        })
        .collect()
}

#[test]
fn point_mass_policy_picks_what_select_picks_with_propensity_one() {
    let pods = registry("p", &["alpha", "zeta", "mid"]);
    let selected = PodRouter
        .select(
            &pods,
            &ProjectId::from("p"),
            &CapabilityId::from("Echo<Text>"),
            &TypeId::from("Text"),
        )
        .unwrap()
        .pod_id;
    let router = sampler(7, 1);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    run(&mut runtime, "r1", "p", b"x", &pods, &router).unwrap();

    let events = routed(&runtime);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].pod, selected.to_string());
    assert_eq!(events[0].propensity, 1.0);
    assert_eq!(events[0].key_id, router.key_id());
    assert_eq!(events[0].key_id, SamplingKey::from_bytes([7; 32]).key_id());

    // PodRouted precedes PodInvoked for the same Pod.
    let order: Vec<_> = runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::PodRouted { pod, .. } => Some(("routed", pod.clone())),
            RuntimeEvent::PodInvoked(pod) => Some(("invoked", pod.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        order,
        [
            ("routed", selected.to_string()),
            ("invoked", selected.to_string())
        ]
    );
}

#[test]
fn nonce_increments_across_requests() {
    let pods = registry("p", &["echo"]);
    let router = sampler(7, 3);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    run(&mut runtime, "r1", "p", b"x", &pods, &router).unwrap();
    run(&mut runtime, "r2", "p", b"x", &pods, &router).unwrap();

    let events = routed(&runtime);
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].nonce[..8], [3; 8]);
    assert_eq!(events[0].nonce[8..], 0u64.to_be_bytes());
    assert_eq!(events[1].nonce[8..], 1u64.to_be_bytes());
    assert_ne!(events[0].draw, events[1].draw);
}

#[test]
fn same_key_nonce_sequence_and_request_replay_the_same_draw() {
    let pods = registry("p", &["echo"]);
    let replayed = |key: u8, payload: &'static [u8]| {
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        run(&mut runtime, "r1", "p", payload, &pods, &sampler(key, 3)).unwrap();
        routed(&runtime).remove(0)
    };
    let first = replayed(7, b"x");
    assert_eq!(first, replayed(7, b"x"));
    assert_ne!(first.draw, replayed(8, b"x").draw);
    assert_ne!(first.key_id, replayed(8, b"x").key_id);
    // The Pod request payload is part of the request digest.
    assert_ne!(first.draw, replayed(7, b"y").draw);
}

struct Names(&'static str);
impl Router for Names {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::certain(PodId::from(self.0)))
    }
}

#[test]
fn failures_never_name_a_pod() {
    // Another project's pod is unavailable in the same words as no pod at all.
    let other = registry("other", &["secret-pod"]);
    let router = sampler(7, 1);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let across = run(&mut runtime, "r1", "p", b"x", &other, &router);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let none = run(
        &mut runtime,
        "r1",
        "p",
        b"x",
        &PodRegistry::default(),
        &router,
    );
    assert_eq!(
        across,
        Err(RuntimeError::PodUnavailable {
            capability: "Echo<Text>".into(),
            input_type: "Text".into(),
        })
    );
    assert_eq!(across, none);
    assert!(routed(&runtime).is_empty());

    // A policy that names a pod outside the project's candidates fails closed, without the name.
    let policy = CheckedRouter::new(
        Names("secret-pod"),
        SamplingKey::from_bytes([7; 32]),
        NonceSequence::starting_at([1; 8], 0),
    );
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let failed = run(
        &mut runtime,
        "r1",
        "p",
        b"x",
        &registry("p", &["echo"]),
        &policy,
    )
    .unwrap_err();
    assert!(matches!(failed, RuntimeError::RouteFailed { .. }));
    assert!(!format!("{failed:?}").contains("secret-pod"));
    assert!(routed(&runtime).is_empty());
    assert!(!runtime
        .events()
        .iter()
        .any(|envelope| matches!(envelope.event, RuntimeEvent::PodInvoked(_))));
}
