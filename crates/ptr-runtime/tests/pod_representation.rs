//! Learned pod representations are neural state: bound to their pod's exact generation, denied
//! per pod when that generation is revoked or superseded, and routed through the trusted boundary.
use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    CheckedRouter, EncoderRef, GateReport, LearnedPodRouter, LearnedRouterConfig, NonceSequence,
    PodRepresentation, PodRepresentationParts, PolicyFailure, RepresentationStore, RouteError,
    SamplingKey, ScoreWeights,
};
use ptr_runtime::neural::{Denial, NeuralError, NeuralState, NeuralStateCache};
use ptr_runtime::{
    pod_lifecycle_target, BindPodRepresentation, EmbeddingSampler, EmbeddingSamplerError,
    PodRepresentationError, PtrRuntime, QueryEmbedder, RepresentationDenial, RepresentationRefusal,
    RouteQuery, RuntimeError,
};
use ptr_types::{
    CapabilityId, CapsuleId, ClusterKey, CodebookVersion, Effect, Generation, NamespaceId,
    PodAddress, PodId, PodIdentity, PodRevisionAddress, Probability, ProjectId, TypeId,
    VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::Arc;

fn encoder() -> EncoderRef {
    EncoderRef {
        id: "test-encoder".into(),
        revision: "r1".into(),
    }
}

fn address(pod: &str, generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("demo"),
            NamespaceId::from("ns"),
            PodId::from(pod),
        )
        .unwrap(),
        semantic_revision: [1; 32],
        generation: Generation(generation),
    }
}

fn representation(pod: &str, z: Vec<f32>, generation: u64) -> PodRepresentation {
    PodRepresentation::new(PodRepresentationParts {
        project: ProjectId::from("demo"),
        pod: PodId::from(pod),
        identity: PodIdentity::from(format!("pod:{pod}").as_str()),
        cluster_key: ClusterKey([3; 32]),
        encoder: encoder(),
        generation: Generation(generation),
        z,
        bias: 10.0,
        weights: ScoreWeights::default(),
        gate: GateReport {
            alias_threshold_permille: 800,
            alias_min_permille: 950,
            negative_threshold_permille: 900,
            negative_reject_permille: 980,
            alias_count: 3,
            negative_count: 12,
        },
        activated: true,
    })
    .unwrap()
}

/// A runtime that knows the lifecycle targets of pods `alpha` and `beta` at generation 1.
fn runtime() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for pod in ["alpha", "beta"] {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "demo".into(),
                capsule: CapsuleId::from(pod_lifecycle_target(&address(pod, 1)).as_str()),
                generation: Generation(1),
            })
            .unwrap();
    }
    runtime
}

fn bind(runtime: &PtrRuntime, representation: &PodRepresentation) -> NeuralState {
    runtime
        .bind_pod_representation(BindPodRepresentation {
            representation,
            pod: &address(
                representation.pod().0.as_str(),
                representation.generation().0,
            ),
            codebook: CodebookVersion::V1,
        })
        .unwrap()
}

fn cache(runtime: &PtrRuntime) -> NeuralStateCache {
    let mut cache = NeuralStateCache::default();
    cache.insert(
        "alpha",
        bind(runtime, &representation("alpha", vec![1.0, 0.0, 0.0], 1)),
    );
    cache.insert(
        "beta",
        bind(runtime, &representation("beta", vec![0.0, 1.0, 0.0], 1)),
    );
    cache
}

#[test]
fn a_bound_representation_is_admitted_and_comes_back_equal() {
    let runtime = runtime();
    let original = representation("alpha", vec![1.0, 0.0, 0.0], 1);
    let state = bind(&runtime, &original);
    assert_eq!(runtime.admit_pod_representation(&state), Ok(original));
}

#[test]
fn a_representation_for_another_pod_or_generation_does_not_bind() {
    let runtime = runtime();
    let alpha = representation("alpha", vec![1.0, 0.0, 0.0], 1);
    let wrong_pod = runtime.bind_pod_representation(BindPodRepresentation {
        representation: &alpha,
        pod: &address("beta", 1),
        codebook: CodebookVersion::V1,
    });
    assert_eq!(
        wrong_pod.unwrap_err(),
        RuntimeError::PodRepresentation(PodRepresentationError::PodMismatch {
            representation: "alpha".into(),
            address: "beta".into(),
        })
    );
    let wrong_generation = runtime.bind_pod_representation(BindPodRepresentation {
        representation: &alpha,
        pod: &address("alpha", 2),
        codebook: CodebookVersion::V1,
    });
    assert_eq!(
        wrong_generation.unwrap_err(),
        RuntimeError::PodRepresentation(PodRepresentationError::GenerationMismatch {
            representation: 1,
            address: 2,
        })
    );
}

#[test]
fn a_pod_without_a_committed_lifecycle_target_cannot_be_bound() {
    let runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let alpha = representation("alpha", vec![1.0, 0.0, 0.0], 1);
    let error = runtime
        .bind_pod_representation(BindPodRepresentation {
            representation: &alpha,
            pod: &address("alpha", 1),
            codebook: CodebookVersion::V1,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::Neural(NeuralError::UndeclarableTarget { .. })
    ));
}

#[test]
fn a_representation_for_a_generation_that_is_not_live_does_not_bind() {
    let mut runtime = runtime();
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(pod_lifecycle_target(&address("alpha", 1)).as_str()),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    let stale = representation("alpha", vec![1.0, 0.0, 0.0], 1);
    let error = runtime
        .bind_pod_representation(BindPodRepresentation {
            representation: &stale,
            pod: &address("alpha", 1),
            codebook: CodebookVersion::V1,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::Neural(NeuralError::Unbindable(Denial::StaleGeneration { .. }))
    ));
}

#[test]
fn revoking_one_pod_generation_denies_exactly_that_representation() {
    let mut runtime = runtime();
    let cache = cache(&runtime);
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&address("alpha", 1)),
            generation: Generation(1),
        })
        .unwrap();

    let admitted = runtime.admitted_representations(encoder(), &cache);
    assert_eq!(admitted.store.len(), 1);
    assert!(admitted
        .store
        .get(&ProjectId::from("demo"), &PodId::from("beta"))
        .is_some());
    assert!(admitted
        .store
        .get(&ProjectId::from("demo"), &PodId::from("alpha"))
        .is_none());
    assert_eq!(admitted.refused.len(), 1);
    let (key, refusal) = &admitted.refused[0];
    assert_eq!(key, "alpha");
    assert!(matches!(
        refusal,
        RepresentationRefusal::Admission(RepresentationDenial::State(
            Denial::RevokedGeneration { .. }
        ))
    ));
}

#[test]
fn a_new_pod_generation_denies_the_old_representation_until_it_is_retrained() {
    let mut runtime = runtime();
    let cache = cache(&runtime);
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(pod_lifecycle_target(&address("beta", 1)).as_str()),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    let admitted = runtime.admitted_representations(encoder(), &cache);
    assert!(admitted
        .store
        .get(&ProjectId::from("demo"), &PodId::from("alpha"))
        .is_some());
    assert!(admitted
        .store
        .get(&ProjectId::from("demo"), &PodId::from("beta"))
        .is_none());
    assert!(matches!(
        &admitted.refused[0].1,
        RepresentationRefusal::Admission(RepresentationDenial::State(
            Denial::StaleGeneration { .. }
        ))
    ));

    // A representation trained for generation 2 binds and is admitted.
    let retrained = representation("beta", vec![0.0, 1.0, 0.0], 2);
    let state = runtime
        .bind_pod_representation(BindPodRepresentation {
            representation: &retrained,
            pod: &address("beta", 2),
            codebook: CodebookVersion::V1,
        })
        .unwrap();
    assert_eq!(runtime.admit_pod_representation(&state), Ok(retrained));
}

#[test]
fn a_payload_swapped_under_a_valid_binding_is_refused() {
    let runtime = runtime();
    let alpha = bind(&runtime, &representation("alpha", vec![1.0, 0.0, 0.0], 1));
    let beta_payload = representation("beta", vec![0.0, 1.0, 0.0], 1).to_payload();
    let swapped = NeuralState::new(alpha.binding().clone(), beta_payload);
    assert_eq!(
        runtime.admit_pod_representation(&swapped),
        Err(RepresentationDenial::BindingMismatch)
    );
    let garbage = NeuralState::new(alpha.binding().clone(), vec![1, 2, 3]);
    assert!(matches!(
        runtime.admit_pod_representation(&garbage),
        Err(RepresentationDenial::Payload(_))
    ));
}

#[test]
fn an_admitted_representation_that_is_not_activated_stays_out_of_the_router_store() {
    let runtime = runtime();
    let inactive = PodRepresentation::new(PodRepresentationParts {
        gate: GateReport {
            alias_min_permille: 100,
            ..representation("alpha", vec![1.0, 0.0, 0.0], 1).gate()
        },
        activated: false,
        ..parts_of(&representation("alpha", vec![1.0, 0.0, 0.0], 1))
    })
    .unwrap();
    let mut cache = NeuralStateCache::default();
    cache.insert("alpha", bind(&runtime, &inactive));
    let admitted = runtime.admitted_representations(encoder(), &cache);
    assert!(admitted.store.is_empty());
    assert!(matches!(
        &admitted.refused[0].1,
        RepresentationRefusal::Store(RouteError::InactiveRepresentation { .. })
    ));
}

fn parts_of(representation: &PodRepresentation) -> PodRepresentationParts {
    PodRepresentationParts {
        project: representation.project().clone(),
        pod: representation.pod().clone(),
        identity: representation.identity().clone(),
        cluster_key: representation.cluster_key(),
        encoder: representation.encoder().clone(),
        generation: representation.generation(),
        z: representation.z().to_vec(),
        bias: representation.bias(),
        weights: representation.weights(),
        gate: representation.gate(),
        activated: representation.activated(),
    }
}

// --- Routing through the runtime -------------------------------------------------------------

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

fn pods() -> PodRegistry {
    let mut registry = PodRegistry::default();
    for id in ["alpha", "beta"] {
        registry.register(Arc::new(Echo(PodManifest {
            project: ProjectId::from("demo"),
            id: PodId::from(id),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        })));
    }
    registry
}

/// A stand-in encoder: the request payload names its own axis.
struct AxisEmbedder;
impl QueryEmbedder for AxisEmbedder {
    fn encoder(&self) -> &EncoderRef {
        static ENCODER: std::sync::OnceLock<EncoderRef> = std::sync::OnceLock::new();
        ENCODER.get_or_init(encoder)
    }

    fn embed(&self, query: &RouteQuery<'_>) -> Result<Vec<f32>, RouteError> {
        match query.payload {
            b"alpha question" => Ok(vec![1.0, 0.0, 0.0]),
            b"beta question" => Ok(vec![0.0, 1.0, 0.0]),
            b"unrelated question" => Ok(vec![0.0, 0.0, 1.0]),
            b"wrong size" => Ok(vec![1.0, 0.0]),
            _ => Err(RouteError::policy(
                PolicyFailure::Unavailable,
                "encoder offline",
            )),
        }
    }
}

fn sampler(store: RepresentationStore) -> EmbeddingSampler<LearnedPodRouter> {
    EmbeddingSampler::new(
        CheckedRouter::new(
            LearnedPodRouter::new(store, LearnedRouterConfig::default()).unwrap(),
            SamplingKey::from_bytes([9; 32]),
            NonceSequence::starting_at([4; 8], 0),
        ),
        Arc::new(AxisEmbedder),
    )
    .unwrap()
}

fn run(
    runtime: &mut PtrRuntime,
    payload: &'static [u8],
    sampler: &EmbeddingSampler<LearnedPodRouter>,
) -> Result<(), RuntimeError> {
    runtime
        .run_resumable_with_pods_using_router(
            "request-1".into(),
            &ProjectId::from("demo"),
            "go",
            &Ask(payload),
            &pods(),
            sampler,
            &Pass,
            2,
        )
        .map(|_| ())
}

fn routed_pods(runtime: &PtrRuntime) -> Vec<String> {
    runtime
        .events()
        .iter()
        .filter_map(|envelope| match &envelope.event {
            RuntimeEvent::PodRouted { pod, .. } => Some(pod.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_runtime_routes_each_question_to_the_pod_whose_address_matches() {
    let mut runtime = runtime();
    let admitted = runtime.admitted_representations(encoder(), &cache(&runtime));
    assert!(admitted.refused.is_empty());
    let sampler = sampler(admitted.store);
    run(&mut runtime, b"alpha question", &sampler).unwrap();
    run(&mut runtime, b"beta question", &sampler).unwrap();
    assert_eq!(routed_pods(&runtime), ["alpha", "beta"]);
}

#[test]
fn after_a_revocation_the_runtime_abstains_instead_of_routing_to_the_revoked_pod() {
    let mut runtime = runtime();
    let cache = cache(&runtime);
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(&address("alpha", 1)),
            generation: Generation(1),
        })
        .unwrap();
    let admitted = runtime.admitted_representations(encoder(), &cache);
    let kept_beta = admitted
        .store
        .get(&ProjectId::from("demo"), &PodId::from("beta"))
        .is_some();
    let sampler = sampler(admitted.store);
    // The alpha question no longer finds alpha's address; beta's does not match it either.
    let error = run(&mut runtime, b"alpha question", &sampler).unwrap_err();
    assert_eq!(
        error,
        RuntimeError::RouteFailed {
            reason: "routing abstained"
        }
    );
    assert!(routed_pods(&runtime).is_empty());
    // alpha is still a registry candidate but has no admitted address, so the field is only
    // partly covered and beta's question abstains too instead of ranking part of it.
    let error = run(&mut runtime, b"beta question", &sampler).unwrap_err();
    assert_eq!(
        error,
        RuntimeError::RouteFailed {
            reason: "routing abstained"
        }
    );
    assert!(routed_pods(&runtime).is_empty());
    // beta's own address is unaffected by alpha's revocation.
    assert!(kept_beta);
}

#[test]
fn an_ambiguous_or_unrelated_question_abstains() {
    let mut runtime = runtime();
    let admitted = runtime.admitted_representations(encoder(), &cache(&runtime));
    let sampler = sampler(admitted.store);
    let error = run(&mut runtime, b"unrelated question", &sampler).unwrap_err();
    assert_eq!(
        error,
        RuntimeError::RouteFailed {
            reason: "routing abstained"
        }
    );
}

#[test]
fn an_encoder_failure_or_a_wrongly_sized_query_is_a_pod_free_route_failure() {
    let mut runtime = runtime();
    let admitted = runtime.admitted_representations(encoder(), &cache(&runtime));
    let sampler = sampler(admitted.store);
    assert_eq!(
        run(&mut runtime, b"wrong size", &sampler).unwrap_err(),
        RuntimeError::RouteFailed {
            reason: "routing features invalid"
        }
    );
    assert_eq!(
        run(&mut runtime, b"anything else", &sampler).unwrap_err(),
        RuntimeError::RouteFailed {
            reason: "routing policy failed"
        }
    );
    assert!(routed_pods(&runtime).is_empty());
}

#[test]
fn the_debug_output_of_a_route_query_does_not_contain_the_payload() {
    let (project, capability, input_type, request_id) = (
        ProjectId::from("demo"),
        CapabilityId::from("Echo<Text>"),
        TypeId::from("Text"),
        ptr_types::RequestId::from("request-1"),
    );
    let secret = b"my passport number is X1234567";
    let query = RouteQuery {
        project: &project,
        capability: &capability,
        input_type: &input_type,
        request_id: &request_id,
        payload: secret,
    };
    let shown = format!("{query:?} {query:#?}");
    assert!(!shown.contains("passport"), "{shown}");
    assert!(!shown.contains("X1234567"), "{shown}");
    assert!(!shown.contains(&format!("{secret:?}")), "{shown}");
    assert!(shown.contains("payload_len"));
    assert!(shown.contains(&secret.len().to_string()));
}

struct OtherEncoder(EncoderRef);
impl QueryEmbedder for OtherEncoder {
    fn encoder(&self) -> &EncoderRef {
        &self.0
    }

    fn embed(&self, _: &RouteQuery<'_>) -> Result<Vec<f32>, RouteError> {
        Ok(vec![1.0, 0.0, 0.0])
    }
}

#[test]
fn an_embedder_of_another_model_or_revision_is_refused_even_at_the_same_dimension() {
    let router = |store| {
        CheckedRouter::new(
            LearnedPodRouter::new(store, LearnedRouterConfig::default()).unwrap(),
            SamplingKey::from_bytes([9; 32]),
            NonceSequence::starting_at([4; 8], 0),
        )
    };
    let mut store = RepresentationStore::new(encoder());
    store
        .insert(representation("alpha", vec![1.0, 0.0, 0.0], 1))
        .unwrap();
    for other in [
        EncoderRef {
            id: "test-encoder".into(),
            revision: "r2".into(),
        },
        EncoderRef {
            id: "another-encoder".into(),
            revision: "r1".into(),
        },
    ] {
        // Same three dimensions as the store: only the identity of the encoder differs.
        let error =
            EmbeddingSampler::new(router(store.clone()), Arc::new(OtherEncoder(other.clone())))
                .err()
                .expect("a mismatched encoder is refused");
        assert_eq!(
            error,
            EmbeddingSamplerError::EncoderMismatch {
                embedder: other,
                store: encoder(),
            }
        );
        assert!(error.to_string().contains("test-encoder@r1"), "{error}");
    }
    // The matching encoder is accepted.
    assert!(EmbeddingSampler::new(router(store), Arc::new(OtherEncoder(encoder()))).is_ok());
}
