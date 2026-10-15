//! The learned pod router is advisory: it ranks registry candidates, abstains when unsure, loads
//! only activated representations and keeps each pod's representation independent.
mod common;

use common::{key, nonce, registry, registry_in, request, route};
use ptr_router::{
    AbstentionReason, ActivationPolicy, EncoderRef, GateReport, LearnedPodRouter,
    LearnedRouterConfig, PodRepresentation, PodRepresentationParts, RepresentationError,
    RepresentationStore, RouteError, RouteFeatures, Router, ScoreWeights, MAX_MANIFEST_BYTES,
    MAX_REPRESENTATIONS,
};
use ptr_types::{
    CapabilityId, ClusterKey, Generation, NamespaceId, PodId, PodIdentity, PodIdentityDescriptor,
    ProjectId, RequestId, TypeId, POD_IDENTITY_SCHEMA_VERSION,
};

fn encoder() -> EncoderRef {
    EncoderRef {
        id: "test-encoder".into(),
        revision: "r1".into(),
    }
}

fn passing_gate() -> GateReport {
    GateReport {
        alias_threshold_permille: 800,
        alias_min_permille: 930,
        negative_threshold_permille: 900,
        negative_reject_permille: 960,
        alias_count: 4,
        negative_count: 20,
    }
}

fn failing_gate() -> GateReport {
    GateReport {
        alias_min_permille: 700,
        ..passing_gate()
    }
}

fn descriptor(pod: &str) -> PodIdentityDescriptor {
    PodIdentityDescriptor {
        schema_version: POD_IDENTITY_SCHEMA_VERSION,
        domain: "supply".into(),
        kind: "Retrieval".into(),
        predicate: Some(format!("answers_{pod}")),
        semantic_role: "reader".into(),
        aliases: [format!("{pod} alias")].into(),
    }
}

fn parts(pod: &str, z: Vec<f32>, bias: f32, generation: u64) -> PodRepresentationParts {
    let descriptor = descriptor(pod);
    PodRepresentationParts {
        project: ProjectId::from("demo"),
        namespace: NamespaceId::from("ns"),
        pod: PodId::from(pod),
        identity: descriptor.identity(),
        cluster_key: descriptor.cluster_key(),
        encoder: encoder(),
        generation: Generation(generation),
        z,
        bias,
        weights: ScoreWeights::default(),
        gate: passing_gate(),
        activated: true,
    }
}

fn representation(pod: &str, z: Vec<f32>, bias: f32, generation: u64) -> PodRepresentation {
    PodRepresentation::new(parts(pod, z, bias, generation)).unwrap()
}

fn store(entries: Vec<PodRepresentation>) -> RepresentationStore {
    let mut store = RepresentationStore::new(encoder());
    for entry in entries {
        store.insert(entry).unwrap();
    }
    store
}

fn router(entries: Vec<PodRepresentation>) -> LearnedPodRouter {
    LearnedPodRouter::new(store(entries), LearnedRouterConfig::default()).unwrap()
}

fn demo() -> ProjectId {
    ProjectId::from("demo")
}

fn sigmoid(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

struct Scene {
    project: ProjectId,
    capability: CapabilityId,
    input_type: TypeId,
    request_id: RequestId,
}

fn scene() -> Scene {
    Scene {
        project: ProjectId::from("demo"),
        capability: CapabilityId::from("read"),
        input_type: TypeId::from("Text"),
        request_id: RequestId::from("request-1"),
    }
}

fn features(vector: Vec<f32>) -> RouteFeatures {
    RouteFeatures::new().with_vector(vector).unwrap()
}

#[test]
fn the_score_is_the_sigmoid_of_weighted_cosine_plus_metadata_minus_bias() {
    let router = router(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1)]);
    let score = router
        .score(&demo(), &PodId::from("alpha"), &[1.0, 0.0, 0.0])
        .unwrap()
        .unwrap();
    assert!(
        (score - sigmoid(12.0 + 3.0 - 10.0)).abs() < 1e-12,
        "{score}"
    );
    // Scaling the query does not change a cosine.
    let scaled = router
        .score(&demo(), &PodId::from("alpha"), &[5.0, 0.0, 0.0])
        .unwrap()
        .unwrap();
    assert!((scaled - score).abs() < 1e-12);
    // A pod without a representation has no score.
    assert_eq!(
        router
            .score(&demo(), &PodId::from("beta"), &[1.0, 0.0, 0.0])
            .unwrap(),
        None
    );
}

#[test]
fn only_registry_candidates_are_scored_even_when_an_outsider_would_score_higher() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    // "intruder" is not in the registry and matches the query perfectly; alpha matches less.
    let router = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 12.0, 1),
        representation("beta", vec![0.0, 1.0, 0.0], 12.0, 1),
        representation("intruder", vec![1.0, 0.0, 0.0], -50.0, 1),
    ]);
    let features = features(vec![1.0, 0.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let choice = route(&router, &request).unwrap().unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("alpha"));
    // A deterministic point mass: the trusted boundary logs a propensity of exactly one.
    assert_eq!(choice.propensity().get(), 1.0);
}

#[test]
fn a_registry_without_candidates_yields_no_choice_rather_than_an_error() {
    let scene = scene();
    let empty = registry(&scene.project, &[]);
    let router = router(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1)]);
    let features = features(vec![1.0, 0.0, 0.0]);
    let request = request(
        &empty,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    assert!(route(&router, &request).unwrap().is_none());
}

#[test]
fn the_router_abstains_when_the_margin_to_the_runner_up_is_below_the_minimum() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let router = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1),
        representation("beta", vec![0.9, 0.1, 0.0], 10.0, 1),
    ]);
    let features = features(vec![1.0, 0.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let error = route(&router, &request).unwrap_err();
    match error.error() {
        RouteError::Abstained {
            best,
            runner_up,
            best_score,
            margin,
            reason,
        } => {
            assert_eq!(*reason, AbstentionReason::SmallMargin);
            assert_eq!(best, &PodId::from("alpha"));
            assert_eq!(runner_up.as_ref(), Some(&PodId::from("beta")));
            assert!(*best_score > 0.5, "the best pod is confident on its own");
            assert!(*margin < 0.15, "margin {margin}");
        }
        other => panic!("expected an abstention, got {other:?}"),
    }
}

#[test]
fn the_router_abstains_when_the_best_score_is_below_the_threshold() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha"]);
    let router = router(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1)]);
    let features = features(vec![-1.0, 0.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let error = route(&router, &request).unwrap_err();
    assert!(matches!(
        error.error(),
        RouteError::Abstained {
            runner_up: None,
            best_score,
            margin,
            reason: AbstentionReason::LowScore,
            ..
        } if *best_score < 0.5 && (best_score - margin).abs() < 1e-12
    ));
}

#[test]
fn a_clear_winner_is_chosen_and_the_runner_up_is_not() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let router = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1),
        representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1),
    ]);
    for (query, expected) in [
        (vec![1.0, 0.0, 0.0], "alpha"),
        (vec![0.0, 1.0, 0.0], "beta"),
    ] {
        let features = features(query);
        let request = request(
            &registry,
            &scene.project,
            &scene.capability,
            &scene.input_type,
            &scene.request_id,
            &features,
        );
        let choice = route(&router, &request).unwrap().unwrap();
        assert_eq!(choice.route().pod_id, PodId::from(expected));
    }
}

#[test]
fn a_router_without_representations_reports_that_nothing_was_scored() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let router = LearnedPodRouter::new(
        RepresentationStore::new(encoder()),
        LearnedRouterConfig::default(),
    )
    .unwrap();
    let features = features(vec![1.0, 0.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let error = route(&router, &request).unwrap_err();
    assert!(matches!(
        error.error(),
        RouteError::NoRepresentation { candidates: 2 }
    ));
}

#[test]
fn candidates_without_a_representation_are_not_scored() {
    // beta is covered and matches perfectly, alpha has no address: the field is only partly
    // covered, so the router declines instead of ranking part of it.
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let router = router(vec![representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1)]);
    let features = features(vec![0.0, 1.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let error = route(&router, &request).unwrap_err();
    assert!(matches!(
        error.error(),
        RouteError::Abstained {
            best,
            reason: AbstentionReason::PartialCoverage,
            ..
        } if best == &PodId::from("beta")
    ));
}

#[test]
fn partial_coverage_is_its_own_abstention_reason_and_full_coverage_still_routes() {
    let scene = scene();
    let features = features(vec![0.0, 1.0, 0.0]);
    let beta = || representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1);
    let alpha = || representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);

    // One of two candidates uncovered: partial coverage, whatever the covered score is.
    let two = registry(&scene.project, &["alpha", "beta"]);
    let partial = router(vec![beta()]);
    let request_two = request(
        &two,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let reason = match partial.policy(&request_two).unwrap_err() {
        RouteError::Abstained { reason, .. } => reason,
        other => panic!("expected an abstention, got {other:?}"),
    };
    assert_eq!(reason, AbstentionReason::PartialCoverage);

    // Covering the second candidate makes the same query a clear win for beta.
    let full = router(vec![alpha(), beta()]);
    assert_eq!(
        route(&full, &request_two).unwrap().unwrap().route().pod_id,
        PodId::from("beta")
    );

    // A registry that only lists the covered pod is fully covered.
    let one = registry(&scene.project, &["beta"]);
    let request_one = request(
        &one,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    assert_eq!(
        route(&partial, &request_one)
            .unwrap()
            .unwrap()
            .route()
            .pod_id,
        PodId::from("beta")
    );
}

#[test]
fn two_projects_with_the_same_pod_id_do_not_collide_and_route_only_with_their_own() {
    let other = ProjectId::from("other");
    // Same pod id "alpha" in both projects, with opposite addresses.
    let demo_alpha = representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    let mut other_parts = parts("alpha", vec![0.0, 1.0, 0.0], 10.0, 1);
    other_parts.project = other.clone();
    let other_alpha = PodRepresentation::new(other_parts).unwrap();
    assert_ne!(demo_alpha.digest(), other_alpha.digest());
    // The lifecycle namespace is part of the digest too: the same training in another namespace
    // is another representation.
    let mut other_namespace = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    other_namespace.namespace = NamespaceId::from("ns-b");
    let other_namespace = PodRepresentation::new(other_namespace).unwrap();
    assert_eq!(other_namespace.namespace(), &NamespaceId::from("ns-b"));
    assert_ne!(demo_alpha.digest(), other_namespace.digest());

    let mut store = RepresentationStore::new(encoder());
    assert!(store.insert(demo_alpha.clone()).unwrap().is_none());
    assert!(store.insert(other_alpha.clone()).unwrap().is_none());
    assert_eq!(store.len(), 2);
    assert_eq!(store.get(&demo(), &PodId::from("alpha")), Some(&demo_alpha));
    assert_eq!(store.get(&other, &PodId::from("alpha")), Some(&other_alpha));

    let router = LearnedPodRouter::new(store, LearnedRouterConfig::default()).unwrap();
    let query = [1.0, 0.0, 0.0];
    let in_demo = router
        .score(&demo(), &PodId::from("alpha"), &query)
        .unwrap()
        .unwrap();
    let in_other = router
        .score(&other, &PodId::from("alpha"), &query)
        .unwrap()
        .unwrap();
    assert!(in_demo > 0.9 && in_other < 0.5, "{in_demo} {in_other}");

    // Routing in "other" uses only other's alpha: the query matches demo's alpha, not other's.
    let scene = scene();
    let other_registry = registry(&other, &["alpha"]);
    let features = features(query.to_vec());
    let request = request(
        &other_registry,
        &other,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    assert!(matches!(
        router.policy(&request).unwrap_err(),
        RouteError::Abstained {
            reason: AbstentionReason::LowScore,
            ..
        }
    ));

    // Removing one project's pod leaves the other's untouched.
    let mut store = router.store().clone();
    assert!(store.remove(&other, &PodId::from("alpha")).is_some());
    assert!(store.get(&demo(), &PodId::from("alpha")).is_some());
}

#[test]
fn a_missing_wrongly_sized_or_zero_query_is_invalid_features() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha"]);
    let router = router(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1)]);
    for query in [vec![], vec![1.0, 0.0], vec![0.0, 0.0, 0.0]] {
        let features = RouteFeatures::new().with_vector(query).unwrap();
        let request = request(
            &registry,
            &scene.project,
            &scene.capability,
            &scene.input_type,
            &scene.request_id,
            &features,
        );
        let error = router.policy(&request).unwrap_err();
        assert!(
            matches!(error, RouteError::InvalidFeatures { .. }),
            "{error}"
        );
    }
}

#[test]
fn a_representation_that_is_not_activated_is_never_loaded() {
    let mut store = RepresentationStore::new(encoder());
    let mut inactive = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    inactive.gate = failing_gate();
    inactive.activated = false;
    let inactive = PodRepresentation::new(inactive).unwrap();
    assert!(!inactive.activated());
    let error = store.insert(inactive).unwrap_err();
    assert!(matches!(
        error,
        RouteError::InactiveRepresentation { ref pod } if pod == &PodId::from("alpha")
    ));
    assert!(store.is_empty());
}

#[test]
fn an_activated_flag_that_contradicts_the_measured_gate_does_not_construct() {
    let mut forged = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    forged.gate = failing_gate();
    forged.activated = true;
    assert_eq!(
        PodRepresentation::new(forged),
        Err(RepresentationError::GateInconsistent)
    );
    let mut understated = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    understated.activated = false;
    assert_eq!(
        PodRepresentation::new(understated),
        Err(RepresentationError::GateInconsistent)
    );
}

#[test]
fn the_store_rechecks_the_measured_gate_against_its_own_minima() {
    // The artifact claims thresholds it passes, but they are lower than the store requires.
    let mut lenient = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    lenient.gate = GateReport {
        alias_threshold_permille: 100,
        alias_min_permille: 500,
        negative_threshold_permille: 100,
        negative_reject_permille: 500,
        alias_count: 3,
        negative_count: 3,
    };
    let lenient = PodRepresentation::new(lenient).unwrap();
    assert!(lenient.activated());
    let mut strict = RepresentationStore::new(encoder());
    assert!(matches!(
        strict.insert(lenient.clone()),
        Err(RouteError::InactiveRepresentation { .. })
    ));
    // The activation gate can be made stricter, never weaker.
    for (alias, negative, field) in [
        (500, 900, "min_alias_permille"),
        (800, 500, "min_negative_reject_permille"),
    ] {
        assert_eq!(
            RepresentationStore::with_policy(
                encoder(),
                ActivationPolicy {
                    min_alias_permille: alias,
                    min_negative_reject_permille: negative,
                },
            )
            .unwrap_err(),
            RepresentationError::PolicyBelowMinimum { field }
        );
    }
    let mut stricter = RepresentationStore::with_policy(
        encoder(),
        ActivationPolicy {
            min_alias_permille: 990,
            min_negative_reject_permille: 990,
        },
    )
    .unwrap();
    // A representation that passes the built-in gate but not this one is refused.
    let passing = representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    assert!(matches!(
        stricter.insert(passing),
        Err(RouteError::InactiveRepresentation { .. })
    ));
}

#[test]
fn a_store_holds_a_bounded_number_of_pods() {
    let mut store = RepresentationStore::new(encoder());
    for index in 0..MAX_REPRESENTATIONS {
        store
            .insert(representation(
                &format!("pod-{index}"),
                vec![1.0, 0.0, 0.0],
                10.0,
                1,
            ))
            .unwrap();
    }
    assert_eq!(store.len(), MAX_REPRESENTATIONS);
    assert!(matches!(
        store.insert(representation("one-too-many", vec![1.0, 0.0, 0.0], 10.0, 1)),
        Err(RouteError::InvalidRepresentation {
            error: RepresentationError::TooManyRepresentations {
                max: MAX_REPRESENTATIONS
            },
            ..
        })
    ));
    // A newer generation of a pod already held still replaces it.
    store
        .insert(representation("pod-0", vec![0.0, 1.0, 0.0], 10.0, 2))
        .unwrap();
}

#[test]
fn an_oversized_manifest_is_refused_before_it_is_parsed() {
    let manifest = " ".repeat(MAX_MANIFEST_BYTES + 1);
    assert_eq!(
        PodRepresentation::from_artifact(&manifest, &[]),
        Err(RepresentationError::ManifestTooLarge {
            max: MAX_MANIFEST_BYTES,
            actual: MAX_MANIFEST_BYTES + 1,
        })
    );
}

#[test]
fn invalid_representations_are_refused_with_the_reason() {
    let refuse = |edit: &dyn Fn(&mut PodRepresentationParts)| {
        let mut broken = parts("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
        edit(&mut broken);
        PodRepresentation::new(broken).unwrap_err()
    };
    assert_eq!(
        refuse(&|p| p.z = vec![0.0, 0.0, 0.0]),
        RepresentationError::ZeroVector
    );
    assert_eq!(
        refuse(&|p| p.z = vec![1.0, f32::NAN, 0.0]),
        RepresentationError::NonFinite { field: "z" }
    );
    assert_eq!(
        refuse(&|p| p.bias = f32::INFINITY),
        RepresentationError::NonFinite { field: "bias" }
    );
    assert_eq!(
        refuse(&|p| p.z = Vec::new()),
        RepresentationError::InvalidDimensions {
            actual: 0,
            max: 4096
        }
    );
    assert_eq!(
        refuse(&|p| p.namespace = NamespaceId::from("")),
        RepresentationError::EmptyField { field: "namespace" }
    );
    assert_eq!(
        refuse(&|p| p.encoder.revision.clear()),
        RepresentationError::EmptyField {
            field: "encoder.revision"
        }
    );
    assert_eq!(
        refuse(&|p| p.weights.cosine = 0.0),
        RepresentationError::NonFinite {
            field: "cosine_weight"
        }
    );
}

#[test]
fn a_representation_of_another_dimension_or_encoder_is_refused() {
    let mut store = store(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1)]);
    let wrong_dims = representation("beta", vec![1.0, 0.0, 0.0, 0.0], 10.0, 1);
    assert!(matches!(
        store.insert(wrong_dims),
        Err(RouteError::InvalidRepresentation {
            error: RepresentationError::DimensionMismatch {
                expected: 3,
                actual: 4
            },
            ..
        })
    ));
    let mut other_encoder = parts("beta", vec![0.0, 1.0, 0.0], 10.0, 1);
    other_encoder.encoder.revision = "r2".into();
    assert!(matches!(
        store.insert(PodRepresentation::new(other_encoder).unwrap()),
        Err(RouteError::InvalidRepresentation {
            error: RepresentationError::EncoderMismatch { .. },
            ..
        })
    ));
    assert_eq!(store.len(), 1);
}

#[test]
fn a_representation_replaces_only_a_strictly_older_generation() {
    let mut store = store(vec![representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 2)]);
    for generation in [1, 2] {
        assert!(matches!(
            store.insert(representation(
                "alpha",
                vec![0.0, 1.0, 0.0],
                10.0,
                generation
            )),
            Err(RouteError::InvalidRepresentation {
                error: RepresentationError::StaleGeneration { .. },
                ..
            })
        ));
    }
    let replaced = store
        .insert(representation("alpha", vec![0.0, 1.0, 0.0], 10.0, 3))
        .unwrap()
        .unwrap();
    assert_eq!(replaced.generation(), Generation(2));
    assert_eq!(
        store
            .get(&demo(), &PodId::from("alpha"))
            .unwrap()
            .generation(),
        Generation(3)
    );
}

#[test]
fn removing_one_pods_representation_leaves_the_others_scoring_identically() {
    let router_before = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1),
        representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1),
        representation("gamma", vec![0.0, 0.0, 1.0], 10.0, 1),
    ]);
    let query = [0.2, 0.7, 0.1];
    let beta_before = router_before
        .score(&demo(), &PodId::from("beta"), &query)
        .unwrap();
    let gamma_before = router_before
        .score(&demo(), &PodId::from("gamma"), &query)
        .unwrap();

    let mut revoked = router_before.store().clone();
    assert!(revoked.remove(&demo(), &PodId::from("alpha")).is_some());
    let router_after = LearnedPodRouter::new(revoked, LearnedRouterConfig::default()).unwrap();

    assert_eq!(
        router_after
            .score(&demo(), &PodId::from("alpha"), &query)
            .unwrap(),
        None
    );
    assert_eq!(
        router_after
            .score(&demo(), &PodId::from("beta"), &query)
            .unwrap(),
        beta_before
    );
    assert_eq!(
        router_after
            .score(&demo(), &PodId::from("gamma"), &query)
            .unwrap(),
        gamma_before
    );
    // And routing still works for the pods that kept their representation.
    let scene = scene();
    let all_three = registry(&scene.project, &["alpha", "beta", "gamma"]);
    let registry = registry(&scene.project, &["beta", "gamma"]);
    let features = features(vec![0.0, 1.0, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let choice = route(&router_after, &request).unwrap().unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("beta"));
    // The revoked pod is still a candidate in its registry: the router no longer ranks that field.
    let request = common::request(
        &all_three,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    assert!(matches!(
        router_after.policy(&request).unwrap_err(),
        RouteError::Abstained {
            reason: AbstentionReason::PartialCoverage,
            ..
        }
    ));
}

#[test]
fn the_same_request_routes_to_the_same_pod_with_the_same_draw_inputs() {
    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let router = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1),
        representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1),
    ]);
    let features = features(vec![0.9, 0.1, 0.0]);
    let request = request(
        &registry,
        &scene.project,
        &scene.capability,
        &scene.input_type,
        &scene.request_id,
        &features,
    );
    let first = route(&router, &request).unwrap().unwrap();
    let again = ptr_router::route_checked(&router, &request, &key(), &nonce())
        .unwrap()
        .unwrap();
    assert_eq!(first, again);
}

#[test]
fn router_thresholds_must_be_probabilities() {
    for config in [
        LearnedRouterConfig {
            abstain_below: 1.5,
            ..LearnedRouterConfig::default()
        },
        LearnedRouterConfig {
            min_margin: -0.1,
            ..LearnedRouterConfig::default()
        },
        LearnedRouterConfig {
            abstain_below: f64::NAN,
            ..LearnedRouterConfig::default()
        },
    ] {
        assert!(matches!(
            LearnedPodRouter::new(RepresentationStore::new(encoder()), config),
            Err(RouteError::InvalidFeatures { .. })
        ));
    }
}

#[test]
fn identity_and_cluster_key_survive_in_the_representation() {
    let alpha = representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    let descriptor = descriptor("alpha");
    assert_eq!(alpha.identity(), &descriptor.identity());
    assert_eq!(alpha.cluster_key(), descriptor.cluster_key());
    let _: (&PodIdentity, ClusterKey) = (alpha.identity(), alpha.cluster_key());
}

#[test]
fn the_admitted_set_of_a_request_narrows_scoring_and_never_widens_it() {
    use ptr_router::AdmittedRepresentation;
    use std::collections::BTreeSet;

    let scene = scene();
    let registry = registry(&scene.project, &["alpha", "beta"]);
    let alpha = representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1);
    let beta = representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1);
    let router = router(vec![alpha.clone(), beta.clone()]);
    let named = |representation: &PodRepresentation| AdmittedRepresentation {
        project: scene.project.clone(),
        pod: representation.pod().clone(),
        digest: *representation.digest(),
    };
    let outcome = |admitted: Option<BTreeSet<AdmittedRepresentation>>| {
        let mut features = features(vec![0.0, 1.0, 0.0]);
        if let Some(admitted) = admitted {
            features = features.with_admitted_representations(admitted);
        }
        let request = request(
            &registry,
            &scene.project,
            &scene.capability,
            &scene.input_type,
            &scene.request_id,
            &features,
        );
        router.policy(&request).map(|_| ())
    };
    let reason = |result: Result<(), RouteError>| match result.unwrap_err() {
        RouteError::Abstained { reason, .. } => Some(reason),
        RouteError::NoRepresentation { .. } => None,
        other => panic!("unexpected {other:?}"),
    };

    // No restriction: both are scored and beta wins clearly.
    assert!(outcome(None).is_ok());
    assert!(outcome(Some([named(&alpha), named(&beta)].into())).is_ok());
    // Only beta admitted: alpha is uncovered, so the field abstains as partial coverage.
    assert_eq!(
        reason(outcome(Some([named(&beta)].into()))),
        Some(AbstentionReason::PartialCoverage)
    );
    // Nothing admitted: nothing is scored at all.
    assert_eq!(reason(outcome(Some(BTreeSet::new()))), None);
    // A digest that is not the loaded representation's does not admit it.
    let mut stale = named(&alpha);
    stale.digest = [0xAA; 32];
    assert_eq!(
        reason(outcome(Some([stale, named(&beta)].into()))),
        Some(AbstentionReason::PartialCoverage)
    );
    // Naming a pod that has no loaded representation adds nothing: eligibility is the registry's.
    let intruder = AdmittedRepresentation {
        project: scene.project.clone(),
        pod: PodId::from("intruder"),
        digest: [1; 32],
    };
    assert_eq!(
        reason(outcome(Some([named(&beta), intruder].into()))),
        Some(AbstentionReason::PartialCoverage)
    );
}

/// A representation is learned for one lifecycle namespace. It scores a registry candidate only
/// when the registry names that namespace for the candidate; an implementation registered for
/// another namespace, or without one, is a candidate without a representation.
#[test]
fn a_representation_scores_only_the_implementation_registered_for_its_namespace() {
    let scene = scene();
    let router = router(vec![
        representation("alpha", vec![1.0, 0.0, 0.0], 10.0, 1),
        representation("beta", vec![0.0, 1.0, 0.0], 10.0, 1),
    ]);
    let features = features(vec![1.0, 0.0, 0.0]);
    let policy = |registry: &ptr_pods::PodRegistry| {
        router.policy(&request(
            registry,
            &scene.project,
            &scene.capability,
            &scene.input_type,
            &scene.request_id,
            &features,
        ))
    };
    // Registered for the representations' own namespace: alpha is chosen.
    let own = registry(&scene.project, &["alpha", "beta"]);
    assert_eq!(
        policy(&own).unwrap().probability(&PodId::from("alpha")),
        1.0
    );
    // The same pod ids registered for another namespace: no representation applies.
    let foreign = registry_in(&scene.project, "ns-b", &["alpha", "beta"]);
    assert!(matches!(
        policy(&foreign).unwrap_err(),
        RouteError::NoRepresentation { candidates: 2 }
    ));
    // Registered without a namespace: no representation applies either.
    let mut unnamed = ptr_pods::PodRegistry::default();
    for id in ["alpha", "beta"] {
        let pod = own.get(&scene.project, &PodId::from(id)).unwrap();
        unnamed.register(pod);
    }
    assert!(matches!(
        policy(&unnamed).unwrap_err(),
        RouteError::NoRepresentation { candidates: 2 }
    ));
    // Only alpha moved to another namespace: the field is partly covered and abstains.
    let mut mixed = registry(&scene.project, &["beta"]);
    mixed.register_in_namespace(
        NamespaceId::from("ns-b"),
        own.get(&scene.project, &PodId::from("alpha")).unwrap(),
    );
    assert!(matches!(
        policy(&mixed).unwrap_err(),
        RouteError::Abstained {
            reason: AbstentionReason::PartialCoverage,
            ..
        }
    ));
}
