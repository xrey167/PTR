use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, CheckedRouter, DistributionError, PodRouter, PolicyFailure, Propensity,
    RouteDistribution, RouteError, RouteFeatures, RouteRequest, RoutedError, Router,
};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, RequestId, TypeId};
use std::sync::Arc;

struct TestPod(PodManifest);
impl DynPod for TestPod {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

fn registry(project: &ProjectId, ids: &[&str]) -> PodRegistry {
    let mut registry = PodRegistry::default();
    for id in ids {
        registry.register(Arc::new(TestPod(PodManifest {
            project: project.clone(),
            id: PodId::from(*id),
            capabilities: vec![CapabilityId::from("read")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        })));
    }
    registry
}

fn request<'a>(
    registry: &'a PodRegistry,
    project: &'a ProjectId,
    cap: &'a CapabilityId,
    ty: &'a TypeId,
    id: &'a RequestId,
    features: &'a RouteFeatures,
) -> RouteRequest<'a> {
    RouteRequest {
        registry,
        project,
        capability: cap,
        input_type: ty,
        request_id: id,
        request_digest: [7; 32],
        features,
    }
}

#[test]
fn router_matches_select_and_logs_certain_propensity() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let selected = PodRouter.select(&registry, &project, &cap, &ty).unwrap();
    let distribution = PodRouter.policy(&req).unwrap();
    assert_eq!(distribution.len(), 1);
    assert_eq!(distribution.probability(&PodId::from("zeta")), 1.0);
    let choice = route_checked(&PodRouter, &req).unwrap().unwrap();
    assert_eq!(choice.route(), &selected);
    assert_eq!(choice.route().pod_id, PodId::from("zeta"));
    assert_eq!(choice.propensity().get(), 1.0);
}

#[test]
fn router_is_usable_as_trait_object() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let router: &dyn Router = &PodRouter;
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let choice = route_checked(router, &req).unwrap().unwrap();
    assert_eq!(choice.into_route().pod_id, PodId::from("alpha"));
}

#[test]
fn router_without_candidates_returns_ok_none_like_select() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &[]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(PodRouter.select(&registry, &project, &cap, &ty).is_none());
    assert!(PodRouter.policy(&req).unwrap().is_empty());
    assert!(matches!(route_checked(&PodRouter, &req), Ok(None)));
}

/// Chooses by token budget: a small budget fully determines the lesser pod id (a point mass),
/// otherwise the policy is stochastic with weights 1:3 for alpha:zeta.
struct FeatureRouter;
impl Router for FeatureRouter {
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        let small = request.features.token_budget().is_some_and(|b| b < 100);
        let weights = if small {
            vec![(PodId::from("alpha"), 1.0)]
        } else {
            vec![(PodId::from("alpha"), 1.0), (PodId::from("zeta"), 3.0)]
        };
        Ok(RouteDistribution::new(weights)?)
    }
}

#[test]
fn request_features_reach_the_router() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let id = RequestId::from("r1");
    let small = RouteFeatures::new()
        .with_token_budget(10)
        .with_uncertainty(Probability::new(0.4).unwrap())
        .with_vector(vec![0.5, -1.0])
        .unwrap();
    let large = RouteFeatures::new().with_token_budget(10_000);
    assert_eq!(small.vector(), &[0.5, -1.0]);
    assert_eq!(small.uncertainty(), Probability::new(0.4));

    let a = FeatureRouter
        .policy(&request(&registry, &project, &cap, &ty, &id, &small))
        .unwrap();
    let b = FeatureRouter
        .policy(&request(&registry, &project, &cap, &ty, &id, &large))
        .unwrap();
    assert_eq!(a.probability(&PodId::from("alpha")), 1.0);
    assert_eq!(b.probability(&PodId::from("alpha")), 0.25);
    assert_eq!(b.probability(&PodId::from("zeta")), 0.75);
}

#[test]
fn deterministic_policy_logs_certain_propensity_not_an_arbitrary_number() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, small) = (
        RequestId::from("r1"),
        RouteFeatures::new().with_token_budget(10),
    );
    let choice = route_checked(
        &FeatureRouter,
        &request(&registry, &project, &cap, &ty, &id, &small),
    )
    .unwrap()
    .unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("alpha"));
    assert_eq!(choice.propensity().get(), 1.0);
}

#[test]
fn recorded_propensity_is_the_sampled_pods_probability() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let large = RouteFeatures::new().with_token_budget(10_000);
    let mut seen = std::collections::BTreeSet::new();
    for n in 0..50 {
        let id = RequestId::from(format!("req-{n}").as_str());
        let choice = route_checked(
            &FeatureRouter,
            &request(&registry, &project, &cap, &ty, &id, &large),
        )
        .unwrap()
        .unwrap();
        let expected = if choice.route().pod_id == PodId::from("alpha") {
            0.25
        } else {
            0.75
        };
        assert_eq!(choice.propensity().get(), expected);
        assert_eq!(choice.route().score.get(), expected as f32);
        seen.insert(choice.route().pod_id.clone());
    }
    assert_eq!(seen.len(), 2, "both pods are sampled across requests");
}

#[test]
fn sampling_is_deterministic_and_matches_weights_empirically() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let large = RouteFeatures::new().with_token_budget(10_000);
    let sample = |n: usize| {
        let id = RequestId::from(format!("request-{n}").as_str());
        route_checked(
            &FeatureRouter,
            &request(&registry, &project, &cap, &ty, &id, &large),
        )
        .unwrap()
        .unwrap()
        .into_route()
        .pod_id
    };
    let mut zeta = 0usize;
    for n in 0..2000 {
        let pod = sample(n);
        assert_eq!(pod, sample(n), "same request replays the same choice");
        zeta += usize::from(pod == PodId::from("zeta"));
    }
    let frequency = zeta as f64 / 2000.0;
    assert!(
        (frequency - 0.75).abs() < 0.05,
        "zeta frequency {frequency}"
    );
}

#[test]
fn sampling_depends_on_request_digest() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, large) = (
        RequestId::from("same"),
        RouteFeatures::new().with_token_budget(10_000),
    );
    let mut seen = std::collections::BTreeSet::new();
    for byte in 0..=255u8 {
        let mut req = request(&registry, &project, &cap, &ty, &id, &large);
        req.request_digest = [byte; 32];
        seen.insert(
            route_checked(&FeatureRouter, &req)
                .unwrap()
                .unwrap()
                .into_route()
                .pod_id,
        );
    }
    assert_eq!(seen.len(), 2);
}

#[test]
fn distribution_normalizes_and_rejects_invalid_weights() {
    let d = RouteDistribution::new(vec![
        (PodId::from("b"), 3.0),
        (PodId::from("a"), 1.0),
        (PodId::from("z"), 0.0),
    ])
    .unwrap();
    assert_eq!(d.len(), 2, "weight-zero entries are dropped");
    assert_eq!(d.probability(&PodId::from("a")), 0.25);
    assert_eq!(d.probability(&PodId::from("b")), 0.75);
    assert_eq!(d.probability(&PodId::from("z")), 0.0);
    assert!(RouteDistribution::new(vec![]).unwrap().is_empty());

    let pod = |n: &str| PodId::from(n);
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), f64::NAN)]),
        Err(DistributionError::NonFiniteWeight { .. })
    ));
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), f64::INFINITY)]),
        Err(DistributionError::NonFiniteWeight { .. })
    ));
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), -0.1), (pod("b"), 1.0)]),
        Err(DistributionError::NegativeWeight { .. })
    ));
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), 0.0), (pod("b"), 0.0)]),
        Err(DistributionError::ZeroTotalWeight)
    ));
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), 1.0), (pod("a"), 2.0)]),
        Err(DistributionError::DuplicatePod { .. })
    ));
    assert!(matches!(
        RouteDistribution::new(vec![(pod("a"), f64::MAX), (pod("b"), f64::MAX)]),
        Err(DistributionError::OverflowingTotalWeight)
    ));
    let error: RouteError = DistributionError::ZeroTotalWeight.into();
    assert!(matches!(error, RouteError::InvalidDistribution(_)));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn non_finite_feature_vector_is_rejected() {
    for bad in [f32::NAN, f32::INFINITY] {
        assert!(matches!(
            RouteFeatures::new().with_vector(vec![0.0, bad]),
            Err(RouteError::InvalidFeatures { .. })
        ));
    }
}

/// Fails whenever candidates exist; reports `Ok(None)` otherwise.
struct FailingRouter;
impl Router for FailingRouter {
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        let has_candidate = PodRouter
            .select(
                request.registry,
                request.project,
                request.capability,
                request.input_type,
            )
            .is_some();
        if !has_candidate {
            return Ok(RouteDistribution::none());
        }
        Err(RouteError::policy(
            PolicyFailure::Unavailable,
            "inference backend unavailable",
        ))
    }
}

#[test]
fn router_failure_is_distinguishable_from_no_candidate() {
    let project = ProjectId::from("demo");
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let router: &dyn Router = &FailingRouter;

    let with = registry(&project, &["alpha"]);
    let err =
        route_checked(router, &request(&with, &project, &cap, &ty, &id, &features)).unwrap_err();
    assert!(matches!(err.error(), RouteError::Policy { .. }));
    assert!(err.to_string().contains("inference backend unavailable"));

    let without = registry(&project, &[]);
    assert!(matches!(
        route_checked(
            router,
            &request(&without, &project, &cap, &ty, &id, &features)
        ),
        Ok(None)
    ));
}

#[test]
fn routing_failures_keep_request_context() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("req-ctx-9"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let err = route_checked(&FailingRouter, &req).unwrap_err();
    assert_eq!(err.request_id(), &id);
    assert_eq!(err.request_digest(), [7; 32]);
    assert!(matches!(
        err.error(),
        RouteError::Policy {
            kind: PolicyFailure::Unavailable,
            ..
        }
    ));
    let text = err.to_string();
    assert!(text.contains("req-ctx-9"), "{text}");
    assert!(text.contains(&"07".repeat(32)), "{text}");
    let source = std::error::Error::source(&err).expect("inner RouteError is the source");
    assert!(source.downcast_ref::<RouteError>().is_some());
    assert!(matches!(err.into_error(), RouteError::Policy { .. }));

    let err = CheckedRouter::new(FixedRouter("ghost"))
        .route(&req)
        .unwrap_err();
    assert_eq!(err.request_id(), &id);
    assert!(matches!(err.error(), RouteError::IneligibleChoice { .. }));
}

#[test]
fn invalid_propensity_converts_to_route_error() {
    let error: RouteError = Propensity::new(0.0).unwrap_err().into();
    assert!(matches!(error, RouteError::InvalidPropensity(_)));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn propensity_rejects_values_outside_zero_exclusive_one_inclusive() {
    for bad in [
        0.0,
        -0.0,
        -0.5,
        1.0000001,
        2.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(Propensity::new(bad).is_err(), "{bad} must be rejected");
    }
}

#[test]
fn propensity_accepts_one_and_small_positive() {
    assert_eq!(Propensity::new(1.0).unwrap().get(), 1.0);
    assert_eq!(Propensity::new(1e-12).unwrap().get(), 1e-12);
    assert_eq!(
        Propensity::new(f64::MIN_POSITIVE).unwrap().get(),
        f64::MIN_POSITIVE
    );
    assert_eq!(
        Propensity::new(0.25).unwrap(),
        Propensity::new(0.25).unwrap()
    );
}

#[test]
fn propensity_rejects_subnormals_with_infinite_reciprocal() {
    for bad in [f64::from_bits(1), 1e-310] {
        assert!(bad.is_subnormal());
        assert!(Propensity::new(bad).is_err(), "{bad:e} must be rejected");
    }
    let p = Propensity::new(f64::MIN_POSITIVE).unwrap();
    assert!((1.0 / p.get()).is_finite());
}

#[derive(Debug)]
struct BackendTimeout {
    millis: u64,
}
impl std::fmt::Display for BackendTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backend timed out after {} ms", self.millis)
    }
}
impl std::error::Error for BackendTimeout {}

#[test]
fn policy_error_preserves_kind_and_source_chain() {
    let error = RouteError::policy(PolicyFailure::Timeout, BackendTimeout { millis: 250 });
    match &error {
        RouteError::Policy { kind, .. } => assert_eq!(*kind, PolicyFailure::Timeout),
        other => panic!("unexpected {other:?}"),
    }
    let source = std::error::Error::source(&error).expect("source preserved");
    let timeout = source
        .downcast_ref::<BackendTimeout>()
        .expect("downcast to the original error type");
    assert_eq!(timeout.millis, 250);
    assert!(error.to_string().contains("timeout"));
    assert!(error.to_string().contains("250 ms"));

    let other = RouteError::policy(PolicyFailure::InvalidOutput, "bad logits");
    assert!(matches!(
        other,
        RouteError::Policy {
            kind: PolicyFailure::InvalidOutput,
            ..
        }
    ));
}

/// Puts all weight on a fixed pod id regardless of the registry.
struct FixedRouter(&'static str);
impl Router for FixedRouter {
    fn policy(&self, _request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::certain(PodId::from(self.0)))
    }
}

fn registry_with_unsupported(project: &ProjectId) -> PodRegistry {
    let mut registry = registry(project, &["alpha"]);
    registry.register(Arc::new(TestPod(PodManifest {
        project: project.clone(),
        id: PodId::from("writer"),
        capabilities: vec![CapabilityId::from("write")],
        accepts: vec![TypeId::from("Image")],
        produces: vec![TypeId::from("Text")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    })));
    registry
}

#[test]
fn checked_route_rejects_unknown_pod() {
    let project = ProjectId::from("demo");
    let registry = registry_with_unsupported(&project);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let err = route_checked(&FixedRouter("ghost"), &req).unwrap_err();
    assert!(
        matches!(err.error(), RouteError::IneligibleChoice { pod } if *pod == PodId::from("ghost"))
    );
    assert!(err.to_string().contains("ghost"));
}

#[test]
fn checked_route_rejects_registered_pod_lacking_capability_or_input_type() {
    let project = ProjectId::from("demo");
    let registry = registry_with_unsupported(&project);
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());

    // `writer` is registered but supports neither `read` nor `Text`.
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(matches!(
        route_checked(&FixedRouter("writer"), &req).map_err(RoutedError::into_error),
        Err(RouteError::IneligibleChoice { .. })
    ));

    // Capability matches but the input type does not.
    let (cap, ty) = (CapabilityId::from("write"), TypeId::from("Text"));
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(matches!(
        route_checked(&FixedRouter("writer"), &req).map_err(RoutedError::into_error),
        Err(RouteError::IneligibleChoice { .. })
    ));
}

#[test]
fn checked_route_passes_valid_candidate_and_wrapper_matches() {
    let project = ProjectId::from("demo");
    let registry = registry_with_unsupported(&project);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let choice = route_checked(&FixedRouter("alpha"), &req).unwrap().unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("alpha"));

    let wrapped = CheckedRouter::new(FixedRouter("ghost"));
    assert!(matches!(
        wrapped.route(&req).map_err(RoutedError::into_error),
        Err(RouteError::IneligibleChoice { .. })
    ));
    let dynamic: &dyn Router = &FixedRouter("alpha");
    assert!(matches!(
        CheckedRouter::new(FixedRouter("alpha")).route(&req),
        Ok(Some(_))
    ));
    assert!(matches!(route_checked(dynamic, &req), Ok(Some(_))));
}

struct NoneRouter;
impl Router for NoneRouter {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::none())
    }
}

/// Lists `alpha` with weight 1 and the given extra pod with the given weight.
struct WithExtra(&'static str, f64);
impl Router for WithExtra {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::new(vec![
            (PodId::from("alpha"), 1.0),
            (PodId::from(self.0), self.1),
        ])?)
    }
}

/// Returns an invalid distribution error, as a policy that fails validation would.
struct BrokenRouter;
impl Router for BrokenRouter {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::new(vec![(
            PodId::from("alpha"),
            f64::NAN,
        )])?)
    }
}

#[test]
fn ineligible_pod_with_positive_weight_is_rejected_zero_weight_is_ignored() {
    let project = ProjectId::from("demo");
    let registry = registry_with_unsupported(&project);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let err = route_checked(&WithExtra("writer", 1e-9), &req).unwrap_err();
    assert!(matches!(err.error(), RouteError::IneligibleChoice { .. }));
    let err = route_checked(&WithExtra("ghost", 5.0), &req).unwrap_err();
    assert!(matches!(err.error(), RouteError::IneligibleChoice { .. }));

    let choice = route_checked(&WithExtra("ghost", 0.0), &req)
        .unwrap()
        .unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("alpha"));
    assert_eq!(choice.propensity().get(), 1.0);
}

#[test]
fn invalid_distribution_surfaces_as_typed_error_with_context() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let err = route_checked(&BrokenRouter, &req).unwrap_err();
    assert!(matches!(
        err.error(),
        RouteError::InvalidDistribution(DistributionError::NonFiniteWeight { .. })
    ));
    assert_eq!(err.request_id(), &id);
}

#[test]
fn checked_route_rejects_false_no_candidate_result() {
    let project = ProjectId::from("demo");
    let registry = registry_with_unsupported(&project);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let err = route_checked(&NoneRouter, &req).unwrap_err();
    assert!(matches!(
        err.error(),
        RouteError::MissingChoice { candidates: 1 }
    ));
    assert!(err.to_string().contains("1 eligible candidates"));
    assert!(matches!(
        CheckedRouter::new(NoneRouter)
            .route(&req)
            .map_err(RoutedError::into_error),
        Err(RouteError::MissingChoice { candidates: 1 })
    ));
}

#[test]
fn checked_route_passes_no_candidate_result_when_registry_is_empty() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &[]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    assert!(matches!(route_checked(&NoneRouter, &req), Ok(None)));
    assert!(matches!(
        CheckedRouter::new(NoneRouter).route(&req),
        Ok(None)
    ));
}

#[test]
fn checked_pod_router_behaves_exactly_like_unchecked() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    let selected = PodRouter.select(&registry, &project, &cap, &ty).unwrap();
    let checked = route_checked(&PodRouter, &req).unwrap().unwrap();
    assert_eq!(checked.route(), &selected);
    assert_eq!(checked.route().pod_id, PodId::from("zeta"));
    assert_eq!(checked.propensity().get(), 1.0);

    let empty = self::registry(&project, &[]);
    let req = request(&empty, &project, &cap, &ty, &id, &features);
    assert!(matches!(route_checked(&PodRouter, &req), Ok(None)));
}

#[test]
fn checked_route_propagates_policy_errors_unchanged() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(matches!(
        route_checked(&FailingRouter, &req).map_err(RoutedError::into_error),
        Err(RouteError::Policy {
            kind: PolicyFailure::Unavailable,
            ..
        })
    ));
}
