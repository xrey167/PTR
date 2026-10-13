use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, CheckedRouter, DecisionNonce, DistributionError, NonceSequence, PodRouter,
    PolicyFailure, Propensity, RouteDistribution, RouteError, RouteFeatures, RouteRequest,
    RoutedChoice, RoutedError, Router, SamplingKey,
};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, RequestId, TypeId};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct TestPod(PodManifest);
impl DynPod for TestPod {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

fn key() -> SamplingKey {
    SamplingKey::from_bytes([0x5a; 32])
}

fn nonce() -> DecisionNonce {
    DecisionNonce::from_bytes([0x11; 16])
}

fn nonces() -> NonceSequence {
    NonceSequence::starting_at([0x22; 8], 0)
}

fn route<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
) -> Result<Option<RoutedChoice>, RoutedError> {
    route_checked(router, request, &key(), &nonce())
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
    let choice = route(&PodRouter, &req).unwrap().unwrap();
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
    let choice = route(router, &req).unwrap().unwrap();
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
    assert!(matches!(route(&PodRouter, &req), Ok(None)));
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
    let choice = route(
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
        let choice = route(
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
        route(
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
            route(&FeatureRouter, &req)
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
    let err = route(router, &request(&with, &project, &cap, &ty, &id, &features)).unwrap_err();
    assert!(matches!(err.error(), RouteError::Policy { .. }));
    assert!(err.to_string().contains("inference backend unavailable"));

    let without = registry(&project, &[]);
    assert!(matches!(
        route(
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

    let err = route(&FailingRouter, &req).unwrap_err();
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

    let err = CheckedRouter::new(FixedRouter("ghost"), key(), nonces())
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

    let err = route(&FixedRouter("ghost"), &req).unwrap_err();
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
        route(&FixedRouter("writer"), &req).map_err(RoutedError::into_error),
        Err(RouteError::IneligibleChoice { .. })
    ));

    // Capability matches but the input type does not.
    let (cap, ty) = (CapabilityId::from("write"), TypeId::from("Text"));
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(matches!(
        route(&FixedRouter("writer"), &req).map_err(RoutedError::into_error),
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

    let choice = route(&FixedRouter("alpha"), &req).unwrap().unwrap();
    assert_eq!(choice.route().pod_id, PodId::from("alpha"));

    let wrapped = CheckedRouter::new(FixedRouter("ghost"), key(), nonces());
    assert!(matches!(
        wrapped.route(&req).map_err(RoutedError::into_error),
        Err(RouteError::IneligibleChoice { .. })
    ));
    let dynamic: &dyn Router = &FixedRouter("alpha");
    assert!(matches!(
        CheckedRouter::new(FixedRouter("alpha"), key(), nonces()).route(&req),
        Ok(Some(_))
    ));
    assert!(matches!(route(dynamic, &req), Ok(Some(_))));
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

    let err = route(&WithExtra("writer", 1e-9), &req).unwrap_err();
    assert!(matches!(err.error(), RouteError::IneligibleChoice { .. }));
    let err = route(&WithExtra("ghost", 5.0), &req).unwrap_err();
    assert!(matches!(err.error(), RouteError::IneligibleChoice { .. }));

    let choice = route(&WithExtra("ghost", 0.0), &req).unwrap().unwrap();
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
    let err = route(&BrokenRouter, &req).unwrap_err();
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

    let err = route(&NoneRouter, &req).unwrap_err();
    assert!(matches!(
        err.error(),
        RouteError::MissingChoice { candidates: 1 }
    ));
    assert!(err.to_string().contains("1 eligible candidates"));
    assert!(matches!(
        CheckedRouter::new(NoneRouter, key(), nonces())
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

    assert!(matches!(route(&NoneRouter, &req), Ok(None)));
    assert!(matches!(
        CheckedRouter::new(NoneRouter, key(), nonces()).route(&req),
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
    let checked = route(&PodRouter, &req).unwrap().unwrap();
    assert_eq!(checked.route(), &selected);
    assert_eq!(checked.route().pod_id, PodId::from("zeta"));
    assert_eq!(checked.propensity().get(), 1.0);

    let empty = self::registry(&project, &[]);
    let req = request(&empty, &project, &cap, &ty, &id, &features);
    assert!(matches!(route(&PodRouter, &req), Ok(None)));
}

#[test]
fn checked_route_propagates_policy_errors_unchanged() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    assert!(matches!(
        route(&FailingRouter, &req).map_err(RoutedError::into_error),
        Err(RouteError::Policy {
            kind: PolicyFailure::Unavailable,
            ..
        })
    ));
}

const RESOLUTION: u64 = RouteDistribution::RESOLUTION;

#[test]
fn quantized_masses_sum_to_exactly_the_resolution() {
    assert_eq!(RESOLUTION, 1 << 53);
    let cases: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![1.0, 3.0],
        vec![1.0, 1.0, 1.0],
        vec![0.1, 0.2, 0.7],
        vec![1.0; 7],
        vec![1e-9, 1.0, 3.3e5, 0.123456789],
        vec![1e-12, 1.0],
        (1..=50).map(f64::from).collect(),
    ];
    for weights in cases {
        let pods: Vec<PodId> = (0..weights.len())
            .map(|n| PodId::from(format!("pod-{n:02}").as_str()))
            .collect();
        let d = RouteDistribution::new(pods.iter().cloned().zip(weights.clone()).collect())
            .unwrap_or_else(|e| panic!("{weights:?}: {e}"));
        let sum: u64 = pods.iter().map(|pod| d.width(pod)).sum();
        assert_eq!(sum, RESOLUTION, "{weights:?}");
        assert!(pods.iter().all(|pod| d.width(pod) >= 1));
        for pod in &pods {
            assert_eq!(d.probability(pod), d.width(pod) as f64 / RESOLUTION as f64);
        }
    }
    let certain = RouteDistribution::certain(PodId::from("a"));
    assert_eq!(certain.width(&PodId::from("a")), RESOLUTION);
    assert_eq!(certain.probability(&PodId::from("a")), 1.0);
}

#[test]
fn quantization_is_exact_for_dyadic_weights_and_ties_go_to_the_lesser_pod() {
    let d = RouteDistribution::new(vec![(PodId::from("b"), 3.0), (PodId::from("a"), 1.0)]).unwrap();
    assert_eq!(d.width(&PodId::from("a")), RESOLUTION / 4);
    assert_eq!(d.width(&PodId::from("b")), RESOLUTION / 4 * 3);
    // 2^53 = 2 (mod 3): three equal shares leave two units over, which go to the first two pods
    // in PodId order.
    let third = RouteDistribution::new(vec![
        (PodId::from("c"), 1.0),
        (PodId::from("a"), 1.0),
        (PodId::from("b"), 1.0),
    ])
    .unwrap();
    let (a, b, c) = (
        third.width(&PodId::from("a")),
        third.width(&PodId::from("b")),
        third.width(&PodId::from("c")),
    );
    assert_eq!(a + b + c, RESOLUTION);
    assert_eq!((a, b), (c + 1, c + 1));
}

#[test]
fn probability_below_the_resolution_is_rejected() {
    let err = RouteDistribution::new(vec![(PodId::from("a"), 1.0), (PodId::from("tiny"), 1e-300)])
        .unwrap_err();
    assert_eq!(
        err,
        DistributionError::ProbabilityBelowResolution {
            pod: PodId::from("tiny")
        }
    );
    assert!(err.to_string().contains("tiny"));
    let error: RouteError = err.into();
    assert!(matches!(error, RouteError::InvalidDistribution(_)));
    // Just below 2^-53 is rejected, one interval (2^-53) and above is accepted.
    let below = 2f64.powi(-54);
    assert!(matches!(
        RouteDistribution::new(vec![(PodId::from("a"), 1.0), (PodId::from("b"), below)]),
        Err(DistributionError::ProbabilityBelowResolution { .. })
    ));
    let edge = RouteDistribution::new(vec![
        (PodId::from("a"), 1.0),
        (PodId::from("b"), 2f64.powi(-52)),
    ])
    .unwrap();
    assert!(edge.width(&PodId::from("b")) >= 1);
    assert_eq!(
        edge.width(&PodId::from("a")) + edge.width(&PodId::from("b")),
        RESOLUTION
    );
}

#[test]
fn router_with_a_sub_resolution_pod_fails_routing_with_context() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha", "tiny"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let err = route(&WithExtra("tiny", 1e-300), &req).unwrap_err();
    assert!(matches!(
        err.error(),
        RouteError::InvalidDistribution(DistributionError::ProbabilityBelowResolution { .. })
    ));
    assert_eq!(err.request_id(), &id);
}

#[test]
fn logged_propensity_is_the_interval_width_over_the_resolution() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let large = RouteFeatures::new().with_token_budget(10_000);
    let (alpha, zeta) = (PodId::from("alpha"), PodId::from("zeta"));
    let distribution =
        RouteDistribution::new(vec![(alpha.clone(), 1.0), (zeta.clone(), 3.0)]).unwrap();
    let alpha_width = distribution.width(&alpha);
    for n in 0..300 {
        let id = RequestId::from(format!("req-{n}").as_str());
        let choice = route(
            &FeatureRouter,
            &request(&registry, &project, &cap, &ty, &id, &large),
        )
        .unwrap()
        .unwrap();
        assert!(choice.draw() < RESOLUTION);
        // alpha owns [0, alpha_width), zeta the rest.
        let (expected_pod, expected_width) = if choice.draw() < alpha_width {
            (&alpha, alpha_width)
        } else {
            (&zeta, distribution.width(&zeta))
        };
        assert_eq!(&choice.route().pod_id, expected_pod);
        assert_eq!(
            choice.propensity().get(),
            expected_width as f64 / RESOLUTION as f64
        );
        assert!(choice.propensity().get() == 0.25 || choice.propensity().get() == 0.75);
    }
}

#[test]
fn draw_replays_with_the_same_key_and_differs_with_another() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let large = RouteFeatures::new().with_token_budget(10_000);
    let other = SamplingKey::from_bytes([0xa5; 32]);
    for n in 0..20 {
        let id = RequestId::from(format!("req-{n}").as_str());
        let req = request(&registry, &project, &cap, &ty, &id, &large);
        let first = route(&FeatureRouter, &req).unwrap().unwrap();
        let again = route(&FeatureRouter, &req).unwrap().unwrap();
        assert_eq!(first, again);
        assert_eq!(first.draw(), again.draw());
        let different = route_checked(&FeatureRouter, &req, &other, &nonce())
            .unwrap()
            .unwrap();
        assert_ne!(first.draw(), different.draw());
    }
}

#[test]
fn sampling_key_debug_does_not_print_the_secret() {
    let text = format!("{:?}", SamplingKey::from_bytes([0x5a; 32]));
    assert_eq!(text, "SamplingKey(..)");
    let checked = format!("{:?}", CheckedRouter::new(PodRouter, key(), nonces()));
    assert!(checked.contains("SamplingKey(..)"), "{checked}");
}

/// The attack on a public draw: reproduces the previous public hash of (request id, digest)
/// and returns the larger weight on a pod chosen by the draw it computed.
struct GamingRouter;

/// The pre-fix public sampling point (FNV-1a + splitmix64).
fn old_public_point(request_id: &RequestId, digest: &[u8; 32]) -> f64 {
    let id = request_id.0.as_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut absorb = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    absorb(&(id.len() as u64).to_le_bytes());
    absorb(id);
    absorb(digest);
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 31;
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

impl Router for GamingRouter {
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        let low = old_public_point(request.request_id, &request.request_digest) < 0.5;
        let (alpha, zeta) = if low { (0.75, 0.25) } else { (0.25, 0.75) };
        Ok(RouteDistribution::new(vec![
            (PodId::from("alpha"), alpha),
            (PodId::from("zeta"), zeta),
        ])?)
    }
}

#[test]
fn router_that_recomputes_the_old_public_hash_cannot_bias_selection() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let features = RouteFeatures::new();
    let alpha = PodId::from("alpha");
    let (mut alpha_count, mut predicted_low, mut alpha_given_low) = (0usize, 0usize, 0usize);
    let mut propensity_sum = 0.0;
    let total = 2000usize;
    for n in 0..total {
        let id = RequestId::from(format!("gaming-{n}").as_str());
        let req = request(&registry, &project, &cap, &ty, &id, &features);
        let low = old_public_point(&id, &req.request_digest) < 0.5;
        let choice = route(&GamingRouter, &req).unwrap().unwrap();
        let is_alpha = choice.route().pod_id == alpha;
        alpha_count += usize::from(is_alpha);
        if low {
            predicted_low += 1;
            alpha_given_low += usize::from(is_alpha);
        }
        propensity_sum += choice.propensity().get();
    }
    // With the draw public the router would always win with alpha (frequency 1.0).
    let alpha_frequency = alpha_count as f64 / total as f64;
    assert!(
        (alpha_frequency - 0.5).abs() < 0.05,
        "alpha frequency {alpha_frequency}"
    );
    // The router's prediction carries no information about the draw: conditional on it saying
    // "alpha 0.75" alpha is chosen with exactly that logged probability.
    let conditional = alpha_given_low as f64 / predicted_low as f64;
    assert!(
        (conditional - 0.75).abs() < 0.07,
        "conditional {conditional}"
    );
    // Each request logs 0.75 with probability 0.75 and 0.25 with probability 0.25, so the
    // mean logged propensity is 0.75^2 + 0.25^2 = 0.625 (a public draw would give 0.75).
    let mean_propensity = propensity_sum / total as f64;
    assert!(
        (mean_propensity - 0.625).abs() < 0.03,
        "mean propensity {mean_propensity}"
    );
}

/// A policy that was handed the draw of the previous routing of a request (as a stateful policy
/// with access to the earlier record would be) and puts weight .75 on the pod whose interval
/// contains it, hoping that the next draw repeats.
struct ReplayGamingRouter {
    known_draw: Rc<Cell<Option<u64>>>,
}

impl Router for ReplayGamingRouter {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        // alpha owns [0, .75R) when heavy and [0, .25R) otherwise.
        let alpha_heavy = self
            .known_draw
            .get()
            .is_some_and(|draw| draw < RESOLUTION / 2);
        let (alpha, zeta) = if alpha_heavy {
            (0.75, 0.25)
        } else {
            (0.25, 0.75)
        };
        Ok(RouteDistribution::new(vec![
            (PodId::from("alpha"), alpha),
            (PodId::from("zeta"), zeta),
        ])?)
    }
}

fn gaming_frequency(
    mut route_once: impl FnMut() -> RoutedChoice,
    known: &Cell<Option<u64>>,
) -> f64 {
    let total = 2000usize;
    let mut heavy = 0usize;
    for _ in 0..total {
        let before = known.get();
        let choice = route_once();
        let heavy_pod = if before.is_some_and(|draw| draw < RESOLUTION / 2) {
            "alpha"
        } else {
            "zeta"
        };
        heavy += usize::from(choice.route().pod_id == PodId::from(heavy_pod));
        known.set(Some(choice.draw()));
    }
    heavy as f64 / total as f64
}

#[test]
fn replay_gaming_router_cannot_force_the_choice_with_fresh_nonces() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let features = RouteFeatures::new();
    // The SAME request id and digest every time.
    let id = RequestId::from("same-request");
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let known = Rc::new(Cell::new(None));
    let checked = CheckedRouter::new(
        ReplayGamingRouter {
            known_draw: Rc::clone(&known),
        },
        key(),
        nonces(),
    );
    let frequency = gaming_frequency(|| checked.route(&req).unwrap().unwrap(), &known);
    // The policy logs 0.75 for its heavy pod, and the heavy pod is chosen with exactly that
    // frequency, not 1.0.
    assert!(
        (frequency - 0.75).abs() < 0.05,
        "heavy action frequency {frequency}"
    );
}

#[test]
fn replay_gaming_router_would_force_the_choice_if_the_nonce_were_reused() {
    // Documents why the nonce must be fresh: a reused nonce reproduces the draw.
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let features = RouteFeatures::new();
    let id = RequestId::from("same-request");
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let known = Rc::new(Cell::new(None));
    let router = ReplayGamingRouter {
        known_draw: Rc::clone(&known),
    };
    let frequency = gaming_frequency(
        || {
            route_checked(&router, &req, &key(), &nonce())
                .unwrap()
                .unwrap()
        },
        &known,
    );
    // After the first routing every later one is forced (the first one is the unknown case).
    assert!(frequency > 0.99, "frequency {frequency}");
}

#[test]
fn checked_router_uses_a_fresh_nonce_and_independent_draw_per_decision() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let features = RouteFeatures::new();
    let id = RequestId::from("same-request");
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let checked = CheckedRouter::new(FeatureRouter, key(), nonces());

    let first = checked.route(&req).unwrap().unwrap();
    let second = checked.route(&req).unwrap().unwrap();
    assert_ne!(first.nonce(), second.nonce());
    assert_eq!(&first.nonce().as_bytes()[..8], &[0x22; 8]);
    assert_eq!(&first.nonce().as_bytes()[8..], &0u64.to_be_bytes());
    assert_eq!(&second.nonce().as_bytes()[8..], &1u64.to_be_bytes());

    let trials = 2000usize;
    let mut draws = vec![first.draw(), second.draw()];
    let mut nonces_seen = vec![first.nonce(), second.nonce()];
    for _ in 2..trials {
        let choice = checked.route(&req).unwrap().unwrap();
        draws.push(choice.draw());
        nonces_seen.push(choice.nonce());
    }
    let mut unique = nonces_seen
        .iter()
        .map(|n| *n.as_bytes())
        .collect::<Vec<_>>();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), trials);
    // Draws are uniform and a draw does not predict the next one.
    let below_half = draws.iter().filter(|d| **d < RESOLUTION / 2).count() as f64;
    assert!((below_half / trials as f64 - 0.5).abs() < 0.05);
    let successive_lower = draws.windows(2).filter(|pair| pair[1] < pair[0]).count() as f64;
    assert!((successive_lower / (trials - 1) as f64 - 0.5).abs() < 0.05);
    let repeats = draws.windows(2).filter(|pair| pair[0] == pair[1]).count();
    assert_eq!(repeats, 0);
}

#[test]
fn explicit_key_nonce_and_request_reproduce_the_decision_exactly() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let features = RouteFeatures::new();
    let id = RequestId::from("replayed");
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let checked = CheckedRouter::new(FeatureRouter, key(), NonceSequence::starting_at([9; 8], 41));
    for _ in 0..20 {
        let original = checked.route(&req).unwrap().unwrap();
        let replayed = route_checked(&FeatureRouter, &req, &key(), &original.nonce())
            .unwrap()
            .unwrap();
        assert_eq!(original, replayed);
        let other_nonce = DecisionNonce::from_bytes([0xee; 16]);
        let other = route_checked(&FeatureRouter, &req, &key(), &other_nonce)
            .unwrap()
            .unwrap();
        assert_ne!(original.draw(), other.draw());
    }
}

#[test]
fn exhausted_nonce_sequence_fails_routing_instead_of_reusing_a_nonce() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);
    let checked = CheckedRouter::new(
        PodRouter,
        key(),
        NonceSequence::starting_at([1; 8], u64::MAX - 1),
    );
    assert!(checked.route(&req).unwrap().is_some());
    let err = checked.route(&req).unwrap_err();
    assert!(matches!(err.error(), RouteError::NonceExhausted(_)));
    assert_eq!(err.request_id(), &id);
}

#[test]
fn eligibility_check_scales_to_many_candidates_and_still_rejects_an_ineligible_pod() {
    let project = ProjectId::from("demo");
    let ids: Vec<String> = (0..10_000).map(|n| format!("pod-{n:05}")).collect();
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let registry = registry(&project, &id_refs);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("big"), RouteFeatures::new());
    let req = request(&registry, &project, &cap, &ty, &id, &features);

    struct AllPods(Vec<PodId>, Option<PodId>);
    impl Router for AllPods {
        fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
            let mut weights: Vec<(PodId, f64)> =
                self.0.iter().map(|pod| (pod.clone(), 1.0)).collect();
            weights.extend(self.1.iter().map(|pod| (pod.clone(), 1.0)));
            Ok(RouteDistribution::new(weights)?)
        }
    }
    let pods: Vec<PodId> = ids.iter().map(|id| PodId::from(id.as_str())).collect();
    let started = Instant::now();
    let choice = route(&AllPods(pods.clone(), None), &req).unwrap().unwrap();
    assert!((choice.propensity().get() - 1.0 / 10_000.0).abs() < 1e-12);
    // Ineligible pods sorting first, in the middle and last are all found.
    for ghost in ["aaa-ghost", "pod-05000x", "zzz-ghost"] {
        let err = route(&AllPods(pods.clone(), Some(PodId::from(ghost))), &req).unwrap_err();
        assert!(
            matches!(err.error(), RouteError::IneligibleChoice { pod } if *pod == PodId::from(ghost))
        );
    }
    assert!(started.elapsed() < Duration::from_secs(30));
}

/// Interval widths in `PodId` order, read from the exact quantized masses in linear time.
fn widths_of(distribution: &RouteDistribution) -> Vec<u64> {
    distribution
        .iter()
        .map(|(_, mass)| (mass * RESOLUTION as f64) as u64)
        .collect()
}

/// Distribution over pods named `p0..` with `weights[i]` assigned to the pod at `order[i]`.
fn distribution_in_order(
    weights: &[f64],
    order: &[usize],
) -> Result<RouteDistribution, DistributionError> {
    RouteDistribution::new(
        weights
            .iter()
            .zip(order)
            .map(|(weight, slot)| (PodId::from(format!("p{slot:04}").as_str()), *weight))
            .collect(),
    )
}

#[test]
fn quantization_depends_on_the_weights_not_on_the_pod_order() {
    let weights = [0.31, 1.7, 2.9, 0.0042, 5.5, 0.77, 3.3, 0.1];
    let n = weights.len();
    let forward: Vec<usize> = (0..n).collect();
    let backward: Vec<usize> = (0..n).rev().collect();
    let shuffled = [3usize, 7, 0, 5, 1, 6, 2, 4];
    let reference = distribution_in_order(&weights, &forward).unwrap();
    for order in [&backward[..], &shuffled[..]] {
        let other = distribution_in_order(&weights, order).unwrap();
        for (index, slot) in order.iter().enumerate() {
            let pod_other = PodId::from(format!("p{slot:04}").as_str());
            let pod_reference = PodId::from(format!("p{index:04}").as_str());
            assert_eq!(
                other.width(&pod_other),
                reference.width(&pod_reference),
                "weight {} got a different width",
                weights[index]
            );
        }
    }
}

#[test]
fn dominant_weight_with_sub_resolution_tails_is_rejected_in_every_order() {
    // 1.0 plus 1000 tails of exactly 2^-53: the tails' share of the total is just under 2^-53
    // (the total exceeds 1.0), which a left-to-right sum with the dominant weight first would
    // round away and accept.
    let tail = 2f64.powi(-53);
    let mut weights = vec![1.0];
    weights.extend(std::iter::repeat_n(tail, 1000));
    let n = weights.len();
    let forward: Vec<usize> = (0..n).collect();
    let backward: Vec<usize> = (0..n).rev().collect();
    // Dominant weight in the middle.
    let mut middle: Vec<usize> = (0..n).collect();
    middle.swap(0, n / 2);
    for order in [&forward, &backward, &middle] {
        assert!(
            matches!(
                distribution_in_order(&weights, order),
                Err(DistributionError::ProbabilityBelowResolution { .. })
            ),
            "dominant weight with sub-resolution tails must be rejected"
        );
    }
}

#[test]
fn dominant_weight_with_representable_tails_is_accepted_with_equal_widths_in_every_order() {
    let tail = 2f64.powi(-52);
    let mut weights = vec![1.0];
    weights.extend(std::iter::repeat_n(tail, 1000));
    let n = weights.len();
    let forward: Vec<usize> = (0..n).collect();
    let backward: Vec<usize> = (0..n).rev().collect();
    let a = distribution_in_order(&weights, &forward).unwrap();
    let b = distribution_in_order(&weights, &backward).unwrap();
    for distribution in [&a, &b] {
        let widths = widths_of(distribution);
        assert_eq!(widths.iter().sum::<u64>(), RESOLUTION);
        assert!(widths.iter().all(|width| *width >= 1));
    }
    // The dominant weight sits at p0000 in `a` and at p1000 in `b`.
    assert_eq!(
        a.width(&PodId::from("p0000")),
        b.width(&PodId::from("p1000"))
    );
    let (mut widths_a, mut widths_b) = (widths_of(&a), widths_of(&b));
    widths_a.sort_unstable();
    widths_b.sort_unstable();
    assert_eq!(widths_a, widths_b);
}

#[test]
fn width_correction_handles_a_large_distribution_quickly() {
    let weights: Vec<f64> = (0..50_000).map(|n| 1.0 + f64::from(n % 7) / 3.0).collect();
    let order: Vec<usize> = (0..weights.len()).collect();
    let started = Instant::now();
    let distribution = distribution_in_order(&weights, &order).unwrap();
    assert_eq!(widths_of(&distribution).iter().sum::<u64>(), RESOLUTION);
    assert!(started.elapsed() < Duration::from_secs(60));
}

#[test]
fn checked_router_exposes_the_key_id_and_is_shareable_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CheckedRouter<PodRouter>>();
    let router = CheckedRouter::new(PodRouter, key(), nonces());
    assert_eq!(router.key_id(), key().key_id());
    assert_ne!(
        router.key_id(),
        SamplingKey::from_bytes([0x5b; 32]).key_id()
    );
}
