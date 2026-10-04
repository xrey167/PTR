use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    PodRouter, Propensity, RouteError, RouteFeatures, RouteRequest, RoutedChoice, Router,
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

    let selected = PodRouter.select(&registry, &project, &cap, &ty).unwrap();
    let choice = PodRouter
        .route(&request(&registry, &project, &cap, &ty, &id, &features))
        .unwrap()
        .unwrap();
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
    let choice = router
        .route(&request(&registry, &project, &cap, &ty, &id, &features))
        .unwrap()
        .unwrap();
    assert_eq!(choice.into_route().pod_id, PodId::from("alpha"));
}

#[test]
fn router_without_candidates_returns_ok_none_like_select() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &[]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    assert!(PodRouter.select(&registry, &project, &cap, &ty).is_none());
    let result = PodRouter.route(&request(&registry, &project, &cap, &ty, &id, &features));
    assert_eq!(result, Ok(None));
}

/// Picks the lesser pod id when the token budget is small, else the greater, with a
/// request-dependent propensity.
struct FeatureRouter;
impl Router for FeatureRouter {
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError> {
        let mut candidates = PodRouter.candidates(
            request.registry,
            request.project,
            request.capability,
            request.input_type,
        );
        candidates.sort_by(|a, b| a.pod_id.cmp(&b.pod_id));
        let small = request.features.token_budget().is_some_and(|b| b < 100);
        let chosen = if small {
            candidates.first()
        } else {
            candidates.last()
        };
        let Some(route) = chosen.cloned() else {
            return Ok(None);
        };
        let p = if small { 0.25 } else { 0.75 };
        Ok(Some(RoutedChoice::new(route, Propensity::new(p)?)))
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
        .route(&request(&registry, &project, &cap, &ty, &id, &small))
        .unwrap()
        .unwrap();
    let b = FeatureRouter
        .route(&request(&registry, &project, &cap, &ty, &id, &large))
        .unwrap()
        .unwrap();
    assert_eq!(a.route().pod_id, PodId::from("alpha"));
    assert_eq!(a.propensity().get(), 0.25);
    assert_eq!(b.route().pod_id, PodId::from("zeta"));
    assert_eq!(b.propensity().get(), 0.75);
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
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError> {
        let has_candidate = PodRouter
            .select(
                request.registry,
                request.project,
                request.capability,
                request.input_type,
            )
            .is_some();
        if !has_candidate {
            return Ok(None);
        }
        Err(RouteError::Policy {
            message: "inference backend unavailable".into(),
        })
    }
}

#[test]
fn router_failure_is_distinguishable_from_no_candidate() {
    let project = ProjectId::from("demo");
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (id, features) = (RequestId::from("r1"), RouteFeatures::new());
    let router: &dyn Router = &FailingRouter;

    let with = registry(&project, &["alpha"]);
    let err = router
        .route(&request(&with, &project, &cap, &ty, &id, &features))
        .unwrap_err();
    assert!(matches!(err, RouteError::Policy { .. }));
    assert!(err.to_string().contains("inference backend unavailable"));

    let without = registry(&project, &[]);
    assert_eq!(
        router.route(&request(&without, &project, &cap, &ty, &id, &features)),
        Ok(None)
    );
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
