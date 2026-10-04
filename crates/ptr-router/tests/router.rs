use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{PodRouter, Propensity, Router};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};
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

#[test]
fn router_matches_select_and_logs_certain_propensity() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["zeta", "alpha"]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));

    let selected = PodRouter.select(&registry, &project, &cap, &ty).unwrap();
    let choice = PodRouter.route(&registry, &project, &cap, &ty).unwrap();
    assert_eq!(choice.route(), &selected);
    assert_eq!(choice.route().pod_id, PodId::from("zeta"));
    assert_eq!(choice.propensity().get(), 1.0);
}

#[test]
fn router_is_usable_as_trait_object() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["alpha"]);
    let router: &dyn Router = &PodRouter;
    let choice = router
        .route(
            &registry,
            &project,
            &CapabilityId::from("read"),
            &TypeId::from("Text"),
        )
        .unwrap();
    assert_eq!(choice.into_route().pod_id, PodId::from("alpha"));
}

#[test]
fn router_without_candidates_returns_none_like_select() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &[]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    assert!(PodRouter.select(&registry, &project, &cap, &ty).is_none());
    assert!(PodRouter.route(&registry, &project, &cap, &ty).is_none());
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
