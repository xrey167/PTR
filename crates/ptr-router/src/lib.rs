use ptr_pods::PodRegistry;
use ptr_types::{CapabilityId, PodId, Probability, ProjectId, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub struct RouteScore {
    pub target: String,
    pub score: Probability,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteDecision {
    pub candidates: Vec<RouteScore>,
}
impl RouteDecision {
    pub fn best(&self) -> Option<&RouteScore> {
        self.candidates.iter().max_by(|a, b| {
            a.score
                .get()
                .total_cmp(&b.score.get())
                .then_with(|| a.target.cmp(&b.target))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PodRoute {
    pub pod_id: PodId,
    pub score: Probability,
}

/// The runtime's minimal deterministic Pod selector. Candidate discovery is
/// deliberately delegated to the project-scoped registry; this router never
/// accepts an out-of-registry Pod identity.
#[derive(Clone, Copy, Debug, Default)]
pub struct PodRouter;

impl PodRouter {
    pub fn candidates(
        &self,
        registry: &PodRegistry,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Vec<PodRoute> {
        registry
            .candidates(project, capability, input_type)
            .into_iter()
            .map(|pod_id| PodRoute {
                pod_id,
                score: Probability::new(1.0).expect("constant score is valid"),
            })
            .collect()
    }

    pub fn select(
        &self,
        registry: &PodRegistry,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Option<PodRoute> {
        self.candidates(registry, project, capability, input_type)
            .into_iter()
            .max_by(|a, b| {
                a.score
                    .get()
                    .total_cmp(&b.score.get())
                    .then_with(|| a.pod_id.cmp(&b.pod_id))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_pods::{DynPod, PodManifest};
    use ptr_protocol::TypedPayload;
    use ptr_types::{Effect, ProjectId};
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

    #[test]
    fn pod_router_breaks_equal_scores_by_pod_id() {
        let project = ProjectId::from("demo");
        let mut registry = PodRegistry::default();
        for id in ["zeta", "alpha"] {
            registry.register(Arc::new(TestPod(PodManifest {
                project: project.clone(),
                id: PodId::from(id),
                capabilities: vec![CapabilityId::from("read")],
                accepts: vec![TypeId::from("Text")],
                produces: vec![TypeId::from("Text")],
                effects: vec![Effect::Pure],
                protocol_version: 1,
            })));
        }

        let selected = PodRouter
            .select(
                &registry,
                &project,
                &CapabilityId::from("read"),
                &TypeId::from("Text"),
            )
            .unwrap();
        assert_eq!(selected.pod_id, PodId::from("zeta"));
    }
}

#[derive(Clone, Debug)]
pub struct RoutingPolicy {
    pub max_parallel: usize,
    pub uncertainty_trigger: f32,
}
impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            uncertainty_trigger: 0.35,
        }
    }
}
