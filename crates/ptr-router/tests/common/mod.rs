#![allow(dead_code)]

use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    BackendCapability, BackendId, CapabilitySet, CheckedCandidate, CheckedRouter, NonceSequence,
    PodRouter, RouteFeatures, RouteOption, RouteRequest, SamplingKey,
};
use ptr_types::{CapabilityId, Effect, PodId, Probability, ProjectId, RequestId, TypeId};
use std::sync::Arc;

pub struct TestPod(pub PodManifest);

impl DynPod for TestPod {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }
    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

pub fn registry(project: &ProjectId, ids: &[&str]) -> PodRegistry {
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

pub fn key_bytes() -> [u8; 32] {
    [0xa7; 32]
}

pub fn checked_router() -> CheckedRouter<PodRouter> {
    CheckedRouter::new(
        PodRouter,
        SamplingKey::from_bytes(key_bytes()),
        NonceSequence::starting_at([0x33; 8], 0),
    )
}

/// A checked candidate for request `id` over a registry holding Pod `pod`.
pub fn candidate(id: &str, pod: &str) -> CheckedCandidate {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &[pod]);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (request_id, features) = (RequestId::from(id), RouteFeatures::new());
    let request = RouteRequest {
        registry: &registry,
        project: &project,
        capability: &cap,
        input_type: &ty,
        request_id: &request_id,
        request_digest: [9; 32],
        features: &features,
    };
    CheckedCandidate::route(&checked_router(), &request)
        .unwrap()
        .unwrap()
}

pub fn backend(name: &str) -> BackendId {
    BackendId::new(name).unwrap()
}

pub fn prob(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

pub fn caps(list: &[BackendCapability]) -> CapabilitySet {
    list.iter()
        .fold(CapabilitySet::empty(), |set, cap| set.with(*cap))
}

pub fn option(
    name: &str,
    cost: u32,
    latency: u32,
    unc: f32,
    with: &[BackendCapability],
) -> RouteOption {
    RouteOption::new(backend(name), cost, latency, caps(with), prob(unc))
}
