//! Shared fixtures of the router integration tests. No test lives here.
#![allow(dead_code)]

use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, DecisionNonce, NonceSequence, RouteFeatures, RouteRequest, RoutedChoice,
    RoutedError, Router, SamplingKey,
};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, RequestId, TypeId};
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

pub fn key() -> SamplingKey {
    SamplingKey::from_bytes([0x5a; 32])
}

pub fn nonce() -> DecisionNonce {
    DecisionNonce::from_bytes([0x11; 16])
}

pub fn nonces() -> NonceSequence {
    NonceSequence::starting_at([0x22; 8], 0)
}

pub fn route<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
) -> Result<Option<RoutedChoice>, RoutedError> {
    route_checked(router, request, &key(), &nonce())
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

pub fn request<'a>(
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
