mod common;

use common::{checked_router, key_bytes, registry};
use ptr_router::{
    CheckedCandidate, DecisionNonce, PodRouter, RouteDistribution, RouteError, RouteFeatures,
    RouteRequest, Router, SamplingKey,
};
use ptr_types::{CapabilityId, PodId, ProjectId, RequestId, TypeId};

struct Ghost;
impl Router for Ghost {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(RouteDistribution::certain(PodId::from("ghost")))
    }
}

fn with_request<T>(pods: &[&str], id: &str, f: impl FnOnce(&RouteRequest<'_>) -> T) -> T {
    let project = ProjectId::from("demo");
    let registry = registry(&project, pods);
    let (cap, ty) = (CapabilityId::from("read"), TypeId::from("Text"));
    let (request_id, features) = (RequestId::from(id), RouteFeatures::new());
    f(&RouteRequest {
        registry: &registry,
        project: &project,
        capability: &cap,
        input_type: &ty,
        request_id: &request_id,
        request_digest: [4; 32],
        features: &features,
    })
}

#[test]
fn a_candidate_carries_the_audit_fields_of_the_checked_decision() {
    let router = checked_router();
    let candidate = with_request(&["zeta", "alpha"], "req-1", |request| {
        CheckedCandidate::route(&router, request).unwrap().unwrap()
    });
    assert_eq!(candidate.pod(), &PodId::from("zeta"));
    assert_eq!(candidate.capability(), &CapabilityId::from("read"));
    assert_eq!(candidate.propensity().get(), 1.0);
    assert_eq!(candidate.request_id(), &RequestId::from("req-1"));
    assert_eq!(candidate.request_digest(), [4; 32]);
    assert_eq!(candidate.key_id(), router.key_id());
    assert_eq!(
        candidate.nonce(),
        DecisionNonce::from_bytes({
            let mut n = [0x33; 16];
            n[8..].copy_from_slice(&0u64.to_be_bytes());
            n
        })
    );
}

#[test]
fn route_with_replays_and_matches_the_checked_router_path() {
    let key = SamplingKey::from_bytes(key_bytes());
    let nonce = DecisionNonce::from_bytes([5; 16]);
    let (a, b) = with_request(&["alpha", "zeta"], "r", |request| {
        (
            CheckedCandidate::route_with(&PodRouter, request, &key, &nonce)
                .unwrap()
                .unwrap(),
            CheckedCandidate::route_with(&PodRouter, request, &key, &nonce)
                .unwrap()
                .unwrap(),
        )
    });
    assert_eq!(a, b);
    assert_eq!(a.key_id(), key.key_id());
}

#[test]
fn no_candidate_in_the_registry_is_none() {
    let router = checked_router();
    let result = with_request(&[], "r", |request| {
        CheckedCandidate::route(&router, request).unwrap()
    });
    assert!(result.is_none());
}

#[test]
fn an_ineligible_pod_never_becomes_a_candidate() {
    let key = SamplingKey::from_bytes(key_bytes());
    let nonce = DecisionNonce::from_bytes([5; 16]);
    let error = with_request(&["alpha"], "r", |request| {
        CheckedCandidate::route_with(&Ghost, request, &key, &nonce).unwrap_err()
    });
    assert!(matches!(
        error.error(),
        RouteError::IneligibleChoice { pod } if pod == &PodId::from("ghost")
    ));
}

#[test]
fn debug_output_hides_the_key_but_may_show_its_id() {
    let router = checked_router();
    let candidate = with_request(&["alpha"], "r", |request| {
        CheckedCandidate::route(&router, request).unwrap().unwrap()
    });
    let text = format!("{candidate:?}");
    assert!(!text.contains("167, 167, 167, 167, 167, 167, 167, 167, 167, 167"));
}
