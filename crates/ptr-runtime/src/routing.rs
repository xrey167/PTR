//! The trusted Pod routing boundary of the runtime.
//!
//! [`PodSampler`] is the object-safe face of a [`ptr_router::CheckedRouter`]: it owns the
//! sampling key and nonce sequence, so the runtime can route through
//! [`ptr_router::route_checked`] without ever holding the key (Invariant 14: a recorded
//! decision replays from key, nonce and request; Invariant 13: the key never leaves the router).

use crate::{PtrRuntime, RuntimeError};
use ptr_events::RuntimeEvent;
use ptr_pods::{DynPod, PodRegistry};
use ptr_router::{
    CheckedRouter, PodRouter, RouteError, RouteFeatures, RouteRequest, RoutedChoice, RoutedError,
    Router,
};
use ptr_types::{CapabilityId, Digest, ProjectId, RequestId, TypeId};
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

/// What a [`PodSampler`] may derive request features from: the scope of one Pod request and its
/// payload. It carries no key, nonce or registry.
#[derive(Clone, Copy, Debug)]
pub struct RouteQuery<'a> {
    pub project: &'a ProjectId,
    pub capability: &'a CapabilityId,
    pub input_type: &'a TypeId,
    pub request_id: &'a RequestId,
    pub payload: &'a [u8],
}

/// Turns a Pod request into the query vector a learned router scores candidates with. The encoder
/// behind it is frozen and pinned; this crate defines no encoder of its own.
pub trait QueryEmbedder: Send + Sync {
    /// The embedding of `query`, or the reason there is none.
    fn embed(&self, query: &RouteQuery<'_>) -> Result<Vec<f32>, RouteError>;
}

const REQUEST_DIGEST_TAG: &[u8] = b"ptr-runtime/pod-route-request/v1";

/// Samples a Pod for a routing request through the trusted checked path.
pub trait PodSampler: Send + Sync {
    /// One routing decision with a fresh nonce. `Ok(None)` only when the request has no
    /// candidate Pod.
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError>;

    /// The non-secret identifier of the sampling key, recorded next to every decision.
    fn key_id(&self) -> [u8; 8];

    /// The request features handed to the router for this Pod request. The default carries none,
    /// which is what routers that ignore features (such as [`PodRouter`]) expect; a sampler over
    /// a learned router supplies the query vector here (see [`EmbeddingSampler`]).
    fn features(&self, _query: &RouteQuery<'_>) -> Result<RouteFeatures, RouteError> {
        Ok(RouteFeatures::default())
    }
}

/// A [`CheckedRouter`] whose requests are embedded by `embedder` before routing, so a learned
/// router receives its query vector. The embedder runs inside the trusted side, before the
/// distribution is requested, and cannot reach the sampling key.
pub struct EmbeddingSampler<R> {
    router: CheckedRouter<R>,
    embedder: Arc<dyn QueryEmbedder>,
}

impl<R> EmbeddingSampler<R> {
    pub fn new(router: CheckedRouter<R>, embedder: Arc<dyn QueryEmbedder>) -> Self {
        Self { router, embedder }
    }
}

impl<R: Router + Send + Sync> PodSampler for EmbeddingSampler<R> {
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError> {
        self.router.route(request)
    }

    fn key_id(&self) -> [u8; 8] {
        self.router.key_id()
    }

    fn features(&self, query: &RouteQuery<'_>) -> Result<RouteFeatures, RouteError> {
        RouteFeatures::new().with_vector(self.embedder.embed(query)?)
    }
}

impl<R: Router + Send + Sync> PodSampler for CheckedRouter<R> {
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError> {
        CheckedRouter::route(self, request)
    }

    fn key_id(&self) -> [u8; 8] {
        CheckedRouter::key_id(self)
    }
}

/// SHA-256 over a domain tag and the length-prefixed project, capability, input type, request id
/// and Pod request payload, so the draw is bound to the full content of the request.
pub(crate) fn pod_request_digest(
    project: &ProjectId,
    capability: &CapabilityId,
    input_type: &TypeId,
    request_id: &RequestId,
    payload: &[u8],
) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(REQUEST_DIGEST_TAG);
    for part in [
        project.to_string().as_bytes(),
        capability.to_string().as_bytes(),
        input_type.to_string().as_bytes(),
        request_id.0.as_bytes(),
        payload,
    ] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// A stable reason for a routing failure. It never names a Pod: Pod identities are not
/// disclosed across projects, and `IneligibleChoice` carries one.
pub(crate) fn route_failure_reason(error: &RoutedError) -> &'static str {
    route_error_reason(error.error())
}

fn route_error_reason(error: &RouteError) -> &'static str {
    match error {
        RouteError::Policy { .. } => "routing policy failed",
        RouteError::InvalidPropensity(_) => "routing propensity invalid",
        RouteError::InvalidDistribution(_) => "routing distribution invalid",
        RouteError::InvalidFeatures { .. } => "routing features invalid",
        RouteError::IneligibleChoice { .. } => "routing policy chose an ineligible pod",
        RouteError::MissingChoice { .. } => "routing policy made no choice",
        RouteError::NonceExhausted(_) => "routing nonce exhausted",
        RouteError::Abstained { .. } => "routing abstained",
        RouteError::NoRepresentation { .. } => "routing has no learned representation",
        RouteError::InactiveRepresentation { .. } | RouteError::InvalidRepresentation { .. } => {
            "routing representation refused"
        }
        _ => "routing failed",
    }
}

/// How the resumable loop picks a Pod.
pub(crate) enum PodSelection<'a> {
    /// The plain, unsampled [`ptr_router::PodRouter::select`], recording nothing.
    Plain,
    /// The trusted checked path; every decision is recorded as [`RuntimeEvent::PodRouted`].
    Sampled(&'a dyn PodSampler),
}

impl PtrRuntime {
    /// Selects the Pod for one Pod request. A Pod in another project is unavailable in exactly
    /// the same words as one that does not exist, and a routing failure names no Pod.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select_pod(
        &mut self,
        selection: &PodSelection<'_>,
        pods: &PodRegistry,
        project: &ProjectId,
        request_id: &RequestId,
        capability: &CapabilityId,
        input_type: &TypeId,
        payload: &[u8],
    ) -> Result<Arc<dyn DynPod>, RuntimeError> {
        let unavailable = || RuntimeError::PodUnavailable {
            capability: capability.to_string(),
            input_type: input_type.to_string(),
        };
        match selection {
            PodSelection::Plain => PodRouter
                .select(pods, project, capability, input_type)
                .and_then(|route| pods.get(project, &route.pod_id))
                .ok_or_else(unavailable),
            PodSelection::Sampled(sampler) => {
                let features = sampler
                    .features(&RouteQuery {
                        project,
                        capability,
                        input_type,
                        request_id,
                        payload,
                    })
                    .map_err(|error| RuntimeError::RouteFailed {
                        reason: route_error_reason(&error),
                    })?;
                let request = RouteRequest {
                    registry: pods,
                    project,
                    capability,
                    input_type,
                    request_id,
                    request_digest: pod_request_digest(
                        project, capability, input_type, request_id, payload,
                    ),
                    features: &features,
                };
                let choice = sampler
                    .route(&request)
                    .map_err(|error| RuntimeError::RouteFailed {
                        reason: route_failure_reason(&error),
                    })?
                    .ok_or_else(unavailable)?;
                let pod = pods
                    .get(project, &choice.route().pod_id)
                    .ok_or_else(unavailable)?;
                self.emit(RuntimeEvent::PodRouted {
                    pod: choice.route().pod_id.to_string(),
                    propensity: choice.propensity().get(),
                    nonce: *choice.nonce().as_bytes(),
                    draw: choice.draw(),
                    key_id: sampler.key_id(),
                });
                Ok(pod)
            }
        }
    }
}
