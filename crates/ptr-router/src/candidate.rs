//! The opaque token that proves a Pod came out of the checked routing path.

use crate::{
    route_checked, CheckedRouter, DecisionNonce, Propensity, RouteRequest, RoutedChoice,
    RoutedError, Router, SamplingKey,
};
use ptr_types::{CapabilityId, Digest, PodId, RequestId};

/// A Pod chosen by [`route_checked`] (directly or through a [`CheckedRouter`]) together with the
/// audit fields of that decision.
///
/// There is no public constructor, no `Default` and no `From` impl: the only ways to obtain one
/// are [`CheckedCandidate::route`] and [`CheckedCandidate::route_with`], which both go through
/// [`route_checked`] (registry eligibility, keyed draw, recorded propensity). The recipe
/// planner accepts a Pod only as a `CheckedCandidate`, so a caller of the new planning path
/// cannot name an unchecked Pod. The existing convenience wrappers (`PodRouter::select`) are
/// unchanged and still return unchecked routes.
///
/// It stores the key *identifier* ([`SamplingKey::key_id`]) and never the key (Invariant 13).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedCandidate {
    pod: PodId,
    capability: CapabilityId,
    propensity: Propensity,
    nonce: DecisionNonce,
    key_id: [u8; 8],
    request_id: RequestId,
    request_digest: Digest,
}

impl CheckedCandidate {
    /// Routes through a [`CheckedRouter`] (fresh nonce, owned key). `Ok(None)` when the
    /// registry offers no candidate.
    pub fn route<R: Router>(
        router: &CheckedRouter<R>,
        request: &RouteRequest<'_>,
    ) -> Result<Option<Self>, RoutedError> {
        let choice = router.route(request)?;
        Ok(choice.map(|choice| Self::from_checked(choice, request, router.key_id())))
    }

    /// Routes through [`route_checked`] with an explicit key and nonce (for replay).
    pub fn route_with<R: Router + ?Sized>(
        router: &R,
        request: &RouteRequest<'_>,
        key: &SamplingKey,
        nonce: &DecisionNonce,
    ) -> Result<Option<Self>, RoutedError> {
        let choice = route_checked(router, request, key, nonce)?;
        Ok(choice.map(|choice| Self::from_checked(choice, request, key.key_id())))
    }

    fn from_checked(choice: RoutedChoice, request: &RouteRequest<'_>, key_id: [u8; 8]) -> Self {
        let propensity = choice.propensity();
        let nonce = choice.nonce();
        Self {
            pod: choice.into_route().pod_id,
            capability: request.capability.clone(),
            propensity,
            nonce,
            key_id,
            request_id: request.request_id.clone(),
            request_digest: request.request_digest,
        }
    }

    pub fn pod(&self) -> &PodId {
        &self.pod
    }

    /// The capability the request asked for; it becomes the Pod class of a recipe stage.
    pub fn capability(&self) -> &CapabilityId {
        &self.capability
    }

    pub fn propensity(&self) -> Propensity {
        self.propensity
    }

    pub fn nonce(&self) -> DecisionNonce {
        self.nonce
    }

    /// Non-secret identifier of the sampling key.
    pub fn key_id(&self) -> [u8; 8] {
        self.key_id
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> Digest {
        self.request_digest
    }
}
