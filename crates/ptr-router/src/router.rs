//! The `Router` contract: pod selection that logs its own propensity.
//!
//! # Propensity semantics
//!
//! Every selection returns a [`RoutedChoice`] holding the chosen [`PodRoute`] and the
//! [`Propensity`], the probability with which this router would choose that pod given the same
//! inputs. Logged choices allow off-policy evaluation of other routing policies.
//!
//! - A deterministic router (such as [`PodRouter`]) logs exactly `1.0`.
//! - A stochastic learned router logs the probability with which it sampled the chosen pod.
//! - A router must never claim propensity `0` for a chosen action; [`Propensity`] cannot
//!   represent it, and it also rejects values so small that the inverse-propensity weight
//!   `1 / p` would overflow to infinity.
//!
//! # Result semantics
//!
//! [`Router::route`] returns `Result<Option<RoutedChoice>, RouteError>`:
//!
//! - `Ok(Some(choice))`: a pod was chosen.
//! - `Ok(None)`: the registry offers no candidate for the request. This is not a failure.
//! - `Err(error)`: candidates may exist but the router failed (model inference, scoring,
//!   sampling, or an invalid propensity). Callers must not treat this as "no candidate".
//!
//! # Request state
//!
//! A [`RouteRequest`] carries the registry scope plus per-request state ([`RouteFeatures`] and
//! the request identity) so a learned router can condition both its choice and its propensity on
//! the actual request.

use crate::{PodRoute, PodRouter, Propensity, PropensityError};
use ptr_pods::PodRegistry;
use ptr_types::{CapabilityId, Digest, Probability, ProjectId, RequestId, TypeId};
use std::fmt;

/// A selected pod together with the propensity of selecting it.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedChoice {
    route: PodRoute,
    propensity: Propensity,
}

impl RoutedChoice {
    pub fn new(route: PodRoute, propensity: Propensity) -> Self {
        Self { route, propensity }
    }

    pub fn route(&self) -> &PodRoute {
        &self.route
    }

    pub fn propensity(&self) -> Propensity {
        self.propensity
    }

    pub fn into_route(self) -> PodRoute {
        self.route
    }
}

/// Per-request inputs a router may condition on. It defines no semantics beyond that: a router
/// decides whether and how to use each field. Absent values mean "not available".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteFeatures {
    uncertainty: Option<Probability>,
    token_budget: Option<u32>,
    vector: Vec<f32>,
}

impl RouteFeatures {
    /// Features with nothing set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the request's estimated uncertainty.
    pub fn with_uncertainty(mut self, uncertainty: Probability) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }

    /// Sets the token budget available to the request.
    pub fn with_token_budget(mut self, token_budget: u32) -> Self {
        self.token_budget = Some(token_budget);
        self
    }

    /// Sets an opaque feature vector for learned routers. Every element must be finite.
    pub fn with_vector(mut self, vector: Vec<f32>) -> Result<Self, RouteError> {
        if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
            return Err(RouteError::InvalidFeatures {
                message: format!("feature vector element {index} is not finite"),
            });
        }
        self.vector = vector;
        Ok(self)
    }

    pub fn uncertainty(&self) -> Option<Probability> {
        self.uncertainty
    }

    pub fn token_budget(&self) -> Option<u32> {
        self.token_budget
    }

    /// The opaque feature vector; empty when none was supplied.
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }
}

/// Everything a [`Router`] receives for one routing decision.
#[derive(Clone, Copy)]
pub struct RouteRequest<'a> {
    pub registry: &'a PodRegistry,
    pub project: &'a ProjectId,
    pub capability: &'a CapabilityId,
    pub input_type: &'a TypeId,
    /// Identity of the request being routed.
    pub request_id: &'a RequestId,
    /// Digest of the request content, for replayable routing logs.
    pub request_digest: Digest,
    pub features: &'a RouteFeatures,
}

/// A router failure while candidates may exist. Distinct from `Ok(None)` (no candidate).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RouteError {
    /// The routing policy or model failed (inference, scoring or sampling).
    Policy { message: String },
    /// The policy produced a propensity that [`Propensity`] rejects.
    InvalidPropensity(PropensityError),
    /// The request features were invalid.
    InvalidFeatures { message: String },
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy { message } => write!(f, "routing policy failed: {message}"),
            Self::InvalidPropensity(error) => write!(f, "routing policy produced {error}"),
            Self::InvalidFeatures { message } => write!(f, "invalid route features: {message}"),
        }
    }
}

impl std::error::Error for RouteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPropensity(error) => Some(error),
            Self::Policy { .. } | Self::InvalidFeatures { .. } => None,
        }
    }
}

impl From<PropensityError> for RouteError {
    fn from(error: PropensityError) -> Self {
        Self::InvalidPropensity(error)
    }
}

/// Pod selection policy that reports the propensity of its choice. Object safe.
pub trait Router {
    /// Chooses a pod for the request. `Ok(None)` means the registry offers no candidate;
    /// `Err` means the router itself failed.
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError>;
}

impl Router for PodRouter {
    /// Deterministic and infallible: ignores `request.features`, never returns `Err`.
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError> {
        Ok(self
            .select(
                request.registry,
                request.project,
                request.capability,
                request.input_type,
            )
            .map(|route| RoutedChoice::new(route, Propensity::CERTAIN)))
    }
}
