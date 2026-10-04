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
//! - `Ok(None)`: the registry offers no candidate for the request. This is not a failure, and
//!   it is reserved for that case: [`route_checked`] rejects `Ok(None)` while eligible
//!   candidates exist with [`RouteError::MissingChoice`].
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
use ptr_types::{CapabilityId, Digest, PodId, Probability, ProjectId, RequestId, TypeId};
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

/// Stable category of a routing-policy failure, so callers can distinguish (for example) a
/// timeout from an invalid model output without parsing messages.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum PolicyFailure {
    /// The policy or model did not answer within its deadline.
    Timeout,
    /// The policy or model backend could not be reached or was not ready.
    Unavailable,
    /// The policy or model answered with output that could not be used.
    InvalidOutput,
    /// Any other internal failure of the policy.
    Internal,
}

impl fmt::Display for PolicyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Timeout => "timeout",
            Self::Unavailable => "unavailable",
            Self::InvalidOutput => "invalid output",
            Self::Internal => "internal error",
        })
    }
}

/// A router failure while candidates may exist. Distinct from `Ok(None)` (no candidate).
#[derive(Debug)]
#[non_exhaustive]
pub enum RouteError {
    /// The routing policy or model failed (inference, scoring or sampling). The original error
    /// is preserved as the [`std::error::Error::source`].
    Policy {
        kind: PolicyFailure,
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// The policy produced a propensity that [`Propensity`] rejects.
    InvalidPropensity(PropensityError),
    /// The request features were invalid.
    InvalidFeatures { message: String },
    /// The router chose a pod that is not an eligible candidate of the request (unknown to the
    /// registry, or lacking the capability or input type). Raised by [`route_checked`].
    IneligibleChoice { pod: PodId },
    /// The router returned `Ok(None)` although the registry offers eligible candidates, which
    /// the trait defines as "no candidate". Raised by [`route_checked`].
    MissingChoice { candidates: usize },
}

impl RouteError {
    /// A policy failure of the given category, keeping `source` as the error chain.
    pub fn policy(
        kind: PolicyFailure,
        source: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Self {
        Self::Policy {
            kind,
            source: source.into(),
        }
    }
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy { kind, source } => write!(f, "routing policy failed ({kind}): {source}"),
            Self::InvalidPropensity(error) => write!(f, "routing policy produced {error}"),
            Self::InvalidFeatures { message } => write!(f, "invalid route features: {message}"),
            Self::IneligibleChoice { pod } => {
                write!(
                    f,
                    "router chose pod {pod:?}, which is not an eligible candidate"
                )
            }
            Self::MissingChoice { candidates } => write!(
                f,
                "router returned no choice although {candidates} eligible candidates exist"
            ),
        }
    }
}

impl std::error::Error for RouteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Policy { source, .. } => Some(source.as_ref()),
            Self::InvalidPropensity(error) => Some(error),
            Self::InvalidFeatures { .. }
            | Self::IneligibleChoice { .. }
            | Self::MissingChoice { .. } => None,
        }
    }
}

impl From<PropensityError> for RouteError {
    fn from(error: PropensityError) -> Self {
        Self::InvalidPropensity(error)
    }
}

/// Pod selection policy that reports the propensity of its choice. Object safe.
///
/// # Untrusted output
///
/// The result of [`Router::route`] is untrusted: [`RoutedChoice::new`] accepts any
/// [`PodRoute`], so a custom or learned router can return a pod that is unknown to the registry
/// or does not support the request's capability and input type. Callers must obtain choices
/// through [`route_checked`] (or a [`CheckedRouter`]) and never act on a raw `route` result.
/// The eligibility check lives outside the policy so a policy cannot bypass it.
pub trait Router {
    /// Chooses a pod for the request. `Ok(None)` means exclusively that the registry offers no
    /// candidate (returning it while candidates exist is a contract violation that
    /// [`route_checked`] rejects); `Err` means the router itself failed. The choice is not validated here; see the trait
    /// documentation.
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError>;
}

/// Routes through `router` and rejects any choice whose pod is not among
/// `request.registry.candidates(project, capability, input_type)`, and any `Ok(None)` while that
/// set is nonempty.
///
/// This is the trusted path: the candidate set is recomputed from the registry here, not taken
/// from the policy. A rejected choice yields [`RouteError::IneligibleChoice`]; a false
/// no-candidate result yields [`RouteError::MissingChoice`].
pub fn route_checked<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
) -> Result<Option<RoutedChoice>, RouteError> {
    let result = router.route(request)?;
    let candidates =
        request
            .registry
            .candidates(request.project, request.capability, request.input_type);
    match result {
        None if candidates.is_empty() => Ok(None),
        None => Err(RouteError::MissingChoice {
            candidates: candidates.len(),
        }),
        Some(choice) if candidates.contains(&choice.route().pod_id) => Ok(Some(choice)),
        Some(choice) => Err(RouteError::IneligibleChoice {
            pod: choice.into_route().pod_id,
        }),
    }
}

/// A [`Router`] that validates every choice of the wrapped router with [`route_checked`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckedRouter<R> {
    inner: R,
}

impl<R: Router> CheckedRouter<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }
}

impl<R: Router> Router for CheckedRouter<R> {
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RouteError> {
        route_checked(&self.inner, request)
    }
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
