//! The `Router` contract: a policy returns a distribution, trusted routing samples from it.
//!
//! # Propensity semantics
//!
//! A [`Router`] does not choose a pod and does not report a propensity. It returns a
//! [`RouteDistribution`], the probability distribution over candidate pods that it would sample
//! from for this request. The trusted boundary, [`route_checked`], samples the pod itself and
//! records the normalized probability of the sampled pod as the [`Propensity`] of the
//! [`RoutedChoice`]. A router therefore cannot log a propensity that differs from the
//! distribution actually sampled, which keeps inverse-propensity evaluation unbiased.
//!
//! - A deterministic router (such as [`PodRouter`]) returns a point mass and logs exactly `1.0`.
//! - A stochastic router logs the probability of the pod that was sampled.
//! - [`RoutedChoice`] can only be constructed inside this crate.
//!
//! # Sampling
//!
//! The sample is deterministic and replayable. A uniform `u` in `[0, 1)` is derived from
//! `request_id` and `request_digest` (64-bit FNV-1a over the little-endian id length, the id
//! bytes and the 32 digest bytes, followed by the splitmix64 finalizer; the top 53 bits give
//! `u`). The pod is the first, in ascending `PodId` order, whose cumulative normalized weight
//! exceeds `u`. Replaying the same request therefore reproduces the same choice, while distinct
//! requests sample independently enough to realize the weights over many requests. The
//! derivation uses no external hash crate and is part of this crate's contract.
//!
//! # Result semantics
//!
//! [`route_checked`] returns `Result<Option<RoutedChoice>, RoutedError>`:
//!
//! - `Ok(Some(choice))`: a pod was sampled from the policy's distribution.
//! - `Ok(None)`: the registry offers no candidate and the policy returned an empty distribution.
//!   An empty distribution while candidates exist is [`RouteError::MissingChoice`].
//! - `Err(error)`: the policy failed, returned an invalid or ineligible distribution. The
//!   [`RoutedError`] carries the request id and digest so the failure can be attributed.
//!
//! # Request state
//!
//! A [`RouteRequest`] carries the registry scope plus per-request state ([`RouteFeatures`] and
//! the request identity) so a learned router can condition its distribution on the request. It
//! has no revision or generation field and `PodRegistry` exposes none, so neither can appear in
//! [`RoutedError`] yet.

use crate::{PodRoute, PodRouter, Propensity, PropensityError};
use ptr_pods::PodRegistry;
use ptr_types::{CapabilityId, Digest, PodId, Probability, ProjectId, RequestId, TypeId};
use std::fmt;

/// A pod sampled by the trusted routing path together with the probability of sampling it.
/// Only [`route_checked`] constructs it. `route().score` is the propensity rounded to `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedChoice {
    route: PodRoute,
    propensity: Propensity,
}

impl RoutedChoice {
    pub(crate) fn new(route: PodRoute, propensity: Propensity) -> Self {
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
    /// The policy produced a propensity that [`Propensity`] rejects (a sampled pod whose
    /// normalized probability is too small for a finite inverse-propensity weight).
    InvalidPropensity(PropensityError),
    /// The policy returned a distribution that [`RouteDistribution::new`] rejects.
    InvalidDistribution(DistributionError),
    /// The request features were invalid.
    InvalidFeatures { message: String },
    /// The policy gave positive weight to a pod that is not an eligible candidate of the request
    /// (unknown to the registry, or lacking the capability or input type). Raised by
    /// [`route_checked`].
    IneligibleChoice { pod: PodId },
    /// The policy returned an empty distribution although the registry offers eligible
    /// candidates. Raised by [`route_checked`].
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
            Self::InvalidDistribution(error) => write!(f, "routing policy produced {error}"),
            Self::InvalidFeatures { message } => write!(f, "invalid route features: {message}"),
            Self::IneligibleChoice { pod } => {
                write!(
                    f,
                    "router gave positive weight to pod {pod:?}, which is not an eligible candidate"
                )
            }
            Self::MissingChoice { candidates } => write!(
                f,
                "router returned an empty distribution although {candidates} eligible candidates exist"
            ),
        }
    }
}

impl std::error::Error for RouteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Policy { source, .. } => Some(source.as_ref()),
            Self::InvalidPropensity(error) => Some(error),
            Self::InvalidDistribution(error) => Some(error),
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

impl From<DistributionError> for RouteError {
    fn from(error: DistributionError) -> Self {
        Self::InvalidDistribution(error)
    }
}

/// A [`RouteError`] together with the request it occurred for, attached at the mandatory
/// routing boundary ([`route_checked`]).
///
/// It keeps the request id and digest so a failure can be attributed in logs and replayed.
/// `RouteRequest` carries no revision or generation (and `PodRegistry` exposes none), so none is
/// recorded here.
#[derive(Debug)]
pub struct RoutedError {
    request_id: RequestId,
    request_digest: Digest,
    error: RouteError,
}

impl RoutedError {
    fn new(request: &RouteRequest<'_>, error: RouteError) -> Self {
        Self {
            request_id: request.request_id.clone(),
            request_digest: request.request_digest,
            error,
        }
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn request_digest(&self) -> Digest {
        self.request_digest
    }

    pub fn error(&self) -> &RouteError {
        &self.error
    }

    pub fn into_error(self) -> RouteError {
        self.error
    }
}

impl fmt::Display for RoutedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "routing request {} (digest ", self.request_id)?;
        for byte in self.request_digest {
            write!(f, "{byte:02x}")?;
        }
        write!(f, ") failed: {}", self.error)
    }
}

impl std::error::Error for RoutedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Why [`RouteDistribution::new`] rejected its entries.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DistributionError {
    /// A weight is NaN or infinite.
    NonFiniteWeight { pod: PodId },
    /// A weight is negative.
    NegativeWeight { pod: PodId },
    /// The same pod is listed twice.
    DuplicatePod { pod: PodId },
    /// Entries exist but their weights sum to zero, so nothing can be normalized.
    ZeroTotalWeight,
    /// The weights sum to infinity.
    OverflowingTotalWeight,
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteWeight { pod } => write!(f, "weight of pod {pod:?} is not finite"),
            Self::NegativeWeight { pod } => write!(f, "weight of pod {pod:?} is negative"),
            Self::DuplicatePod { pod } => write!(f, "pod {pod:?} is listed more than once"),
            Self::ZeroTotalWeight => f.write_str("distribution weights sum to zero"),
            Self::OverflowingTotalWeight => f.write_str("distribution weights sum to infinity"),
        }
    }
}

impl std::error::Error for DistributionError {}

/// A validated probability distribution over pods: the output of a [`Router`].
///
/// Weights are finite and non-negative, pods are unique, and the total is positive; they are
/// normalized to sum to one on construction. Entries with weight zero are dropped. The empty
/// distribution ([`RouteDistribution::none`]) means "no choice". Entries are kept sorted by
/// `PodId`, the order [`route_checked`] samples in.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteDistribution {
    entries: Vec<(PodId, f64)>,
}

impl RouteDistribution {
    /// The empty distribution: the policy makes no choice.
    pub fn none() -> Self {
        Self::default()
    }

    /// A point mass on `pod`, the distribution of a deterministic router.
    pub fn certain(pod: PodId) -> Self {
        Self {
            entries: vec![(pod, 1.0)],
        }
    }

    /// Validates and normalizes `(pod, weight)` pairs. An empty list yields the empty
    /// distribution.
    pub fn new(weights: Vec<(PodId, f64)>) -> Result<Self, DistributionError> {
        let mut entries = weights;
        for (pod, weight) in &entries {
            if !weight.is_finite() {
                return Err(DistributionError::NonFiniteWeight { pod: pod.clone() });
            }
            if *weight < 0.0 {
                return Err(DistributionError::NegativeWeight { pod: pod.clone() });
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(DistributionError::DuplicatePod {
                pod: pair[0].0.clone(),
            });
        }
        if entries.is_empty() {
            return Ok(Self::none());
        }
        let total: f64 = entries.iter().map(|(_, weight)| weight).sum();
        if total.is_infinite() {
            return Err(DistributionError::OverflowingTotalWeight);
        }
        if total <= 0.0 {
            return Err(DistributionError::ZeroTotalWeight);
        }
        entries.retain(|(_, weight)| *weight > 0.0);
        for (_, weight) in &mut entries {
            *weight /= total;
        }
        Ok(Self { entries })
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The normalized probability of `pod`; zero when it is not listed.
    pub fn probability(&self, pod: &PodId) -> f64 {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == pod)
            .map_or(0.0, |(_, probability)| *probability)
    }

    /// Positive-probability pods in ascending `PodId` order with normalized probabilities.
    pub fn iter(&self) -> impl Iterator<Item = (&PodId, f64)> {
        self.entries
            .iter()
            .map(|(pod, probability)| (pod, *probability))
    }

    fn sample(&self, u: f64) -> Option<&(PodId, f64)> {
        let mut cumulative = 0.0;
        for entry in &self.entries {
            cumulative += entry.1;
            if u < cumulative {
                return Some(entry);
            }
        }
        // Rounding can leave the cumulative sum a hair below one.
        self.entries.last()
    }
}

/// Uniform value in `[0, 1)` derived from the request identity; see the module documentation.
fn sampling_point(request_id: &RequestId, request_digest: &Digest) -> f64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let id = request_id.0.as_bytes();
    let mut hash = FNV_OFFSET;
    let mut absorb = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    absorb(&(id.len() as u64).to_le_bytes());
    absorb(id);
    absorb(request_digest);
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 31;
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

/// Pod selection policy: returns the distribution over pods it would sample from. Object safe.
///
/// # Untrusted output
///
/// The distribution is untrusted: it may list pods unknown to the registry or lacking the
/// request's capability or input type. The policy neither samples nor reports a propensity;
/// callers obtain choices only through [`route_checked`] (or a [`CheckedRouter`]), which
/// validates eligibility, samples, and records the sampled probability as the propensity.
pub trait Router {
    /// The distribution over pods for the request. The empty distribution means "no choice" and
    /// is accepted only when the registry offers no candidate; `Err` means the router failed.
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError>;
}

/// Routes through `router`: validates its distribution against the registry, samples a pod and
/// records the sampled pod's probability as the propensity.
///
/// This is the trusted path: the candidate set is recomputed from
/// `request.registry.candidates(project, capability, input_type)` here, not taken from the
/// policy. A pod with positive weight outside that set yields [`RouteError::IneligibleChoice`]
/// (weight-zero entries are dropped by [`RouteDistribution::new`] and ignored); an empty
/// distribution while candidates exist yields [`RouteError::MissingChoice`]. Every failure is
/// wrapped in a [`RoutedError`] with the request id and digest.
pub fn route_checked<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
) -> Result<Option<RoutedChoice>, RoutedError> {
    route_inner(router, request).map_err(|error| RoutedError::new(request, error))
}

fn route_inner<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
) -> Result<Option<RoutedChoice>, RouteError> {
    let distribution = router.policy(request)?;
    let candidates =
        request
            .registry
            .candidates(request.project, request.capability, request.input_type);
    if let Some((pod, _)) = distribution
        .iter()
        .find(|(pod, _)| !candidates.contains(pod))
    {
        return Err(RouteError::IneligibleChoice { pod: pod.clone() });
    }
    if distribution.is_empty() {
        return if candidates.is_empty() {
            Ok(None)
        } else {
            Err(RouteError::MissingChoice {
                candidates: candidates.len(),
            })
        };
    }
    let u = sampling_point(request.request_id, &request.request_digest);
    let Some((pod, probability)) = distribution.sample(u) else {
        return Ok(None);
    };
    let propensity = Propensity::new(*probability)?;
    let score =
        Probability::new(propensity.get() as f32).expect("a propensity in (0, 1] is a probability");
    Ok(Some(RoutedChoice::new(
        PodRoute {
            pod_id: pod.clone(),
            score,
        },
        propensity,
    )))
}

/// Wraps a [`Router`] so it can only be used through [`route_checked`].
///
/// It deliberately does not implement [`Router`]: a router yields distributions, a
/// `CheckedRouter` yields trusted, sampled choices.
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckedRouter<R> {
    inner: R,
}

impl<R: Router> CheckedRouter<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// [`route_checked`] on the wrapped router.
    pub fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError> {
        route_checked(&self.inner, request)
    }
}

impl Router for PodRouter {
    /// Deterministic and infallible: a point mass on the pod [`PodRouter::select`] chooses (the
    /// empty distribution without candidates). Ignores `request.features`.
    fn policy(&self, request: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Ok(self
            .select(
                request.registry,
                request.project,
                request.capability,
                request.input_type,
            )
            .map_or_else(RouteDistribution::none, |route| {
                RouteDistribution::certain(route.pod_id)
            }))
    }
}
