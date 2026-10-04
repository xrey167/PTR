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
//!   represent it.

use crate::{PodRoute, PodRouter, Propensity};
use ptr_pods::PodRegistry;
use ptr_types::{CapabilityId, ProjectId, TypeId};

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

/// Pod selection policy that reports the propensity of its choice. Object safe.
pub trait Router {
    /// Chooses a pod for the request, or `None` when the registry offers no candidate.
    fn route(
        &self,
        registry: &PodRegistry,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Option<RoutedChoice>;
}

impl Router for PodRouter {
    fn route(
        &self,
        registry: &PodRegistry,
        project: &ProjectId,
        capability: &CapabilityId,
        input_type: &TypeId,
    ) -> Option<RoutedChoice> {
        self.select(registry, project, capability, input_type)
            .map(|route| RoutedChoice::new(route, Propensity::CERTAIN))
    }
}
