//! The call-time eligibility check around an expert adapter.

use ptr_router::PodEligibility;

use super::still_eligible;
use crate::vcw::{ProposeCall, RepairCall, RolePods};

/// An adapter that can answer the loop's calls provided a predicate still holds when the Pod is
/// about to run. [`Guarded`] supplies that predicate.
pub(crate) trait GuardedExperts<P> {
    /// Answer a proposal call. `eligible` is asked immediately before the Pod's generator runs;
    /// when it is false the Pod is not run and the call is recorded as ineligible.
    fn propose_while(&mut self, call: ProposeCall<'_>, eligible: &dyn Fn() -> bool) -> Option<P>;

    /// Answer a repair call, with the same `eligible` rule.
    fn repair_while(&mut self, call: RepairCall<'_, P>, eligible: &dyn Fn() -> bool) -> Option<P>;
}

/// An expert adapter bound to the eligibility view the loop uses. `standing` must be the loop's
/// own view, the one passed to `VcwLoop::run` for the same request: the guard checks the view it
/// is given, so a different view would check different Pods. It is the only way to hand an
/// adapter to the loop: every call is checked against the *current* generation of the Pod it
/// names, so a Pod revoked or superseded after routing (even between two actions of one repair
/// plan) is declined and its generator never runs.
pub struct Guarded<'a, T> {
    experts: &'a mut T,
    standing: &'a dyn PodEligibility,
}

impl<'a, T> Guarded<'a, T> {
    #[must_use]
    pub fn new(experts: &'a mut T, standing: &'a dyn PodEligibility) -> Self {
        Self { experts, standing }
    }

    /// The adapter, for its call log.
    #[must_use]
    pub fn experts(&self) -> &T {
        self.experts
    }
}

/// The guarded `RolePods` of one adapter type. It is written out per adapter, not over the
/// crate-private [`GuardedExperts`] trait, so that trait and its unchecked methods stay out of
/// the public API: the only way to hand an adapter to the loop is through this check.
macro_rules! guarded_role_pods {
    ([$($generics:tt)*] $payload:ty, $adapter:ty) => {
        impl<$($generics)*> RolePods<$payload> for Guarded<'_, $adapter> {
            fn propose(&mut self, call: ProposeCall<'_>) -> Option<$payload> {
                let standing = self.standing;
                let pod = call.pod.clone();
                self.experts
                    .propose_while(call, &|| still_eligible(standing, &pod))
            }

            fn repair(&mut self, call: RepairCall<'_, $payload>) -> Option<$payload> {
                let standing = self.standing;
                let pod = call.pod.clone();
                self.experts
                    .repair_while(call, &|| still_eligible(standing, &pod))
            }
        }
    };
}

guarded_role_pods!([] super::TextPayload, super::LlmExpertPods);
guarded_role_pods!([P: super::DefectPayload] P, super::SimulatedExperts<P>);
