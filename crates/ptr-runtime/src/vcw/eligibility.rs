//! One view of which Pods may be chosen and may write: the Pod registry view.
//!
//! The finding router needs a [`PodEligibility`] (which Pod generations may be routed to) and the
//! workspace needs an [`AuthorRegistry`] (which authors may write). Both come from the same
//! object here, [`AuthorStanding`] over one `&dyn PodEligibility`, so a Pod the router may not
//! choose is also a Pod the workspace refuses, and the two cannot drift apart.
//!
//! [`RuntimeStanding`] derives the view from the runtime's own state:
//!
//! * a Pod must be registered in the [`PodRegistry`] of the project (a Pod that is not there
//!   cannot be called);
//! * a Pod the ledger-backed expert projection knows is eligible only while its state is
//!   `Serving` or `Frozen`, and only at the generation the projection holds (a `Fenced`,
//!   `Retired`, `Superseded`, `Candidate` or `Gated` expert, or a stale generation, is not);
//! * an expert's generation is also checked against the lifecycle authority (`expert:{project}/{pod}`),
//!   which can only withdraw it;
//! * any other Pod is eligible only at the generation the host declared for it **and** only while
//!   the lifecycle authority reports exactly that generation live. A Pod the authority does not
//!   know, or a declared generation ahead of the live one, is not eligible (fail closed).
//!
//! The view reads state; nothing in this module writes it.

use std::collections::BTreeMap;

use ptr_pods::{AuthorRegistry, Bid, ExpertState, PodRegistry, Standing};
use ptr_router::PodEligibility;
use ptr_types::{Generation, PodId, ProjectId};

use crate::PtrRuntime;

/// Whether an expert in `state` may be routed to.
pub(crate) fn state_is_routable(state: ExpertState) -> bool {
    match state {
        ExpertState::Serving | ExpertState::Frozen => true,
        ExpertState::Candidate
        | ExpertState::Gated
        | ExpertState::Fenced
        | ExpertState::Superseded
        | ExpertState::Retired(_) => false,
    }
}

/// The eligibility view of a runtime and its Pod registry for one project.
pub struct RuntimeStanding<'a> {
    runtime: &'a PtrRuntime,
    pods: &'a PodRegistry,
    project: &'a ProjectId,
    base: BTreeMap<PodId, Generation>,
}

impl<'a> RuntimeStanding<'a> {
    pub fn new(runtime: &'a PtrRuntime, pods: &'a PodRegistry, project: &'a ProjectId) -> Self {
        Self {
            runtime,
            pods,
            project,
            base: BTreeMap::new(),
        }
    }

    /// Declare a Pod that is not an expert (a generator, say) at its current generation. The
    /// lifecycle authority can still revoke it.
    #[must_use]
    pub fn with_base_pod(mut self, pod: PodId, generation: Generation) -> Self {
        self.base.insert(pod, generation);
        self
    }
}

impl PodEligibility for RuntimeStanding<'_> {
    fn current_generation(&self, pod: &PodId) -> Option<Generation> {
        self.pods.get(self.project, pod)?;
        if let Some(entry) = self.runtime.expert_registry().get(self.project, pod) {
            if !state_is_routable(entry.state) {
                return None;
            }
            // The expert's own lifecycle is the projection; the lifecycle authority can only
            // take it away (revoked, superseded, disputed). An authority that does not know the
            // expert says nothing.
            let target = format!("expert:{}/{}", self.project.0, pod.0);
            return match self.runtime.generation_validity(&target, entry.generation) {
                None | Some(ptr_types::Validity::Live) => Some(entry.generation),
                Some(_) => None,
            };
        }
        // Any other Pod must be live in the lifecycle authority at exactly the declared
        // generation. Unknown to the authority, or ahead of it, is not eligible.
        let generation = self.base.get(pod).copied()?;
        match self.runtime.generation_validity(&pod.0, generation) {
            Some(ptr_types::Validity::Live) => Some(generation),
            Some(_) | None => None,
        }
    }
}

/// The workspace's [`AuthorRegistry`] over the same [`PodEligibility`] the router uses.
///
/// An author is `Current` exactly when the view lists it at the generation it wrote with.
pub struct AuthorStanding<'a> {
    view: &'a dyn PodEligibility,
    bid_ceiling: Bid,
}

impl<'a> AuthorStanding<'a> {
    pub fn new(view: &'a dyn PodEligibility, bid_ceiling: Bid) -> Self {
        Self { view, bid_ceiling }
    }
}

impl AuthorRegistry for AuthorStanding<'_> {
    fn standing(&self, pod: &PodId, generation: Generation) -> Standing {
        if self.view.current_generation(pod) == Some(generation) {
            Standing::Current
        } else {
            Standing::Revoked
        }
    }

    fn bid_ceiling(&self, _pod: &PodId) -> Bid {
        self.bid_ceiling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_pods::RetiredReason;

    #[test]
    fn only_serving_and_frozen_experts_are_routable() {
        let cases = [
            (ExpertState::Candidate, false),
            (ExpertState::Gated, false),
            (ExpertState::Serving, true),
            (ExpertState::Frozen, true),
            (ExpertState::Fenced, false),
            (ExpertState::Superseded, false),
            (ExpertState::Retired(RetiredReason::InputRevoked), false),
            (ExpertState::Retired(RetiredReason::Pruned), false),
            (ExpertState::Retired(RetiredReason::Superseded), false),
            (ExpertState::Retired(RetiredReason::GateFailed), false),
        ];
        for (state, routable) in cases {
            assert_eq!(state_is_routable(state), routable, "{state:?}");
        }
    }
}
