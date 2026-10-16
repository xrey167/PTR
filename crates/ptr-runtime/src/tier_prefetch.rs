//! Prefetch signals: a router's output as a hint for which weight objects to warm.
//!
//! A [`PrefetchSignal`] names targets (Pods or experts) a router ranked highly. It can only
//! feed *residency planning*: [`plan_prefetch`] reads the controller (it takes `&` and calls
//! read-only methods), keeps the targets that the caller's own [`PrefetchEligibility`] still
//! allows, resolves them to objects of the signal's project, and lets a
//! [`TierPlacementPolicy`] (for example `LookaheadPrefetchPolicy`) choose among them. The
//! result is a [`PrefetchPlan`]: a list of objects to copy. It carries no authority.
//!
//! * Eligibility is never taken from the signal: a target the eligibility source rejects is
//!   dropped, whatever the router ranked.
//! * A plan admits nothing and evicts nothing. Moving an object still needs a [`TierGrant`]
//!   of the object's project (see [`TierResidencyController::execute_prefetch_plan`]); an
//!   object without a grant is skipped with a typed refusal.
//! * The signal changes no gate, threshold, verifier or eligibility (docs/INVARIANTS.md,
//!   AGENTS.md section 3).

use crate::tier::{
    PrefetchOutcome, TierPlacementPolicy, TierPlanningContext, TierPolicyError,
    TierResidencyController, TierRuntimeError,
};
use crate::tier_grant::TierGrant;
use ptr_router::{FindingRouteRecord, PodEligibility, RouteDecision};
use ptr_storage::StorageTier;
use ptr_types::{Digest, PodId, ProjectId, SessionId};
use std::collections::{BTreeMap, BTreeSet};

/// The most targets one signal can name.
pub const MAX_PREFETCH_TARGETS: usize = 256;

/// The most entries one [`PrefetchObjectIndex`] can hold.
pub const MAX_PREFETCH_INDEX_ENTRIES: usize = 65_536;

/// What a router can name before a Pod is called.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PrefetchTarget {
    /// All weight and adapter objects of one Pod.
    Pod(PodId),
    /// One expert of one model artifact.
    Expert {
        artifact_digest: Digest,
        expert_id: u32,
    },
}

/// Where a signal came from. Informational: it does not change how the signal is treated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefetchSignalSource {
    /// The top-k of a route decision.
    RouterTopK,
    /// The eligible set a `FindingRouter` sampled from.
    FindingEligibleSet,
    /// An expert ranking supplied by a caller.
    ExpertRanking,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefetchSignalError {
    TooManyTargets,
    EmptyProject,
}

/// An ordered, bounded, duplicate-free list of targets, most wanted first, for one project.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefetchSignal {
    source: PrefetchSignalSource,
    project: ProjectId,
    targets: Vec<PrefetchTarget>,
}

impl PrefetchSignal {
    pub fn new(
        source: PrefetchSignalSource,
        project: ProjectId,
        targets: impl IntoIterator<Item = PrefetchTarget>,
    ) -> Result<Self, PrefetchSignalError> {
        if project.0.trim().is_empty() {
            return Err(PrefetchSignalError::EmptyProject);
        }
        let mut seen = BTreeSet::new();
        let mut ordered = Vec::new();
        for target in targets {
            if seen.insert(target.clone()) {
                if ordered.len() == MAX_PREFETCH_TARGETS {
                    return Err(PrefetchSignalError::TooManyTargets);
                }
                ordered.push(target);
            }
        }
        Ok(Self {
            source,
            project,
            targets: ordered,
        })
    }

    /// The first `k` candidates of a route decision, best score first (ties by target name).
    pub fn from_route_decision(
        project: ProjectId,
        decision: &RouteDecision,
        k: usize,
    ) -> Result<Self, PrefetchSignalError> {
        let mut ranked: Vec<_> = decision.candidates.iter().collect();
        ranked.sort_by(|a, b| {
            b.score
                .get()
                .total_cmp(&a.score.get())
                .then_with(|| a.target.cmp(&b.target))
        });
        Self::new(
            PrefetchSignalSource::RouterTopK,
            project,
            ranked
                .into_iter()
                .take(k)
                .map(|score| PrefetchTarget::Pod(PodId::from(score.target.as_str()))),
        )
    }

    /// The eligible set a `FindingRouter` sampled from, widest interval first (ties by Pod
    /// id), with the chosen Pod first. The router already restricted its support to Pods its
    /// eligibility view admitted; [`plan_prefetch`] asks the eligibility source again.
    pub fn from_finding_route(
        project: ProjectId,
        record: &FindingRouteRecord,
    ) -> Result<Self, PrefetchSignalError> {
        let mut support: Vec<_> = record.support().to_vec();
        support.sort_by(|(left_pod, left_width), (right_pod, right_width)| {
            right_width
                .cmp(left_width)
                .then_with(|| left_pod.cmp(right_pod))
        });
        Self::new(
            PrefetchSignalSource::FindingEligibleSet,
            project,
            std::iter::once(record.pod().clone())
                .chain(support.into_iter().map(|(pod, _)| pod))
                .map(PrefetchTarget::Pod),
        )
    }

    /// Experts of one model artifact in ranked order.
    pub fn from_expert_ranking(
        project: ProjectId,
        artifact_digest: Digest,
        expert_ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self, PrefetchSignalError> {
        Self::new(
            PrefetchSignalSource::ExpertRanking,
            project,
            expert_ids
                .into_iter()
                .map(|expert_id| PrefetchTarget::Expert {
                    artifact_digest,
                    expert_id,
                }),
        )
    }

    pub fn source(&self) -> PrefetchSignalSource {
        self.source
    }

    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn targets(&self) -> &[PrefetchTarget] {
        &self.targets
    }
}

/// The authoritative answer to "may this target be used for this project right now". Supplied
/// by the caller (registry, security); never derived from a signal.
pub trait PrefetchEligibility {
    fn is_eligible(&self, project: &ProjectId, target: &PrefetchTarget) -> bool;
}

/// Pod targets are eligible when a router's `PodEligibility` view reports a current
/// generation; expert targets are not eligible through this view.
pub struct PodEligibilityGate<'a>(pub &'a dyn PodEligibility);

impl PrefetchEligibility for PodEligibilityGate<'_> {
    fn is_eligible(&self, _project: &ProjectId, target: &PrefetchTarget) -> bool {
        match target {
            PrefetchTarget::Pod(pod) => self.0.current_generation(pod).is_some(),
            PrefetchTarget::Expert { .. } => false,
        }
    }
}

/// An explicit allow-list of (project, target) pairs.
#[derive(Clone, Debug, Default)]
pub struct EligibleTargets {
    allowed: BTreeSet<(ProjectId, PrefetchTarget)>,
}

impl EligibleTargets {
    pub fn allow(&mut self, project: ProjectId, target: PrefetchTarget) {
        self.allowed.insert((project, target));
    }
}

impl PrefetchEligibility for EligibleTargets {
    fn is_eligible(&self, project: &ProjectId, target: &PrefetchTarget) -> bool {
        self.allowed.contains(&(project.clone(), target.clone()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefetchIndexError {
    Full,
}

/// Which tier objects belong to a target of a project (registered by whoever registered the
/// objects; a signal cannot add to it).
#[derive(Clone, Debug, Default)]
pub struct PrefetchObjectIndex {
    objects: BTreeMap<(ProjectId, PrefetchTarget), Vec<Digest>>,
    entries: usize,
}

impl PrefetchObjectIndex {
    pub fn insert(
        &mut self,
        project: ProjectId,
        target: PrefetchTarget,
        object: Digest,
    ) -> Result<(), PrefetchIndexError> {
        let objects = self.objects.entry((project, target)).or_default();
        if objects.contains(&object) {
            return Ok(());
        }
        if self.entries >= MAX_PREFETCH_INDEX_ENTRIES {
            return Err(PrefetchIndexError::Full);
        }
        objects.push(object);
        self.entries += 1;
        Ok(())
    }

    fn objects(&self, project: &ProjectId, target: &PrefetchTarget) -> &[Digest] {
        self.objects
            .get(&(project.clone(), target.clone()))
            .map_or(&[], Vec::as_slice)
    }
}

/// Why a signalled target is not in the plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefetchDropReason {
    /// The eligibility source does not allow the target.
    Ineligible,
    /// The index holds no object for the target in the signal's project.
    Unresolved,
    /// The object is not registered with the controller, or is registered under another
    /// project (or none).
    NotInNamespace,
    /// The object already has an available replica on the destination tier.
    AlreadyResident,
    /// The policy did not choose it (for example past `max_objects`).
    NotChosen,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefetchDrop {
    pub target: PrefetchTarget,
    pub object: Option<Digest>,
    pub reason: PrefetchDropReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedPrefetch {
    pub target: PrefetchTarget,
    pub object: Digest,
}

/// Objects to copy, nothing else: no eviction, no admission, no grant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefetchPlan {
    pub project: ProjectId,
    pub destinations: Vec<StorageTier>,
    pub items: Vec<PlannedPrefetch>,
    pub dropped: Vec<PrefetchDrop>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefetchPlanError {
    Controller(TierRuntimeError),
    Policy(TierPolicyError),
}

/// What `plan_prefetch` reads.
pub struct PrefetchPlanRequest<'a> {
    pub signal: &'a PrefetchSignal,
    pub index: &'a PrefetchObjectIndex,
    pub eligibility: &'a dyn PrefetchEligibility,
    pub controller: &'a TierResidencyController,
    pub policy: &'a dyn TierPlacementPolicy,
    /// Tiers a prefetched object should reach; a target whose object already has an available
    /// replica on one of them is not planned.
    pub requested_tiers: Vec<StorageTier>,
}

/// Plans residency from a signal. Reads the controller only; changes nothing in it.
pub fn plan_prefetch(request: &PrefetchPlanRequest<'_>) -> Result<PrefetchPlan, PrefetchPlanError> {
    let project = request.signal.project();
    let mut dropped = Vec::new();
    let mut candidates: Vec<PlannedPrefetch> = Vec::new();
    let mut seen_objects = BTreeSet::new();
    for target in request.signal.targets() {
        if !request.eligibility.is_eligible(project, target) {
            dropped.push(PrefetchDrop {
                target: target.clone(),
                object: None,
                reason: PrefetchDropReason::Ineligible,
            });
            continue;
        }
        let objects = request.index.objects(project, target);
        if objects.is_empty() {
            dropped.push(PrefetchDrop {
                target: target.clone(),
                object: None,
                reason: PrefetchDropReason::Unresolved,
            });
            continue;
        }
        for object in objects {
            let reason = match classify_object(request, project, object)? {
                Some(reason) => reason,
                None => {
                    if seen_objects.insert(*object) {
                        candidates.push(PlannedPrefetch {
                            target: target.clone(),
                            object: *object,
                        });
                    }
                    continue;
                }
            };
            dropped.push(PrefetchDrop {
                target: target.clone(),
                object: Some(*object),
                reason,
            });
        }
    }
    let context = TierPlanningContext {
        requested_tiers: request.requested_tiers.clone(),
        lookahead: candidates.iter().map(|item| item.object).collect(),
        eviction_candidates: Vec::new(),
        available_replicas: 0,
    };
    let plan = request
        .policy
        .plan(&context)
        .map_err(PrefetchPlanError::Policy)?;
    // Only objects the eligible candidates offered can be chosen, whatever the policy returns;
    // the policy's eviction choice is ignored.
    let chosen: BTreeSet<Digest> = plan.prefetch.iter().copied().collect();
    let mut items = Vec::new();
    for candidate in candidates {
        if chosen.contains(&candidate.object) {
            items.push(candidate);
        } else {
            dropped.push(PrefetchDrop {
                target: candidate.target,
                object: Some(candidate.object),
                reason: PrefetchDropReason::NotChosen,
            });
        }
    }
    Ok(PrefetchPlan {
        project: project.clone(),
        destinations: plan.destinations,
        items,
        dropped,
    })
}

/// `None` when the object may be planned, otherwise why not.
fn classify_object(
    request: &PrefetchPlanRequest<'_>,
    project: &ProjectId,
    object: &Digest,
) -> Result<Option<PrefetchDropReason>, PrefetchPlanError> {
    match request.controller.object_project(object) {
        Ok(Some(owner)) if owner == *project => {}
        Ok(_) | Err(TierRuntimeError::UnknownObject) => {
            return Ok(Some(PrefetchDropReason::NotInNamespace))
        }
        Err(error) => return Err(PrefetchPlanError::Controller(error)),
    }
    let record = request
        .controller
        .residency(object)
        .map_err(PrefetchPlanError::Controller)?;
    let resident = record.replicas.iter().any(|replica| {
        replica.state == crate::tier::ReplicaState::Available
            && request.requested_tiers.contains(&replica.tier)
    });
    Ok(resident.then_some(PrefetchDropReason::AlreadyResident))
}

impl TierResidencyController {
    /// Runs a plan with the grants an authority issued for it. An item without a grant for
    /// its object is skipped with `GrantRequired`; the plan itself authorizes nothing. Items
    /// are processed in order, one transfer at a time.
    pub async fn execute_prefetch_plan(
        &self,
        plan: &PrefetchPlan,
        grants: &[TierGrant],
        presenter: &SessionId,
    ) -> Vec<(Digest, PrefetchOutcome)> {
        let mut outcomes = Vec::with_capacity(plan.items.len());
        for item in &plan.items {
            let grant = grants.iter().find(|grant| grant.object == item.object);
            let outcome = match grant {
                Some(grant) => self.prefetch_with_grant(grant, presenter).await,
                None => PrefetchOutcome::Skipped(TierRuntimeError::GrantRequired),
            };
            outcomes.push((item.object, outcome));
        }
        outcomes
    }
}
