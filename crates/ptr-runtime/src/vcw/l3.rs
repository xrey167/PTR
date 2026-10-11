//! L3: verifier statistics to structural requests, and the way a committed structure reaches the
//! router again.
//!
//! The loop counts, per observation window, which Pod was routed which finding code and whether
//! the repaired candidate then passed, how often each Pod was used, and how often a code had no
//! eligible repair Pod (taken from the router's own counters). [`VcwLoop::structure_step`] turns
//! the closed windows and a read-only view of the expert projection into a
//! [`StructureObservation`] and runs the structure policy. It returns the policy's *requests*
//! to the host:
//!
//! * the signature takes an [`ExpertRegistryProjection`] by reference, never the runtime, so the
//!   loop cannot append, fence or admit anything;
//! * a request that creates an expert is not an `ExpertProposal` until the host completes it
//!   with a trained manifest digest, binds a gate report to the completed proposal and calls
//!   `PtrRuntime::admit_expert_proposal` itself;
//! * after the commit, [`VcwLoop::sync_roster`] enrols the born expert for its codes, but only if
//!   the projection shows it routable, so a request that was refused or never completed changes
//!   nothing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use ptr_pods::{ExpertProposal, ExpertState};
use ptr_router::{
    AdmissibleScorer, FindingCode, FindingRouteRecord, RegistryChange, RegistryError,
    RegistryUpdate, RouteClass,
};
use ptr_types::{CommitIndex, PodId, ProjectId};

use super::eligibility::state_is_routable;
use super::{VcwError, VcwLoop};
use crate::expert::{
    BirthContext, CodeWindowStats, ExpertRegistryProjection, ObservationLimits, PodCodeStats,
    RegisteredExpert, StructureObservation, StructureProposal, StructureRationale, StructureReport,
    SubspaceOverlap, WindowObservation,
};
use ptr_pods::ProposerRef;

use super::closed_loop::MAX_ROSTER;

/// What one request contributes to the current window.
#[derive(Default)]
pub(crate) struct RequestTally {
    /// (code, repair Pod) to (routed, cleared).
    repair: BTreeMap<(FindingCode, PodId), (u64, u64)>,
    usage: BTreeMap<PodId, u64>,
    pub(super) unroutable: BTreeMap<FindingCode, u64>,
    pub(super) untracked: u64,
}

impl RequestTally {
    /// Count the routed Pods of one attempt. `cleared[i]` is whether the verifier's verdict on
    /// the repaired candidate shows that `records[i]` cleared the code it was routed (see
    /// `VcwLoop::judge`): the whole candidate is not the unit, because a code that was repaired
    /// is not "uncleared" merely because another code of the same candidate is still defective.
    pub(super) fn note(&mut self, records: &[FindingRouteRecord], cleared: &[bool]) {
        for (index, record) in records.iter().enumerate() {
            let cleared = cleared.get(index).copied().unwrap_or(false);
            let used = self.usage.entry(record.pod().clone()).or_default();
            *used = used.saturating_add(1);
            if let RouteClass::Repair(code) = record.class() {
                let entry = self
                    .repair
                    .entry((code.clone(), record.pod().clone()))
                    .or_default();
                entry.0 = entry.0.saturating_add(1);
                entry.1 = entry.1.saturating_add(u64::from(cleared));
            }
        }
    }
}

/// The current window and the ring of closed ones. Bounded by the policy's observation limits:
/// anything that would not fit is counted in [`WindowRecorder::overflow`], never stored.
pub(crate) struct WindowRecorder {
    window_requests: u32,
    max_windows: usize,
    limits: ObservationLimits,
    current: WindowObservation,
    in_window: u32,
    closed: VecDeque<WindowObservation>,
    overflow: u64,
}

impl WindowRecorder {
    pub(super) fn new(
        window_requests: u32,
        max_windows: usize,
        limits: &ObservationLimits,
    ) -> Self {
        Self {
            window_requests,
            max_windows,
            limits: *limits,
            current: WindowObservation::default(),
            in_window: 0,
            closed: VecDeque::new(),
            overflow: 0,
        }
    }

    pub(super) fn closed(&self) -> &VecDeque<WindowObservation> {
        &self.closed
    }

    pub(super) fn overflow(&self) -> u64 {
        self.overflow
    }

    /// Add one request. Returns whether it closed the window.
    pub(super) fn record(&mut self, tally: RequestTally) -> bool {
        let max_pods = self.limits.max_pods_per_code;
        for ((code, pod), (routed, cleared)) in tally.repair {
            match self.code_slot(&code) {
                Some(stats) => {
                    if stats.pods.contains_key(&pod) || stats.pods.len() < max_pods {
                        let entry = stats.pods.entry(pod).or_insert(PodCodeStats::default());
                        entry.routed = entry.routed.saturating_add(routed);
                        entry.cleared = entry.cleared.saturating_add(cleared);
                    } else {
                        self.overflow = self.overflow.saturating_add(1);
                    }
                }
                None => self.overflow = self.overflow.saturating_add(1),
            }
        }
        for (code, count) in tally.unroutable {
            match self.code_slot(&code) {
                Some(stats) => stats.unroutable = stats.unroutable.saturating_add(count),
                None => {
                    self.current.unroutable_untracked =
                        self.current.unroutable_untracked.saturating_add(count);
                }
            }
        }
        self.current.unroutable_untracked = self
            .current
            .unroutable_untracked
            .saturating_add(tally.untracked);
        for (pod, count) in tally.usage {
            if self.current.usage.contains_key(&pod)
                || self.current.usage.len() < self.limits.max_experts
            {
                let used = self.current.usage.entry(pod).or_default();
                *used = used.saturating_add(count);
            } else {
                self.overflow = self.overflow.saturating_add(1);
            }
        }
        self.in_window += 1;
        if self.in_window < self.window_requests {
            return false;
        }
        if self.closed.len() >= self.max_windows {
            self.closed.pop_front();
        }
        self.closed.push_back(std::mem::take(&mut self.current));
        self.in_window = 0;
        true
    }

    fn code_slot(&mut self, code: &FindingCode) -> Option<&mut CodeWindowStats> {
        if !self.current.codes.contains_key(code)
            && self.current.codes.len() >= self.limits.max_codes_per_window
        {
            return None;
        }
        Some(self.current.codes.entry(code.clone()).or_default())
    }
}

/// An expert-creating request the host has been handed and has not yet resolved.
struct Outstanding {
    /// The operation id the request was handed out under.
    key: [u8; 32],
    /// The loop's closed-window count when the request was handed out.
    created_window: u64,
    /// The experts the request would create, with the codes each repairs.
    results: Vec<(PodId, BTreeSet<FindingCode>)>,
    /// The codes that keep the policy from proposing the same Birth again.
    holds: BTreeSet<FindingCode>,
}

/// Requests handed out and not yet resolved, by operation id; bounded.
pub(crate) struct OutstandingRequests {
    max: usize,
    by_operation: BTreeMap<[u8; 32], Outstanding>,
    /// Experts the ledger committed whose install the host abandoned: not routable, not in the
    /// roster. Their codes stay held so the policy neither sees them as uncovered nor births a
    /// second expert for them, until the projection shows the expert gone.
    stranded: BTreeMap<PodId, BTreeSet<FindingCode>>,
}

impl OutstandingRequests {
    pub(super) fn new(max: usize) -> Self {
        Self {
            max,
            by_operation: BTreeMap::new(),
            stranded: BTreeMap::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.by_operation.len()
    }

    /// Remove and return the entries handed out `ttl` or more windows ago.
    fn take_expired(&mut self, now: u64, ttl: u64) -> Vec<Outstanding> {
        let keys: Vec<[u8; 32]> = self
            .by_operation
            .iter()
            .filter(|(_, entry)| now.saturating_sub(entry.created_window) >= ttl)
            .map(|(key, _)| *key)
            .collect();
        keys.iter()
            .filter_map(|key| self.by_operation.remove(key))
            .collect()
    }

    fn holds(&self) -> BTreeSet<FindingCode> {
        self.by_operation
            .values()
            .flat_map(|entry| entry.holds.iter().cloned())
            .chain(self.stranded.values().flatten().cloned())
            .collect()
    }
}

/// The read-only inputs of one structure step that the loop cannot know.
pub struct StructureHost<'a> {
    pub project: ProjectId,
    pub proposer: ProposerRef,
    /// The commit index the registry view was read at.
    pub observed_at: CommitIndex,
    /// The expert projection as of `observed_at`. A reference to a projection, not to the
    /// runtime: nothing here can write.
    pub registry: &'a ExpertRegistryProjection,
    /// Needed for any Birth; without it none is proposed.
    pub birth: Option<BirthContext>,
    /// AIMER score per expert, computed by the host from the lineage crate.
    pub aimer: BTreeMap<PodId, f64>,
    pub overlaps: Vec<SubspaceOverlap>,
}

/// What [`VcwLoop::sync_roster`] enrolled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RosterSync {
    pub enrolled: Vec<(PodId, Vec<FindingCode>)>,
    /// Experts removed because the projection shows them retired or superseded.
    pub removed: Vec<PodId>,
}

impl<S: AdmissibleScorer> VcwLoop<S> {
    /// Closed observation windows, oldest first.
    pub fn closed_windows(&self) -> impl Iterator<Item = &WindowObservation> {
        self.windows.closed().iter()
    }

    /// Tallies that did not fit the observation limits and were dropped (counted, not stored).
    pub fn window_overflow(&self) -> u64 {
        self.windows.overflow()
    }

    /// Requests handed to the host and not yet resolved.
    pub fn outstanding_requests(&self) -> usize {
        self.outstanding.len()
    }

    /// Windows an unresolved request may stay outstanding before the loop drops it.
    pub fn outstanding_ttl_windows(&self) -> u32 {
        self.config.outstanding_ttl_windows
    }

    /// Whether the request that `proposal` completes is still outstanding in this loop.
    pub fn is_outstanding(&self, proposal: &ExpertProposal) -> bool {
        self.outstanding
            .by_operation
            .contains_key(proposal.operation_id().as_bytes())
    }

    /// Committed experts the host abandoned installing, with the codes each was born for. Their
    /// codes are held (no second Birth) until the projection shows the expert gone.
    pub fn stranded_experts(&self) -> &BTreeMap<PodId, BTreeSet<FindingCode>> {
        &self.outstanding.stranded
    }

    /// The host gives up installing a committed expert (`pods`, born for `codes`): its request
    /// is forgotten and the experts are recorded as stranded, so the structure policy neither
    /// treats the codes as uncovered nor births another expert for them. The stranded record
    /// ends when the projection shows the expert retired, superseded or absent (the host retired
    /// it through `admit_expert_proposal`), at the next [`Self::structure_step`].
    ///
    /// # Errors
    /// [`VcwError::Config`] when 256 experts are already stranded; then nothing changed.
    pub fn strand_committed(
        &mut self,
        committed: &ExpertProposal,
        pods: &[PodId],
        codes: &BTreeSet<FindingCode>,
    ) -> Result<(), VcwError> {
        let fresh = pods
            .iter()
            .filter(|pod| !self.outstanding.stranded.contains_key(*pod))
            .count();
        if self.outstanding.stranded.len() + fresh > MAX_ROSTER {
            return Err(VcwError::Config("more than 256 stranded experts"));
        }
        for pod in pods {
            self.outstanding
                .stranded
                .entry(pod.clone())
                .or_default()
                .extend(codes.iter().cloned());
        }
        self.outstanding
            .by_operation
            .remove(committed.operation_id().as_bytes());
        Ok(())
    }

    /// Run the structure policy on the closed windows. The returned requests are inert: the loop
    /// admits nothing and has no handle that could.
    ///
    /// # Errors
    /// [`VcwError`] when no window has closed, the observation is refused by the policy's limits,
    /// or the requests would not fit the outstanding table (then nothing is recorded).
    pub fn structure_step(&mut self, host: StructureHost<'_>) -> Result<StructureReport, VcwError> {
        if self.windows.closed().is_empty() {
            return Err(VcwError::NoClosedWindow);
        }
        // A stranded expert the projection shows retired, superseded or absent is resolved. A
        // fenced one may be readmitted, so it stays held.
        self.outstanding.stranded.retain(|pod, _| {
            host.registry.get(&host.project, pod).is_some_and(|expert| {
                !matches!(
                    expert.state,
                    ExpertState::Retired(_) | ExpertState::Superseded
                )
            })
        });
        // An entry that outlived its window budget is dropped, unless the host did commit its
        // experts and only forgot to sync: those are enrolled now, so a committed expert is not
        // stranded outside the roster.
        let expired = self.outstanding.take_expired(
            self.counters.windows_closed,
            u64::from(self.config.outstanding_ttl_windows),
        );
        for entry in expired {
            let committed =
                Self::all_routable(host.registry, &host.project, &entry.results).is_ok();
            if committed {
                if let Err(error) = self.enrol(&entry.results) {
                    // Nothing was lost: keep it for the next step.
                    let key = Self::entry_key(&entry);
                    self.outstanding.by_operation.insert(key, entry);
                    return Err(error);
                }
                self.counters.outstanding_reconciled =
                    self.counters.outstanding_reconciled.saturating_add(1);
            } else {
                self.counters.outstanding_expired =
                    self.counters.outstanding_expired.saturating_add(1);
            }
        }
        let experts: Vec<RegisteredExpert> = self
            .repairers
            .iter()
            .filter_map(|(pod, codes)| {
                RegisteredExpert::from_projection(host.registry, &host.project, pod, codes.clone())
            })
            .collect();
        let observation = StructureObservation {
            project: host.project,
            proposer: host.proposer,
            observed_at: host.observed_at,
            windows: self.windows.closed().iter().cloned().collect(),
            experts,
            aimer: host.aimer,
            overlaps: host.overlaps,
            birth: host.birth,
            outstanding: self.outstanding.holds(),
        };
        let report = self
            .structure
            .propose(&observation)
            .map_err(VcwError::Structure)?;
        let mut pending: Vec<([u8; 32], Outstanding)> = Vec::new();
        for proposal in report.proposals() {
            if let Some(entry) = self.outstanding_entry(proposal) {
                pending.push((*proposal.request().operation_id().as_bytes(), entry));
            }
        }
        let fresh = pending
            .iter()
            .filter(|(op, _)| !self.outstanding.by_operation.contains_key(op))
            .count();
        if self.outstanding.len() + fresh > self.outstanding.max {
            self.counters.requests_refused_outstanding_full = self
                .counters
                .requests_refused_outstanding_full
                .saturating_add(1);
            return Err(VcwError::OutstandingFull);
        }
        for (operation, entry) in pending {
            self.outstanding.by_operation.insert(operation, entry);
        }
        Ok(report)
    }

    /// The outstanding record of a request that creates an expert.
    fn outstanding_entry(&self, proposal: &StructureProposal) -> Option<Outstanding> {
        if !proposal.needs_training() {
            return None;
        }
        let codes_of = |pod: &PodId| self.repairers.get(pod).cloned().unwrap_or_default();
        let results: Vec<(PodId, BTreeSet<FindingCode>)> = match proposal.rationale() {
            StructureRationale::Birth { code, .. } => proposal
                .request()
                .op()
                .results()
                .iter()
                .map(|r| (r.pod().clone(), BTreeSet::from([code.clone()])))
                .collect(),
            StructureRationale::Reactivate { pod, code, .. } => {
                let mut codes = codes_of(pod);
                codes.insert(code.clone());
                proposal
                    .request()
                    .op()
                    .results()
                    .iter()
                    .map(|r| (r.pod().clone(), codes.clone()))
                    .collect()
            }
            StructureRationale::Merge { first, second, .. } => {
                let mut codes = codes_of(first);
                codes.extend(codes_of(second));
                proposal
                    .request()
                    .op()
                    .results()
                    .iter()
                    .map(|r| (r.pod().clone(), codes.clone()))
                    .collect()
            }
            StructureRationale::Prune { .. } | StructureRationale::Freeze { .. } => return None,
        };
        let holds = match proposal.rationale() {
            StructureRationale::Birth { code, .. }
            | StructureRationale::Reactivate { code, .. } => BTreeSet::from([code.clone()]),
            _ => BTreeSet::new(),
        };
        Some(Outstanding {
            key: *proposal.request().operation_id().as_bytes(),
            created_window: self.counters.windows_closed,
            results,
            holds,
        })
    }

    /// Forget an unresolved request by the request the policy made, for a host that could not
    /// even complete it (training failed, the held-out suite could not run).
    pub fn withdraw_structure_request(&mut self, request: &StructureProposal) -> bool {
        self.outstanding
            .by_operation
            .remove(request.request().operation_id().as_bytes())
            .is_some()
    }

    /// Forget an unresolved request (the host dropped it or its gate failed).
    pub fn withdraw_request(&mut self, proposal: &ExpertProposal) -> bool {
        self.outstanding
            .by_operation
            .remove(proposal.operation_id().as_bytes())
            .is_some()
    }

    /// After the host admitted `committed` (through `PtrRuntime::admit_expert_proposal`), enrol
    /// the experts it created for the codes the request named.
    ///
    /// The experts must be in `registry` in a routable state, so a proposal the ledger did not
    /// commit enrols nothing. This changes the router's registry only; whether a Pod can then be
    /// chosen is still decided by the eligibility view of each request.
    ///
    /// # Errors
    /// [`VcwError`] when the operation was not handed out by this loop, an expert is not
    /// committed and routable, or the router's registry is full.
    pub fn sync_roster(
        &mut self,
        project: &ProjectId,
        registry: &ExpertRegistryProjection,
        committed: &ExpertProposal,
    ) -> Result<RosterSync, VcwError> {
        let key = *committed.operation_id().as_bytes();
        let results = self
            .outstanding
            .by_operation
            .get(&key)
            .ok_or(VcwError::UnknownOperation)?
            .results
            .clone();
        Self::all_routable(registry, project, &results)?;
        // Removal first: it is idempotent, so a failure here changes nothing a retry cannot
        // redo, and the outstanding entry is still there. It also frees the roster slots of
        // retired experts, so the capacity check comes after it: a host that pruned an expert
        // to make room can retry a failed enrolment. Enrolment and the removal of the entry
        // then follow with no fallible step in between that could lose the entry.
        let removed = self.retire_dead_experts(project, registry)?;
        for (pod, _) in &results {
            if !self.repairers.contains_key(pod) && self.repairers.len() >= MAX_ROSTER {
                return Err(VcwError::Config("more than 256 repair experts"));
            }
        }
        let enrolled = self.enrol(&results)?;
        self.outstanding.by_operation.remove(&key);
        Ok(RosterSync { enrolled, removed })
    }

    fn entry_key(entry: &Outstanding) -> [u8; 32] {
        entry.key
    }

    /// Every created expert is in the projection in a routable state.
    fn all_routable(
        registry: &ExpertRegistryProjection,
        project: &ProjectId,
        results: &[(PodId, BTreeSet<FindingCode>)],
    ) -> Result<(), VcwError> {
        for (pod, _) in results {
            let routable = registry
                .get(project, pod)
                .is_some_and(|expert| state_is_routable(expert.state));
            if !routable {
                return Err(VcwError::NotCommitted { pod: pod.clone() });
            }
        }
        Ok(())
    }

    /// Register the experts for their codes in the router and the roster. Re-registration of a
    /// pair is a no-op.
    fn enrol(
        &mut self,
        results: &[(PodId, BTreeSet<FindingCode>)],
    ) -> Result<Vec<(PodId, Vec<FindingCode>)>, VcwError> {
        let mut enrolled = Vec::new();
        for (pod, codes) in results {
            if !self.repairers.contains_key(pod) && self.repairers.len() >= MAX_ROSTER {
                return Err(VcwError::Config("more than 256 repair experts"));
            }
            let mut added = Vec::new();
            for code in codes {
                let update = RegistryUpdate {
                    expected: self.router.registry().generation(),
                    change: RegistryChange::AddRepair {
                        code: code.clone(),
                        pod: pod.clone(),
                    },
                };
                match self.router.update_registry(update) {
                    Ok(_) => added.push(code.clone()),
                    Err(RegistryError::AlreadyRegistered) => {}
                    Err(error) => return Err(VcwError::Registry(error)),
                }
                self.repairers
                    .entry(pod.clone())
                    .or_default()
                    .insert(code.clone());
            }
            enrolled.push((pod.clone(), added));
        }
        Ok(enrolled)
    }

    /// Take every expert the projection shows as `Retired` or `Superseded` (terminal states) out
    /// of the router's registry, so it can no longer be routed to, and out of the roster, except
    /// an expert retired for `GateFailed`: that is the one state the structure policy can
    /// reactivate, and it needs the expert in its view, so it stays in the roster (with its codes)
    /// and only leaves routing. A `Fenced` expert stays everywhere: it can be readmitted, and the
    /// eligibility view already keeps it from being chosen. Returns the Pods taken out of routing.
    ///
    /// # Errors
    /// [`VcwError::Registry`] if the router's registry refuses a removal.
    pub fn retire_dead_experts(
        &mut self,
        project: &ProjectId,
        registry: &ExpertRegistryProjection,
    ) -> Result<Vec<PodId>, VcwError> {
        #[cfg(test)]
        if self.fault.retire {
            return Err(VcwError::Registry(RegistryError::GenerationExhausted));
        }
        let dead: Vec<PodId> = self
            .repairers
            .keys()
            .filter(|pod| {
                registry.get(project, pod).is_some_and(|expert| {
                    matches!(
                        expert.state,
                        ExpertState::Retired(_) | ExpertState::Superseded
                    )
                })
            })
            .cloned()
            .collect();
        let mut removed = Vec::new();
        for pod in &dead {
            let mut taken_out = false;
            let codes = self.repairers.get(pod).cloned().unwrap_or_default();
            for code in codes {
                let update = RegistryUpdate {
                    expected: self.router.registry().generation(),
                    change: RegistryChange::RemoveRepair {
                        code,
                        pod: pod.clone(),
                    },
                };
                match self.router.update_registry(update) {
                    Ok(_) => taken_out = true,
                    Err(RegistryError::NotRegistered) => {}
                    Err(error) => return Err(VcwError::Registry(error)),
                }
            }
            if taken_out {
                removed.push(pod.clone());
            }
            let reactivatable = registry.get(project, pod).is_some_and(|expert| {
                matches!(
                    expert.state,
                    ExpertState::Retired(ptr_pods::RetiredReason::GateFailed)
                )
            });
            if !reactivatable {
                self.repairers.remove(pod);
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{codes, make_loop, request, view, CodeVerifier, Proposer};
    use super::*;
    use crate::expert::{EnabledOps, ExpertEntry, StructurePolicyConfig};
    use ptr_pods::{
        BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ManifestDigest,
        ProposerRef, RationaleDigest, RetiredReason, StructuralOp,
    };
    use ptr_types::{CommitIndex, Generation};

    fn project() -> ProjectId {
        ProjectId::from("proj")
    }

    fn base() -> BaseModelRef {
        BaseModelRef::new(
            BaseModelId::new("base-a").unwrap(),
            BaseModelDigest::new([9; 32]).unwrap(),
            1,
        )
        .unwrap()
    }

    fn proposer() -> ProposerRef {
        ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap()
    }

    fn entry(state: ExpertState) -> ExpertEntry {
        ExpertEntry {
            state,
            state_version: 1,
            generation: Generation(1),
            manifest_digest: ManifestDigest::new([3; 32]).unwrap(),
            level: ExpertLevel::Segment,
            base_model: base(),
        }
    }

    fn projection(experts: &[(&str, ExpertState)]) -> ExpertRegistryProjection {
        ExpertRegistryProjection::from_entries_all_legacy(
            experts
                .iter()
                .map(|(pod, state)| (("proj".to_owned(), (*pod).to_owned()), entry(*state)))
                .collect(),
        )
    }

    fn only(ops: EnabledOps) -> StructurePolicyConfig {
        StructurePolicyConfig {
            min_unroutable: 1,
            birth_windows: 1,
            ops,
            ..StructurePolicyConfig::default()
        }
    }

    const NONE: EnabledOps = EnabledOps {
        birth: false,
        merge: false,
        prune: false,
        freeze: false,
        reactivate: false,
    };

    #[test]
    fn a_gate_failed_expert_stays_in_the_policy_view_so_it_can_be_reactivated() {
        let mut vcw = make_loop(
            1,
            &[("gf", &["x"]), ("pr", &["y"])],
            only(EnabledOps {
                reactivate: true,
                ..NONE
            }),
        );
        let registry = projection(&[
            ("gf", ExpertState::Retired(RetiredReason::GateFailed)),
            ("pr", ExpertState::Retired(RetiredReason::Pruned)),
        ]);
        let removed = vcw.retire_dead_experts(&project(), &registry).unwrap();
        assert_eq!(removed, vec![PodId::from("gf"), PodId::from("pr")]);
        // Both left routing; only the one the policy can reactivate stays in the roster.
        assert!(vcw
            .router
            .registry()
            .repair_pods(&FindingCode::from("x"))
            .is_empty());
        assert!(vcw.repairers.contains_key(&PodId::from("gf")));
        assert!(!vcw.repairers.contains_key(&PodId::from("pr")));
        // A second removal pass has nothing left to take out of routing.
        assert!(vcw
            .retire_dead_experts(&project(), &registry)
            .unwrap()
            .is_empty());

        // Code `x` has no routable repairer, so it is unroutable and the window records it.
        let run = vcw
            .run(
                &request("reactivate", 2, 2),
                &view(&["gen"]),
                &CodeVerifier,
                &mut Proposer("x"),
            )
            .unwrap();
        assert!(!matches!(
            run.outcome,
            super::super::VcwOutcome::Verified(_)
        ));
        let report = vcw
            .structure_step(StructureHost {
                project: project(),
                proposer: proposer(),
                observed_at: CommitIndex(1),
                registry: &registry,
                birth: None,
                aimer: BTreeMap::new(),
                overlaps: Vec::new(),
            })
            .unwrap();
        assert_eq!(report.count(ptr_pods::OpKind::Reactivate), 1, "{report:?}");
    }

    fn birth_proposal(born: &str) -> ExpertProposal {
        let spec = ExpertPodSpec::new(
            project(),
            PodId::from(born),
            ManifestDigest::new([3; 32]).unwrap(),
            ExpertLevel::Segment,
            base(),
            Generation(1),
        )
        .unwrap();
        ExpertProposal::new(
            project(),
            proposer(),
            CommitIndex(1),
            StructuralOp::Birth { result: spec },
            RationaleDigest::new([7; 32]).unwrap(),
        )
        .unwrap()
    }

    fn track(vcw: &mut VcwLoop, proposal: &ExpertProposal, born: &str, code: &str) {
        let key = *proposal.operation_id().as_bytes();
        vcw.outstanding.by_operation.insert(
            key,
            Outstanding {
                key,
                created_window: vcw.counters.windows_closed,
                results: vec![(PodId::from(born), codes(&[code]))],
                holds: codes(&[code]),
            },
        );
    }

    #[test]
    fn a_failed_removal_does_not_lose_the_outstanding_request_and_a_retry_enrols() {
        let mut vcw = make_loop(1, &[], only(NONE_PLUS_BIRTH));
        let proposal = birth_proposal("born");
        track(&mut vcw, &proposal, "born", "x");
        let registry = projection(&[("born", ExpertState::Serving)]);

        vcw.fault.retire = true;
        let failed = vcw.sync_roster(&project(), &registry, &proposal);
        assert!(matches!(failed, Err(VcwError::Registry(_))));
        // Nothing was enrolled and the request is still there to retry.
        assert_eq!(vcw.outstanding_requests(), 1);
        assert!(!vcw.repairers.contains_key(&PodId::from("born")));

        vcw.fault.retire = false;
        let synced = vcw.sync_roster(&project(), &registry, &proposal).unwrap();
        assert_eq!(
            synced.enrolled,
            vec![(PodId::from("born"), vec![FindingCode::from("x")])]
        );
        assert_eq!(vcw.outstanding_requests(), 0);
        assert!(vcw.repairers.contains_key(&PodId::from("born")));
        // The request is resolved: a second sync is an unknown operation, not a half state.
        assert!(matches!(
            vcw.sync_roster(&project(), &registry, &proposal),
            Err(VcwError::UnknownOperation)
        ));
    }

    const NONE_PLUS_BIRTH: EnabledOps = EnabledOps {
        birth: true,
        ..NONE
    };
}
