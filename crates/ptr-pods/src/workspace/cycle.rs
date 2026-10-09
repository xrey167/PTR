//! The two-phase cycle: plan without mutating, then commit with no fallible step.

use super::config::{Displacement, DisplacementPolicy, Revalidation, SlotDecay};
use super::gate::{check_verification_gate, level_rank};
use super::proposal::{Bid, Encoder, Proposal, SlotIndex, SubscriberId, WorkspaceProposalId};
use super::snapshot::{
    build_snapshot, records, report_digest, FindingRecord, HostFinding, Refusal, RefusalReason,
    RetireReason, RetiredRecord, SlotContent, SnapshotParts, VerifierIdentity, WorkspaceSnapshot,
    MAX_FINDINGS_PER_REFUSAL,
};
use super::{Workspace, WorkspaceError, WorkspaceRevision};
use ptr_types::{Digest, Generation, PodId, Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus, Verifier};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

const TIE_DOMAIN: &[u8] = b"ptr.pods.workspace.tie.v1";
/// Host findings returned by one cycle; the rest are counted in
/// [`CycleReport::host_findings_dropped`].
pub const MAX_HOST_FINDINGS_PER_CYCLE: usize = 4096;
/// The finding code of a verifier call that panicked.
pub const VERIFIER_PANIC_CODE: &str = "verifier-panic";

/// Whether a Pod generation may still write and listen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Standing {
    Current,
    /// Revoked or superseded (invariant 3): it neither writes, nor keeps slot content, nor stays
    /// subscribed.
    Revoked,
}

/// The host's view of Pod identity and authority. It must not panic and must be deterministic for
/// replay.
pub trait AuthorRegistry {
    fn standing(&self, pod: &PodId, generation: Generation) -> Standing;

    /// The highest bid this author may have. The effective bid is the minimum of the requested
    /// bid, this ceiling and [`WorkspaceConfig::max_bid`](super::WorkspaceConfig::max_bid): bids
    /// are host-derived, not self-declared.
    fn bid_ceiling(&self, _pod: &PodId) -> Bid {
        Bid::new(u32::MAX)
    }
}

/// A registry that accepts every generation with no ceiling. For tests and single-author hosts.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnrestrictedAuthors;

impl AuthorRegistry for UnrestrictedAuthors {
    fn standing(&self, _pod: &PodId, _generation: Generation) -> Standing {
        Standing::Current
    }
}

/// Everything the host brings to a cycle. None of it comes from a proposal.
#[derive(Clone, Copy)]
pub struct CycleHost<'a> {
    pub verifier: &'a dyn Verifier<Proposal>,
    pub identity: &'a VerifierIdentity,
    pub authors: &'a dyn AuthorRegistry,
}

/// What one cycle did.
#[derive(Clone, Debug)]
pub struct CycleReport {
    /// The snapshot every subscriber was offered; the same allocation is shared by all.
    pub snapshot: Arc<WorkspaceSnapshot>,
    /// Subscribers whose mailbox took the snapshot.
    pub delivered: Vec<SubscriberId>,
    /// Subscribers whose full mailbox refused the snapshot.
    pub overflowed: Vec<SubscriberId>,
    /// Subscriptions dropped because their Pod generation is revoked.
    pub revoked_subscribers: Vec<SubscriberId>,
    /// Finding messages, for the host only (invariant 13). Not part of any snapshot.
    pub host_findings: Vec<HostFinding>,
    pub host_findings_dropped: usize,
}

struct Plan {
    slots: Vec<Option<SlotContent>>,
    last_change: Vec<u64>,
    pinned: Vec<bool>,
    admitted: Vec<WorkspaceProposalId>,
    refusals: Vec<Refusal>,
    retired: Vec<RetiredRecord>,
    revoked_subscribers: Vec<SubscriberId>,
    host_findings: Vec<HostFinding>,
    host_findings_dropped: usize,
    panics: u64,
}

struct Candidate<'p> {
    proposal: &'p Proposal,
    level: VerificationLevel,
    bid: Bid,
    report_digest: Digest,
    findings: Vec<FindingRecord>,
    findings_dropped: usize,
}

impl Workspace {
    /// Run one cycle, judging every pending proposal and, if configured, every incumbent.
    ///
    /// Nothing in the workspace changes until the whole plan exists; the only step that can fail
    /// after planning is none. A revision overflow returns an error before any change.
    pub fn run_cycle(&mut self, host: CycleHost<'_>) -> Result<CycleReport, WorkspaceError> {
        let current = self.revision;
        let next = WorkspaceRevision(
            current
                .0
                .checked_add(1)
                .ok_or(WorkspaceError::RevisionExhausted)?,
        );
        let plan = self.plan(&host, current, next);
        Ok(self.commit(plan, next, host.identity))
    }

    fn is_stale(&self, proposal: &Proposal, current: WorkspaceRevision) -> bool {
        let behind = current.0.saturating_sub(proposal.computed_against.0);
        behind > self.config.max_staleness()
            || (behind > 0
                && self
                    .last_change
                    .iter()
                    .any(|&c| c > proposal.computed_against.0))
    }

    fn plan(
        &self,
        host: &CycleHost<'_>,
        current: WorkspaceRevision,
        next: WorkspaceRevision,
    ) -> Plan {
        let cfg = &self.config;
        let visibility = cfg.finding_visibility();
        let mut plan = Plan {
            slots: self.slots.clone(),
            last_change: self.last_change.clone(),
            pinned: self.pinned.clone(),
            admitted: Vec::new(),
            refusals: Vec::new(),
            retired: Vec::new(),
            revoked_subscribers: self
                .subscribers
                .iter()
                .filter(|(_, s)| host.authors.standing(&s.pod, s.generation) == Standing::Revoked)
                .map(|(id, _)| *id)
                .collect(),
            host_findings: Vec::new(),
            host_findings_dropped: 0,
            panics: 0,
        };

        // Retire incumbents.
        for index in 0..plan.slots.len() {
            let Some(content) = plan.slots[index].as_ref() else {
                continue;
            };
            let age = next.0.saturating_sub(content.admitted_at.0);
            let mut findings = Vec::new();
            let reason = if host
                .authors
                .standing(content.author(), content.author_generation())
                == Standing::Revoked
            {
                Some(RetireReason::AuthorRevoked)
            } else if matches!(cfg.decay(), SlotDecay::AfterCycles(n) if age >= n.get()) {
                Some(RetireReason::Aged)
            } else if self.retire_requests.contains(&SlotIndex::new(index)) {
                Some(RetireReason::Explicit)
            } else if match cfg.revalidation() {
                Revalidation::Never => false,
                Revalidation::EveryCycle => true,
                Revalidation::EveryCycles(n) => age % n.get() == 0,
            } {
                let report = judge(host.verifier, &content.origin, &mut plan.panics);
                match check_verification_gate(&report, cfg.required_level()) {
                    Ok(()) => None,
                    Err(_) => {
                        findings = records(&report.findings, visibility).0;
                        push_host(
                            &mut plan.host_findings,
                            &mut plan.host_findings_dropped,
                            content.origin.id,
                            &report.findings,
                        );
                        Some(RetireReason::Contradicted)
                    }
                }
            } else {
                None
            };
            if let Some(reason) = reason {
                let content = plan.slots[index].take().expect("checked occupied");
                plan.retired.push(RetiredRecord {
                    content,
                    reason,
                    retired_at: next,
                    findings,
                });
                plan.last_change[index] = next.0;
                plan.pinned[index] = false;
            }
        }

        // Judge pending proposals.
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for proposal in self.pending.values() {
            if self.is_stale(proposal, current) {
                plan.refusals.push(refusal(
                    proposal,
                    RefusalReason::Stale {
                        computed_against: proposal.computed_against,
                        current,
                    },
                    Vec::new(),
                    0,
                ));
                continue;
            }
            if host
                .authors
                .standing(&proposal.author, proposal.author_generation)
                == Standing::Revoked
            {
                plan.refusals.push(refusal(
                    proposal,
                    RefusalReason::AuthorRevoked,
                    Vec::new(),
                    0,
                ));
                continue;
            }
            let report = judge(host.verifier, proposal, &mut plan.panics);
            push_host(
                &mut plan.host_findings,
                &mut plan.host_findings_dropped,
                proposal.id,
                &report.findings,
            );
            let (findings, findings_dropped) = records(&report.findings, visibility);
            match check_verification_gate(&report, cfg.required_level()) {
                Ok(()) => {
                    let ceiling = host.authors.bid_ceiling(&proposal.author);
                    candidates.push(Candidate {
                        proposal,
                        level: report.level,
                        bid: proposal.bid.min(cfg.max_bid()).min(ceiling),
                        report_digest: report_digest(&report),
                        findings,
                        findings_dropped,
                    });
                }
                Err(gate) => plan.refusals.push(refusal(
                    proposal,
                    RefusalReason::Gate(gate),
                    findings,
                    findings_dropped,
                )),
            }
        }

        // Write competition.
        let seed = cfg.tie_break_seed();
        candidates.sort_by_cached_key(|c| {
            (
                Reverse(c.bid),
                tie_break(seed, next, c.proposal.id),
                c.proposal.id,
            )
        });
        let mut held: BTreeSet<Digest> =
            plan.slots.iter().flatten().map(|s| *s.content()).collect();
        let mut per_author: BTreeMap<PodId, usize> = BTreeMap::new();
        for slot in plan.slots.iter().flatten() {
            *per_author.entry(slot.author().clone()).or_default() += 1;
        }
        let free_slots = plan.slots.iter().filter(|s| s.is_none()).count();
        let mut displaced = 0usize;
        let mut displaced_by: BTreeMap<PodId, usize> = BTreeMap::new();
        for c in candidates {
            let p = c.proposal;
            if held.contains(&p.content) {
                plan.refusals.push(refusal(
                    p,
                    RefusalReason::Duplicate,
                    c.findings,
                    c.findings_dropped,
                ));
                continue;
            }
            let cap = cfg.max_slots_per_author();
            if per_author.get(&p.author).copied().unwrap_or(0) >= cap {
                plan.refusals.push(refusal(
                    p,
                    RefusalReason::AuthorSlotCap { cap },
                    c.findings,
                    c.findings_dropped,
                ));
                continue;
            }
            let index = match plan.slots.iter().position(Option::is_none) {
                Some(i) => i,
                None => {
                    let victim = match cfg.displacement() {
                        Displacement::Off => None,
                        Displacement::Policy(policy)
                            if displaced < policy.max_per_cycle
                                && displaced_by.get(&p.author).copied().unwrap_or(0)
                                    < policy.max_per_author =>
                        {
                            pick_victim(&plan, &policy, next, &c)
                        }
                        Displacement::Policy(_) => None,
                    };
                    let Some(i) = victim else {
                        plan.refusals.push(refusal(
                            p,
                            RefusalReason::Outbid { free_slots },
                            c.findings,
                            c.findings_dropped,
                        ));
                        continue;
                    };
                    let old = plan.slots[i].take().expect("victim is occupied");
                    held.remove(old.content());
                    if let Some(n) = per_author.get_mut(old.author()) {
                        *n -= 1;
                    }
                    plan.retired.push(RetiredRecord {
                        content: old,
                        reason: RetireReason::Displaced { by: p.id },
                        retired_at: next,
                        findings: Vec::new(),
                    });
                    displaced += 1;
                    *displaced_by.entry(p.author.clone()).or_default() += 1;
                    i
                }
            };
            held.insert(p.content);
            *per_author.entry(p.author.clone()).or_default() += 1;
            plan.admitted.push(p.id);
            plan.slots[index] = Some(SlotContent {
                slot: SlotIndex::new(index),
                origin: p.clone(),
                bid: c.bid,
                level: c.level,
                admitted_at: next,
                report_digest: c.report_digest,
                findings: c.findings,
            });
            plan.last_change[index] = next.0;
            plan.pinned[index] = false;
        }
        plan.refusals.sort_by_key(|r| r.proposal);
        plan
    }

    /// Apply a finished plan. Nothing here can fail or panic on input.
    fn commit(
        &mut self,
        plan: Plan,
        next: WorkspaceRevision,
        identity: &VerifierIdentity,
    ) -> CycleReport {
        self.slots = plan.slots.clone();
        self.last_change = plan.last_change;
        self.pinned = plan.pinned;
        self.pending.clear();
        self.retire_requests.clear();
        for id in &plan.revoked_subscribers {
            self.subscribers.remove(id);
        }

        let c = &mut self.counters;
        c.cycles += 1;
        c.admitted += plan.admitted.len() as u64;
        c.retired += plan.retired.len() as u64;
        c.verifier_panics += plan.panics;
        c.subscriptions_revoked += plan.revoked_subscribers.len() as u64;
        for r in &plan.refusals {
            c.refused[r.reason.kind().index()] += 1;
        }
        for record in &plan.retired {
            match record.reason {
                RetireReason::Displaced { .. } => c.displaced += 1,
                RetireReason::Contradicted => c.contradicted += 1,
                RetireReason::AuthorRevoked => c.author_revoked_retired += 1,
                RetireReason::Aged | RetireReason::Explicit => {}
            }
            if self.retired_log.len() >= self.config.retired_log() {
                self.retired_log.pop_front();
                c.retired_log_evicted += 1;
            }
            self.retired_log.push_back(record.clone());
        }
        self.revision = next;

        let snapshot = Arc::new(build_snapshot(SnapshotParts {
            revision: next,
            slots: plan.slots,
            admitted: plan.admitted,
            refusals: plan.refusals,
            retired: plan.retired,
            verifier: identity.clone(),
        }));
        let mut delivered = Vec::new();
        let mut overflowed = Vec::new();
        for (id, subscriber) in &mut self.subscribers {
            if subscriber.mailbox.len() >= self.config.mailbox_depth() {
                subscriber.overflowed += 1;
                self.counters.mailbox_overflow += 1;
                overflowed.push(*id);
            } else {
                subscriber.mailbox.push_back(Arc::clone(&snapshot));
                subscriber.delivered += 1;
                self.counters.delivered += 1;
                delivered.push(*id);
            }
        }
        debug_assert!(self.check_invariants().is_ok());
        CycleReport {
            snapshot,
            delivered,
            overflowed,
            revoked_subscribers: plan.revoked_subscribers,
            host_findings: plan.host_findings,
            host_findings_dropped: plan.host_findings_dropped,
        }
    }
}

/// Run the host verifier; a panic is a refusal, not a crash: `Unknown` at `Unverified` with the
/// hard finding [`VERIFIER_PANIC_CODE`].
fn judge(
    verifier: &dyn Verifier<Proposal>,
    proposal: &Proposal,
    panics: &mut u64,
) -> VerificationReport {
    match catch_unwind(AssertUnwindSafe(|| verifier.verify(proposal))) {
        Ok(report) => report,
        Err(_) => {
            *panics += 1;
            VerificationReport {
                status: VerificationStatus::Unknown,
                level: VerificationLevel::Unverified,
                score: Probability::new(0.0).expect("zero probability"),
                findings: vec![Finding {
                    code: VERIFIER_PANIC_CODE.into(),
                    message: "the verifier panicked on this proposal".into(),
                    hard: true,
                }],
            }
        }
    }
}

fn push_host(
    out: &mut Vec<HostFinding>,
    dropped: &mut usize,
    proposal: WorkspaceProposalId,
    findings: &[Finding],
) {
    let mut ordered: Vec<&Finding> = findings.iter().collect();
    ordered.sort_by_key(|f| !f.hard);
    for finding in ordered.into_iter().take(MAX_FINDINGS_PER_REFUSAL) {
        if out.len() >= MAX_HOST_FINDINGS_PER_CYCLE {
            *dropped += 1;
            continue;
        }
        out.push(HostFinding {
            proposal,
            code: super::snapshot::bounded_code(&finding.code),
            message: super::snapshot::bounded_message(&finding.message),
            hard: finding.hard,
        });
    }
}

fn refusal(
    proposal: &Proposal,
    reason: RefusalReason,
    findings: Vec<FindingRecord>,
    findings_dropped: usize,
) -> Refusal {
    Refusal {
        proposal: proposal.id,
        author: proposal.author.clone(),
        author_generation: proposal.author_generation,
        content: proposal.content,
        reason,
        findings,
        findings_dropped,
    }
}

fn tie_break(seed: u64, revision: WorkspaceRevision, id: WorkspaceProposalId) -> Digest {
    let mut e = Encoder::new(TIE_DOMAIN);
    e.u64(seed);
    e.u64(revision.0);
    e.raw(&id.0);
    e.finish()
}

/// The incumbent a challenger may displace, if any: occupied, not pinned, resident long enough,
/// not deeper-verified than the challenger, and outbid by at least the margin. The weakest such
/// incumbent by (bid, level, admission revision, slot index) is chosen.
fn pick_victim(
    plan: &Plan,
    policy: &DisplacementPolicy,
    next: WorkspaceRevision,
    challenger: &Candidate<'_>,
) -> Option<usize> {
    plan.slots
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_ref().map(|s| (i, s)))
        .filter(|(i, s)| {
            !plan.pinned[*i]
                && next.0.saturating_sub(s.admitted_at.0) >= policy.min_residency.get()
                && level_rank(challenger.level) >= level_rank(s.level)
                && u64::from(challenger.bid.get())
                    >= u64::from(s.bid.get()) + u64::from(policy.margin)
        })
        .min_by_key(|(i, s)| (s.bid, level_rank(s.level), s.admitted_at, *i))
        .map(|(i, _)| i)
}
