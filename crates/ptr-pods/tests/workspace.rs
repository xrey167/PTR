//! Public-API tests of the global workspace contract (doc 37): bounded slots, verifier-gated
//! admission, deterministic write competition, host-derived bids, generation safety,
//! revalidation, redaction, panic atomicity, opt-in displacement and immutable broadcast.

use ptr_feedback::Candidate;
use ptr_pods::{
    check_verification_gate, AuthorRegistry, Authority, Bid, CycleHost, CycleReport, Displacement,
    DisplacementPolicy, FindingVisibility, GateRefusal, PayloadRef, PinRequest, Proposal,
    ProposalInit, RefusalKind, RefusalReason, RetireReason, RetireRequest, Revalidation, SlotDecay,
    SlotIndex, Standing, SubmitError, SubscribeRequest, SubscriberId, UnboundedResidencyRisk,
    UnrestrictedAuthors, VerifierIdentity, Workspace, WorkspaceConfig, WorkspaceConfigError,
    WorkspaceConfigInit, WorkspaceError, WorkspaceRevision, WorkspaceSnapshot,
    MAX_FINDING_CODE_BYTES, MAX_SNAPSHOT_FINDINGS, VERIFIER_PANIC_CODE,
};
use ptr_types::{CandidateId, Generation, PodId, Probability, TypeId, VerificationLevel as L};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus as S, Verifier};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::sync::Arc;

/// Host-side verifier: reports keyed by the first byte of the content digest; everything else
/// passes deterministically. `panics` lists contents the verifier panics on.
#[derive(Default)]
struct Table {
    reports: BTreeMap<u8, VerificationReport>,
    panics: BTreeSet<u8>,
    soft_everywhere: bool,
}

impl Table {
    fn with(tag: u8, report: VerificationReport) -> Self {
        let mut t = Self::default();
        t.reports.insert(tag, report);
        t
    }
}

impl Verifier<Proposal> for Table {
    fn verify(&self, p: &Proposal) -> VerificationReport {
        let tag = p.content()[0];
        assert!(!self.panics.contains(&tag), "scripted verifier panic");
        let mut r = self
            .reports
            .get(&tag)
            .cloned()
            .unwrap_or_else(|| report(S::Pass, L::Deterministic, vec![]));
        if self.soft_everywhere {
            r.findings.push(finding("note", false));
        }
        r
    }
}

#[derive(Default)]
struct Registry {
    revoked: BTreeSet<(String, u64)>,
    ceilings: BTreeMap<String, u32>,
}

impl AuthorRegistry for Registry {
    fn standing(&self, pod: &PodId, generation: Generation) -> Standing {
        if self.revoked.contains(&(pod.0.clone(), generation.0)) {
            Standing::Revoked
        } else {
            Standing::Current
        }
    }

    fn bid_ceiling(&self, pod: &PodId) -> Bid {
        Bid::new(self.ceilings.get(&pod.0).copied().unwrap_or(u32::MAX))
    }
}

fn report(status: S, level: L, findings: Vec<Finding>) -> VerificationReport {
    VerificationReport {
        status,
        level,
        // A perfect score everywhere: the gate must not be moved by it.
        score: Probability::new(1.0).unwrap(),
        findings,
    }
}

fn finding(code: &str, hard: bool) -> Finding {
    Finding {
        code: code.into(),
        message: format!("{code} message"),
        hard,
    }
}

fn identity() -> VerifierIdentity {
    VerifierIdentity::new("test-table", [9; 32]).unwrap()
}

fn cycle_with(
    ws: &mut Workspace,
    verifier: &dyn Verifier<Proposal>,
    authors: &dyn AuthorRegistry,
) -> CycleReport {
    let id = identity();
    ws.run_cycle(CycleHost {
        verifier,
        identity: &id,
        authors,
    })
    .unwrap()
}

fn cycle(ws: &mut Workspace, verifier: &dyn Verifier<Proposal>) -> CycleReport {
    cycle_with(ws, verifier, &UnrestrictedAuthors)
}

fn init(slots: usize) -> WorkspaceConfigInit {
    WorkspaceConfigInit {
        max_pending: 64,
        max_pending_per_author: 64,
        max_subscribers: 4,
        mailbox_depth: 2,
        retired_log: 4,
        tie_break_seed: 42,
        ..WorkspaceConfigInit::conservative(slots)
    }
}

fn build(init: WorkspaceConfigInit) -> Workspace {
    Workspace::new(WorkspaceConfig::new(init).unwrap())
}

fn workspace(slots: usize) -> Workspace {
    build(init(slots))
}

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

fn by(author: &str, generation: u64, tag: u8, rev: WorkspaceRevision, bid: u32) -> Proposal {
    Proposal::new(ProposalInit {
        author: PodId::from(author),
        author_generation: Generation(generation),
        content: [tag; 32],
        payload: PayloadRef::new(TypeId::from("note.v1"), [tag; 32]),
        computed_against: rev,
        bid: Bid::new(bid),
    })
    .unwrap()
}

fn proposal(tag: u8, rev: WorkspaceRevision, bid: u32) -> Proposal {
    by(&format!("pod-{tag}"), 1, tag, rev, bid)
}

fn admitted_contents(snapshot: &WorkspaceSnapshot) -> BTreeSet<u8> {
    snapshot
        .slots()
        .iter()
        .flatten()
        .map(|s| s.content()[0])
        .collect()
}

fn subscribe(ws: &mut Workspace, name: &str) -> SubscriberId {
    ws.subscribe(SubscribeRequest {
        pod: PodId::from(name),
        generation: Generation(1),
    })
    .unwrap()
}

// --- capacity, competition, determinism -----------------------------------------------------

#[test]
fn capacity_is_respected_under_more_proposals_than_slots() {
    let mut ws = workspace(3);
    for tag in 1..=12u8 {
        ws.submit(proposal(tag, ws.revision(), u32::from(tag)))
            .unwrap();
    }
    let report = cycle(&mut ws, &Table::default());
    assert_eq!(report.snapshot.slots().len(), 3);
    assert_eq!(report.snapshot.occupied(), 3);
    assert_eq!(report.snapshot.admitted().len(), 3);
    assert_eq!(
        admitted_contents(&report.snapshot),
        BTreeSet::from([10, 11, 12])
    );
    ws.check_invariants().unwrap();
    for _ in 0..5 {
        for tag in 20..40u8 {
            let _ = ws.submit(proposal(tag, ws.revision(), 1000));
        }
        let r = cycle(&mut ws, &Table::default());
        assert_eq!(r.snapshot.occupied(), 3);
        ws.check_invariants().unwrap();
    }
}

#[test]
fn every_loser_has_a_counted_explicit_refusal() {
    let mut ws = workspace(2);
    for tag in 1..=7u8 {
        ws.submit(proposal(tag, ws.revision(), u32::from(tag)))
            .unwrap();
    }
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.admitted().len() + snap.refusals().len(), 7);
    for r in snap.refusals() {
        assert_eq!(r.reason(), &RefusalReason::Outbid { free_slots: 2 });
    }
    assert_eq!(ws.counters().refused(RefusalKind::Outbid), 5);
    assert_eq!(ws.counters().refused_total(), 5);
    assert_eq!(ws.counters().admitted(), 2);
    assert_eq!(ws.counters().submitted(), 7);
    assert_eq!(ws.pending_len(), 0);
}

/// A tiny deterministic shuffle (seeded LCG), so permutations are reproducible.
fn shuffled<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
    let mut v = items.to_vec();
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..v.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.swap(i, (state >> 33) as usize % (i + 1));
    }
    v
}

#[test]
fn winner_set_is_independent_of_submission_order_over_many_permutations() {
    let table = {
        let mut t = Table::with(
            4,
            report(S::Fail, L::Deterministic, vec![finding("x", true)]),
        );
        t.reports
            .insert(5, report(S::Unknown, L::Unverified, vec![]));
        t
    };
    let run = |order: u64| {
        let mut ws = workspace(3);
        let old = ws.revision();
        cycle(&mut ws, &table); // revision 1, so `old` is behind
        let now = ws.revision();
        let mut set = vec![proposal(30, old, 9)]; // stale
                                                  // valid, equal bids (tie-break decides), a gate failure, an unknown, duplicates
        set.extend((2..=8u8).map(|t| proposal(t, now, 5)));
        set.push(proposal(1, now, 100));
        set.push(by("copycat", 1, 1, now, 100)); // same content as tag 1, other author
        for p in shuffled(&set, order) {
            ws.submit(p).unwrap();
        }
        cycle(&mut ws, &table).snapshot
    };
    let baseline = run(0);
    assert_eq!(baseline.occupied(), 3);
    assert!(baseline
        .refusals()
        .iter()
        .any(|r| r.reason().kind() == RefusalKind::Stale));
    assert!(baseline
        .refusals()
        .iter()
        .any(|r| r.reason().kind() == RefusalKind::Duplicate));
    for seed in 1..10 {
        assert_eq!(run(seed).digest(), baseline.digest(), "permutation {seed}");
    }
}

#[test]
fn tie_break_is_seeded() {
    let winners = |seed: u64| {
        let mut ws = build(WorkspaceConfigInit {
            tie_break_seed: seed,
            ..init(1)
        });
        for tag in 1..=16u8 {
            ws.submit(proposal(tag, ws.revision(), 5)).unwrap();
        }
        admitted_contents(&cycle(&mut ws, &Table::default()).snapshot)
    };
    assert_eq!(winners(1), winners(1));
    let distinct: BTreeSet<_> = (0..16u64).map(winners).collect();
    assert!(distinct.len() > 1, "the seed must influence equal-bid ties");
}

// --- the gate -------------------------------------------------------------------------------

#[test]
fn bid_orders_only_proposals_that_passed_the_gate() {
    let table = Table::with(
        9,
        report(S::Fail, L::Deterministic, vec![finding("x", true)]),
    );
    let mut ws = workspace(1);
    ws.submit(proposal(9, ws.revision(), u32::MAX)).unwrap();
    ws.submit(proposal(1, ws.revision(), 0)).unwrap();
    assert_eq!(
        admitted_contents(&cycle(&mut ws, &table).snapshot),
        BTreeSet::from([1])
    );
}

#[test]
fn fail_unknown_disputed_never_occupy_a_slot() {
    for status in [S::Fail, S::Unknown, S::Disputed] {
        let table = Table::with(1, report(status, L::Deterministic, vec![]));
        let mut ws = workspace(4);
        ws.submit(proposal(1, ws.revision(), u32::MAX)).unwrap();
        let snap = cycle(&mut ws, &table).snapshot;
        assert_eq!(snap.occupied(), 0, "{status:?}");
        assert_eq!(
            snap.refusals()[0].reason(),
            &RefusalReason::Gate(GateRefusal::NotPassed(status))
        );
    }
}

#[test]
fn forged_pass_with_a_hard_finding_is_refused_and_the_code_is_broadcast() {
    let table = Table::with(
        1,
        report(
            S::Pass,
            L::Deterministic,
            vec![finding("soft", false), finding("target", true)],
        ),
    );
    let mut ws = workspace(2);
    let sub = subscribe(&mut ws, "router");
    ws.submit(proposal(1, ws.revision(), 100)).unwrap();
    cycle(&mut ws, &table);
    let snap = ws.receive(sub).unwrap().unwrap();
    assert_eq!(snap.occupied(), 0);
    let refusal = &snap.refusals()[0];
    assert_eq!(
        refusal.reason(),
        &RefusalReason::Gate(GateRefusal::HardFinding)
    );
    let codes: Vec<_> = refusal
        .findings()
        .iter()
        .map(|f| (f.code(), f.hard()))
        .collect();
    assert_eq!(codes, [("target", true), ("soft", false)]);
}

#[test]
fn below_required_level_is_refused_and_the_boundary_level_is_admitted() {
    let mut table = Table::default();
    table
        .reports
        .insert(1, report(S::Pass, L::LatentAgreement, vec![]));
    table
        .reports
        .insert(2, report(S::Pass, L::Unverified, vec![]));
    table
        .reports
        .insert(3, report(S::Pass, L::SampleVerified, vec![]));
    let mut ws = workspace(4);
    for tag in 1..=3 {
        ws.submit(proposal(tag, ws.revision(), 1)).unwrap();
    }
    let snap = cycle(&mut ws, &table).snapshot;
    assert_eq!(admitted_contents(&snap), BTreeSet::from([3]));
    assert_eq!(ws.counters().refused(RefusalKind::BelowRequiredLevel), 2);
}

#[test]
fn an_empty_verifier_fabric_admits_nothing() {
    let fabric = ptr_verifier::VerifierFabric::<Proposal>::default();
    let mut ws = workspace(2);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    assert_eq!(cycle(&mut ws, &fabric).snapshot.occupied(), 0);
    assert_eq!(ws.counters().refused(RefusalKind::NotPassed), 1);
}

#[test]
fn gate_equals_candidate_passes_over_the_full_matrix() {
    let levels = [
        L::Unverified,
        L::LatentAgreement,
        L::SampleVerified,
        L::FullSemantic,
        L::Deterministic,
    ];
    let sets: [Vec<Finding>; 3] = [
        vec![],
        vec![finding("s", false)],
        vec![finding("s", false), finding("h", true)],
    ];
    let mut checked = 0;
    for status in [S::Pass, S::Fail, S::Disputed, S::Unknown] {
        for level in levels {
            for required in levels {
                for fs in &sets {
                    for score in [0.0, 0.5, 1.0] {
                        let mut r = report(status, level, fs.clone());
                        r.score = Probability::new(score).unwrap();
                        let candidate = Candidate {
                            id: CandidateId::from("c"),
                            value: (),
                            verification: Some(r.clone()),
                        };
                        assert_eq!(
                            check_verification_gate(&r, required).is_ok(),
                            candidate.passes(required),
                            "{status:?} {level:?} {required:?} {fs:?} {score}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 5 * 3 * 3);
}

// --- staleness ------------------------------------------------------------------------------

#[test]
fn stale_revision_is_refused_and_future_revision_is_rejected_at_submit() {
    let mut ws = workspace(2);
    let other = {
        let mut o = workspace(1);
        cycle(&mut o, &Table::default());
        cycle(&mut o, &Table::default());
        cycle(&mut o, &Table::default());
        o.revision()
    };
    let r0 = ws.revision();
    cycle(&mut ws, &Table::default());
    ws.submit(proposal(1, r0, 99)).unwrap();
    ws.submit(proposal(2, ws.revision(), 1)).unwrap();
    assert!(matches!(
        ws.submit(proposal(3, other, 1)),
        Err(SubmitError::FutureRevision { .. })
    ));
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(admitted_contents(&snap), BTreeSet::from([2]));
    assert_eq!(snap.refusals()[0].reason().kind(), RefusalKind::Stale);
    assert!(snap.refusals()[0].findings().is_empty());
    assert_eq!(ws.counters().submit_rejected(), 1);
}

#[test]
fn staleness_window_and_slot_change_rules() {
    // Within the window and nothing changed: admitted.
    let mut ws = build(WorkspaceConfigInit {
        max_staleness: 3,
        ..init(2)
    });
    let r0 = ws.revision();
    cycle(&mut ws, &Table::default());
    cycle(&mut ws, &Table::default());
    ws.submit(proposal(1, r0, 1)).unwrap(); // 2 behind
    assert_eq!(
        admitted_contents(&cycle(&mut ws, &Table::default()).snapshot),
        BTreeSet::from([1])
    );

    // Beyond the window: stale.
    let mut ws = build(WorkspaceConfigInit {
        max_staleness: 1,
        ..init(2)
    });
    let r0 = ws.revision();
    cycle(&mut ws, &Table::default());
    cycle(&mut ws, &Table::default());
    ws.submit(proposal(1, r0, 1)).unwrap(); // 2 behind, window 1
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.refusals()[0].reason().kind(), RefusalKind::Stale);

    // Inside the window but a slot changed since: stale; the loser resubmits against the new
    // revision and is judged on its merits.
    let mut ws = build(WorkspaceConfigInit {
        max_staleness: 5,
        ..init(2)
    });
    let r0 = ws.revision();
    ws.submit(proposal(7, r0, 1)).unwrap();
    cycle(&mut ws, &Table::default()); // slot changed at revision 1
    ws.submit(proposal(1, r0, 1)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.refusals()[0].reason().kind(), RefusalKind::Stale);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    assert_eq!(cycle(&mut ws, &Table::default()).snapshot.occupied(), 2);
}

#[test]
fn default_staleness_is_strict() {
    assert_eq!(WorkspaceConfigInit::conservative(1).max_staleness, 0);
    let mut ws = workspace(1);
    let r0 = ws.revision();
    cycle(&mut ws, &Table::default());
    ws.submit(proposal(1, r0, 1)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.refusals()[0].reason().kind(), RefusalKind::Stale);
}

// --- duplicates and bounds ------------------------------------------------------------------

#[test]
fn duplicate_content_keeps_the_first_author() {
    let mut ws = workspace(4);
    ws.submit(by("first", 1, 1, ws.revision(), 5)).unwrap();
    ws.submit(by("copycat", 1, 1, ws.revision(), 4)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.occupied(), 1);
    assert_eq!(snap.refusals()[0].reason(), &RefusalReason::Duplicate);
    // Later the original holds the slot, and an even higher bid cannot take over its content.
    ws.submit(by("copycat", 1, 1, ws.revision(), 99)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.slots()[0].as_ref().unwrap().author().0, "first");
    assert_eq!(ws.counters().refused(RefusalKind::Duplicate), 2);
}

#[test]
fn pending_area_and_author_quota_are_bounded_and_counted() {
    let mut ws = build(WorkspaceConfigInit {
        max_pending: 3,
        max_pending_per_author: 2,
        ..init(2)
    });
    ws.submit(by("a", 1, 1, ws.revision(), 1)).unwrap();
    ws.submit(by("a", 1, 2, ws.revision(), 1)).unwrap();
    assert_eq!(
        ws.submit(by("a", 1, 3, ws.revision(), 1)),
        Err(SubmitError::AuthorPendingFull)
    );
    ws.submit(by("b", 1, 4, ws.revision(), 1)).unwrap();
    assert_eq!(
        ws.submit(by("c", 1, 5, ws.revision(), 1)),
        Err(SubmitError::PendingFull)
    );
    assert_eq!(
        ws.submit(by("b", 1, 4, ws.revision(), 1)),
        Err(SubmitError::Duplicate)
    );
    assert_eq!(ws.pending_len(), 3);
    assert_eq!(ws.counters().submit_author_full(), 1);
    assert_eq!(ws.counters().submit_pending_full(), 1);
    assert_eq!(ws.counters().submit_rejected(), 1);
}

#[test]
fn config_rejects_unbounded_or_zero_capacities_and_unsafe_combinations() {
    let base = init(2);
    assert!(WorkspaceConfig::new(base).is_ok());
    let bad = |c: WorkspaceConfigInit| WorkspaceConfig::new(c).unwrap_err();
    assert_eq!(
        bad(WorkspaceConfigInit { slots: 0, ..base }),
        WorkspaceConfigError::Slots(0)
    );
    assert!(matches!(
        bad(WorkspaceConfigInit {
            slots: usize::MAX,
            ..base
        }),
        WorkspaceConfigError::Slots(_)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            mailbox_depth: 0,
            ..base
        }),
        WorkspaceConfigError::MailboxDepth(0)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            max_pending: 0,
            ..base
        }),
        WorkspaceConfigError::MaxPending(0)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            max_slots_per_author: 3,
            ..base
        }),
        WorkspaceConfigError::MaxSlotsPerAuthor(3)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            max_pending_per_author: 65,
            ..base
        }),
        WorkspaceConfigError::MaxPendingPerAuthor(65)
    ));
    let never = SlotDecay::NeverUnsafe(UnboundedResidencyRisk::acknowledge());
    assert_eq!(
        bad(WorkspaceConfigInit {
            decay: never,
            revalidation: Revalidation::Never,
            ..base
        }),
        WorkspaceConfigError::UnboundedResidencyWithoutRevalidation
    );
    assert!(WorkspaceConfig::new(WorkspaceConfigInit {
        decay: never,
        ..base
    })
    .is_ok());
    let policy = |margin, per_cycle, per_author| {
        Displacement::Policy(DisplacementPolicy {
            margin,
            min_residency: nz(1),
            max_per_cycle: per_cycle,
            max_per_author: per_author,
        })
    };
    assert!(matches!(
        bad(WorkspaceConfigInit {
            displacement: policy(0, 1, 1),
            ..base
        }),
        WorkspaceConfigError::Displacement(_)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            displacement: policy(1, 3, 1),
            ..base
        }),
        WorkspaceConfigError::Displacement(_)
    ));
    assert!(matches!(
        bad(WorkspaceConfigInit {
            displacement: policy(1, 1, 2),
            ..base
        }),
        WorkspaceConfigError::Displacement(_)
    ));
    assert!(WorkspaceConfig::new(WorkspaceConfigInit {
        displacement: policy(1, 2, 1),
        ..base
    })
    .is_ok());
}

// --- authority ------------------------------------------------------------------------------

#[test]
fn proposals_are_pending_and_slot_content_is_provisional() {
    let mut ws = workspace(2);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    assert_eq!(ws.occupied(), 0);
    assert!(ws.slot(SlotIndex::new(0)).is_none());
    assert_eq!(ws.pending_len(), 1);
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(snap.authority(), Authority::Provisional);
    assert_eq!(
        snap.slots()[0].as_ref().unwrap().authority(),
        Authority::Provisional
    );
}

#[test]
fn no_submitted_value_can_change_the_gate_or_the_config() {
    // Compile-time pin: the proposal input has exactly these fields. Adding a field that could
    // carry a gate, threshold, level, verifier or report breaks this destructuring and demands a
    // review of invariants 4, 5 and 11.
    let ProposalInit {
        author: _,
        author_generation: _,
        content: _,
        payload: _,
        computed_against: _,
        bid: _,
    } = ProposalInit {
        author: PodId::from("p"),
        author_generation: Generation(1),
        content: [1; 32],
        payload: PayloadRef::new(TypeId::from("t"), [1; 32]),
        computed_against: workspace(1).revision(),
        bid: Bid::new(0),
    };

    // Behavioural pin: a hostile proposal with the maximum bid never gets in while the host
    // verifier fails it, round after round, and the configuration is the one it started with.
    let mut ws = workspace(2);
    let table = Table::with(
        7,
        report(S::Fail, L::Deterministic, vec![finding("hard", true)]),
    );
    for round in 0..4u8 {
        ws.submit(proposal(7, ws.revision(), u32::MAX)).unwrap();
        ws.submit(proposal(8 + round, ws.revision(), 0)).unwrap();
        cycle(&mut ws, &table);
    }
    assert!(ws
        .slot(SlotIndex::new(0))
        .into_iter()
        .chain(ws.slot(SlotIndex::new(1)))
        .all(|s| s.content()[0] != 7));
    assert_eq!(ws.counters().refused(RefusalKind::NotPassed), 4);
}

// --- bids ----------------------------------------------------------------------------------

#[test]
fn bids_are_clamped_by_the_config_and_by_the_host_ceiling() {
    let mut ws = build(WorkspaceConfigInit {
        max_bid: Bid::new(50),
        ..init(1)
    });
    let mut authors = Registry::default();
    authors.ceilings.insert("capped".into(), 10);
    ws.submit(by("capped", 1, 1, ws.revision(), u32::MAX))
        .unwrap();
    ws.submit(by("modest", 1, 2, ws.revision(), 20)).unwrap();
    let snap = cycle_with(&mut ws, &Table::default(), &authors).snapshot;
    // `capped` asked for MAX but is limited to 10, so `modest` (20) wins.
    assert_eq!(admitted_contents(&snap), BTreeSet::from([2]));
    assert_eq!(snap.slots()[0].as_ref().unwrap().bid().get(), 20);
    ws.request_retire(RetireRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    ws.submit(by("greedy", 1, 3, ws.revision(), u32::MAX))
        .unwrap();
    let snap = cycle_with(&mut ws, &Table::default(), &authors).snapshot;
    assert_eq!(
        snap.slots()[0].as_ref().unwrap().bid().get(),
        50,
        "clamped to max_bid"
    );
}

#[test]
fn one_author_cannot_monopolise_the_slots() {
    let mut ws = build(WorkspaceConfigInit {
        max_slots_per_author: 1,
        ..init(3)
    });
    for tag in 1..=6u8 {
        ws.submit(by("greedy", 1, tag, ws.revision(), u32::MAX))
            .unwrap();
    }
    ws.submit(by("honest", 1, 20, ws.revision(), 1)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    let authors: BTreeSet<_> = snap
        .slots()
        .iter()
        .flatten()
        .map(|s| s.author().0.clone())
        .collect();
    assert_eq!(
        authors,
        BTreeSet::from(["greedy".to_string(), "honest".to_string()])
    );
    assert_eq!(snap.occupied(), 2);
    assert_eq!(ws.counters().refused(RefusalKind::AuthorSlotCap), 5);
    // The cap also holds across cycles.
    ws.submit(by("greedy", 1, 30, ws.revision(), u32::MAX))
        .unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(
        snap.refusals()[0].reason(),
        &RefusalReason::AuthorSlotCap { cap: 1 }
    );
}

// --- generations and revalidation ------------------------------------------------------------

#[test]
fn a_revoked_author_generation_cannot_write_and_loses_its_slot_and_subscription() {
    let mut ws = workspace(3);
    let g1 = subscribe(&mut ws, "worker");
    let other = subscribe(&mut ws, "other");
    ws.submit(by("worker", 1, 1, ws.revision(), 1)).unwrap();
    ws.submit(by("other", 1, 2, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    assert_eq!(ws.occupied(), 2);

    let mut authors = Registry::default();
    authors.revoked.insert(("worker".into(), 1));
    ws.submit(by("worker", 1, 3, ws.revision(), 99)).unwrap(); // a revoked author tries again
    ws.submit(by("worker", 2, 4, ws.revision(), 1)).unwrap(); // its successor generation is fine
    let report = cycle_with(&mut ws, &Table::default(), &authors);
    let snap = &report.snapshot;
    assert_eq!(admitted_contents(snap), BTreeSet::from([2, 4]));
    assert_eq!(snap.retired().len(), 1);
    assert_eq!(snap.retired()[0].reason(), RetireReason::AuthorRevoked);
    assert_eq!(snap.retired()[0].content().content()[0], 1);
    assert_eq!(snap.refusals().len(), 1);
    assert_eq!(snap.refusals()[0].reason(), &RefusalReason::AuthorRevoked);
    assert_eq!(report.revoked_subscribers, vec![g1]);
    assert_eq!(ws.counters().subscriptions_revoked(), 1);
    assert_eq!(ws.counters().author_revoked_retired(), 1);
    assert_eq!(ws.receive(g1), Err(WorkspaceError::UnknownSubscriber(g1)));
    assert!(ws.receive(other).unwrap().is_some());
}

#[test]
fn incumbents_are_revalidated_and_a_later_contradiction_retires_them() {
    let mut ws = workspace(2);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    ws.submit(proposal(2, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    assert_eq!(ws.occupied(), 2);
    // The host's verifier now contradicts content 1 and no longer reaches the required level for 2.
    let mut stricter = Table::with(
        1,
        report(S::Fail, L::Deterministic, vec![finding("withdrawn", true)]),
    );
    stricter
        .reports
        .insert(2, report(S::Pass, L::LatentAgreement, vec![]));
    let snap = cycle(&mut ws, &stricter).snapshot;
    assert_eq!(snap.occupied(), 0);
    assert_eq!(snap.retired().len(), 2);
    assert!(snap
        .retired()
        .iter()
        .all(|r| r.reason() == RetireReason::Contradicted));
    let withdrawn = snap
        .retired()
        .iter()
        .find(|r| r.content().content()[0] == 1)
        .unwrap();
    assert_eq!(withdrawn.findings()[0].code(), "withdrawn");
    assert_eq!(ws.counters().contradicted(), 2);
}

#[test]
fn revalidation_off_keeps_a_contradicted_incumbent_and_every_n_checks_on_schedule() {
    let table = Table::with(
        1,
        report(S::Fail, L::Deterministic, vec![finding("x", true)]),
    );
    // Never: the incumbent stays (this is exactly the risk revalidation removes).
    let mut ws = build(WorkspaceConfigInit {
        revalidation: Revalidation::Never,
        ..init(1)
    });
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    assert_eq!(cycle(&mut ws, &table).snapshot.occupied(), 1);
    // Every 2 cycles: age 1 is skipped, age 2 retires it.
    let mut ws = build(WorkspaceConfigInit {
        revalidation: Revalidation::EveryCycles(nz(2)),
        ..init(1)
    });
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    assert_eq!(cycle(&mut ws, &table).snapshot.occupied(), 1);
    assert_eq!(cycle(&mut ws, &table).snapshot.occupied(), 0);
}

#[test]
fn aged_and_explicit_retirement_are_recorded_and_aging_wins_over_a_request() {
    let mut ws = build(WorkspaceConfigInit {
        decay: SlotDecay::AfterCycles(nz(2)),
        ..init(2)
    });
    let v = Table::default();
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &v); // admitted at revision 1
    assert_eq!(cycle(&mut ws, &v).snapshot.occupied(), 1); // revision 2: age 1
    ws.request_retire(RetireRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    let s3 = cycle(&mut ws, &v).snapshot; // revision 3: age 2 -> aged (wins over the request)
    assert_eq!(s3.retired().len(), 1);
    assert_eq!(s3.retired()[0].reason(), RetireReason::Aged);
    assert_eq!(s3.retired()[0].retired_at().get(), 3);

    ws.submit(proposal(2, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &v);
    ws.request_retire(RetireRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    ws.submit(proposal(3, ws.revision(), 1)).unwrap();
    let snap = cycle(&mut ws, &v).snapshot;
    assert_eq!(snap.retired()[0].reason(), RetireReason::Explicit);
    assert_eq!(admitted_contents(&snap), BTreeSet::from([3]));
    assert_eq!(
        ws.request_retire(RetireRequest {
            slot: SlotIndex::new(9)
        }),
        Err(WorkspaceError::SlotOutOfRange(SlotIndex::new(9)))
    );
    assert_eq!(
        workspace(1).request_retire(RetireRequest {
            slot: SlotIndex::new(0)
        }),
        Err(WorkspaceError::SlotEmpty(SlotIndex::new(0)))
    );
}

#[test]
fn retired_log_is_bounded_and_evictions_are_counted() {
    let mut ws = build(WorkspaceConfigInit {
        retired_log: 2,
        decay: SlotDecay::AfterCycles(nz(1)),
        ..init(1)
    });
    for tag in 1..=5u8 {
        ws.submit(proposal(tag, ws.revision(), 1)).unwrap();
        cycle(&mut ws, &Table::default());
    }
    cycle(&mut ws, &Table::default());
    assert_eq!(ws.counters().retired(), 5);
    assert_eq!(ws.retired_log().count(), 2);
    assert_eq!(ws.counters().retired_log_evicted(), 3);
}

// --- redaction ------------------------------------------------------------------------------

struct Leaky;
impl Verifier<Proposal> for Leaky {
    fn verify(&self, _: &Proposal) -> VerificationReport {
        report(
            S::Fail,
            L::Deterministic,
            vec![Finding {
                code: "token-mismatch".into(),
                message: "token SECRET-MARKER-1234 does not match".into(),
                hard: true,
            }],
        )
    }
}

#[test]
fn finding_messages_stay_host_side_by_default() {
    let mut ws = workspace(2);
    let sub = subscribe(&mut ws, "low-privilege");
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    let report = cycle(&mut ws, &Leaky);
    let delivered = ws.receive(sub).unwrap().unwrap();
    for snap in [&report.snapshot, &delivered] {
        assert!(!format!("{snap:?}").contains("SECRET-MARKER"));
        assert_eq!(snap.refusals()[0].findings()[0].code(), "token-mismatch");
        assert!(snap.refusals()[0].findings()[0].detail().is_none());
    }
    // The host still gets the message.
    assert!(report
        .host_findings
        .iter()
        .any(|f| f.message.contains("SECRET-MARKER-1234") && f.hard));
}

#[test]
fn publishing_messages_is_an_explicit_opt_in_and_changes_the_digest() {
    let digest_of = |visibility| {
        let mut ws = build(WorkspaceConfigInit {
            finding_visibility: visibility,
            ..init(2)
        });
        ws.submit(proposal(1, ws.revision(), 1)).unwrap();
        cycle(&mut ws, &Leaky).snapshot
    };
    let hidden = digest_of(FindingVisibility::HostOnly);
    let public = digest_of(FindingVisibility::Public);
    assert!(format!("{public:?}").contains("SECRET-MARKER"));
    assert_ne!(hidden.digest(), public.digest());
}

#[test]
fn finding_codes_are_bounded_and_the_snapshot_finding_budget_is_enforced() {
    struct Noisy;
    impl Verifier<Proposal> for Noisy {
        fn verify(&self, _: &Proposal) -> VerificationReport {
            let mut fs = vec![Finding {
                code: "x".repeat(10_000),
                message: String::new(),
                hard: true,
            }];
            fs.extend((0..4).map(|i| finding(&format!("f{i}"), true)));
            report(S::Fail, L::Deterministic, fs)
        }
    }
    let mut ws = build(WorkspaceConfigInit {
        max_pending: 1024,
        max_pending_per_author: 1024,
        ..init(1)
    });
    for i in 0..600u16 {
        let tag = (i % 250) as u8 + 1;
        let author = format!("a{i}");
        let mut content = [tag; 32];
        content[1] = (i >> 8) as u8;
        content[2] = i as u8;
        let p = Proposal::new(ProposalInit {
            author: PodId::from(author.as_str()),
            author_generation: Generation(1),
            content,
            payload: PayloadRef::new(TypeId::from("t"), content),
            computed_against: ws.revision(),
            bid: Bid::new(1),
        })
        .unwrap();
        ws.submit(p).unwrap();
    }
    let snap = cycle(&mut ws, &Noisy).snapshot;
    let kept: usize = snap.refusals().iter().map(|r| r.findings().len()).sum();
    let dropped: usize = snap.refusals().iter().map(|r| r.findings_dropped()).sum();
    assert_eq!(kept, MAX_SNAPSHOT_FINDINGS);
    assert_eq!(kept + dropped, 600 * 5);
    assert!(snap
        .refusals()
        .iter()
        .flat_map(|r| r.findings())
        .all(|f| f.code().len() <= MAX_FINDING_CODE_BYTES));
}

// --- verifier panic --------------------------------------------------------------------------

#[test]
fn a_panicking_verifier_is_a_counted_refusal_and_loses_no_state() {
    let mut ws = workspace(3);
    let sub = subscribe(&mut ws, "listener");
    let mut table = Table::default();
    table.panics.insert(2);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    ws.submit(proposal(2, ws.revision(), 9)).unwrap();
    ws.submit(proposal(3, ws.revision(), 1)).unwrap();
    let report = cycle(&mut ws, &table);
    let snap = &report.snapshot;
    assert_eq!(snap.revision().get(), 1);
    assert_eq!(admitted_contents(snap), BTreeSet::from([1, 3]));
    assert_eq!(
        snap.admitted().len() + snap.refusals().len(),
        3,
        "nothing lost"
    );
    let refusal = &snap.refusals()[0];
    assert_eq!(
        refusal.reason(),
        &RefusalReason::Gate(GateRefusal::NotPassed(S::Unknown))
    );
    assert_eq!(refusal.findings()[0].code(), VERIFIER_PANIC_CODE);
    assert!(refusal.findings()[0].hard());
    assert_eq!(ws.counters().verifier_panics(), 1);
    assert_eq!(ws.pending_len(), 0);
    assert!(ws.receive(sub).unwrap().is_some());
    ws.check_invariants().unwrap();
}

#[test]
fn a_verifier_panic_on_an_incumbent_fails_closed_and_is_recorded() {
    let mut ws = workspace(2);
    ws.submit(proposal(1, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    let mut table = Table::default();
    table.panics.insert(1);
    ws.request_retire(RetireRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    let snap = cycle(&mut ws, &table).snapshot;
    // An explicit retirement takes precedence and never runs the verifier.
    assert_eq!(snap.retired()[0].reason(), RetireReason::Explicit);
    ws.submit(proposal(5, ws.revision(), 1)).unwrap();
    cycle(&mut ws, &Table::default());
    let mut table = Table::default();
    table.panics.insert(5);
    let snap = cycle(&mut ws, &table).snapshot;
    assert_eq!(snap.retired()[0].reason(), RetireReason::Contradicted);
    assert_eq!(snap.retired()[0].findings()[0].code(), VERIFIER_PANIC_CODE);
    assert_eq!(ws.counters().verifier_panics(), 1);
}

// --- displacement ----------------------------------------------------------------------------

fn displacing(slots: usize) -> WorkspaceConfigInit {
    WorkspaceConfigInit {
        displacement: Displacement::Policy(DisplacementPolicy {
            margin: 5,
            min_residency: nz(2),
            max_per_cycle: 1,
            max_per_author: 1,
        }),
        ..init(slots)
    }
}

/// Fill a one-slot workspace with `tag` 1 (bid 10) and age it to the protected boundary.
fn aged_incumbent(config: WorkspaceConfigInit, table: &Table) -> Workspace {
    let mut ws = build(config);
    ws.submit(proposal(1, ws.revision(), 10)).unwrap();
    cycle(&mut ws, table); // admitted at revision 1
    cycle(&mut ws, table); // revision 2
    ws
}

#[test]
fn without_a_policy_a_full_workspace_is_never_displaced() {
    let mut ws = aged_incumbent(init(1), &Table::default());
    ws.submit(proposal(2, ws.revision(), 1_000_000)).unwrap();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(admitted_contents(&snap), BTreeSet::from([1]));
    assert_eq!(
        snap.refusals()[0].reason(),
        &RefusalReason::Outbid { free_slots: 0 }
    );
}

#[test]
fn a_better_verified_challenger_displaces_and_the_displacement_is_recorded() {
    let mut ws = aged_incumbent(displacing(1), &Table::default());
    ws.submit(proposal(2, ws.revision(), 15)).unwrap(); // 10 + margin 5
    let challenger = proposal(2, ws.revision(), 15).id();
    let snap = cycle(&mut ws, &Table::default()).snapshot;
    assert_eq!(admitted_contents(&snap), BTreeSet::from([2]));
    assert_eq!(snap.retired().len(), 1);
    assert_eq!(
        snap.retired()[0].reason(),
        RetireReason::Displaced { by: challenger }
    );
    assert_eq!(snap.retired()[0].content().content()[0], 1);
    assert_eq!(ws.counters().displaced(), 1);
}

#[test]
fn displacement_safety_conditions_each_block_it() {
    let table = {
        let mut t = Table::default();
        t.reports
            .insert(3, report(S::Pass, L::SampleVerified, vec![])); // below the incumbent's level
        t.reports.insert(
            4,
            report(S::Fail, L::Deterministic, vec![finding("x", true)]),
        );
        t
    };
    // Margin not met.
    let mut ws = aged_incumbent(displacing(1), &table);
    ws.submit(proposal(2, ws.revision(), 14)).unwrap();
    assert_eq!(
        cycle(&mut ws, &table).snapshot.slots()[0]
            .as_ref()
            .unwrap()
            .content()[0],
        1
    );
    // A bid alone never displaces: shallower verification loses whatever it bids.
    ws.submit(proposal(3, ws.revision(), 1_000_000)).unwrap();
    let snap = cycle(&mut ws, &table).snapshot;
    assert_eq!(admitted_contents(&snap), BTreeSet::from([1]));
    // Nor does a failing proposal.
    ws.submit(proposal(4, ws.revision(), 1_000_000)).unwrap();
    assert_eq!(
        admitted_contents(&cycle(&mut ws, &table).snapshot),
        BTreeSet::from([1])
    );
    // Minimum residency: a fresh incumbent is protected.
    let mut fresh = build(displacing(1));
    fresh.submit(proposal(1, fresh.revision(), 10)).unwrap();
    cycle(&mut fresh, &table); // admitted at revision 1
    fresh.submit(proposal(2, fresh.revision(), 100)).unwrap();
    assert_eq!(
        admitted_contents(&cycle(&mut fresh, &table).snapshot),
        BTreeSet::from([1])
    );
    // Pinned content is never displaced; unpinned it is.
    let mut ws = aged_incumbent(displacing(1), &table);
    ws.pin(PinRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    assert!(ws.is_pinned(SlotIndex::new(0)));
    ws.submit(proposal(2, ws.revision(), 100)).unwrap();
    assert_eq!(
        admitted_contents(&cycle(&mut ws, &table).snapshot),
        BTreeSet::from([1])
    );
    ws.unpin(PinRequest {
        slot: SlotIndex::new(0),
    })
    .unwrap();
    ws.submit(proposal(2, ws.revision(), 100)).unwrap();
    assert_eq!(
        admitted_contents(&cycle(&mut ws, &table).snapshot),
        BTreeSet::from([2])
    );
}

#[test]
fn an_incumbent_contradicted_this_cycle_is_retired_not_displaced() {
    let mut ws = aged_incumbent(displacing(1), &Table::default());
    let withdrawn = Table::with(
        1,
        report(S::Fail, L::Deterministic, vec![finding("w", true)]),
    );
    ws.submit(proposal(2, ws.revision(), 1)).unwrap(); // a low bid suffices for a freed slot
    let snap = cycle(&mut ws, &withdrawn).snapshot;
    assert_eq!(snap.retired()[0].reason(), RetireReason::Contradicted);
    assert_eq!(admitted_contents(&snap), BTreeSet::from([2]));
    assert_eq!(ws.counters().displaced(), 0);
}

#[test]
fn displacement_is_capped_per_cycle_and_per_author_and_is_deterministic() {
    let run = |order: u64| {
        let mut ws = build(WorkspaceConfigInit {
            displacement: Displacement::Policy(DisplacementPolicy {
                margin: 5,
                min_residency: nz(1),
                max_per_cycle: 2,
                max_per_author: 1,
            }),
            ..init(3)
        });
        for tag in 1..=3u8 {
            ws.submit(proposal(tag, ws.revision(), 10)).unwrap();
        }
        cycle(&mut ws, &Table::default());
        cycle(&mut ws, &Table::default());
        let now = ws.revision();
        let set = vec![
            by("x", 1, 10, now, 100),
            by("x", 1, 11, now, 99), // same author: its second displacement is refused
            by("y", 1, 12, now, 98),
            by("z", 1, 13, now, 97), // third displacement of the cycle: refused
        ];
        for p in shuffled(&set, order) {
            ws.submit(p).unwrap();
        }
        let snap = cycle(&mut ws, &Table::default()).snapshot;
        (snap, ws.counters().displaced())
    };
    let (baseline, displaced) = run(0);
    assert_eq!(displaced, 2);
    assert_eq!(
        admitted_contents(&baseline)
            .intersection(&BTreeSet::from([10, 12]))
            .count(),
        2
    );
    assert_eq!(baseline.refusals().len(), 2);
    for seed in 1..8 {
        assert_eq!(
            run(seed).0.digest(),
            baseline.digest(),
            "permutation {seed}"
        );
    }
}

// --- broadcast ------------------------------------------------------------------------------

#[test]
fn broadcast_is_one_identical_immutable_snapshot_for_every_subscriber() {
    let mut ws = workspace(2);
    let a = subscribe(&mut ws, "a");
    let b = subscribe(&mut ws, "b");
    for tag in 1..=3u8 {
        ws.submit(proposal(tag, ws.revision(), u32::from(tag)))
            .unwrap();
    }
    let report = cycle(&mut ws, &Table::default());
    assert_eq!(report.delivered, vec![a, b]);
    let sa = ws.receive(a).unwrap().unwrap();
    let sb = ws.receive(b).unwrap().unwrap();
    assert!(Arc::ptr_eq(&sa, &sb));
    assert!(Arc::ptr_eq(&sa, &report.snapshot));
    assert_eq!(sa.verifier(), &identity());
    let before = *sa.digest();
    cycle(&mut ws, &Table::default());
    assert_eq!(*sa.digest(), before);
    assert_eq!(ws.receive(a).unwrap().unwrap().revision().get(), 2);
}

#[test]
fn a_slow_subscriber_overflows_visibly_without_blocking_the_cycle() {
    let mut ws = workspace(1);
    let slow = subscribe(&mut ws, "slow");
    let fast = subscribe(&mut ws, "fast");
    for n in 1..=5u64 {
        let report = cycle(&mut ws, &Table::default());
        assert_eq!(report.snapshot.revision().get(), n);
        assert_eq!(ws.receive(fast).unwrap().unwrap().revision().get(), n);
        if n > 2 {
            assert_eq!(report.overflowed, vec![slow]);
        }
    }
    let stats = ws.subscriber_stats(slow).unwrap();
    assert_eq!((stats.queued, stats.delivered, stats.overflowed), (2, 2, 3));
    assert_eq!(ws.counters().mailbox_overflow(), 3);
    assert_eq!(ws.receive(slow).unwrap().unwrap().revision().get(), 1);
    ws.check_invariants().unwrap();
}

#[test]
fn subscriber_limits_and_unknown_ids() {
    let mut ws = build(WorkspaceConfigInit {
        max_subscribers: 1,
        ..init(1)
    });
    let id = subscribe(&mut ws, "a");
    assert_eq!(
        ws.subscribe(SubscribeRequest {
            pod: PodId::from("b"),
            generation: Generation(1)
        }),
        Err(WorkspaceError::SubscribersFull)
    );
    ws.unsubscribe(id).unwrap();
    assert_eq!(
        ws.unsubscribe(id),
        Err(WorkspaceError::UnknownSubscriber(id))
    );
    let again = subscribe(&mut ws, "a");
    assert_ne!(again, id);
}

#[test]
fn unsubscribing_between_cycles_stops_delivery() {
    let mut ws = workspace(1);
    let a = subscribe(&mut ws, "a");
    cycle(&mut ws, &Table::default());
    ws.unsubscribe(a).unwrap();
    let report = cycle(&mut ws, &Table::default());
    assert!(report.delivered.is_empty());
}

// --- audit and replay ------------------------------------------------------------------------

#[test]
fn snapshots_record_the_verifier_identity_report_digest_and_soft_findings_of_every_outcome() {
    let table = Table {
        soft_everywhere: true,
        ..Table::default()
    };
    let mut ws = workspace(1);
    ws.submit(by("a", 1, 1, ws.revision(), 9)).unwrap();
    ws.submit(by("b", 1, 2, ws.revision(), 1)).unwrap(); // outbid
    ws.submit(by("c", 1, 1, ws.revision(), 0)).unwrap(); // duplicate of tag 1
    let snap = cycle(&mut ws, &table).snapshot;
    let slot = snap.slots()[0].as_ref().unwrap();
    assert_eq!(slot.findings()[0].code(), "note");
    assert_ne!(slot.report_digest(), &[0; 32]);
    assert_eq!(snap.refusals().len(), 2);
    assert!(snap
        .refusals()
        .iter()
        .all(|r| r.findings().len() == 1 && r.findings()[0].code() == "note"));
    // A different verifier identity is a different snapshot.
    let other = VerifierIdentity::new("other", [1; 32]).unwrap();
    let mut ws2 = workspace(1);
    ws2.submit(by("a", 1, 1, ws2.revision(), 9)).unwrap();
    ws2.submit(by("b", 1, 2, ws2.revision(), 1)).unwrap();
    ws2.submit(by("c", 1, 1, ws2.revision(), 0)).unwrap();
    let snap2 = ws2
        .run_cycle(CycleHost {
            verifier: &table,
            identity: &other,
            authors: &UnrestrictedAuthors,
        })
        .unwrap()
        .snapshot;
    assert_ne!(snap.digest(), snap2.digest());
    assert!(VerifierIdentity::new(" ", [0; 32]).is_none());
}

#[test]
fn replay_gives_byte_identical_snapshot_digests() {
    let script = |config: WorkspaceConfigInit| -> Vec<[u8; 32]> {
        let mut ws = build(config);
        let r0 = ws.revision();
        let mut table = Table::with(
            3,
            report(S::Fail, L::Deterministic, vec![finding("c", true)]),
        );
        table
            .reports
            .insert(4, report(S::Unknown, L::Unverified, vec![]));
        let mut authors = Registry::default();
        let mut digests = Vec::new();
        for round in 0..8u8 {
            if round == 5 {
                authors.revoked.insert(("pod-9".into(), 1));
            }
            for k in 1..=6u8 {
                let tag = (round * 3 + k) % 11 + 1;
                let rev = if k == 6 && round > 0 {
                    r0
                } else {
                    ws.revision()
                };
                let _ = ws.submit(by(&format!("pod-{}", tag % 4 + 6), 1, tag, rev, 1));
            }
            if round == 3 {
                let _ = ws.request_retire(RetireRequest {
                    slot: SlotIndex::new(0),
                });
            }
            digests.push(*cycle_with(&mut ws, &table, &authors).snapshot.digest());
        }
        digests
    };
    let mut config = init(2);
    config.decay = SlotDecay::AfterCycles(nz(3));
    config.displacement = Displacement::Policy(DisplacementPolicy {
        margin: 2,
        min_residency: nz(1),
        max_per_cycle: 1,
        max_per_author: 1,
    });
    let a = script(config);
    assert_eq!(a, script(config));
    assert_eq!(
        a.iter().collect::<BTreeSet<_>>().len(),
        a.len(),
        "snapshots differ per cycle"
    );
    // The seed is an input: some other seed gives a different history.
    assert!((0..8u64).any(|seed| {
        a != script(WorkspaceConfigInit {
            tie_break_seed: seed,
            ..config
        })
    }));
}
