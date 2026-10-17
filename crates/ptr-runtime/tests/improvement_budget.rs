//! The improvement-cycle budget is counted from committed history and a clock the caller does not
//! control: an aborted cycle and a running one spend it, a replayed runtime counts the same, a
//! clock that goes back (or a history from the future) is refused, and going forward and back
//! cannot buy a second cycle on a day.
use ptr_config::PtrConfig;
use ptr_feedback::{BudgetExhausted, ImprovementCycleBudget};
use ptr_ledger::{ImprovementCycleOutcome, LedgerEvent};
use ptr_runtime::adapter::{AdapterError, BeginImprovementCycle, CycleClock};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::ProjectId;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A clock the test sets; `u64::MAX` reads as "cannot say".
#[derive(Debug)]
struct ManualClock(AtomicU64);

impl ManualClock {
    fn at(day: u64) -> Arc<Self> {
        Arc::new(Self(AtomicU64::new(day)))
    }

    fn set(&self, day: u64) {
        self.0.store(day, Ordering::SeqCst);
    }
}

impl CycleClock for ManualClock {
    fn today(&self) -> Option<u64> {
        let day = self.0.load(Ordering::SeqCst);
        (day != u64::MAX).then_some(day)
    }
}

fn begin(project: &str) -> BeginImprovementCycle {
    BeginImprovementCycle {
        project: ProjectId::from(project),
        policy_digest: [7; 32],
    }
}

fn runtime(per_day: u32, day: u64) -> (PtrRuntime, Arc<ManualClock>) {
    let clock = ManualClock::at(day);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    runtime
        .set_improvement_budget(ImprovementCycleBudget::new(per_day), clock.clone())
        .expect("sets once");
    (runtime, clock)
}

fn exhausted(day: u64, used: u64, per_day: u32) -> RuntimeError {
    RuntimeError::Adapter(AdapterError::BudgetExhausted(BudgetExhausted {
        day,
        used,
        per_day,
    }))
}

fn regressed(high_water_day: u64, today: u64) -> RuntimeError {
    RuntimeError::Adapter(AdapterError::ClockRegressed {
        high_water_day,
        today,
    })
}

#[test]
fn no_cycle_is_admitted_without_a_configured_budget_and_clock() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(RuntimeError::Adapter(AdapterError::NoImprovementBudget))
    );
    let (mut zero, _clock) = self::runtime(0, 10);
    assert_eq!(
        zero.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 0, 0))
    );
}

#[test]
fn an_unreadable_clock_admits_nothing() {
    let (mut runtime, clock) = runtime(5, 10);
    clock.set(u64::MAX);
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(RuntimeError::Adapter(AdapterError::ClockUnavailable))
    );
}

#[test]
fn an_exhausted_budget_refuses_the_next_cycle_even_when_the_earlier_ones_aborted() {
    let (mut runtime, _clock) = runtime(2, 10);
    let first = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("first");
    runtime
        .record_improvement_cycle(
            &first,
            ImprovementCycleOutcome::Aborted {
                reason: "evaluation diverged".into(),
            },
        )
        .expect("records the abort");
    let second = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("second");
    runtime
        .record_improvement_cycle(&second, ImprovementCycleOutcome::Completed)
        .expect("records completion");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 2, 2))
    );
    assert_eq!(
        runtime.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        2
    );
}

#[test]
fn a_running_cycle_and_a_dropped_ticket_count_against_the_budget() {
    let (mut runtime, _clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1))
    );
    drop(ticket);
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1)),
        "a ticket nobody recorded stays counted"
    );
}

#[test]
fn going_to_the_next_day_and_back_does_not_buy_a_second_cycle_on_the_first_day() {
    // per_day = 1: a cycle runs on day 10, one is admitted on day 11, and then the clock reads
    // day 10 again. The open count of day 10 was dropped when day 11 began; the high-water day
    // is what refuses day 10 now.
    let (mut runtime, clock) = runtime(1, 10);
    let _running = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("day 10");
    clock.set(11);
    let _next = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("day 11 has its own budget");
    clock.set(10);
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(regressed(11, 10))
    );
    assert_eq!(
        runtime.improvement_cycles_used(&ProjectId::from("alpha"), 11),
        1
    );
}

#[test]
fn the_caller_cannot_count_upwards_to_a_fresh_budget() {
    // There is no day in the request, and the clock says day 10 however often one asks.
    let (mut runtime, _clock) = runtime(1, 10);
    let _first = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    for _ in 0..5 {
        assert_eq!(
            runtime.begin_improvement_cycle(begin("alpha")),
            Err(exhausted(10, 1, 1))
        );
    }
}

#[test]
fn the_count_is_rebuilt_from_the_ledger_and_the_next_day_starts_fresh() {
    let (mut runtime, _clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    runtime
        .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
        .expect("records");
    let history = runtime.committed_events().to_vec();

    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    let clock = ManualClock::at(10);
    replayed
        .set_improvement_budget(ImprovementCycleBudget::new(1), clock.clone())
        .expect("sets once");
    assert_eq!(
        replayed.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        1
    );
    assert_eq!(
        replayed.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1))
    );
    clock.set(11);
    let _next_day = replayed
        .begin_improvement_cycle(begin("alpha"))
        .expect("a new day has a fresh budget");
}

#[test]
fn a_history_from_the_future_refuses_cycles_until_the_clock_catches_up() {
    let (mut runtime, _clock) = runtime(1, 20);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    runtime
        .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
        .expect("records");
    let history = runtime.committed_events().to_vec();

    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    let clock = ManualClock::at(19);
    replayed
        .set_improvement_budget(ImprovementCycleBudget::new(1), clock.clone())
        .expect("sets once");
    assert_eq!(
        replayed.begin_improvement_cycle(begin("alpha")),
        Err(regressed(20, 19))
    );
    clock.set(21);
    assert!(replayed.begin_improvement_cycle(begin("alpha")).is_ok());
}

#[test]
fn budgets_are_per_project() {
    let (mut runtime, _clock) = runtime(1, 10);
    let _alpha = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("alpha");
    let _beta = runtime
        .begin_improvement_cycle(begin("beta"))
        .expect("beta has its own budget");
}

#[test]
fn a_zero_policy_digest_and_the_public_commit_are_refused() {
    let (mut runtime, _clock) = runtime(3, 10);
    let mut request = begin("alpha");
    request.policy_digest = [0; 32];
    assert_eq!(
        runtime.begin_improvement_cycle(request),
        Err(RuntimeError::Adapter(AdapterError::ZeroPolicyDigest))
    );
    assert_eq!(
        runtime.commit(LedgerEvent::ImprovementCycleRecorded {
            project: ProjectId::from("alpha"),
            day: 10,
            outcome: ImprovementCycleOutcome::Completed,
            policy_digest: [7; 32],
        }),
        Err(RuntimeError::StructuralRecordOutsideAdmissionPath)
    );
}

#[test]
fn a_malformed_abort_reason_is_refused_and_leaves_the_cycle_counted() {
    let (mut runtime, _clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    assert!(runtime
        .record_improvement_cycle(
            &ticket,
            ImprovementCycleOutcome::Aborted {
                reason: String::new()
            }
        )
        .is_err());
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1))
    );
}

#[test]
fn the_tracked_projects_are_bounded_for_admitted_and_for_recorded_cycles() {
    use ptr_runtime::adapter::MAX_TRACKED_CYCLE_PROJECTS;
    let too_many = RuntimeError::Adapter(AdapterError::TooManyCycleProjects {
        limit: MAX_TRACKED_CYCLE_PROJECTS,
    });

    // Admitted and never recorded: the table stops at the bound.
    let (mut open, _clock) = runtime(1, 1);
    let mut held = Vec::new();
    for n in 0..MAX_TRACKED_CYCLE_PROJECTS {
        held.push(
            open.begin_improvement_cycle(begin(&format!("p{n}")))
                .expect("within the bound"),
        );
    }
    assert_eq!(
        open.begin_improvement_cycle(begin("one-more")),
        Err(too_many.clone())
    );

    // Recorded: the same bound holds.
    let (mut recorded, _clock) = runtime(1, 1);
    for n in 0..MAX_TRACKED_CYCLE_PROJECTS {
        let ticket = recorded
            .begin_improvement_cycle(begin(&format!("p{n}")))
            .expect("within the bound");
        recorded
            .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
            .expect("records");
    }
    assert_eq!(
        recorded.begin_improvement_cycle(begin("one-more")),
        Err(too_many)
    );
}

#[test]
fn an_admitted_cycle_that_is_not_recorded_blocks_a_compacted_snapshot() {
    let (mut runtime, _clock) = runtime(1, 10);
    assert!(runtime.export_compacted_snapshot().is_ok());
    let _running = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    assert!(matches!(
        runtime.export_compacted_snapshot(),
        Err(RuntimeError::Adapter(AdapterError::CompactionUnsupported))
    ));
}

#[test]
fn a_ticket_of_another_runtime_is_refused_and_commits_nothing() {
    let (mut a, _clock_a) = runtime(2, 10);
    let (mut b, _clock_b) = runtime(2, 10);
    // B has an admission for the very same project and day, so a count-only check would pass.
    let _b_own = b
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted by b");
    let ticket_of_a = a
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted by a");
    let before = b.committed_events().len();
    assert_eq!(
        b.record_improvement_cycle(&ticket_of_a, ImprovementCycleOutcome::Completed),
        Err(RuntimeError::Adapter(AdapterError::TicketNotAdmitted))
    );
    assert_eq!(b.committed_events().len(), before);
    assert_eq!(b.improvement_cycles_used(&ProjectId::from("alpha"), 10), 1);
}

#[test]
fn an_admission_that_stayed_open_past_the_retention_can_no_longer_be_recorded() {
    use ptr_runtime::adapter::OPEN_CYCLE_RETENTION_DAYS;
    let (mut runtime, clock) = runtime(1, 10);
    let stale = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    clock.set(10 + OPEN_CYCLE_RETENTION_DAYS + 1);
    let _fresh = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("a later day has its own budget");
    let before = runtime.committed_events().len();
    assert_eq!(
        runtime.record_improvement_cycle(&stale, ImprovementCycleOutcome::Completed),
        Err(RuntimeError::Adapter(AdapterError::TicketNotAdmitted))
    );
    assert_eq!(runtime.committed_events().len(), before);
}

#[test]
fn replay_refuses_a_history_that_names_more_projects_than_the_runtime_tracks() {
    use ptr_ledger::CommittedEvent;
    use ptr_runtime::adapter::MAX_TRACKED_CYCLE_PROJECTS;
    use ptr_types::CommitIndex;

    let history: Vec<CommittedEvent> = (0..=MAX_TRACKED_CYCLE_PROJECTS)
        .map(|n| CommittedEvent {
            index: CommitIndex(n as u64 + 1),
            event: LedgerEvent::ImprovementCycleRecorded {
                project: ProjectId::from(format!("p{n}").as_str()),
                day: 1,
                outcome: ImprovementCycleOutcome::Completed,
                policy_digest: [7; 32],
            },
        })
        .collect();
    assert_eq!(
        PtrRuntime::replay(PtrConfig::default(), &history).err(),
        Some(RuntimeError::Adapter(AdapterError::TooManyCycleProjects {
            limit: MAX_TRACKED_CYCLE_PROJECTS,
        }))
    );
    // One project fewer replays.
    assert!(
        PtrRuntime::replay(PtrConfig::default(), &history[..MAX_TRACKED_CYCLE_PROJECTS]).is_ok()
    );
}

#[test]
fn a_ticket_older_than_the_retention_is_refused_at_record_time_without_another_begin() {
    use ptr_runtime::adapter::OPEN_CYCLE_RETENTION_DAYS;
    let (mut runtime, clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    // The clock moves past the retention; nobody calls begin in between.
    clock.set(10 + OPEN_CYCLE_RETENTION_DAYS + 1);
    let before = runtime.committed_events().len();
    assert_eq!(
        runtime.record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed),
        Err(RuntimeError::Adapter(AdapterError::TicketNotAdmitted))
    );
    assert_eq!(runtime.committed_events().len(), before);
}

#[test]
fn a_ticket_within_the_retention_is_still_recorded() {
    use ptr_runtime::adapter::OPEN_CYCLE_RETENTION_DAYS;
    let (mut runtime, clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    clock.set(10 + OPEN_CYCLE_RETENTION_DAYS);
    assert!(runtime
        .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
        .is_ok());
}

#[test]
fn recording_with_an_unreadable_clock_is_refused_and_the_admission_stays_counted() {
    let (mut runtime, clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    clock.set(u64::MAX);
    let before = runtime.committed_events().len();
    assert_eq!(
        runtime.record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed),
        Err(RuntimeError::Adapter(AdapterError::ClockUnavailable))
    );
    assert_eq!(runtime.committed_events().len(), before);
    assert_eq!(
        runtime.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        1
    );
}

#[test]
fn one_projects_expired_admissions_do_not_keep_another_project_out() {
    use ptr_runtime::adapter::{MAX_TRACKED_CYCLE_PROJECTS, OPEN_CYCLE_RETENTION_DAYS};
    let (mut runtime, clock) = runtime(5_000, 1);
    let mut held = Vec::new();
    for _ in 0..MAX_TRACKED_CYCLE_PROJECTS {
        held.push(
            runtime
                .begin_improvement_cycle(begin("alpha"))
                .expect("within the bound"),
        );
    }
    // The table is full while the admissions are live.
    assert!(matches!(
        runtime.begin_improvement_cycle(begin("beta")),
        Err(RuntimeError::Adapter(
            AdapterError::TooManyCycleProjects { .. }
        ))
    ));
    // Once they have expired, they are swept before the bound is looked at.
    clock.set(1 + OPEN_CYCLE_RETENTION_DAYS + 1);
    assert!(runtime.begin_improvement_cycle(begin("beta")).is_ok());
}

#[test]
fn a_cycle_recorded_out_of_day_order_is_counted_on_its_own_day_live_and_on_replay() {
    let (mut runtime, clock) = runtime(2, 10);
    let on_day_10 = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("day 10");
    clock.set(11);
    let on_day_11 = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("day 11");
    runtime
        .record_improvement_cycle(&on_day_11, ImprovementCycleOutcome::Completed)
        .expect("records day 11 first");
    runtime
        .record_improvement_cycle(&on_day_10, ImprovementCycleOutcome::Completed)
        .expect("then day 10");
    let alpha = ProjectId::from("alpha");
    assert_eq!(runtime.improvement_cycles_used(&alpha, 10), 1);
    assert_eq!(runtime.improvement_cycles_used(&alpha, 11), 1);

    let history = runtime.committed_events().to_vec();
    let replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    assert_eq!(replayed.improvement_cycles_used(&alpha, 10), 1);
    assert_eq!(replayed.improvement_cycles_used(&alpha, 11), 1);
}

#[test]
fn projects_kept_only_by_expired_admissions_do_not_fill_the_project_bound() {
    use ptr_runtime::adapter::{MAX_TRACKED_CYCLE_PROJECTS, OPEN_CYCLE_RETENTION_DAYS};
    let (mut runtime, clock) = runtime(5_000, 1);
    // One admission each for as many distinct projects as the bound allows, none recorded.
    let mut held = Vec::new();
    for n in 0..MAX_TRACKED_CYCLE_PROJECTS {
        held.push(
            runtime
                .begin_improvement_cycle(begin(&format!("project-{n}")))
                .expect("within the bound"),
        );
    }
    assert!(matches!(
        runtime.begin_improvement_cycle(begin("newcomer")),
        Err(RuntimeError::Adapter(
            AdapterError::TooManyCycleProjects { .. }
        ))
    ));
    // Their admissions expire: what remains of them is a high-water day, which must not keep a
    // new project out.
    clock.set(1 + OPEN_CYCLE_RETENTION_DAYS + 1);
    assert!(runtime.begin_improvement_cycle(begin("newcomer")).is_ok());
}

#[test]
fn a_record_refused_before_the_append_leaves_the_ticket_usable() {
    let (mut runtime, _clock) = runtime(1, 10);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    // Refused before anything is appended: a malformed abort reason.
    assert!(runtime
        .record_improvement_cycle(
            &ticket,
            ImprovementCycleOutcome::Aborted {
                reason: String::new()
            }
        )
        .is_err());
    let before = runtime.committed_events().len();
    // The same ticket records the cycle once the outcome is right.
    runtime
        .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
        .expect("the ticket was not consumed");
    assert_eq!(runtime.committed_events().len(), before + 1);
    // The budget counts the cycle once, and the ticket cannot be used a second time.
    assert_eq!(
        runtime.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        1
    );
    assert_eq!(
        runtime.record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed),
        Err(RuntimeError::Adapter(AdapterError::TicketNotAdmitted))
    );
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1))
    );
}

#[test]
fn the_budget_is_set_once_and_neither_a_looser_nor_a_tighter_one_replaces_it() {
    let (mut runtime, clock) = runtime(1, 10);
    // Use up the one cycle of the day.
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha"))
        .expect("admitted");
    runtime
        .record_improvement_cycle(&ticket, ImprovementCycleOutcome::Completed)
        .expect("recorded");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1))
    );
    // A looser budget is refused and changes nothing; so is a tighter one.
    for requested in [2, 0] {
        assert_eq!(
            runtime.set_improvement_budget(ImprovementCycleBudget::new(requested), clock.clone()),
            Err(AdapterError::BudgetPinConflict { set: 1, requested })
        );
    }
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha")),
        Err(exhausted(10, 1, 1)),
        "the next cycle stays refused"
    );
    // The same budget again changes nothing.
    assert_eq!(
        runtime.set_improvement_budget(ImprovementCycleBudget::new(1), clock),
        Ok(())
    );
}

#[test]
fn a_project_the_ledger_cannot_record_is_refused_before_anything_changes() {
    let (mut runtime, _clock) = runtime(5, 10);
    let too_long = "x".repeat(257);
    for bad in ["", " ", "a b", "a/b", "tab\there", too_long.as_str()] {
        assert_eq!(
            runtime.begin_improvement_cycle(begin(bad)),
            Err(RuntimeError::Adapter(AdapterError::InvalidProject)),
            "{bad:?}"
        );
    }
    // Nothing was counted or tracked: the budget is intact and a valid project is admitted.
    let alpha = ProjectId::from("alpha");
    assert_eq!(runtime.improvement_cycles_used(&alpha, 10), 0);
    for bad in ["", "a/b"] {
        assert_eq!(
            runtime.improvement_cycles_used(&ProjectId::from(bad), 10),
            0
        );
    }
    assert!(runtime.begin_improvement_cycle(begin("alpha")).is_ok());
    assert_eq!(runtime.improvement_cycles_used(&alpha, 10), 1);
}
