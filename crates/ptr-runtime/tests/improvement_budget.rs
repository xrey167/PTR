//! The improvement-cycle budget is counted from committed history: an aborted cycle and a
//! running one spend it, a replayed runtime counts the same, a clock that goes back is refused.

use ptr_config::PtrConfig;
use ptr_feedback::{BudgetExhausted, ImprovementCycleBudget};
use ptr_ledger::{ImprovementCycleOutcome, LedgerEvent};
use ptr_runtime::adapter::{AdapterError, BeginImprovementCycle};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::ProjectId;

fn begin(project: &str, day: u64) -> BeginImprovementCycle {
    BeginImprovementCycle {
        project: ProjectId::from(project),
        day,
        policy_digest: [7; 32],
    }
}

fn runtime(per_day: u32) -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    runtime.set_improvement_budget(ImprovementCycleBudget::new(per_day));
    runtime
}

fn exhausted(day: u64, used: u64, per_day: u32) -> RuntimeError {
    RuntimeError::Adapter(AdapterError::BudgetExhausted(BudgetExhausted {
        day,
        used,
        per_day,
    }))
}

#[test]
fn no_cycle_is_admitted_without_a_configured_budget() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(RuntimeError::Adapter(AdapterError::NoImprovementBudget))
    );
    let mut zero = self::runtime(0);
    assert_eq!(
        zero.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 0, 0))
    );
}

#[test]
fn an_exhausted_budget_refuses_the_next_cycle_even_when_the_earlier_ones_aborted() {
    let mut runtime = runtime(2);
    let first = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("first");
    runtime
        .record_improvement_cycle(
            first,
            ImprovementCycleOutcome::Aborted {
                reason: "evaluation diverged".into(),
            },
        )
        .expect("records the abort");
    let second = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("second");
    runtime
        .record_improvement_cycle(second, ImprovementCycleOutcome::Completed)
        .expect("records completion");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 2, 2))
    );
    assert_eq!(
        runtime.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        2
    );
}

#[test]
fn a_running_cycle_and_a_dropped_ticket_count_against_the_budget() {
    let mut runtime = runtime(1);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("admitted");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 1, 1))
    );
    drop(ticket);
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 1, 1)),
        "a ticket nobody recorded stays counted"
    );
}

#[test]
fn the_count_is_rebuilt_from_the_ledger_and_the_next_day_starts_fresh() {
    let mut runtime = runtime(1);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("admitted");
    runtime
        .record_improvement_cycle(ticket, ImprovementCycleOutcome::Completed)
        .expect("records");
    let history = runtime.committed_events().to_vec();
    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &history).expect("replays");
    replayed.set_improvement_budget(ImprovementCycleBudget::new(1));
    assert_eq!(
        replayed.improvement_cycles_used(&ProjectId::from("alpha"), 10),
        1
    );
    assert_eq!(
        replayed.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 1, 1))
    );
    let _next_day = replayed
        .begin_improvement_cycle(begin("alpha", 11))
        .expect("a new day has a fresh budget");
}

#[test]
fn a_clock_that_goes_back_does_not_hand_out_a_fresh_budget() {
    let mut runtime = runtime(1);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("admitted");
    runtime
        .record_improvement_cycle(ticket, ImprovementCycleOutcome::Completed)
        .expect("records");
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha", 11))
        .expect("next day");
    runtime
        .record_improvement_cycle(ticket, ImprovementCycleOutcome::Completed)
        .expect("records");
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(RuntimeError::Adapter(AdapterError::ClockRegressed {
            recorded_day: 11,
            requested_day: 10,
        }))
    );
}

#[test]
fn budgets_are_per_project() {
    let mut runtime = runtime(1);
    let _alpha = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("alpha");
    let _beta = runtime
        .begin_improvement_cycle(begin("beta", 10))
        .expect("beta has its own budget");
}

#[test]
fn a_zero_policy_digest_and_the_public_commit_are_refused() {
    let mut runtime = runtime(3);
    let mut request = begin("alpha", 10);
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
    let mut runtime = runtime(1);
    let ticket = runtime
        .begin_improvement_cycle(begin("alpha", 10))
        .expect("admitted");
    assert!(runtime
        .record_improvement_cycle(
            ticket,
            ImprovementCycleOutcome::Aborted {
                reason: String::new()
            }
        )
        .is_err());
    assert_eq!(
        runtime.begin_improvement_cycle(begin("alpha", 10)),
        Err(exhausted(10, 1, 1))
    );
}

#[test]
fn the_tracked_projects_are_bounded_for_admitted_and_for_recorded_cycles() {
    use ptr_runtime::adapter::MAX_TRACKED_CYCLE_PROJECTS;
    let too_many = RuntimeError::Adapter(AdapterError::TooManyCycleProjects {
        limit: MAX_TRACKED_CYCLE_PROJECTS,
    });

    // Admitted and never recorded: the open table stops at the bound.
    let mut open = runtime(1);
    let mut held = Vec::new();
    for n in 0..MAX_TRACKED_CYCLE_PROJECTS {
        held.push(
            open.begin_improvement_cycle(begin(&format!("p{n}"), 1))
                .expect("within the bound"),
        );
    }
    assert_eq!(
        open.begin_improvement_cycle(begin("one-more", 1)),
        Err(too_many.clone())
    );

    // Recorded: the table of latest days stops at the bound too.
    let mut recorded = runtime(1);
    for n in 0..MAX_TRACKED_CYCLE_PROJECTS {
        let ticket = recorded
            .begin_improvement_cycle(begin(&format!("p{n}"), 1))
            .expect("within the bound");
        recorded
            .record_improvement_cycle(ticket, ImprovementCycleOutcome::Completed)
            .expect("records");
    }
    assert_eq!(
        recorded.begin_improvement_cycle(begin("one-more", 1)),
        Err(too_many)
    );
}
