use std::sync::Arc;

use ptr_observe::{
    correlate, fields, ManualClock, PlanRecord, PlannedStep, Span, StepOutcome, Tracer,
    TracerConfig,
};

fn plan() -> PlanRecord {
    PlanRecord::new(
        "plan-1",
        vec![
            PlannedStep::new("s1", "retrieve"),
            PlannedStep::new("s2", "verify"),
            PlannedStep::new("s3", "commit"),
            PlannedStep::new("s4", "notify"),
        ],
    )
    .expect("plan")
}

/// Runs steps in the given order; each entry is `(step id, operation, fails)`.
fn execute(steps: &[(&str, &str, bool)]) -> Vec<Span> {
    let clock = Arc::new(ManualClock::new(0));
    let tracer = Tracer::new(TracerConfig::default(), clock.clone());
    for (step, operation, fails) in steps {
        let span = tracer.root_span(format!("exec-{step}")).expect("open");
        span.set_field(fields::PLAN_STEP, *step);
        span.set_field(fields::OPERATION, *operation);
        clock.advance(10);
        if *fails {
            span.fail("error");
        } else {
            span.finish();
        }
        clock.advance(1);
    }
    tracer.drain_finished()
}

fn outcomes(spans: &[Span]) -> Vec<(String, StepOutcome)> {
    correlate(&plan(), spans)
        .steps
        .into_iter()
        .map(|row| (row.step_id, row.outcome))
        .collect()
}

#[test]
fn clean_execution_reports_all_executed() {
    let spans = execute(&[
        ("s1", "retrieve", false),
        ("s2", "verify", false),
        ("s3", "commit", false),
        ("s4", "notify", false),
    ]);
    let report = correlate(&plan(), &spans);
    assert!(report.is_clean());
    assert_eq!(report.summary.executed, 4);
    assert_eq!(report.summary.planned, 4);
    assert!(report.steps.iter().all(|row| !row.operation_mismatch));
    assert!(report
        .render_text()
        .contains("summary: planned=4 executed=4"));
}

#[test]
fn skipped_failed_out_of_order_and_unplanned_are_distinguished() {
    // Planned: s1 s2 s3 s4. Run: s1, s3 (fails), s2 (late), s9 (unplanned). s4 never runs.
    let spans = execute(&[
        ("s1", "retrieve", false),
        ("s3", "commit", true),
        ("s2", "verify", false),
        ("s9", "extra", false),
    ]);
    let report = correlate(&plan(), &spans);
    assert_eq!(
        outcomes(&spans),
        [
            ("s1".to_owned(), StepOutcome::Executed),
            ("s2".to_owned(), StepOutcome::Executed),
            ("s3".to_owned(), StepOutcome::Failed),
            ("s4".to_owned(), StepOutcome::Skipped),
            ("s9".to_owned(), StepOutcome::Unplanned),
        ]
    );
    assert!(!report.is_clean());
    assert_eq!(report.steps[4].plan_index, None);
    let s = report.summary;
    assert_eq!((s.executed, s.failed, s.skipped, s.unplanned), (2, 1, 1, 1));
}

#[test]
fn step_that_ran_before_an_earlier_planned_step_is_out_of_order() {
    let spans = execute(&[
        ("s1", "retrieve", false),
        ("s3", "commit", false),
        ("s2", "verify", false),
        ("s4", "notify", false),
    ]);
    // s3 started before s2, which is planned earlier, so s3 is out of order.
    let report = correlate(&plan(), &spans);
    let got: Vec<_> = report.steps.iter().map(|r| r.outcome).collect();
    assert_eq!(
        got,
        [
            StepOutcome::Executed,
            StepOutcome::Executed,
            StepOutcome::OutOfOrder,
            StepOutcome::Executed
        ]
    );
    assert_eq!(report.summary.out_of_order, 1);
}

#[test]
fn retry_success_is_executed_and_final_failure_is_failed() {
    let recovered = execute(&[
        ("s1", "retrieve", false),
        ("s2", "verify", true),
        ("s2", "verify", false),
    ]);
    let report = correlate(&plan(), &recovered);
    assert_eq!(report.steps[1].outcome, StepOutcome::Executed);
    assert_eq!(report.steps[1].executions, 2);

    let broken = execute(&[
        ("s1", "retrieve", false),
        ("s2", "verify", false),
        ("s2", "verify", true),
    ]);
    assert_eq!(
        correlate(&plan(), &broken).steps[1].outcome,
        StepOutcome::Failed
    );
}

#[test]
fn operation_mismatch_is_flagged_and_untagged_spans_counted() {
    let mut spans = execute(&[("s1", "wrong-op", false)]);
    let clock = Arc::new(ManualClock::new(0));
    let tracer = Tracer::new(TracerConfig::default(), clock);
    tracer.root_span("untagged").expect("open").finish();
    spans.extend(tracer.drain_finished());
    let report = correlate(&plan(), &spans);
    assert!(report.steps[0].operation_mismatch);
    assert_eq!(report.steps[0].outcome, StepOutcome::Executed);
    assert_eq!(report.summary.untagged_spans, 1);
}

#[test]
fn correlation_is_independent_of_input_order() {
    let spans = execute(&[
        ("s1", "retrieve", false),
        ("s3", "commit", false),
        ("s2", "verify", false),
    ]);
    let mut reversed = spans.clone();
    reversed.reverse();
    assert_eq!(correlate(&plan(), &spans), correlate(&plan(), &reversed));
}
