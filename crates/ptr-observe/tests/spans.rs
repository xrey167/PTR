use std::sync::Arc;

use ptr_observe::{
    ManualClock, SpanError, SpanStatus, TraceEvent, TraceLevel, TraceValue, Tracer, TracerConfig,
    TracerLimits,
};

fn tracer_with(limits: TracerLimits) -> (Tracer, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new(0));
    let tracer = Tracer::new(
        TracerConfig {
            id_seed: 42,
            limits,
        },
        clock.clone(),
    );
    (tracer, clock)
}

#[test]
fn nested_spans_share_trace_and_link_parents() {
    let (tracer, clock) = tracer_with(TracerLimits::default());
    let root = tracer.root_span("request").expect("root");
    let a = root.child("route").expect("a");
    clock.advance(3);
    let b = a.child("pod").expect("b");
    clock.advance(4);
    b.event(TraceEvent::new(TraceLevel::Info, "lease").with_field("n", 1u64));
    b.finish();
    a.finish();
    root.finish();

    let spans = tracer.drain_finished();
    let names: Vec<_> = spans.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["pod", "route", "request"]);
    assert!(spans.iter().all(|s| s.trace_id == spans[0].trace_id));
    assert_eq!(spans[0].parent, Some(spans[1].id));
    assert_eq!(spans[1].parent, Some(spans[2].id));
    assert_eq!(spans[2].parent, None);
    assert_eq!(spans[0].events[0].tick, 7);
    assert_eq!(spans[0].duration(), 4);
    assert_eq!(spans[2].duration(), 7);
    assert!(spans.iter().all(|s| s.status == SpanStatus::Ok));
}

#[test]
fn separate_roots_get_distinct_traces_and_runs_are_reproducible() {
    let run = || {
        let (tracer, _) = tracer_with(TracerLimits::default());
        let a = tracer.root_span("a").expect("a");
        let b = tracer.root_span("b").expect("b");
        (a.context(), b.context())
    };
    let (a, b) = run();
    assert_ne!(a.trace_id, b.trace_id);
    assert_ne!(a.span_id, b.span_id);
    assert_eq!(run(), (a, b));
}

#[test]
fn dropping_guards_closes_spans_and_frees_open_slots() {
    let limits = TracerLimits {
        max_open_spans: 2,
        ..TracerLimits::default()
    };
    let (tracer, _) = tracer_with(limits);
    {
        let root = tracer.root_span("r").expect("r");
        let _child = root.child("c").expect("c");
        assert_eq!(
            root.child("overflow").expect_err("limit"),
            SpanError::TooManyOpenSpans { limit: 2 }
        );
        assert_eq!(tracer.open_span_count(), 2);
    }
    assert_eq!(tracer.open_span_count(), 0);
    assert_eq!(tracer.finished().len(), 2);
    assert!(tracer.root_span("again").is_ok());
}

#[test]
fn bounds_are_counted_and_deterministic() {
    let limits = TracerLimits {
        max_fields_per_event: 2,
        max_events_per_span: 2,
        max_value_len: 8,
        max_finished_spans: 3,
        ..TracerLimits::default()
    };
    let (tracer, _) = tracer_with(limits);
    for i in 0..5u64 {
        let span = tracer.root_span(format!("span-{i}")).expect("open");
        span.set_field("long", "x".repeat(100));
        for j in 0..4u64 {
            span.event(TraceEvent::new(TraceLevel::Debug, "e").with_field("j", j));
        }
        span.finish();
    }
    let spans = tracer.finished();
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0].name, "span-2");
    assert_eq!(spans[0].fields["long"], TraceValue::String("x".repeat(8)));
    assert_eq!(spans[0].events.len(), 2);
    let stats = tracer.stats();
    assert_eq!(stats.spans_finished, 5);
    assert_eq!(stats.dropped_finished, 2);
    assert_eq!(stats.dropped_events, 10);
    assert!(stats.truncated_values >= 5);
}

#[test]
fn oversized_request_id_is_truncated_like_strings() {
    let limits = TracerLimits {
        max_value_len: 16,
        ..TracerLimits::default()
    };
    let (tracer, _) = tracer_with(limits);
    let span = tracer.root_span("request").expect("open");
    span.set_field("rid", ptr_types::RequestId("r".repeat(10_000)));
    span.event(TraceEvent::request(
        TraceLevel::Info,
        "ev",
        ptr_types::RequestId("q".repeat(10_000)),
        ptr_types::Revision(1),
    ));
    span.finish();
    let spans = tracer.finished();
    assert_eq!(
        spans[0].fields["rid"],
        TraceValue::RequestId(ptr_types::RequestId("r".repeat(16)))
    );
    assert_eq!(
        spans[0].events[0].event.fields["ptr.request_id"],
        TraceValue::RequestId(ptr_types::RequestId("q".repeat(16)))
    );
    assert!(tracer.stats().truncated_values >= 2);
}

#[test]
fn child_of_finished_parent_is_typed_error() {
    let (tracer, _) = tracer_with(TracerLimits::default());
    let root = tracer.root_span("r").expect("root");
    let ctx = root.context();
    drop(root);
    let error = tracer
        .start_span("late", Some(ctx))
        .expect_err("closed parent");
    assert_eq!(
        error,
        SpanError::ParentNotOpen {
            span_id: ctx.span_id
        }
    );
}
