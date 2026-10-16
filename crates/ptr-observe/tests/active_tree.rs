use std::sync::Arc;

use ptr_observe::{ManualClock, Tracer, TracerConfig, STUCK_PATH_MAX_BYTES, STUCK_PATH_MAX_DEPTH};

fn setup() -> (Tracer, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new(1_000));
    (Tracer::new(TracerConfig::default(), clock.clone()), clock)
}

#[test]
fn tree_renders_nesting_and_ages() {
    let (tracer, clock) = setup();
    let root = tracer.root_span("request").expect("root");
    clock.advance(1_000_000);
    let route = root.child("route").expect("route");
    clock.advance(500);
    let _pod = route.child("pod").expect("pod");
    let _verify = root.child("verify").expect("verify");
    clock.advance(1_500);

    let tree = tracer.active_tree();
    assert_eq!(tree.spans.len(), 4);
    assert_eq!(tree.at_tick, 1_000 + 1_000_000 + 500 + 1_500);
    let text = tree.render_text();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(
        lines[0],
        format!("active spans at tick {}: 4", tree.at_tick)
    );
    assert!(lines[1].starts_with("- request ["));
    assert!(lines[1].ends_with("open 1.002ms"), "{}", lines[1]);
    assert!(lines[2].starts_with("  - route ["));
    assert!(lines[3].starts_with("    - pod ["));
    assert!(lines[3].ends_with("open 1.500us"), "{}", lines[3]);
    assert!(lines[4].starts_with("  - verify ["));
}

#[test]
fn finished_spans_leave_the_tree_and_orphans_become_roots() {
    let (tracer, _) = setup();
    let root = tracer.root_span("parent").expect("root");
    let child = root.child("child").expect("child");
    drop(root);
    let text = tracer.active_tree().render_text();
    assert!(text.contains("- child ["));
    assert!(text.contains("(parent closed)"));
    assert!(!text.contains("- parent"));
    drop(child);
    let empty = tracer.active_tree();
    assert!(empty.spans.is_empty());
    assert!(empty.stuck(0).is_empty());
}

#[test]
fn stuck_detection_uses_threshold_and_reports_path() {
    let (tracer, clock) = setup();
    let root = tracer.root_span("request").expect("root");
    clock.advance(1);
    let slow = root.child("exec").expect("slow");
    clock.advance(1);
    let _leaf = slow.child("model.call").expect("leaf");
    clock.advance(10_000);
    let fresh = root.child("fresh").expect("fresh");
    clock.advance(100);

    let tree = tracer.active_tree();
    let stuck = tree.stuck(5_000);
    let paths: Vec<_> = stuck.iter().map(|s| s.path.join("/")).collect();
    assert_eq!(
        paths,
        ["request", "request/exec", "request/exec/model.call"]
    );
    assert!(stuck.iter().all(|s| s.age >= 10_000));
    assert!(!stuck.iter().any(|s| s.context == fresh.context()));
    assert_eq!(tree.stuck(100).len(), 4);
    assert!(tree.stuck(1_000_000).is_empty());
}

#[test]
fn stuck_paths_stay_bounded_for_deep_chains_at_default_limits() {
    let (tracer, clock) = setup();
    let limits = tracer.limits();
    let name = "n".repeat(limits.max_value_len);
    let mut guards: Vec<ptr_observe::SpanGuard> = Vec::new();
    for depth in 0..limits.max_open_spans {
        let guard = match guards.last() {
            None => tracer.root_span(name.clone()),
            Some(parent) => parent.child(name.clone()),
        }
        .unwrap_or_else(|e| panic!("depth {depth}: {e}"));
        guards.push(guard);
        clock.advance(1);
    }
    clock.advance(10);
    let stuck = tracer.active_tree().stuck(0);
    assert_eq!(stuck.len(), limits.max_open_spans);
    let total: usize = stuck
        .iter()
        .map(|s| s.path.iter().map(String::len).sum::<usize>())
        .sum();
    assert!(
        total <= limits.max_open_spans * (STUCK_PATH_MAX_BYTES + limits.max_value_len),
        "stuck output is {total} bytes"
    );
    assert!(stuck.iter().all(|s| s.path.len() <= STUCK_PATH_MAX_DEPTH));
    // The deepest span reports its own name last and is flagged as truncated.
    let deepest = stuck.last().expect("deepest");
    assert!(deepest.path_truncated);
    assert_eq!(deepest.path.last(), Some(&name));
    // A shallow span has its complete path and no flag.
    assert!(!stuck[0].path_truncated);
    assert_eq!(stuck[0].path.len(), 1);
}
