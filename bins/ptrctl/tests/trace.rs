//! `ptrctl trace` against files written by the real `ptr_observe::OtlpJsonFileExporter`.

mod common;

use common::{code, ptrctl, stderr, stdout, TempDir};
use ptr_observe::{
    ManualClock, OtlpFileConfig, OtlpJsonFileExporter, Span, SpanStatus, TraceEvent, TraceLevel,
    Tracer, TracerConfig,
};
use std::ffi::OsStr;
use std::sync::Arc;

fn run(args: &[&OsStr]) -> std::process::Output {
    let mut all = vec![OsStr::new("trace")];
    all.extend_from_slice(args);
    ptrctl(&all)
}

/// A root span (100 ns) with a failing child (50 ns, one event) and an ok sibling.
fn sample_spans() -> Vec<Span> {
    let clock = Arc::new(ManualClock::new(1_000));
    let tracer = Tracer::new(
        TracerConfig {
            id_seed: 3,
            ..TracerConfig::default()
        },
        clock.clone(),
    );
    let root = tracer.root_span("request \"quoted\"").expect("root");
    clock.advance(10);
    let failing = root.child("verify").expect("child");
    failing.event(TraceEvent::new(TraceLevel::Warn, "slow"));
    clock.advance(50);
    failing.fail("boom");
    let fine = root.child("commit").expect("child");
    clock.advance(40);
    fine.set_status(SpanStatus::Ok);
    fine.finish();
    root.finish();
    tracer.drain_finished()
}

#[test]
fn summarizes_the_span_tree_durations_and_status_of_exporter_output() {
    let dir = TempDir::new("trace-tree");
    let path = dir.path().join("spans.jsonl");
    let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
    let spans = sample_spans();
    assert_eq!(exporter.export(&spans).expect("export"), 3);

    let output = run(&[path.as_os_str()]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        text.contains("1 trace(s), 3 span(s), 1 error span(s)"),
        "{text}"
    );
    let root = text
        .lines()
        .position(|l| l.starts_with("  request"))
        .expect("root line");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[root].contains("[100 ns] OK"), "{text}");
    // Children are indented one level deeper, in start order.
    assert!(
        lines[root + 1].starts_with("    verify [50 ns] ERROR \"boom\" events=1"),
        "{text}"
    );
    assert!(
        lines[root + 2].starts_with("    commit [40 ns] OK events=0"),
        "{text}"
    );
    assert!(text.contains("trace "), "{text}");
}

#[test]
fn several_exports_and_traces_are_all_listed() {
    let dir = TempDir::new("trace-multi");
    let path = dir.path().join("spans.jsonl");
    let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
    let spans = sample_spans();
    exporter.export(&spans[..1]).expect("first batch");
    exporter.export(&spans[1..]).expect("second batch");
    let other = sample_spans_with_seed(9);
    exporter.export(&other).expect("other trace");

    let text = stdout(&run(&[path.as_os_str()]));
    assert!(
        text.contains("2 trace(s), 5 span(s), 2 error span(s)"),
        "{text}"
    );
}

fn sample_spans_with_seed(seed: u64) -> Vec<Span> {
    let clock = Arc::new(ManualClock::new(5));
    let tracer = Tracer::new(
        TracerConfig {
            id_seed: seed,
            ..TracerConfig::default()
        },
        clock.clone(),
    );
    let root = tracer.root_span("other").expect("root");
    clock.advance(7);
    let child = root.child("inner").expect("child");
    child.fail("bad");
    root.finish();
    tracer.drain_finished()
}

#[test]
fn a_span_whose_parent_is_not_in_the_file_is_shown_as_a_root() {
    let dir = TempDir::new("trace-orphan");
    let path = dir.path().join("spans.jsonl");
    let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
    let spans = sample_spans();
    // Only the children: their parent (the root) was never exported.
    exporter.export(&spans[..2]).expect("export");
    let text = stdout(&run(&[path.as_os_str()]));
    assert!(text.contains("not in file"), "{text}");
}

#[test]
fn damaged_missing_and_empty_files_exit_one() {
    let dir = TempDir::new("trace-bad");
    let missing = dir.path().join("absent.jsonl");
    assert_eq!(code(&run(&[missing.as_os_str()])), 1);

    let empty = dir.file("empty.jsonl", b"");
    let output = run(&[empty.as_os_str()]);
    assert_eq!(code(&output), 1);
    assert!(stderr(&output).contains("no spans"), "{}", stderr(&output));

    let truncated = dir.file("cut.jsonl", b"{\"resourceSpans\":[{\"scopeSpans\":");
    let output = run(&[truncated.as_os_str()]);
    assert_eq!(code(&output), 1);
    assert!(stderr(&output).contains("line 1"), "{}", stderr(&output));

    let wrong_shape = dir.file("shape.jsonl", b"{\"hello\":1}\n");
    assert_eq!(code(&run(&[wrong_shape.as_os_str()])), 1);

    let not_utf8 = dir.file("bin.jsonl", &[0xff, 0xfe, 0x00]);
    assert_eq!(code(&run(&[not_utf8.as_os_str()])), 1);
}

#[test]
fn usage_errors_exit_two_and_help_exits_zero() {
    assert_eq!(code(&run(&[])), 2);
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("b")])), 2);
    let help = run(&[OsStr::new("--help")]);
    assert_eq!(code(&help), 0);
    let text = stdout(&help);
    assert!(text.contains("usage: ptrctl trace"), "{text}");
    assert!(text.contains("exit codes:"), "{text}");
}

/// A valid chain of `count` spans where span `i` is the child of span `i - 1`, packed into one
/// line with short ids so it stays under the 16 MiB file bound.
fn chain_file(count: usize) -> String {
    let mut spans = Vec::with_capacity(count);
    for i in 1..=count {
        let parent = if i == 1 {
            String::new()
        } else {
            format!("\"parentSpanId\":\"{}\",", i - 1)
        };
        spans.push(format!(
            "{{\"traceId\":\"t\",\"spanId\":\"{i}\",{parent}\"name\":\"n\",\
             \"startTimeUnixNano\":\"1\",\"endTimeUnixNano\":\"2\",\"events\":[],\
             \"status\":{{\"code\":0}}}}"
        ));
    }
    format!(
        "{{\"resourceSpans\":[{{\"scopeSpans\":[{{\"spans\":[{}]}}]}}]}}\n",
        spans.join(",")
    )
}

#[test]
fn a_very_deep_parent_chain_renders_with_bounded_indentation_and_output() {
    let dir = TempDir::new("trace-deep");
    let content = chain_file(100_000);
    assert!(content.len() < 16 * 1024 * 1024, "fixture must be accepted");
    let path = dir.file("deep.jsonl", content.as_bytes());

    let started = std::time::Instant::now();
    let output = run(&[path.as_os_str()]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(30),
        "rendering took {:?}",
        started.elapsed()
    );
    // Every span is still listed, but indentation is capped, so output stays linear.
    assert_eq!(
        text.lines().filter(|l| l.contains("span=")).count(),
        100_000
    );
    assert!(
        text.contains("[depth 100000]"),
        "deep spans carry a depth marker"
    );
    assert!(
        text.lines().all(|l| l.len() < 300),
        "no line is longer than its fields"
    );
    assert!(
        text.len() < 100_000 * 200,
        "output is linear in spans, got {} bytes",
        text.len()
    );
}

#[test]
fn control_characters_in_identifiers_never_reach_the_terminal() {
    // JSON escapes decode to ESC/BEL: an OSC title sequence, a screen clear and a colour code.
    let span = |id: &str, parent: Option<&str>| {
        let parent = parent
            .map(|p| format!("\"parentSpanId\":\"{p}\","))
            .unwrap_or_default();
        format!(
            "{{\"traceId\":\"t\\u001b]0;pwned\\u0007\",\"spanId\":\"{id}\",{parent}\
             \"name\":\"n\",\"startTimeUnixNano\":\"1\",\"endTimeUnixNano\":\"2\",\
             \"events\":[],\"status\":{{\"code\":0}}}}"
        )
    };
    let spans = [
        // A root, an orphan whose parent is not in the file, and a two-span parent cycle.
        span("root\\u001b[2J", None),
        span("orphan", Some("gone\\u001b[31m")),
        span("c1\\u001b[0m", Some("c2\\u0007")),
        span("c2\\u0007", Some("c1\\u001b[0m")),
    ];
    let content = format!(
        "{{\"resourceSpans\":[{{\"scopeSpans\":[{{\"spans\":[{}]}}]}}]}}\n",
        spans.join(",")
    );
    let dir = TempDir::new("trace-escape");
    let path = dir.file("hostile.jsonl", content.as_bytes());

    let output = run(&[path.as_os_str()]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        !text.contains(['\u{1b}', '\u{7}']),
        "raw control characters were printed: {text:?}"
    );
    // The identifiers are still shown, escaped.
    assert!(text.contains("\\u{1b}]0;pwned\\u{7}"), "{text}");
    assert!(text.contains("gone\\u{1b}[31m not in file"), "{text}");
    assert!(text.contains("unreachable (parent cycle)"), "{text}");
}
