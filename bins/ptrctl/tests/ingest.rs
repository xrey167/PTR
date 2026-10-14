//! `ptrctl ingest` against the real binary on files created in the test.

mod common;

use common::{code, ptrctl, stderr, stdout, TempDir};
use std::ffi::OsStr;

fn run(args: &[&std::ffi::OsStr]) -> std::process::Output {
    let mut all = vec![OsStr::new("ingest")];
    all.extend_from_slice(args);
    ptrctl(&all)
}

#[test]
fn json_file_renders_segments_with_raw_spans_and_a_decision() {
    let dir = TempDir::new("ingest-json");
    let path = dir.file("doc.json", br#"{"name":"x","items":[1,2]}"#);
    let output = run(&[path.as_os_str()]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.starts_with("IngressRecord\n"), "{text}");
    assert!(text.contains("kind: ArtifactKind::Text"), "{text}");
    assert!(text.contains("encoding: \"utf-8\""), "{text}");
    assert!(text.contains("lossy: false"), "{text}");
    assert!(text.contains("kind: SegmentKind::Json"), "{text}");
    assert!(text.contains("raw_start: 0"), "{text}");
    assert!(text.contains("raw_end: 26"), "{text}");
    assert!(
        text.contains("status: CrossCheckStatus::SampleVerified"),
        "{text}"
    );
    assert!(text.contains("label: \"data-analysis\""), "{text}");
}

#[test]
fn raw_content_is_not_printed_unless_a_preview_is_requested() {
    let dir = TempDir::new("ingest-raw");
    let path = dir.file("note.txt", b"hunter2-needle is the value\n");
    let plain = stdout(&run(&[path.as_os_str()]));
    assert!(!plain.contains("hunter2-needle"), "{plain}");
    assert!(plain.contains("fingerprint: \"fnv1a64:"), "{plain}");
    assert!(!plain.contains("preview"), "{plain}");

    let previewed = stdout(&run(&[path.as_os_str(), OsStr::new("--preview")]));
    assert!(
        previewed.contains("preview: \"hunter2-needle is the value"),
        "{previewed}"
    );
}

#[test]
fn utf16_text_is_decoded_and_reported_with_its_encoding() {
    let dir = TempDir::new("ingest-utf16");
    let mut bytes = vec![0xff, 0xfe];
    for unit in "hello wörld\n".encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    let path = dir.file("wide.txt", &bytes);
    let output = run(&[path.as_os_str()]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("encoding: \"utf-16le\""), "{text}");
    assert!(
        text.contains("mime: \"text/plain; charset=utf-16le\""),
        "{text}"
    );
    // The segment's raw span starts after the 2-byte BOM, in raw bytes.
    assert!(text.contains("raw_start: 2"), "{text}");
}

#[test]
fn png_header_is_an_image_and_needs_a_full_check_without_failing() {
    let dir = TempDir::new("ingest-png");
    let path = dir.file("pic.png", b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR");
    let output = run(&[path.as_os_str(), OsStr::new("--format"), OsStr::new("json")]);
    // A NeedsFullCheck decision is a result, not an error.
    assert_eq!(code(&output), 0, "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.starts_with("{\"struct\":\"IngressRecord\""), "{text}");
    assert!(text.contains("\"mime\":\"image/png\""), "{text}");
    assert!(text.contains("\"variant\":\"Image\""), "{text}");
    assert!(text.contains("\"variant\":\"NeedsFullCheck\""), "{text}");
    assert!(text.contains("UnclassifiedArtifact"), "{text}");
    assert!(text.contains("\"segments\":[]"), "{text}");
}

#[test]
fn oversized_missing_and_non_file_inputs_exit_one() {
    let dir = TempDir::new("ingest-bad");
    let path = dir.file("big.txt", b"0123456789");
    let output = run(&[path.as_os_str(), OsStr::new("--max-bytes"), OsStr::new("9")]);
    assert_eq!(code(&output), 1);
    assert!(
        stderr(&output).contains("--max-bytes 9"),
        "{}",
        stderr(&output)
    );
    assert!(stdout(&output).is_empty());

    // Exactly at the bound is accepted.
    let output = run(&[
        path.as_os_str(),
        OsStr::new("--max-bytes"),
        OsStr::new("10"),
    ]);
    assert_eq!(code(&output), 0, "{}", stderr(&output));

    let missing = dir.path().join("absent.txt");
    assert_eq!(code(&run(&[missing.as_os_str()])), 1);
    assert_eq!(code(&run(&[dir.path().as_os_str()])), 1);
}

#[test]
fn usage_errors_exit_two_and_help_exits_zero() {
    assert_eq!(code(&run(&[])), 2);
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("b")])), 2);
    assert_eq!(code(&run(&[OsStr::new("a"), OsStr::new("--bogus")])), 2);
    assert_eq!(
        code(&run(&[
            OsStr::new("a"),
            OsStr::new("--format"),
            OsStr::new("xml")
        ])),
        2
    );
    assert_eq!(
        code(&run(&[
            OsStr::new("a"),
            OsStr::new("--max-bytes"),
            OsStr::new("x")
        ])),
        2
    );
    let help = run(&[OsStr::new("--help")]);
    assert_eq!(code(&help), 0);
    let text = stdout(&help);
    assert!(text.contains("usage: ptrctl ingest"), "{text}");
    assert!(text.contains("exit codes:"), "{text}");
}
