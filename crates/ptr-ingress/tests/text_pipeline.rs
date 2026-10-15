mod common;

use common::{assert_provenance_round_trips, ingest, ingest_bytes, utf16};
use ptr_ingress::{
    ArtifactKind, CodeLanguage, DocumentFormat, IngestHints, IngressConfig, IngressPipeline,
    InvalidUtf8Policy, NormalizeError, NormalizeOptions, NulPolicy, ParseIssue, ParseOptions,
    RawInput, SegmentKind, TextEncoding,
};

const MARKDOWN: &str =
    "# Plan\n\nPlease fix the parser bug, because it panics.\n\n```sh\ncargo test\n```\n";

#[test]
fn utf8_lf_markdown_round_trips() {
    let record = ingest(RawInput::Text(MARKDOWN.to_owned()));
    assert_eq!(record.kind, ArtifactKind::Text);
    assert_eq!(
        record.document.as_ref().map(|d| d.format),
        Some(DocumentFormat::Markdown)
    );
    assert_provenance_round_trips(&record);
    assert_eq!(record.segments().len(), 3);
}

#[test]
fn crlf_input_maps_back_to_the_crlf_bytes() {
    let crlf = MARKDOWN.replace('\n', "\r\n");
    let record = ingest_bytes(crlf.as_bytes());
    let normalized = record.normalized.as_ref().expect("normalized");
    assert_eq!(normalized.text(), MARKDOWN);
    assert!(normalized.crlf_normalized() >= 5);
    assert_provenance_round_trips(&record);
    // The multi-line fence segment spans the CRLF bytes inside it.
    let fence = record
        .segments()
        .iter()
        .find(|s| matches!(s.kind, SegmentKind::CodeFence { .. }))
        .expect("fence");
    let raw_slice = fence.span.slice(crlf.as_bytes()).expect("slice");
    assert_eq!(raw_slice, b"```sh\r\ncargo test\r\n```");
}

#[test]
fn utf8_bom_is_consumed_and_offsets_account_for_it() {
    let mut bytes = vec![0xEF, 0xBB, 0xBF];
    bytes.extend_from_slice(MARKDOWN.as_bytes());
    let record = ingest_bytes(&bytes);
    let normalized = record.normalized.as_ref().expect("normalized");
    assert_eq!(
        (normalized.bom_len(), normalized.encoding()),
        (3, TextEncoding::Utf8)
    );
    assert_eq!(record.segments()[0].span.byte_start, 3);
    assert_provenance_round_trips(&record);
}

#[test]
fn utf16_with_and_without_bom_round_trips() {
    let text = "Bitte analysiere das dataset: Mittelwert, Median \u{1F4C8}?\r\nDanke!";
    for (big, bom) in [(false, true), (true, true), (false, false)] {
        let raw = utf16(text, big, bom);
        let record = ingest_bytes(&raw);
        let normalized = record.normalized.as_ref().expect("normalized");
        let expected = if big {
            TextEncoding::Utf16Be
        } else {
            TextEncoding::Utf16Le
        };
        assert_eq!(normalized.encoding(), expected, "big={big} bom={bom}");
        assert_eq!(normalized.text(), text.replace("\r\n", "\n"));
        assert!(!normalized.is_lossy());
        assert_provenance_round_trips(&record);
        // Spans are byte offsets into the UTF-16 raw, so they are even-aligned.
        for seg in record.segments() {
            assert_eq!(seg.span.byte_start % 2, 0);
            assert_eq!(seg.span.byte_end % 2, 0);
        }
    }
}

#[test]
fn invalid_utf8_falls_back_with_an_explicit_lossy_flag() {
    let raw = b"Please show the caf\xE9 menu\r\nthanks".to_vec();
    let record = ingest_bytes(&raw);
    let normalized = record.normalized.as_ref().expect("normalized");
    assert_eq!(normalized.encoding(), TextEncoding::Windows1252);
    assert!(normalized.is_lossy());
    assert!(normalized.text().contains("caf\u{E9}"));
    assert!(record
        .proposal
        .issues
        .iter()
        .any(|i| i.code == "normalize.lossy" && !i.hard));
    assert_provenance_round_trips(&record);
}

#[test]
fn strict_utf8_policy_rejects_instead_of_guessing() {
    let config = IngressConfig {
        normalize: NormalizeOptions {
            invalid_utf8: InvalidUtf8Policy::Reject,
            ..Default::default()
        },
        ..Default::default()
    };
    let record = IngressPipeline::new(config)
        .ingest_bytes(
            b"Please list the caf\xE9 menu".to_vec(),
            &IngestHints::default(),
        )
        .expect("a record is still produced");
    assert_eq!(
        record.normalize_error,
        Some(NormalizeError::InvalidUtf8 { raw_offset: 19 })
    );
    assert!(record.normalized.is_none());
    assert_eq!(
        record.raw_bytes(),
        Some(&b"Please list the caf\xE9 menu"[..])
    );
}

#[test]
fn nul_bytes_are_rejected_or_stripped_by_policy() {
    // The NUL sits beyond the sniffing window, so the input still sniffs as text and it is
    // normalization (not the sniffer) that refuses it.
    let head = "please list files ".repeat(500);
    let raw = format!("{head}\0tail").into_bytes();
    let rejected = ingest_bytes(&raw);
    assert_eq!(
        rejected.normalize_error,
        Some(NormalizeError::NulByte {
            raw_offset: head.len()
        })
    );
    assert!(rejected
        .proposal
        .issues
        .iter()
        .any(|i| i.code == "normalize.rejected" && i.hard));

    let config = IngressConfig {
        normalize: NormalizeOptions {
            nul_policy: NulPolicy::Strip,
            ..Default::default()
        },
        ..Default::default()
    };
    let stripped = IngressPipeline::new(config)
        .ingest_bytes(raw.clone(), &IngestHints::default())
        .expect("ingests");
    let normalized = stripped.normalized.as_ref().expect("normalized");
    assert_eq!(
        (normalized.text(), normalized.nul_stripped()),
        (format!("{head}tail").as_str(), 1)
    );
    assert_provenance_round_trips(&stripped);
}

#[test]
fn empty_and_bom_only_inputs_yield_empty_documents() {
    for raw in [&b""[..], &[0xEF, 0xBB, 0xBF][..], &[0xFF, 0xFE][..]] {
        let record = ingest_bytes(raw);
        if raw.is_empty() {
            assert_eq!(record.kind, ArtifactKind::Unknown);
            continue;
        }
        let normalized = record.normalized.as_ref().expect("normalized");
        assert_eq!(normalized.text(), "");
        assert!(normalized.offset_map().is_empty());
        assert_eq!(
            record.document.as_ref().map(|d| d.format),
            Some(DocumentFormat::Empty)
        );
        assert!(record.segments().is_empty());
        assert_eq!(record.raw_bytes(), Some(raw));
    }
}

#[test]
fn structure_detection_through_the_pipeline() {
    let cases: [(&str, DocumentFormat); 7] = [
        (
            "{\"task\": \"sum\", \"items\": [1, 2, 3]}",
            DocumentFormat::Json,
        ),
        ("# Title\nbody text\n", DocumentFormat::Markdown),
        (
            "#!/usr/bin/env python3\nprint('hi')\n",
            DocumentFormat::SourceCode(CodeLanguage::Python),
        ),
        (
            "pub fn f() -> Self {\n    let mut x = 1;\n}\n",
            DocumentFormat::SourceCode(CodeLanguage::Rust),
        ),
        (
            "[db]\nhost = \"x\"\nport = 5432\n",
            DocumentFormat::KeyValue,
        ),
        ("a,b,c\n1,2,3\n4,5,6\n", DocumentFormat::Csv),
        (
            "Just a sentence.\n\nAnd another paragraph.",
            DocumentFormat::Prose,
        ),
    ];
    for (text, format) in cases {
        let record = ingest(RawInput::Text(text.to_owned()));
        assert_eq!(
            record.document.as_ref().map(|d| d.format),
            Some(format),
            "{text:?}"
        );
        assert_provenance_round_trips(&record);
    }
}

#[test]
fn json_nested_beyond_the_depth_limit_is_flagged_not_trusted() {
    let nested = format!("{}1{}", "[".repeat(10), "]".repeat(10));
    let config = IngressConfig {
        parse: ParseOptions {
            max_json_depth: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let record = IngressPipeline::new(config)
        .ingest_text(nested.clone(), &IngestHints::default())
        .expect("ingests");
    let doc = record.document.as_ref().expect("document");
    assert_ne!(doc.format, DocumentFormat::Json);
    assert_eq!(doc.issues, vec![ParseIssue::JsonTooDeep { limit: 4 }]);
    assert!(record
        .proposal
        .issues
        .iter()
        .any(|i| i.code == "parse.json_too_deep" && i.hard));
    // Within the limit the same shape is accepted.
    let ok = IngressPipeline::default()
        .ingest_text(nested, &IngestHints::default())
        .expect("ok");
    assert_eq!(
        ok.document.as_ref().map(|d| d.format),
        Some(DocumentFormat::Json)
    );
}

#[test]
fn classification_evidence_points_into_the_raw_bytes() {
    let raw = utf16(
        "Why did it fail? Then delete the cache because it is stale.",
        false,
        true,
    );
    let record = ingest_bytes(&raw);
    assert!(!record.evidence.is_empty());
    assert_provenance_round_trips(&record);
    let terms: Vec<_> = record.evidence.iter().map(|e| e.term).collect();
    assert!(terms.contains(&"delete") && terms.contains(&"?") && terms.contains(&"because"));
    for list in [
        &record.proposal.task_types,
        &record.proposal.reasoning_types,
        &record.proposal.effects,
    ] {
        let total: f32 = list.iter().map(|s| s.probability.get()).sum();
        assert!((total - 1.0).abs() < 1e-4, "{list:?}");
    }
}

#[test]
fn identical_input_gives_identical_records() {
    let a = ingest(RawInput::Text(MARKDOWN.to_owned()));
    let b = ingest(RawInput::Text(MARKDOWN.to_owned()));
    assert_eq!(a, b);
}

#[test]
fn nul_inside_the_sniff_window_makes_the_input_binary_not_text() {
    let record = ingest_bytes(b"hello\0 world, please list files");
    assert_eq!(record.kind, ArtifactKind::Unknown);
    assert!(record.normalized.is_none() && record.normalize_error.is_none());
}
