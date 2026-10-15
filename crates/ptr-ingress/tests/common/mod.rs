//! Shared helpers for integration tests (no test annotations here).
#![allow(dead_code)]

use ptr_ingress::{IngestHints, IngressPipeline, IngressRecord, RawInput, TextEncoding};

pub fn ingest(raw: RawInput) -> IngressRecord {
    IngressPipeline::default()
        .ingest(raw, &IngestHints::default())
        .expect("ingest succeeds")
}

pub fn ingest_bytes(bytes: &[u8]) -> IngressRecord {
    ingest(RawInput::Bytes(bytes.to_vec()))
}

pub fn utf16(text: &str, big_endian: bool, bom: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<u8>, unit: u16| {
        let pair = if big_endian {
            unit.to_be_bytes()
        } else {
            unit.to_le_bytes()
        };
        out.extend_from_slice(&pair);
    };
    if bom {
        push(&mut out, 0xFEFF);
    }
    for unit in text.encode_utf16() {
        push(&mut out, unit);
    }
    out
}

/// Independent decoder used to check provenance: decodes `bytes` and folds CRLF to LF.
pub fn decode_independently(bytes: &[u8], encoding: TextEncoding) -> String {
    let text: String = match encoding {
        TextEncoding::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
        TextEncoding::Utf16Le | TextEncoding::Utf16Be => {
            let big = encoding == TextEncoding::Utf16Be;
            let units = bytes.as_chunks::<2>().0.iter().map(|c| {
                if big {
                    u16::from_be_bytes([c[0], c[1]])
                } else {
                    u16::from_le_bytes([c[0], c[1]])
                }
            });
            char::decode_utf16(units)
                .map(|r| r.unwrap_or('\u{FFFD}'))
                .collect()
        }
        TextEncoding::Windows1252 => bytes.iter().map(|&b| char::from(b)).collect(),
    };
    // Spans of a NUL-stripped input legitimately cover the dropped NUL bytes.
    text.replace('\0', "").replace("\r\n", "\n")
}

/// Asserts that every segment and every label evidence span slices back out of the raw bytes
/// and decodes to the segment text (or the matched term).
pub fn assert_provenance_round_trips(record: &IngressRecord) {
    let raw = record.raw_bytes().expect("inline raw");
    let normalized = record.normalized.as_ref().expect("normalized text");
    let encoding = normalized.encoding();
    assert!(
        !record.segments().is_empty(),
        "expected at least one segment"
    );
    for segment in record.segments() {
        let slice = segment.span.slice(raw).expect("segment span inside raw");
        assert_eq!(
            decode_independently(slice, encoding),
            segment.text,
            "segment {:?}",
            segment.kind
        );
        assert!(normalized.span_round_trips(raw, segment.span, &segment.text));
        assert_eq!(
            normalized.text().get(segment.range.clone()),
            Some(segment.text.as_str())
        );
    }
    for ev in &record.evidence {
        let slice = ev.span.slice(raw).expect("evidence span inside raw");
        let decoded = decode_independently(slice, encoding);
        if ev.term.starts_with('<') {
            assert!(!decoded.is_empty());
        } else {
            assert!(
                decoded.eq_ignore_ascii_case(ev.term),
                "evidence {:?} decoded to {decoded:?}",
                ev.term
            );
        }
    }
}
