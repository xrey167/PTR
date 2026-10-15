mod common;

use common::{ingest_bytes, utf16};
use ptr_ingress::{sniff, ArtifactKind, TextEncoding, SNIFF_WINDOW};

fn padded(prefix: &[u8], len: usize) -> Vec<u8> {
    let mut v = prefix.to_vec();
    v.resize(len, 0);
    v
}

#[test]
fn every_supported_container_is_recognized() {
    let mut tar = padded(b"file.txt", 512);
    tar[257..262].copy_from_slice(b"ustar");
    let cases: Vec<(&str, Vec<u8>, ArtifactKind, &str)> = vec![
        (
            "pdf",
            b"%PDF-1.7\n%".to_vec(),
            ArtifactKind::Pdf,
            "application/pdf",
        ),
        (
            "png",
            padded(b"\x89PNG\r\n\x1a\n", 16),
            ArtifactKind::Image,
            "image/png",
        ),
        (
            "jpeg",
            padded(b"\xFF\xD8\xFF\xE0", 16),
            ArtifactKind::Image,
            "image/jpeg",
        ),
        (
            "gif",
            padded(b"GIF87a", 16),
            ArtifactKind::Image,
            "image/gif",
        ),
        (
            "webp",
            padded(b"RIFF\x10\0\0\0WEBPVP8 ", 24),
            ArtifactKind::Image,
            "image/webp",
        ),
        (
            "wav",
            padded(b"RIFF\x10\0\0\0WAVEfmt ", 24),
            ArtifactKind::Audio,
            "audio/wav",
        ),
        (
            "mp3-id3",
            padded(b"ID3\x03\0\0", 16),
            ArtifactKind::Audio,
            "audio/mpeg",
        ),
        (
            "ogg",
            padded(b"OggS\0\x02", 16),
            ArtifactKind::Audio,
            "audio/ogg",
        ),
        (
            "flac",
            padded(b"fLaC\0\0\0\x22", 16),
            ArtifactKind::Audio,
            "audio/flac",
        ),
        (
            "mp4",
            padded(b"\0\0\0\x20ftypisom", 24),
            ArtifactKind::Video,
            "video/mp4",
        ),
        (
            "zip",
            padded(b"PK\x03\x04", 16),
            ArtifactKind::Archive,
            "application/zip",
        ),
        (
            "gzip",
            padded(b"\x1F\x8B\x08\0", 16),
            ArtifactKind::Archive,
            "application/gzip",
        ),
        ("tar", tar, ArtifactKind::Archive, "application/x-tar"),
        (
            "7z",
            padded(b"7z\xBC\xAF\x27\x1C", 16),
            ArtifactKind::Archive,
            "application/x-7z-compressed",
        ),
        (
            "sqlite",
            padded(b"SQLite format 3\0", 100),
            ArtifactKind::Database,
            "application/vnd.sqlite3",
        ),
    ];
    for (name, bytes, kind, mime) in cases {
        let report = sniff(&bytes);
        assert_eq!((report.kind, report.mime), (kind, mime), "{name}");
        assert!(report.confidence.get() >= 0.9, "{name}");
        assert_eq!(report.text_encoding, None, "{name}");
    }
}

#[test]
fn text_encodings_are_detected() {
    let utf8 = sniff("grüße, 世界".as_bytes());
    assert_eq!(
        (utf8.kind, utf8.text_encoding),
        (ArtifactKind::Text, Some(TextEncoding::Utf8))
    );
    let le = sniff(&utf16("hello", false, true));
    assert_eq!(le.text_encoding, Some(TextEncoding::Utf16Le));
    let be = sniff(&utf16("hello", true, true));
    assert_eq!(be.text_encoding, Some(TextEncoding::Utf16Be));
    let bomless = sniff(&utf16("hello world", false, false));
    assert_eq!(bomless.text_encoding, Some(TextEncoding::Utf16Le));
}

#[test]
fn tricky_cases_do_not_overclaim() {
    // Truncated magic numbers are not promoted to a container type.
    for truncated in [
        &b"\x89PNG\r\n"[..],
        b"\xFF\xD8",
        b"RIFF\0\0\0\0WA",
        b"7z\xBC\xAF",
        b"\x1F\x8B",
    ] {
        assert_eq!(
            sniff(truncated).kind,
            ArtifactKind::Unknown,
            "{truncated:?}"
        );
    }
    // Empty input and random binary stay Unknown.
    assert_eq!(sniff(b"").kind, ArtifactKind::Unknown);
    assert_eq!(sniff(&[0, 1, 2, 3, 0xFE, 0x80]).kind, ArtifactKind::Unknown);
    // A text file that merely starts like a magic number is still text.
    assert_eq!(sniff(b"%PD").kind, ArtifactKind::Text);
    assert_eq!(sniff(b"PK").kind, ArtifactKind::Text);
}

#[test]
fn inspection_is_bounded_by_the_window() {
    let mut big = vec![b'x'; 3 * SNIFF_WINDOW];
    big[SNIFF_WINDOW + 1] = 0;
    let report = sniff(&big);
    assert_eq!(report.inspected_bytes, SNIFF_WINDOW);
    assert!(report.truncated);
    assert_eq!(report.kind, ArtifactKind::Text);
}

#[test]
fn pipeline_record_kind_follows_the_sniffer() {
    let png = ingest_bytes(&padded(b"\x89PNG\r\n\x1a\n", 32));
    assert_eq!(png.kind, ArtifactKind::Image);
    assert!(png.normalized.is_none() && png.document.is_none());
    assert!(png.proposal.task_types.is_empty());
    assert!(png
        .proposal
        .issues
        .iter()
        .any(|i| i.code == "ingress.non_text_artifact"));
}
