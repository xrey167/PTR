//! Deterministic magic-byte MIME/kind sniffer with a bounded inspection window.

use ptr_types::Probability;

use crate::model::ArtifactKind;
use crate::normalize::TextEncoding;

/// Number of leading bytes inspected. Longer inputs are only sniffed on this prefix.
pub const SNIFF_WINDOW: usize = 8192;

#[derive(Clone, Debug, PartialEq)]
pub struct SniffReport {
    pub kind: ArtifactKind,
    pub mime: &'static str,
    pub confidence: Probability,
    /// Detected text encoding when `kind` is [`ArtifactKind::Text`].
    pub text_encoding: Option<TextEncoding>,
    pub inspected_bytes: usize,
    /// True when the input was longer than the inspection window.
    pub truncated: bool,
}

struct Hit {
    kind: ArtifactKind,
    mime: &'static str,
    confidence: f32,
    encoding: Option<TextEncoding>,
}

fn hit(kind: ArtifactKind, mime: &'static str, confidence: f32) -> Option<Hit> {
    Some(Hit {
        kind,
        mime,
        confidence,
        encoding: None,
    })
}

/// Sniffs `raw` from magic bytes (first [`SNIFF_WINDOW`] bytes only).
pub fn sniff(raw: &[u8]) -> SniffReport {
    let window = raw.get(..SNIFF_WINDOW).unwrap_or(raw);
    let truncated = raw.len() > window.len();
    let found = if window.is_empty() {
        hit(ArtifactKind::Unknown, "application/x-empty", 1.0)
    } else {
        sniff_magic(window)
            .or_else(|| sniff_utf16(window))
            .or_else(|| sniff_mp3_frame(window))
            .or_else(|| sniff_text(window, truncated))
            .or_else(|| hit(ArtifactKind::Unknown, "application/octet-stream", 0.5))
    };
    let found = found.unwrap_or(Hit {
        kind: ArtifactKind::Unknown,
        mime: "application/octet-stream",
        confidence: 0.5,
        encoding: None,
    });
    SniffReport {
        kind: found.kind,
        mime: found.mime,
        confidence: Probability::new(found.confidence).unwrap_or_default(),
        text_encoding: found.encoding,
        inspected_bytes: window.len(),
        truncated,
    }
}

fn sniff_magic(w: &[u8]) -> Option<Hit> {
    use ArtifactKind::{Archive, Audio, Database, Image, Pdf, Video};
    if w.starts_with(b"%PDF-") {
        return hit(Pdf, "application/pdf", 0.99);
    }
    if w.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return hit(Image, "image/png", 0.99);
    }
    if w.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return hit(Image, "image/jpeg", 0.97);
    }
    if w.starts_with(b"GIF87a") || w.starts_with(b"GIF89a") {
        return hit(Image, "image/gif", 0.99);
    }
    if w.len() >= 12 && w.starts_with(b"RIFF") {
        return match w.get(8..12) {
            Some(b"WEBP") => hit(Image, "image/webp", 0.99),
            Some(b"WAVE") => hit(Audio, "audio/wav", 0.99),
            Some(b"AVI ") => hit(Video, "video/x-msvideo", 0.97),
            _ => None,
        };
    }
    if w.starts_with(b"ID3") {
        return hit(Audio, "audio/mpeg", 0.95);
    }
    if w.starts_with(b"OggS") {
        return hit(Audio, "audio/ogg", 0.97);
    }
    if w.starts_with(b"fLaC") {
        return hit(Audio, "audio/flac", 0.97);
    }
    if w.len() >= 12 && w.get(4..8) == Some(b"ftyp") {
        return match w.get(8..12) {
            Some(b"M4A ") => hit(Audio, "audio/mp4", 0.95),
            Some(b"qt  ") => hit(Video, "video/quicktime", 0.95),
            Some(b"avif") => hit(Image, "image/avif", 0.95),
            Some(b"heic" | b"heix" | b"mif1") => hit(Image, "image/heic", 0.9),
            _ => hit(Video, "video/mp4", 0.95),
        };
    }
    if w.starts_with(b"PK\x03\x04") || w.starts_with(b"PK\x05\x06") {
        return hit(Archive, "application/zip", 0.95);
    }
    if w.starts_with(&[0x1F, 0x8B, 0x08]) {
        return hit(Archive, "application/gzip", 0.97);
    }
    if w.starts_with(&[b'7', b'z', 0xBC, 0xAF, 0x27, 0x1C]) {
        return hit(Archive, "application/x-7z-compressed", 0.99);
    }
    if w.starts_with(b"SQLite format 3\0") {
        return hit(Database, "application/vnd.sqlite3", 0.99);
    }
    if w.len() >= 262 && w.get(257..262) == Some(b"ustar") {
        return hit(Archive, "application/x-tar", 0.9);
    }
    None
}

fn sniff_mp3_frame(w: &[u8]) -> Option<Hit> {
    let (b0, b1, b2) = (*w.first()?, *w.get(1)?, *w.get(2)?);
    let sync = b0 == 0xFF && (b1 & 0xE0) == 0xE0;
    let valid =
        ((b1 >> 3) & 3) != 1 && ((b1 >> 1) & 3) != 0 && (b2 >> 4) != 0xF && ((b2 >> 2) & 3) != 3;
    (sync && valid).then_some(Hit {
        kind: ArtifactKind::Audio,
        mime: "audio/mpeg",
        confidence: 0.6,
        encoding: None,
    })
}

fn is_text_pair(lo: u8, hi: u8, text_is_hi_zero: bool) -> bool {
    let (ch, zero) = if text_is_hi_zero { (lo, hi) } else { (hi, lo) };
    zero == 0 && (matches!(ch, b'\t' | b'\n' | b'\r') || (0x20..0x7F).contains(&ch))
}

fn sniff_utf16(w: &[u8]) -> Option<Hit> {
    let text = |encoding, mime| {
        Some(Hit {
            kind: ArtifactKind::Text,
            mime,
            confidence: 0.95,
            encoding: Some(encoding),
        })
    };
    match w {
        [0xFF, 0xFE, ..] => return text(TextEncoding::Utf16Le, "text/plain; charset=utf-16le"),
        [0xFE, 0xFF, ..] => return text(TextEncoding::Utf16Be, "text/plain; charset=utf-16be"),
        _ => {}
    }
    if w.len() < 4 {
        return None;
    }
    let pairs = w.len() / 2;
    let count = |little: bool| {
        w.as_chunks::<2>()
            .0
            .iter()
            .filter(|p| is_text_pair(p[0], p[1], little))
            .count()
    };
    let (le, be) = (count(true), count(false));
    let (best, encoding, mime) = if le >= be {
        (le, TextEncoding::Utf16Le, "text/plain; charset=utf-16le")
    } else {
        (be, TextEncoding::Utf16Be, "text/plain; charset=utf-16be")
    };
    (best * 10 >= pairs * 9).then_some(Hit {
        kind: ArtifactKind::Text,
        mime,
        confidence: 0.7,
        encoding: Some(encoding),
    })
}

fn has_binary_control(w: &[u8]) -> bool {
    w.iter()
        .any(|&b| matches!(b, 0x00..=0x08 | 0x0B | 0x0E..=0x1A | 0x1C..=0x1F))
}

fn sniff_text(w: &[u8], truncated: bool) -> Option<Hit> {
    let text = |mime, confidence, encoding| {
        Some(Hit {
            kind: ArtifactKind::Text,
            mime,
            confidence,
            encoding: Some(encoding),
        })
    };
    if w.starts_with(&[0xEF, 0xBB, 0xBF]) && !has_binary_control(w) {
        return text("text/plain; charset=utf-8", 0.95, TextEncoding::Utf8);
    }
    if has_binary_control(w) {
        return None;
    }
    match std::str::from_utf8(w) {
        Ok(_) if w.is_ascii() => text("text/plain; charset=utf-8", 0.9, TextEncoding::Utf8),
        Ok(_) => text("text/plain; charset=utf-8", 0.95, TextEncoding::Utf8),
        Err(e) if e.error_len().is_none() && truncated => {
            text("text/plain; charset=utf-8", 0.9, TextEncoding::Utf8)
        }
        Err(_) => {
            // Legacy single-byte text only if it is overwhelmingly ASCII; otherwise stay binary.
            let high = w.iter().filter(|b| !b.is_ascii()).count();
            if high * 10 <= w.len() {
                text(
                    "text/plain; charset=windows-1252",
                    0.5,
                    TextEncoding::Windows1252,
                )
            } else {
                None
            }
        }
    }
}

/// Maps a file extension (without dot, case-insensitive) to the kind it claims.
pub fn kind_for_extension(extension: &str) -> Option<ArtifactKind> {
    let ext = extension.trim_start_matches('.').to_ascii_lowercase();
    let kind = match ext.as_str() {
        "pdf" => ArtifactKind::Pdf,
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "avif" | "heic" => ArtifactKind::Image,
        "wav" | "mp3" | "ogg" | "flac" | "m4a" => ArtifactKind::Audio,
        "mp4" | "mov" | "avi" | "m4v" => ArtifactKind::Video,
        "zip" | "gz" | "tgz" | "tar" | "7z" => ArtifactKind::Archive,
        "sqlite" | "sqlite3" | "db" => ArtifactKind::Database,
        "txt" | "md" | "json" | "toml" | "csv" | "tsv" | "rs" | "py" | "js" | "sh" | "log"
        | "yaml" | "yml" | "xml" | "html" => ArtifactKind::Text,
        _ => return None,
    };
    Some(kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind_mime(raw: &[u8]) -> (ArtifactKind, &'static str) {
        let r = sniff(raw);
        (r.kind, r.mime)
    }

    fn padded(prefix: &[u8], len: usize) -> Vec<u8> {
        let mut v = prefix.to_vec();
        v.resize(len, 0);
        v
    }

    #[test]
    fn magic_table() {
        let mut tar = padded(b"name", 300);
        tar[257..262].copy_from_slice(b"ustar");
        let cases: Vec<(Vec<u8>, ArtifactKind, &str)> = vec![
            (b"%PDF-1.7\n".to_vec(), ArtifactKind::Pdf, "application/pdf"),
            (
                b"\x89PNG\r\n\x1a\n\0\0".to_vec(),
                ArtifactKind::Image,
                "image/png",
            ),
            (
                b"\xFF\xD8\xFF\xE0\0\x10JFIF".to_vec(),
                ArtifactKind::Image,
                "image/jpeg",
            ),
            (b"GIF89a\x01\0".to_vec(), ArtifactKind::Image, "image/gif"),
            (
                b"RIFF\x24\0\0\0WEBPVP8 ".to_vec(),
                ArtifactKind::Image,
                "image/webp",
            ),
            (
                b"RIFF\x24\0\0\0WAVEfmt ".to_vec(),
                ArtifactKind::Audio,
                "audio/wav",
            ),
            (b"ID3\x04\0\0\0".to_vec(), ArtifactKind::Audio, "audio/mpeg"),
            (b"OggS\0\x02".to_vec(), ArtifactKind::Audio, "audio/ogg"),
            (b"fLaC\0\0\0\"".to_vec(), ArtifactKind::Audio, "audio/flac"),
            (
                b"\0\0\0\x18ftypisom\0\0".to_vec(),
                ArtifactKind::Video,
                "video/mp4",
            ),
            (
                b"\0\0\0\x18ftypM4A \0\0".to_vec(),
                ArtifactKind::Audio,
                "audio/mp4",
            ),
            (
                b"PK\x03\x04\x14\0".to_vec(),
                ArtifactKind::Archive,
                "application/zip",
            ),
            (
                b"\x1F\x8B\x08\0\0\0".to_vec(),
                ArtifactKind::Archive,
                "application/gzip",
            ),
            (tar, ArtifactKind::Archive, "application/x-tar"),
            (
                b"7z\xBC\xAF\x27\x1C\0\x04".to_vec(),
                ArtifactKind::Archive,
                "application/x-7z-compressed",
            ),
            (
                b"SQLite format 3\0\x10\0".to_vec(),
                ArtifactKind::Database,
                "application/vnd.sqlite3",
            ),
        ];
        for (raw, kind, mime) in cases {
            assert_eq!(kind_mime(&raw), (kind, mime), "{mime}");
        }
    }

    #[test]
    fn mp3_frame_sync_without_id3() {
        let r = sniff(&[0xFF, 0xFB, 0x90, 0x00, 0x00]);
        assert_eq!((r.kind, r.mime), (ArtifactKind::Audio, "audio/mpeg"));
        assert!(r.confidence.get() < 0.7);
    }

    #[test]
    fn truncated_magic_is_not_identified() {
        for raw in [
            &b"\x89PNG\r\n"[..],
            b"\xFF\xD8",
            b"RIFF\0\0\0\0WE",
            b"\x1F\x8B",
        ] {
            assert_eq!(sniff(raw).kind, ArtifactKind::Unknown, "{raw:?}");
        }
        assert_eq!(sniff(b"%PD").kind, ArtifactKind::Text);
    }

    #[test]
    fn empty_input_is_unknown_with_full_confidence() {
        let r = sniff(b"");
        assert_eq!(
            (r.kind, r.mime, r.inspected_bytes),
            (ArtifactKind::Unknown, "application/x-empty", 0)
        );
        assert_eq!(r.confidence.get(), 1.0);
    }

    #[test]
    fn text_detection_variants() {
        let utf8 = sniff("héllo wörld".as_bytes());
        assert_eq!(
            (utf8.kind, utf8.text_encoding),
            (ArtifactKind::Text, Some(TextEncoding::Utf8))
        );
        let bom8 = sniff(b"\xEF\xBB\xBFabc");
        assert_eq!(bom8.text_encoding, Some(TextEncoding::Utf8));
        let le = sniff(&[0xFF, 0xFE, b'a', 0]);
        assert_eq!(le.text_encoding, Some(TextEncoding::Utf16Le));
        let be = sniff(&[0xFE, 0xFF, 0, b'a']);
        assert_eq!(be.text_encoding, Some(TextEncoding::Utf16Be));
        let bomless = sniff(&[b'h', 0, b'e', 0, b'y', 0, b'!', 0]);
        assert_eq!(bomless.text_encoding, Some(TextEncoding::Utf16Le));
        assert!(bomless.confidence.get() < 0.95);
        let latin = sniff(b"caf\xE9 au lait");
        assert_eq!(latin.text_encoding, Some(TextEncoding::Windows1252));
    }

    #[test]
    fn binary_noise_is_unknown() {
        let r = sniff(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05]);
        assert_eq!(
            (r.kind, r.mime),
            (ArtifactKind::Unknown, "application/octet-stream")
        );
    }

    #[test]
    fn window_is_bounded_and_reports_truncation() {
        let mut raw = vec![b'a'; SNIFF_WINDOW + 10];
        raw[SNIFF_WINDOW + 5] = 0; // outside the window: not inspected
        let r = sniff(&raw);
        assert_eq!(
            (r.kind, r.inspected_bytes, r.truncated),
            (ArtifactKind::Text, SNIFF_WINDOW, true)
        );
        let mut cut = vec![b'a'; SNIFF_WINDOW - 1];
        cut.extend_from_slice(&[0xC3, 0xA9, b'x']); // multibyte char straddles the window end
        let r = sniff(&cut);
        assert_eq!(r.text_encoding, Some(TextEncoding::Utf8));
    }

    #[test]
    fn extension_mapping() {
        assert_eq!(kind_for_extension(".PNG"), Some(ArtifactKind::Image));
        assert_eq!(kind_for_extension("json"), Some(ArtifactKind::Text));
        assert_eq!(kind_for_extension("xyz"), None);
    }
}
