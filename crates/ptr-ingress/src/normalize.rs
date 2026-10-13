//! Deterministic text normalization: BOM detection, strict UTF-8 / UTF-16 decoding, an explicit
//! Windows-1252 fallback, CRLF to LF, NUL policy. Unicode NFC is out of scope. Every normalized
//! character keeps a mapping to the raw bytes it came from.

use std::fmt;
use std::ops::Range;

use crate::provenance::{OffsetMap, RawSpan};

const REPLACEMENT: char = '\u{FFFD}';

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TextEncoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    /// Single-byte fallback (C1 range mapped like Windows-1252, undefined bytes pass through).
    Windows1252,
}

impl TextEncoding {
    pub fn label(self) -> &'static str {
        match self {
            TextEncoding::Utf8 => "utf-8",
            TextEncoding::Utf16Le => "utf-16le",
            TextEncoding::Utf16Be => "utf-16be",
            TextEncoding::Windows1252 => "windows-1252",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NulPolicy {
    /// Fail with [`NormalizeError::NulByte`]; a NUL means the input is probably not text.
    #[default]
    Reject,
    /// Drop NUL characters and count them in [`NormalizedText::nul_stripped`].
    Strip,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InvalidUtf8Policy {
    Reject,
    /// Decode the whole input as Windows-1252 and mark the result lossy.
    #[default]
    FallbackWindows1252,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizeOptions {
    pub nul_policy: NulPolicy,
    pub invalid_utf8: InvalidUtf8Policy,
    /// Skip detection and decode as this encoding (a matching BOM is still consumed).
    pub forced_encoding: Option<TextEncoding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormalizeError {
    NulByte { raw_offset: usize },
    InvalidUtf8 { raw_offset: usize },
}

impl fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormalizeError::NulByte { raw_offset } => {
                write!(f, "NUL character at raw byte {raw_offset}")
            }
            NormalizeError::InvalidUtf8 { raw_offset } => {
                write!(f, "invalid UTF-8 at raw byte {raw_offset}")
            }
        }
    }
}

impl std::error::Error for NormalizeError {}

/// Normalized text plus everything needed to relate it to the raw bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedText {
    text: String,
    encoding: TextEncoding,
    bom_len: usize,
    lossy: bool,
    crlf_normalized: usize,
    nul_stripped: usize,
    offsets: OffsetMap,
}

impl NormalizedText {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn encoding(&self) -> TextEncoding {
        self.encoding
    }

    /// Number of leading raw bytes consumed as a byte-order mark.
    pub fn bom_len(&self) -> usize {
        self.bom_len
    }

    /// True when decoding guessed the encoding (fallback) or substituted U+FFFD.
    pub fn is_lossy(&self) -> bool {
        self.lossy
    }

    pub fn crlf_normalized(&self) -> usize {
        self.crlf_normalized
    }

    pub fn nul_stripped(&self) -> usize {
        self.nul_stripped
    }

    pub fn offset_map(&self) -> &OffsetMap {
        &self.offsets
    }

    /// Raw span of a byte range of [`Self::text`].
    pub fn raw_span(&self, range: Range<usize>) -> Option<RawSpan> {
        self.offsets.raw_span(range)
    }

    /// Decodes `raw` at `span` with this text's encoding, applying the same newline and NUL
    /// handling, and reports whether the result equals `expected`.
    pub fn span_round_trips(&self, raw: &[u8], span: RawSpan, expected: &str) -> bool {
        let Some(bytes) = span.slice(raw) else {
            return false;
        };
        let (decoded, _) = decode_body(bytes, span.byte_start, self.encoding);
        match assemble(&decoded, NulPolicy::Strip) {
            Ok(parts) => parts.text == expected,
            Err(_) => false,
        }
    }
}

#[derive(Clone, Copy)]
struct Decoded {
    ch: char,
    start: usize,
    end: usize,
}

struct Assembled {
    text: String,
    offsets: OffsetMap,
    crlf: usize,
    nul: usize,
}

/// Normalizes `raw` according to `options`. Empty input and a BOM alone give empty text.
pub fn normalize(raw: &[u8], options: &NormalizeOptions) -> Result<NormalizedText, NormalizeError> {
    let (mut encoding, mut bom_len) = match (detect_bom(raw), options.forced_encoding) {
        (Some(found), None) => found,
        (Some((enc, len)), Some(forced)) if enc == forced => (enc, len),
        (_, forced) => (forced.unwrap_or(TextEncoding::Utf8), 0),
    };
    let mut lossy = false;
    let decoded = match encoding {
        TextEncoding::Utf8 => match decode_utf8(raw, bom_len) {
            Ok(decoded) => decoded,
            Err(raw_offset) => match options.invalid_utf8 {
                InvalidUtf8Policy::Reject => {
                    return Err(NormalizeError::InvalidUtf8 { raw_offset });
                }
                InvalidUtf8Policy::FallbackWindows1252 => {
                    encoding = TextEncoding::Windows1252;
                    bom_len = 0;
                    lossy = true;
                    decode_windows_1252(raw, 0)
                }
            },
        },
        TextEncoding::Utf16Le | TextEncoding::Utf16Be => {
            let (decoded, replaced) = decode_body(&raw[bom_len..], bom_len, encoding);
            lossy |= replaced;
            decoded
        }
        TextEncoding::Windows1252 => decode_windows_1252(raw, 0),
    };
    let parts = assemble(&decoded, options.nul_policy)?;
    Ok(NormalizedText {
        text: parts.text,
        encoding,
        bom_len,
        lossy,
        crlf_normalized: parts.crlf,
        nul_stripped: parts.nul,
        offsets: parts.offsets,
    })
}

fn detect_bom(raw: &[u8]) -> Option<(TextEncoding, usize)> {
    match raw {
        [0xEF, 0xBB, 0xBF, ..] => Some((TextEncoding::Utf8, 3)),
        [0xFF, 0xFE, ..] => Some((TextEncoding::Utf16Le, 2)),
        [0xFE, 0xFF, ..] => Some((TextEncoding::Utf16Be, 2)),
        _ => None,
    }
}

/// Decodes a body slice; `base` is the raw offset of `body[0]`. Returns whether replacement
/// characters were substituted.
fn decode_body(body: &[u8], base: usize, encoding: TextEncoding) -> (Vec<Decoded>, bool) {
    match encoding {
        TextEncoding::Utf8 => match std::str::from_utf8(body) {
            Ok(text) => (utf8_chars(text, base), false),
            Err(error) => {
                let (valid, _) = body.split_at(error.valid_up_to());
                let mut out = match std::str::from_utf8(valid) {
                    Ok(text) => utf8_chars(text, base),
                    Err(_) => Vec::new(),
                };
                let start = base + valid.len();
                out.push(Decoded {
                    ch: REPLACEMENT,
                    start,
                    end: base + body.len(),
                });
                (out, true)
            }
        },
        TextEncoding::Utf16Le => decode_utf16(body, base, false),
        TextEncoding::Utf16Be => decode_utf16(body, base, true),
        TextEncoding::Windows1252 => (decode_windows_1252(body, base), false),
    }
}

fn utf8_chars(text: &str, base: usize) -> Vec<Decoded> {
    text.char_indices()
        .map(|(i, ch)| Decoded {
            ch,
            start: base + i,
            end: base + i + ch.len_utf8(),
        })
        .collect()
}

fn decode_utf8(raw: &[u8], bom_len: usize) -> Result<Vec<Decoded>, usize> {
    let body = raw.get(bom_len..).unwrap_or(&[]);
    match std::str::from_utf8(body) {
        Ok(text) => Ok(utf8_chars(text, bom_len)),
        Err(error) => Err(bom_len + error.valid_up_to()),
    }
}

fn unit(body: &[u8], at: usize, big_endian: bool) -> u16 {
    let pair = [
        body.get(at).copied().unwrap_or(0),
        body.get(at + 1).copied().unwrap_or(0),
    ];
    if big_endian {
        u16::from_be_bytes(pair)
    } else {
        u16::from_le_bytes(pair)
    }
}

fn decode_utf16(body: &[u8], base: usize, big_endian: bool) -> (Vec<Decoded>, bool) {
    let mut out = Vec::with_capacity(body.len() / 2);
    let mut replaced = false;
    let even = body.len() / 2 * 2;
    let mut i = 0;
    while i < even {
        let u = unit(body, i, big_endian);
        let start = base + i;
        let mut width = 2;
        let ch = if (0xD800..0xDC00).contains(&u) {
            let low = if i + 2 < even {
                unit(body, i + 2, big_endian)
            } else {
                0
            };
            if (0xDC00..0xE000).contains(&low) {
                width = 4;
                let cp = 0x10000 + (((u32::from(u) - 0xD800) << 10) | (u32::from(low) - 0xDC00));
                char::from_u32(cp).unwrap_or(REPLACEMENT)
            } else {
                replaced = true;
                REPLACEMENT
            }
        } else if (0xDC00..0xE000).contains(&u) {
            replaced = true;
            REPLACEMENT
        } else {
            char::from_u32(u32::from(u)).unwrap_or(REPLACEMENT)
        };
        out.push(Decoded {
            ch,
            start,
            end: start + width,
        });
        i += width;
    }
    if even < body.len() {
        replaced = true;
        out.push(Decoded {
            ch: REPLACEMENT,
            start: base + even,
            end: base + body.len(),
        });
    }
    (out, replaced)
}

fn decode_windows_1252(body: &[u8], base: usize) -> Vec<Decoded> {
    const C1: [u16; 32] = [
        0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
        0x2039, 0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
        0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
    ];
    body.iter()
        .enumerate()
        .map(|(i, &byte)| {
            let cp = match byte {
                0x80..=0x9F => u32::from(C1[usize::from(byte - 0x80)]),
                other => u32::from(other),
            };
            Decoded {
                ch: char::from_u32(cp).unwrap_or(REPLACEMENT),
                start: base + i,
                end: base + i + 1,
            }
        })
        .collect()
}

fn assemble(decoded: &[Decoded], nul_policy: NulPolicy) -> Result<Assembled, NormalizeError> {
    let mut text = String::with_capacity(decoded.len());
    let mut offsets = OffsetMap::default();
    let (mut crlf, mut nul) = (0, 0);
    let mut i = 0;
    while let Some(&current) = decoded.get(i) {
        i += 1;
        if current.ch == '\0' {
            match nul_policy {
                NulPolicy::Reject => {
                    return Err(NormalizeError::NulByte {
                        raw_offset: current.start,
                    });
                }
                NulPolicy::Strip => {
                    nul += 1;
                    continue;
                }
            }
        }
        let mut end = current.end;
        let mut ch = current.ch;
        if ch == '\r' {
            if let Some(next) = decoded.get(i) {
                if next.ch == '\n' && next.start == current.end {
                    ch = '\n';
                    end = next.end;
                    crlf += 1;
                    i += 1;
                }
            }
        }
        let norm_start = text.len();
        text.push(ch);
        offsets.push(
            norm_start,
            ch.len_utf8(),
            RawSpan {
                byte_start: current.start,
                byte_end: end,
            },
        );
    }
    Ok(Assembled {
        text,
        offsets,
        crlf,
        nul,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(raw: &[u8]) -> NormalizedText {
        normalize(raw, &NormalizeOptions::default()).expect("normalizes")
    }

    #[test]
    fn empty_and_bom_only_give_empty_text() {
        assert_eq!(norm(b"").text(), "");
        let bom = norm(&[0xEF, 0xBB, 0xBF]);
        assert_eq!(
            (bom.text(), bom.bom_len(), bom.encoding()),
            ("", 3, TextEncoding::Utf8)
        );
        let le = norm(&[0xFF, 0xFE]);
        assert_eq!((le.text(), le.encoding()), ("", TextEncoding::Utf16Le));
    }

    #[test]
    fn utf8_bom_is_stripped_and_offsets_skip_it() {
        let n = norm(b"\xEF\xBB\xBFhi");
        assert_eq!(n.text(), "hi");
        assert_eq!(
            n.raw_span(0..2),
            Some(RawSpan {
                byte_start: 3,
                byte_end: 5
            })
        );
        assert!(!n.is_lossy());
    }

    #[test]
    fn utf16_both_endians_with_surrogate_pair() {
        let le = norm(&[0xFF, 0xFE, b'a', 0, 0x3D, 0xD8, 0x00, 0xDE]);
        assert_eq!(le.text(), "a\u{1F600}");
        assert_eq!(
            le.raw_span(1..5),
            Some(RawSpan {
                byte_start: 4,
                byte_end: 8
            })
        );
        let be = norm(&[0xFE, 0xFF, 0, b'a', 0xD8, 0x3D, 0xDE, 0x00]);
        assert_eq!(be.text(), "a\u{1F600}");
        assert!(!be.is_lossy());
    }

    #[test]
    fn utf16_lone_surrogate_and_odd_tail_are_lossy() {
        let lone = norm(&[0xFF, 0xFE, 0x3D, 0xD8, b'x', 0]);
        assert_eq!(lone.text(), "\u{FFFD}x");
        assert!(lone.is_lossy());
        let odd = norm(&[0xFF, 0xFE, b'x', 0, 0x41]);
        assert_eq!(odd.text(), "x\u{FFFD}");
        assert!(odd.is_lossy());
    }

    #[test]
    fn invalid_utf8_falls_back_to_windows_1252_flagged_lossy() {
        let n = norm(b"caf\xE9 \x80");
        assert_eq!(n.text(), "caf\u{E9} \u{20AC}");
        assert_eq!(n.encoding(), TextEncoding::Windows1252);
        assert!(n.is_lossy());
        let strict = NormalizeOptions {
            invalid_utf8: InvalidUtf8Policy::Reject,
            ..Default::default()
        };
        assert_eq!(
            normalize(b"ab\xFF", &strict),
            Err(NormalizeError::InvalidUtf8 { raw_offset: 2 })
        );
    }

    #[test]
    fn crlf_collapses_to_lf_with_two_byte_span_and_lone_cr_is_kept() {
        let n = norm(b"a\r\nb\rc");
        assert_eq!(n.text(), "a\nb\rc");
        assert_eq!(n.crlf_normalized(), 1);
        assert_eq!(
            n.raw_span(1..2),
            Some(RawSpan {
                byte_start: 1,
                byte_end: 3
            })
        );
        assert_eq!(
            n.raw_span(2..3),
            Some(RawSpan {
                byte_start: 3,
                byte_end: 4
            })
        );
    }

    #[test]
    fn nul_is_rejected_by_default_and_strippable() {
        assert_eq!(
            normalize(b"ab\0c", &NormalizeOptions::default()),
            Err(NormalizeError::NulByte { raw_offset: 2 })
        );
        let strip = NormalizeOptions {
            nul_policy: NulPolicy::Strip,
            ..Default::default()
        };
        let n = normalize(b"ab\0c", &strip).expect("strips");
        assert_eq!((n.text(), n.nul_stripped()), ("abc", 1));
        assert_eq!(
            n.raw_span(0..3),
            Some(RawSpan {
                byte_start: 0,
                byte_end: 4
            })
        );
    }

    #[test]
    fn forced_utf16_without_bom_decodes() {
        let forced = NormalizeOptions {
            forced_encoding: Some(TextEncoding::Utf16Le),
            ..Default::default()
        };
        let n = normalize(&[b'h', 0, b'i', 0], &forced).expect("decodes");
        assert_eq!((n.text(), n.bom_len()), ("hi", 0));
    }

    #[test]
    fn span_round_trip_detects_mismatch() {
        let raw = b"one\r\ntwo";
        let n = norm(raw);
        let span = n.raw_span(0..7).expect("span");
        assert!(n.span_round_trips(raw, span, "one\ntwo"));
        assert!(!n.span_round_trips(raw, span, "one\r\ntwo"));
        assert!(!n.span_round_trips(
            raw,
            RawSpan {
                byte_start: 0,
                byte_end: 99
            },
            "x"
        ));
    }
}
