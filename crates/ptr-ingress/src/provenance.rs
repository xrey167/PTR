//! Span-level provenance: byte ranges into the original raw input and the offset map from
//! normalized text back to them.

use std::ops::Range;

/// Half-open byte range `[byte_start, byte_end)` into the original raw bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RawSpan {
    pub byte_start: usize,
    pub byte_end: usize,
}

impl RawSpan {
    pub fn new(byte_start: usize, byte_end: usize) -> Option<Self> {
        (byte_start <= byte_end).then_some(Self {
            byte_start,
            byte_end,
        })
    }

    pub fn len(&self) -> usize {
        self.byte_end.saturating_sub(self.byte_start)
    }

    pub fn is_empty(&self) -> bool {
        self.byte_end <= self.byte_start
    }

    /// The raw bytes covered by this span, or `None` when it lies outside `raw`.
    pub fn slice<'a>(&self, raw: &'a [u8]) -> Option<&'a [u8]> {
        raw.get(self.byte_start..self.byte_end)
    }

    pub fn contains(&self, other: &RawSpan) -> bool {
        self.byte_start <= other.byte_start && other.byte_end <= self.byte_end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CharMapping {
    norm_start: usize,
    norm_len: usize,
    raw: RawSpan,
}

/// Maps each normalized character to the raw bytes it was decoded from. A collapsed CRLF maps
/// the single `\n` to both raw bytes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OffsetMap {
    entries: Vec<CharMapping>,
}

impl OffsetMap {
    pub(crate) fn push(&mut self, norm_start: usize, norm_len: usize, raw: RawSpan) {
        self.entries.push(CharMapping {
            norm_start,
            norm_len,
            raw,
        });
    }

    /// Number of mapped characters.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Raw span covering the normalized byte `range`. Returns `None` for an empty range or when
    /// either end is not on a character boundary of the normalized text.
    pub fn raw_span(&self, range: Range<usize>) -> Option<RawSpan> {
        if range.start >= range.end {
            return None;
        }
        let first = self.entries.partition_point(|e| e.norm_start < range.start);
        let after = self.entries.partition_point(|e| e.norm_start < range.end);
        if after <= first {
            return None;
        }
        let head = self.entries.get(first)?;
        let tail = self.entries.get(after - 1)?;
        if head.norm_start != range.start || tail.norm_start + tail.norm_len != range.end {
            return None;
        }
        Some(RawSpan {
            byte_start: head.raw.byte_start,
            byte_end: tail.raw.byte_end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_basics() {
        assert!(RawSpan::new(3, 2).is_none());
        let span = RawSpan::new(1, 3).expect("valid");
        assert_eq!(span.len(), 2);
        assert_eq!(span.slice(b"abcd"), Some(&b"bc"[..]));
        assert_eq!(span.slice(b"ab"), None);
        assert!(span.contains(&RawSpan {
            byte_start: 1,
            byte_end: 2
        }));
        assert!(!span.contains(&RawSpan {
            byte_start: 0,
            byte_end: 2
        }));
    }

    #[test]
    fn offset_map_requires_char_boundaries() {
        let mut map = OffsetMap::default();
        map.push(
            0,
            1,
            RawSpan {
                byte_start: 0,
                byte_end: 2,
            },
        );
        map.push(
            1,
            2,
            RawSpan {
                byte_start: 2,
                byte_end: 4,
            },
        );
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.raw_span(0..3),
            Some(RawSpan {
                byte_start: 0,
                byte_end: 4
            })
        );
        assert_eq!(
            map.raw_span(1..3),
            Some(RawSpan {
                byte_start: 2,
                byte_end: 4
            })
        );
        assert_eq!(map.raw_span(0..2), None);
        assert_eq!(map.raw_span(2..3), None);
        assert_eq!(map.raw_span(1..1), None);
        assert_eq!(map.raw_span(0..9), None);
    }
}
