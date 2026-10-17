//! Hand-written little-endian binary codec shared by the durable stores.
//!
//! Every read is bounds-checked; element counts are validated against the bytes that remain so
//! that a corrupted length can never trigger an oversized allocation.

/// Why a payload could not be decoded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    /// The payload ended before the value was complete.
    Truncated,
    /// A length or element count exceeds the bytes that remain.
    LengthTooLarge,
    /// A string field is not valid UTF-8.
    InvalidUtf8,
    /// An enum tag byte is not assigned.
    InvalidTag { field: &'static str, tag: u8 },
    /// A decoded value violates a domain constraint (for example a probability outside 0..=1).
    InvalidValue(&'static str),
    /// Bytes remain after the value was fully decoded.
    TrailingBytes,
}

/// Largest number of elements reserved up front while decoding. A validated count can still be
/// large (a 64 MiB frame of tiny elements), so vectors grow on demand beyond this.
pub(crate) const MAX_PREALLOCATED_ELEMENTS: usize = 1024;

/// Capacity to reserve for `count` decoded elements.
pub(crate) fn bounded_capacity(count: usize) -> usize {
    count.min(MAX_PREALLOCATED_ELEMENTS)
}

/// Minimum encoded size of a length-prefixed string or byte blob (the `u32` length alone).
pub(crate) const MIN_STRING_BYTES: usize = 4;

#[derive(Default)]
pub(crate) struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.buf.push(value);
    }

    pub(crate) fn bool(&mut self, value: bool) {
        self.buf.push(u8::from(value));
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub(crate) fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Writes a count; counts beyond `u32::MAX` cannot occur because inputs are bounded.
    pub(crate) fn count(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        self.count(bytes.len());
        self.raw(bytes);
    }

    pub(crate) fn str(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    pub(crate) fn opt_str(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.bool(true);
                self.str(value);
            }
            None => self.bool(false),
        }
    }
}

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(crate) fn finish(self) -> Result<(), CodecError> {
        if self.pos == self.buf.len() {
            Ok(())
        } else {
            Err(CodecError::TrailingBytes)
        }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], CodecError> {
        let end = self.pos.checked_add(len).ok_or(CodecError::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(CodecError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, CodecError> {
        let slice = self.take(1)?;
        Ok(slice[0])
    }

    pub(crate) fn bool(&mut self) -> Result<bool, CodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            tag => Err(CodecError::InvalidTag { field: "bool", tag }),
        }
    }

    pub(crate) fn u32(&mut self) -> Result<u32, CodecError> {
        let mut raw = [0_u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(raw))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, CodecError> {
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) fn array32(&mut self) -> Result<[u8; 32], CodecError> {
        let mut raw = [0_u8; 32];
        raw.copy_from_slice(self.take(32)?);
        Ok(raw)
    }

    /// Reads a byte length (strings, byte blobs): at least that many bytes must remain.
    pub(crate) fn count(&mut self) -> Result<usize, CodecError> {
        self.count_of(1)
    }

    /// Reads an element count and checks that the bytes that remain can hold that many elements
    /// of at least `min_element_bytes` encoded bytes each, so a CRC-valid frame with an inflated
    /// count is rejected before anything is reserved for it.
    pub(crate) fn count_of(&mut self, min_element_bytes: usize) -> Result<usize, CodecError> {
        let count = usize::try_from(self.u32()?).map_err(|_| CodecError::LengthTooLarge)?;
        let remaining = self.buf.len() - self.pos;
        if count > remaining / min_element_bytes.max(1) {
            return Err(CodecError::LengthTooLarge);
        }
        Ok(count)
    }

    pub(crate) fn bytes(&mut self) -> Result<Vec<u8>, CodecError> {
        let len = self.count()?;
        Ok(self.take(len)?.to_vec())
    }

    pub(crate) fn string(&mut self) -> Result<String, CodecError> {
        let len = self.count()?;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| CodecError::InvalidUtf8)
    }

    pub(crate) fn opt_string(&mut self) -> Result<Option<String>, CodecError> {
        if self.bool()? {
            Ok(Some(self.string()?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_primitives() {
        let mut w = Writer::new();
        w.u8(7);
        w.bool(true);
        w.u32(0xdead_beef);
        w.u64(u64::MAX - 3);
        w.f32(0.25);
        w.str("héllo");
        w.opt_str(None);
        w.opt_str(Some("x"));
        let bytes = w.finish();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.u8(), Ok(7));
        assert_eq!(r.bool(), Ok(true));
        assert_eq!(r.u32(), Ok(0xdead_beef));
        assert_eq!(r.u64(), Ok(u64::MAX - 3));
        assert_eq!(r.f32(), Ok(0.25));
        assert_eq!(r.string().as_deref(), Ok("héllo"));
        assert_eq!(r.opt_string(), Ok(None));
        assert_eq!(r.opt_string(), Ok(Some("x".to_owned())));
        assert_eq!(r.finish(), Ok(()));
    }

    #[test]
    fn min_string_size_is_the_encoding_of_the_empty_string() {
        let mut w = Writer::new();
        w.str("");
        assert_eq!(w.finish().len(), MIN_STRING_BYTES);
    }

    #[test]
    fn count_of_rejects_counts_the_remaining_bytes_cannot_hold() {
        let mut w = Writer::new();
        w.u32(3);
        w.raw(&[0; 24]);
        let bytes = w.finish();
        assert_eq!(Reader::new(&bytes).count_of(8), Ok(3));
        assert_eq!(
            Reader::new(&bytes).count_of(9),
            Err(CodecError::LengthTooLarge)
        );
        assert_eq!(bounded_capacity(usize::MAX), MAX_PREALLOCATED_ELEMENTS);
        assert_eq!(bounded_capacity(3), 3);
    }

    #[test]
    fn rejects_truncation_oversized_lengths_and_bad_utf8() {
        assert_eq!(Reader::new(&[1, 2]).u32(), Err(CodecError::Truncated));
        let mut w = Writer::new();
        w.u32(1_000_000);
        assert_eq!(
            Reader::new(&w.finish()).bytes(),
            Err(CodecError::LengthTooLarge)
        );
        let mut w = Writer::new();
        w.bytes(&[0xff, 0xfe]);
        assert_eq!(
            Reader::new(&w.finish()).string(),
            Err(CodecError::InvalidUtf8)
        );
        assert_eq!(
            Reader::new(&[2]).bool(),
            Err(CodecError::InvalidTag {
                field: "bool",
                tag: 2
            })
        );
        assert_eq!(
            Reader::new(&[0, 0]).finish(),
            Err(CodecError::TrailingBytes)
        );
    }
}
