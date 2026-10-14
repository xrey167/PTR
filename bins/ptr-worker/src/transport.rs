//! [`StdioTransport`]: control frames over any `Read`/`Write` pair.
//!
//! Each frame is the codec's encoded bytes behind a `u32` big-endian length prefix. The prefix
//! is checked against [`MAX_FRAME_BYTES`] before any body is read, so a hostile peer cannot
//! make the reader allocate more than one frame.

use std::io::{self, Read, Write};

use ptr_protocol::control::{decode, encode, CodecError, ControlFrame, MAX_FRAME_BYTES};
use ptr_protocol::session::{FrameTransport, TransportError};

/// Bytes of the length prefix.
pub const LENGTH_PREFIX_BYTES: usize = 4;
const READ_CHUNK: usize = 8192;

/// Every way moving a frame over a byte stream can fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StdioError {
    /// The length prefix (or an encoded frame) exceeds [`MAX_FRAME_BYTES`].
    Oversize {
        len: usize,
    },
    /// The stream ended inside a prefix or frame.
    UnexpectedEof {
        expected: usize,
        got: usize,
    },
    Io(io::ErrorKind),
    Encode(CodecError),
    Decode(CodecError),
}

impl std::fmt::Display for StdioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for StdioError {}

/// A [`FrameTransport`] over a reader and a writer.
#[derive(Debug)]
pub struct StdioTransport<R, W> {
    reader: R,
    writer: W,
    /// Bytes read from the stream and not yet consumed (at most one frame plus one chunk).
    buf: Vec<u8>,
    last_error: Option<StdioError>,
}

impl<R: Read, W: Write> StdioTransport<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            buf: Vec::new(),
            last_error: None,
        }
    }

    /// The exact cause of the last failure seen through the [`FrameTransport`] impl, whose
    /// error type cannot name I/O or EOF failures.
    pub fn last_error(&self) -> Option<StdioError> {
        self.last_error
    }

    pub fn into_parts(self) -> (R, W) {
        (self.reader, self.writer)
    }

    /// True when a whole frame (or an oversize prefix, which fails the next read) is already
    /// buffered, so reading it cannot block.
    pub fn has_buffered_frame(&self) -> bool {
        match self.buffered_len() {
            None => false,
            Some(len) if len > MAX_FRAME_BYTES => true,
            Some(len) => self.buf.len() >= LENGTH_PREFIX_BYTES + len,
        }
    }

    fn buffered_len(&self) -> Option<usize> {
        let prefix: [u8; LENGTH_PREFIX_BYTES] =
            self.buf.get(..LENGTH_PREFIX_BYTES)?.try_into().ok()?;
        Some(u32::from_be_bytes(prefix) as usize)
    }

    /// Reads bytes until `buf` holds `want`. `Ok(false)` is EOF.
    fn fill_to(&mut self, want: usize) -> Result<bool, StdioError> {
        let mut chunk = [0u8; READ_CHUNK];
        while self.buf.len() < want {
            match self.reader.read(&mut chunk) {
                Ok(0) => return Ok(false),
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(StdioError::Io(error.kind())),
            }
        }
        Ok(true)
    }

    /// Reads one frame. `Ok(None)` is a clean end of stream at a frame boundary.
    pub fn read_frame(&mut self) -> Result<Option<ControlFrame>, StdioError> {
        if !self.fill_to(LENGTH_PREFIX_BYTES)? {
            return if self.buf.is_empty() {
                Ok(None)
            } else {
                Err(StdioError::UnexpectedEof {
                    expected: LENGTH_PREFIX_BYTES,
                    got: self.buf.len(),
                })
            };
        }
        let len = self.buffered_len().unwrap_or(0);
        if len > MAX_FRAME_BYTES {
            return Err(StdioError::Oversize { len });
        }
        let total = LENGTH_PREFIX_BYTES + len;
        if !self.fill_to(total)? {
            return Err(StdioError::UnexpectedEof {
                expected: total,
                got: self.buf.len(),
            });
        }
        let decoded = decode(&self.buf[LENGTH_PREFIX_BYTES..total]);
        self.buf.drain(..total);
        decoded.map(Some).map_err(StdioError::Decode)
    }

    /// Writes and flushes already-framed bytes in one `write_all` (used by clients that batch
    /// several frames, or send deliberately malformed ones).
    pub fn write_raw(&mut self, bytes: &[u8]) -> Result<(), StdioError> {
        self.writer
            .write_all(bytes)
            .and_then(|()| self.writer.flush())
            .map_err(|error| StdioError::Io(error.kind()))
    }

    /// Encodes, prefixes, writes and flushes one frame.
    pub fn write_frame(&mut self, frame: &ControlFrame) -> Result<(), StdioError> {
        let bytes = encode(frame).map_err(StdioError::Encode)?;
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(StdioError::Oversize { len: bytes.len() });
        }
        let mut out = Vec::with_capacity(LENGTH_PREFIX_BYTES + bytes.len());
        out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(&bytes);
        self.write_raw(&out)
    }
}

/// `recv` returns `Ok(None)` at a clean end of stream. Failures that `TransportError` cannot
/// name are mapped to the nearest variant (`Decode(Truncated)`/`Decode(FrameTooLarge)`) and the
/// exact cause stays in [`StdioTransport::last_error`].
impl<R: Read, W: Write> FrameTransport for StdioTransport<R, W> {
    fn send(&mut self, frame: &ControlFrame) -> Result<(), TransportError> {
        self.write_frame(frame).map_err(|error| {
            self.last_error = Some(error);
            match error {
                StdioError::Encode(code) => TransportError::Encode(code),
                StdioError::Oversize { len } => {
                    TransportError::Encode(CodecError::FrameTooLarge { len })
                }
                StdioError::UnexpectedEof { .. } | StdioError::Io(_) | StdioError::Decode(_) => {
                    TransportError::Encode(CodecError::Truncated)
                }
            }
        })
    }

    fn recv(&mut self) -> Result<Option<ControlFrame>, TransportError> {
        self.read_frame().map_err(|error| {
            self.last_error = Some(error);
            match error {
                StdioError::Decode(code) | StdioError::Encode(code) => TransportError::Decode(code),
                StdioError::Oversize { len } => {
                    TransportError::Decode(CodecError::FrameTooLarge { len })
                }
                StdioError::UnexpectedEof { .. } | StdioError::Io(_) => {
                    TransportError::Decode(CodecError::Truncated)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_protocol::control::IdempotencyKey;
    use ptr_protocol::TypedPayload;
    use ptr_types::TypeId;
    use std::io::Cursor;

    fn call(seq: u64, bytes: &[u8]) -> ControlFrame {
        ControlFrame::Call {
            seq,
            idempotency_key: IdempotencyKey::new(format!("k{seq}")).unwrap(),
            payload: TypedPayload {
                type_id: TypeId::from("T"),
                bytes: bytes.to_vec(),
            },
        }
    }

    fn wire(frames: &[ControlFrame]) -> Vec<u8> {
        let mut t = StdioTransport::new(io::empty(), Vec::new());
        for frame in frames {
            t.write_frame(frame).unwrap();
        }
        t.into_parts().1
    }

    fn reader(bytes: Vec<u8>) -> StdioTransport<Cursor<Vec<u8>>, Vec<u8>> {
        StdioTransport::new(Cursor::new(bytes), Vec::new())
    }

    #[test]
    fn round_trip_two_frames_then_clean_eof() {
        let frames = [
            call(1, b"a"),
            ControlFrame::Bye {
                reason: String::new(),
            },
        ];
        let mut t = reader(wire(&frames));
        assert!(!t.has_buffered_frame());
        assert_eq!(t.read_frame().unwrap(), Some(frames[0].clone()));
        assert!(t.has_buffered_frame());
        assert_eq!(t.read_frame().unwrap(), Some(frames[1].clone()));
        assert!(!t.has_buffered_frame());
        assert_eq!(t.read_frame().unwrap(), None);
    }

    #[test]
    fn prefix_is_big_endian_length_of_the_encoded_frame() {
        let bytes = wire(&[call(1, b"x")]);
        let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert_eq!(len, bytes.len() - 4);
        assert_eq!(&bytes[4..8], b"PTRC");
    }

    #[test]
    fn truncation_inside_prefix_and_body_is_unexpected_eof() {
        let bytes = wire(&[call(1, b"payload")]);
        let mut t = reader(bytes[..2].to_vec());
        assert_eq!(
            t.read_frame(),
            Err(StdioError::UnexpectedEof {
                expected: 4,
                got: 2
            })
        );
        let mut t = reader(bytes[..bytes.len() - 1].to_vec());
        assert!(matches!(
            t.read_frame(),
            Err(StdioError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn oversize_prefix_is_refused_without_reading_a_body() {
        let len = MAX_FRAME_BYTES + 1;
        let mut bytes = (len as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(&[0; 16]);
        let mut t = reader(bytes);
        assert_eq!(t.read_frame(), Err(StdioError::Oversize { len }));
        assert_eq!(
            t.recv(),
            Err(TransportError::Decode(CodecError::FrameTooLarge { len }))
        );
        assert_eq!(t.last_error(), Some(StdioError::Oversize { len }));
    }

    #[test]
    fn oversize_frame_is_refused_on_write() {
        let mut t = StdioTransport::new(io::empty(), Vec::new());
        let big = call(1, &vec![0u8; MAX_FRAME_BYTES]);
        assert!(matches!(
            t.write_frame(&big),
            Err(StdioError::Encode(CodecError::FrameTooLarge { .. }))
        ));
        assert!(t.into_parts().1.is_empty());
    }

    #[test]
    fn garbage_body_is_a_decode_error() {
        let mut bytes = 12u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(&[0xAA; 12]);
        assert_eq!(
            reader(bytes).read_frame(),
            Err(StdioError::Decode(CodecError::BadMagic))
        );
    }

    struct Trickle(Cursor<Vec<u8>>);
    impl Read for Trickle {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = out.len().min(1);
            self.0.read(&mut out[..n])
        }
    }

    #[test]
    fn one_byte_reads_still_assemble_frames() {
        let frames = [call(1, b"abc"), call(2, b"def")];
        let mut t = StdioTransport::new(Trickle(Cursor::new(wire(&frames))), Vec::new());
        assert_eq!(t.read_frame().unwrap(), Some(frames[0].clone()));
        assert_eq!(t.read_frame().unwrap(), Some(frames[1].clone()));
        assert_eq!(t.read_frame().unwrap(), None);
    }

    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::ConnectionReset))
        }
    }
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn io_errors_are_typed_on_both_directions() {
        let mut t = StdioTransport::new(Broken, Broken);
        assert_eq!(
            t.read_frame(),
            Err(StdioError::Io(io::ErrorKind::ConnectionReset))
        );
        assert_eq!(
            t.write_frame(&call(1, b"x")),
            Err(StdioError::Io(io::ErrorKind::BrokenPipe))
        );
        assert!(t.send(&call(1, b"x")).is_err());
        assert_eq!(
            t.last_error(),
            Some(StdioError::Io(io::ErrorKind::BrokenPipe))
        );
    }

    #[test]
    fn trait_impl_reads_frames_and_reports_clean_eof() {
        let mut t = reader(wire(&[call(7, b"q")]));
        assert_eq!(FrameTransport::recv(&mut t).unwrap(), Some(call(7, b"q")));
        assert_eq!(FrameTransport::recv(&mut t).unwrap(), None);
    }
}
