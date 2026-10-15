use std::fmt;

/// Why a CUDA carrier step refused or failed. Every refusal is typed; nothing is mapped or
/// returned when one is.
///
/// What is guaranteed: without a driver no cudarc call runs at all (the loader is only reached
/// after [`CudaDriver::open`](super::CudaDriver::open) has shown that a driver is present), and
/// with a driver that lacks an entry point (cudarc's lazy loader panics on it) the public GPU
/// entry points catch the panic and return [`CudaError::MissingEntryPoint`]; destructors
/// swallow it. That catch needs unwinding: in a build with `panic = "abort"` it does not exist.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaError {
    /// There is no usable CUDA driver on this host: `libcuda` cannot be loaded, or `cuInit`
    /// failed. The carrier is unavailable; the tensor goes by another carrier.
    NoDriver { reason: String },
    /// A driver call returned an error code.
    Driver {
        /// The driver entry point, for example `cuMemCreate`.
        call: &'static str,
        /// The `CUresult` value.
        code: u32,
        /// The driver's own name for the code (`cuGetErrorName`), when it could be read.
        name: String,
    },
    /// This device, driver or request cannot do what the carrier needs (no VMM, no POSIX
    /// descriptor handles, no read-only host registration, a legacy export asked for).
    Unsupported { what: &'static str },
    /// The imported extent does not hold the tensor, or the sizes of the record, the locator
    /// and the allocation granularity disagree.
    ExtentMismatch { needed: u64, available: u64 },
    /// The copied bytes do not hash to the handle's `content_digest`.
    DigestMismatch,
    /// A size computation overflowed.
    Overflow,
    /// An empty tensor has no staging allocation.
    EmptyTensor,
    /// A field of the export record disagrees with what the importer expects (session,
    /// operation, device, or an export that was never filled).
    RecordMismatch { field: &'static str },
    /// The handle names another carrier than a CUDA one, or its locator is not a VMM export.
    WrongLocator,
    /// A driver entry point this crate calls is missing from the installed `libcuda` (cudarc's
    /// lazy loader panicked and the panic was caught). Only with `panic = "unwind"`.
    MissingEntryPoint,
    /// The device ordinal does not exist on this host.
    NoSuchDevice { ordinal: u32 },
    /// An operating-system call on a descriptor failed; the text is the OS error.
    Io(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDriver { reason } => write!(formatter, "no usable CUDA driver: {reason}"),
            Self::Driver { call, code, name } => {
                write!(formatter, "{call} failed with CUresult {code} ({name})")
            }
            Self::Unsupported { what } => write!(formatter, "not supported here: {what}"),
            Self::ExtentMismatch { needed, available } => write!(
                formatter,
                "the tensor needs {needed} bytes but the export holds {available}"
            ),
            Self::DigestMismatch => {
                write!(
                    formatter,
                    "the copied bytes do not match the content digest"
                )
            }
            Self::Overflow => write!(formatter, "a size computation overflowed"),
            Self::EmptyTensor => write!(formatter, "an empty tensor has no staging allocation"),
            Self::RecordMismatch { field } => {
                write!(formatter, "the export record disagrees on {field}")
            }
            Self::WrongLocator => {
                write!(
                    formatter,
                    "the handle is not a VMM CudaIpc or DevicePeer handle"
                )
            }
            Self::MissingEntryPoint => {
                write!(
                    formatter,
                    "the CUDA driver lacks an entry point this crate needs"
                )
            }
            Self::NoSuchDevice { ordinal } => write!(formatter, "no CUDA device {ordinal}"),
            Self::Io(text) => write!(formatter, "operating-system error: {text}"),
        }
    }
}

impl std::error::Error for CudaError {}
