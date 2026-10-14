use std::fmt;

/// One of the four seals the `HostShm` carrier requires, in the order they are applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Seal {
    /// `F_SEAL_WRITE`: no byte can change after the export.
    Write,
    /// `F_SEAL_SHRINK`: the file cannot get smaller.
    Shrink,
    /// `F_SEAL_GROW`: the file cannot get larger.
    Grow,
    /// `F_SEAL_SEAL`: no seal can be added or removed afterwards.
    Seal,
}

/// Why a carrier refused to export or import a tensor. Every refusal is typed; nothing is
/// mapped or returned when one is.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FabricError {
    /// An empty tensor has no region and no handle.
    EmptyTensor,
    /// The tensor is larger than the carrier or the handle bound allows.
    TooLarge { len: u64, max: u64 },
    /// The handle names another carrier than the one asked to import it.
    WrongCarrier,
    /// The locator is not the variant of this carrier.
    WrongLocator,
    /// The bytes or the file do not have the length the handle commits to.
    LengthMismatch { expected: u64, found: u64 },
    /// The bytes do not hash to the handle's `content_digest`.
    DigestMismatch,
    /// The tensor does not start at offset 0 of the region.
    NonZeroOffset { offset: u64 },
    /// The region lacks a required seal.
    MissingSeal(Seal),
    /// The descriptor cannot carry seals at all (not a `memfd`: a regular file, a pipe, a
    /// socket), so it is never a `HostShm` region.
    NotSealable,
    /// The length does not fit the address space of this host.
    LengthNotAddressable { len: u64 },
    /// An operating-system call failed; the text is the OS error.
    Io(String),
}

impl fmt::Display for FabricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTensor => write!(formatter, "an empty tensor has no carrier region"),
            Self::TooLarge { len, max } => {
                write!(
                    formatter,
                    "tensor of {len} bytes exceeds the bound of {max}"
                )
            }
            Self::WrongCarrier => write!(formatter, "the handle names another carrier"),
            Self::WrongLocator => write!(formatter, "the locator is not of this carrier"),
            Self::LengthMismatch { expected, found } => {
                write!(
                    formatter,
                    "length {found} where the handle commits to {expected}"
                )
            }
            Self::DigestMismatch => write!(formatter, "content digest mismatch"),
            Self::NonZeroOffset { offset } => {
                write!(formatter, "tensor offset {offset} is not 0")
            }
            Self::MissingSeal(seal) => write!(formatter, "the region lacks the {seal:?} seal"),
            Self::LengthNotAddressable { len } => {
                write!(formatter, "length {len} does not fit this host")
            }
            Self::NotSealable => write!(formatter, "the descriptor is not a sealable memfd"),
            Self::Io(message) => write!(formatter, "os error: {message}"),
        }
    }
}

impl std::error::Error for FabricError {}
