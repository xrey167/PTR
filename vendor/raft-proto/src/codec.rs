// Copyright 2026 PTR contributors. Licensed under Apache-2.0.
//! Narrow compatibility interface used by the vendored Raft implementation.
//! This is implemented by the maintained prost codec, not a parser replacement.
pub type CodecError = prost::DecodeError;

pub trait Message: prost::Message + Default {
    fn compute_size(&self) -> u32 {
        // Preserve upstream bounded-entry sizing contract; never silently wrap.
        u32::try_from(self.encoded_len()).expect("Raft message exceeds u32 wire size")
    }
    fn write_to_bytes(&self) -> Result<Vec<u8>, CodecError> {
        Ok(self.encode_to_vec())
    }
    fn merge_from_bytes(&mut self, bytes: &[u8]) -> Result<(), CodecError> {
        self.merge(bytes)
    }
    fn default_instance() -> &'static Self;
}
macro_rules! codec_impl {
    ($($ty:ty),+ $(,)?) => {$(
        impl Message for $ty {
            fn default_instance() -> &'static Self {
                static INSTANCE: std::sync::OnceLock<$ty> = std::sync::OnceLock::new();
                INSTANCE.get_or_init(Self::default)
            }
        }
    )+};
}
use crate::eraftpb::{Entry, SnapshotMetadata, Snapshot, HardState, ConfState, ConfChange, ConfChangeSingle, ConfChangeV2};
codec_impl!(Entry, SnapshotMetadata, Snapshot, crate::eraftpb::Message, HardState, ConfState,
    ConfChange, ConfChangeSingle, ConfChangeV2);
