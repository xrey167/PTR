use sha2::{Digest, Sha256};

use ptr_semdb::{canonical_input_bytes, SemanticValue};

use crate::error::BranchError;

/// Digest of what a key held when a branch read it, or of its absence.
///
/// It hashes the canonical journal encoding of `key = value`
/// ([`canonical_input_bytes`]), the same bytes neural-state admission digests,
/// so a branch and an admitted neural state can never disagree about whether
/// an input changed. A payload's `source` participates, as it does in the
/// journal: a changed provenance is a changed input.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ValueDigest([u8; 32]);

const DOMAIN: &[u8] = b"ptr-branch/value-digest/v2";
const RANGE_DOMAIN: &[u8] = b"ptr-branch/range-digest/v1";

impl ValueDigest {
    pub fn of(key: &str, value: Option<&SemanticValue>) -> Result<Self, BranchError> {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        match value {
            None => {
                hasher.update([0u8]);
                hasher.update((key.len() as u64).to_le_bytes());
                hasher.update(key.as_bytes());
            }
            Some(value) => {
                let canonical =
                    canonical_input_bytes(key, value).map_err(|_| BranchError::InvalidValue {
                        key: key.to_owned(),
                    })?;
                hasher.update([1u8]);
                hasher.update(&canonical);
            }
        }
        Ok(Self(hasher.finalize().into()))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Rebuild a digest read back from storage. The bytes are trusted to be a
    /// digest this type produced; a store that corrupted them yields a digest
    /// no value matches, which certification reports as a conflict.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Digest of every key under a prefix and of each key's value: what a
/// predicate read saw. A key inserted or removed under the prefix changes it,
/// which is how certification detects phantoms that point reads cannot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RangeDigest([u8; 32]);

impl RangeDigest {
    /// `entries` must be sorted by key, as a snapshot's keys are.
    pub(crate) fn of<'a, I>(prefix: &str, entries: I) -> Result<Self, BranchError>
    where
        I: IntoIterator<Item = (&'a str, &'a SemanticValue)>,
    {
        let mut hasher = Sha256::new();
        hasher.update(RANGE_DOMAIN);
        hasher.update((prefix.len() as u64).to_le_bytes());
        hasher.update(prefix.as_bytes());
        for (key, value) in entries {
            hasher.update(ValueDigest::of(key, Some(value))?.as_bytes());
        }
        Ok(Self(hasher.finalize().into()))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_semdb::SemanticPayload;
    use ptr_types::TypeId;

    fn payload(source: &str, bytes: &[u8]) -> SemanticValue {
        SemanticValue::Payload(SemanticPayload {
            type_id: TypeId::from("t"),
            source: source.into(),
            bytes: bytes.to_vec(),
        })
    }

    #[test]
    fn absence_and_an_empty_text_never_share_a_digest() {
        let empty = SemanticValue::Text(String::new());
        assert_ne!(
            ValueDigest::of("k", None).unwrap(),
            ValueDigest::of("k", Some(&empty)).unwrap()
        );
    }

    #[test]
    fn the_same_value_under_another_key_is_another_input() {
        let value = SemanticValue::Text("x".into());
        assert_ne!(
            ValueDigest::of("a", Some(&value)).unwrap(),
            ValueDigest::of("b", Some(&value)).unwrap()
        );
    }

    #[test]
    fn a_changed_source_is_a_changed_input_as_in_the_journal() {
        assert_ne!(
            ValueDigest::of("k", Some(&payload("a", b"x"))).unwrap(),
            ValueDigest::of("k", Some(&payload("b", b"x"))).unwrap()
        );
    }
}
