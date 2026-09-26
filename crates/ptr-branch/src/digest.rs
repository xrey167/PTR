use std::collections::BTreeSet;

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
const INPUTS_DOMAIN: &[u8] = b"ptr-branch/inputs-digest/v1";

impl ValueDigest {
    /// Digest a key and its value, or its absence when `value` is `None`.
    /// Returns `BranchError::InvalidValue` if a present value cannot be
    /// encoded by the semantic journal.
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

/// Digest of the input set a key's dependency entry declared when a branch
/// touched it: which keys it is derived from, not what they hold.
///
/// A merge publishes a touched key's value but keeps whatever dependency set
/// the target declares for it, so a value computed against one set must not
/// be merged under another. Value digests of the inputs cannot tell: a
/// concurrent delta may rewire a derived key to other inputs while recomputing
/// it to the same value. The digest is domain-tagged and length-delimited over
/// the key, the number of inputs and each input name in ascending order, so a
/// key with no inputs has a digest of its own that no non-empty set shares.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InputsDigest([u8; 32]);

impl InputsDigest {
    /// Digest `key`'s input set. `inputs` may come in any order and repeat a
    /// name; the digest is of the set.
    pub fn of<'a, I>(key: &str, inputs: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let inputs: BTreeSet<&str> = inputs.into_iter().collect();
        let mut hasher = Sha256::new();
        hasher.update(INPUTS_DOMAIN);
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update((inputs.len() as u64).to_le_bytes());
        for input in inputs {
            hasher.update((input.len() as u64).to_le_bytes());
            hasher.update(input.as_bytes());
        }
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Rebuild a digest read back from storage. As for [`ValueDigest`], bytes
    /// a store corrupted match no input set, and certification reports the
    /// key as a conflict.
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

    #[test]
    fn an_input_set_digest_is_of_the_set_and_never_shared_by_another_set_or_key() {
        let empty = InputsDigest::of("d", []);
        assert_ne!(empty, InputsDigest::of("d", [""]));
        assert_ne!(
            InputsDigest::of("d", ["ab"]),
            InputsDigest::of("d", ["a", "b"])
        );
        assert_ne!(InputsDigest::of("d", ["a"]), InputsDigest::of("e", ["a"]));
        assert_ne!(InputsDigest::of("d", ["a"]), InputsDigest::of("d", ["b"]));
        assert_eq!(
            InputsDigest::of("d", ["b", "a", "b"]),
            InputsDigest::of("d", ["a", "b"])
        );
    }
}
