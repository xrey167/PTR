//! Value-free pod identity.
//!
//! A pod keeps one stable [`PodIdentity`] across artifacts and generations. The identity is derived
//! from the hard semantics of the pod (domain, kind, predicate, role and its aliases) and never
//! from values such as the artifact or the generation, so retraining a pod does not rename it.
//! The [`ClusterKey`] groups pods by the same hard semantics but ignores aliases.

use super::{put_text, Digest, PodIdentity};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

/// Version of the descriptor layout hashed into identities and cluster keys.
pub const POD_IDENTITY_SCHEMA_VERSION: u32 = 1;

const IDENTITY_DOMAIN: &str = "ptr.pod.identity";
const CLUSTER_DOMAIN: &str = "ptr.pod.cluster-key";
const IDENTITY_PREFIX: &str = "pod:";

/// Grouping key over the hard semantics of a pod, independent of values and aliases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClusterKey(pub Digest);

impl ClusterKey {
    pub fn digest(&self) -> &Digest {
        &self.0
    }

    /// Lower-case hex spelling used by exported artifacts.
    pub fn to_hex(&self) -> String {
        hex(&self.0)
    }

    /// Parses exactly the spelling [`ClusterKey::to_hex`] prints.
    pub fn from_hex(value: &str) -> Result<Self, PodIdentityError> {
        let bytes = value.as_bytes();
        if bytes.len() != 64
            || !bytes
                .iter()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(PodIdentityError::InvalidClusterKey);
        }
        let nibble = |byte: u8| match byte {
            b'0'..=b'9' => byte - b'0',
            _ => byte - b'a' + 10,
        };
        let mut digest = [0u8; 32];
        for (slot, pair) in digest.iter_mut().zip(bytes.chunks_exact(2)) {
            *slot = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        Ok(Self(digest))
    }
}

impl fmt::Display for ClusterKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// The hard semantics a pod identity is derived from.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PodIdentityDescriptor {
    pub schema_version: u32,
    pub domain: String,
    pub kind: String,
    pub predicate: Option<String>,
    pub semantic_role: String,
    pub aliases: BTreeSet<String>,
}

impl PodIdentityDescriptor {
    pub fn validate(&self) -> Result<(), PodIdentityError> {
        if self.schema_version != POD_IDENTITY_SCHEMA_VERSION {
            return Err(PodIdentityError::UnsupportedSchema {
                expected: POD_IDENTITY_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        for (field, value) in [
            ("domain", &self.domain),
            ("kind", &self.kind),
            ("semantic_role", &self.semantic_role),
        ] {
            if value.is_empty() {
                return Err(PodIdentityError::EmptyField { field });
            }
        }
        if matches!(&self.predicate, Some(predicate) if predicate.is_empty()) {
            return Err(PodIdentityError::EmptyField { field: "predicate" });
        }
        if self.aliases.iter().any(String::is_empty) {
            return Err(PodIdentityError::EmptyField { field: "alias" });
        }
        Ok(())
    }

    /// Groups pods by domain, kind, predicate and role. Aliases are deliberately excluded so pods
    /// that differ only by alias land in the same cluster.
    pub fn cluster_key(&self) -> ClusterKey {
        let mut bytes = Vec::new();
        put_text(&mut bytes, CLUSTER_DOMAIN);
        self.put_hard_semantics(&mut bytes);
        ClusterKey(Sha256::digest(bytes).into())
    }

    /// Stable identity of the pod. Contains no artifact, generation or other value field.
    pub fn identity(&self) -> PodIdentity {
        let mut bytes = Vec::new();
        put_text(&mut bytes, IDENTITY_DOMAIN);
        self.put_hard_semantics(&mut bytes);
        bytes.extend_from_slice(&(self.aliases.len() as u64).to_le_bytes());
        for alias in &self.aliases {
            put_text(&mut bytes, alias);
        }
        let digest: Digest = Sha256::digest(bytes).into();
        PodIdentity(format!("{IDENTITY_PREFIX}{}", hex(&digest)))
    }

    fn put_hard_semantics(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        put_text(bytes, &self.domain);
        put_text(bytes, &self.kind);
        match &self.predicate {
            Some(predicate) => {
                bytes.push(1);
                put_text(bytes, predicate);
            }
            None => bytes.push(0),
        }
        put_text(bytes, &self.semantic_role);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodIdentityError {
    UnsupportedSchema { expected: u32, actual: u32 },
    EmptyField { field: &'static str },
    InvalidClusterKey,
}

impl fmt::Display for PodIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for PodIdentityError {}

fn hex(bytes: &Digest) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> PodIdentityDescriptor {
        PodIdentityDescriptor {
            schema_version: POD_IDENTITY_SCHEMA_VERSION,
            domain: "supply".into(),
            kind: "Retrieval".into(),
            predicate: Some("lead_time".into()),
            semantic_role: "reader".into(),
            aliases: ["lead time".to_owned(), "delivery time".to_owned()].into(),
        }
    }

    #[test]
    fn cluster_key_hex_round_trips_and_rejects_other_spellings() {
        let key = descriptor().cluster_key();
        assert_eq!(ClusterKey::from_hex(&key.to_hex()), Ok(key));
        assert_eq!(
            ClusterKey::from_hex(&key.to_hex().to_uppercase()),
            Err(PodIdentityError::InvalidClusterKey)
        );
        assert_eq!(
            ClusterKey::from_hex("abc"),
            Err(PodIdentityError::InvalidClusterKey)
        );
    }

    #[test]
    fn identity_and_cluster_key_use_separate_hash_domains() {
        let descriptor = descriptor();
        let identity = descriptor.identity();
        assert!(identity.0.starts_with(IDENTITY_PREFIX));
        assert_ne!(
            identity.0.trim_start_matches(IDENTITY_PREFIX),
            descriptor.cluster_key().to_hex()
        );
    }
}
