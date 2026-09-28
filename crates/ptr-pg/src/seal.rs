//! The key a substrate tags stored branches with, so that rows it did not
//! store under the key never load as a sealed branch under it.

use std::fmt;

use ptr_ledger::integrity::{constant_time_eq, hmac_sha256};

/// Domain every branch seal tag is computed under. `ptr_ledger`'s anchor MACs
/// use a domain of their own, so a tag of one kind never verifies as the
/// other under a shared key.
const SEAL_TAG_DOMAIN: &[u8] = b"ptr-pg/branch-seal/v1";

/// Host-held key of the branch seal tags a substrate writes and checks
/// (`PgSubstrate::with_branch_seal_key`).
///
/// The key must come from the embedding host's secret material and must never
/// be stored in the database whose branches it tags: a writer who holds it can
/// tag any rows. A tag authenticates the rows to every substrate holding the
/// same key, whatever its schemas, so a key is shared only by substrates meant
/// to accept each other's branches. [`fmt::Debug`] is redacted and [`Drop`]
/// overwrites the bytes on a best-effort basis, as `ptr_ledger::AnchorKey`
/// does: Rust cannot guarantee erasure of copies the optimizer, the allocator
/// or a page swap retained.
pub struct BranchSealKey([u8; 32]);

impl BranchSealKey {
    /// Construct a host-held key from exactly 256 bits of secret material.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The tag of the sealed branch whose digest is `seal_digest`
    /// (`ptr_branch::SealedBranch::seal_digest`): HMAC-SHA-256 under this
    /// key of [`SEAL_TAG_DOMAIN`] followed by the digest.
    pub(crate) fn tag(&self, seal_digest: &[u8; 32]) -> [u8; 32] {
        let mut message = Vec::with_capacity(SEAL_TAG_DOMAIN.len() + seal_digest.len());
        message.extend_from_slice(SEAL_TAG_DOMAIN);
        message.extend_from_slice(seal_digest);
        hmac_sha256(&self.0, &message)
    }

    /// Whether `stored` is [`tag`](Self::tag) of `seal_digest`, compared
    /// without an early exit.
    pub(crate) fn verifies(&self, seal_digest: &[u8; 32], stored: &[u8]) -> bool {
        constant_time_eq(&self.tag(seal_digest), stored)
    }
}

impl fmt::Debug for BranchSealKey {
    /// Render without exposing key material.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BranchSealKey(<redacted>)")
    }
}

impl Drop for BranchSealKey {
    /// Best-effort overwrite of the owned key bytes.
    fn drop(&mut self) {
        self.0.fill(0);
        let _ = std::hint::black_box(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_is_the_hmac_of_the_domain_and_the_seal_digest() {
        let key = BranchSealKey::from_bytes([7; 32]);
        let digest = [9; 32];
        let mut message = b"ptr-pg/branch-seal/v1".to_vec();
        message.extend_from_slice(&digest);
        assert_eq!(key.tag(&digest), hmac_sha256(&[7; 32], &message));
        assert!(key.verifies(&digest, &key.tag(&digest)));
    }

    #[test]
    fn a_tag_verifies_only_its_own_digest_under_its_own_key() {
        let key = BranchSealKey::from_bytes([7; 32]);
        let tag = key.tag(&[9; 32]);
        let mut flipped = [9; 32];
        flipped[31] ^= 1;
        assert!(!key.verifies(&flipped, &tag));
        assert!(!BranchSealKey::from_bytes([8; 32]).verifies(&[9; 32], &tag));
        let mut altered = tag;
        altered[0] ^= 0x80;
        assert!(!key.verifies(&[9; 32], &altered));
        assert!(!key.verifies(&[9; 32], &tag[..31]));
        assert!(!key.verifies(&[9; 32], &[]));
        // The domain separates it from an HMAC of the bare digest.
        assert_ne!(tag, hmac_sha256(&[7; 32], &[9; 32]));
    }

    #[test]
    fn the_key_never_prints() {
        let key = BranchSealKey::from_bytes([0x5a; 32]);
        let printed = format!("{key:?}");
        assert_eq!(printed, "BranchSealKey(<redacted>)");
        assert!(!printed.contains("5a") && !printed.contains("90"));
    }
}
