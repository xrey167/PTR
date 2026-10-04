//! The secret sampling key and the keyed draw used by the trusted routing path.

use ptr_types::{Digest, RequestId};
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;

const BLOCK: usize = 64;

/// Secret key of the trusted routing path that determines every sampling draw.
///
/// The draw for a request is `HMAC-SHA256(key, len(request_id) || request_id || request_digest)`
/// reduced to its top 53 bits, so nobody without the key (in particular a [`crate::Router`]
/// policy, which sees the request id and digest) can predict the draw and correlate its weights
/// with it. Replaying a routing decision requires the same key: the runtime keeps it in trusted
/// configuration (for example next to the ledger) and passes it to [`crate::route_checked`] or
/// [`crate::CheckedRouter::new`]. It is never part of a [`crate::RouteRequest`].
///
/// The key is deliberately not `Clone`, `Copy`, `Default` or comparable, and its `Debug` output
/// hides the bytes. This crate does not generate keys, since the standard library has no portable
/// entropy source and the crate takes no entropy dependency; the runtime supplies 32 bytes of
/// entropy (or a secret replayed from a trusted store) through [`SamplingKey::from_bytes`].
pub struct SamplingKey([u8; 32]);

impl SamplingKey {
    /// A key from 32 secret bytes, for example fresh OS entropy or a value replayed from a
    /// trusted store.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The uniform draw in `[0, 2^53)` for a request: the top 53 bits of the first eight bytes
    /// (big-endian) of the keyed hash.
    pub(crate) fn draw(&self, request_id: &RequestId, request_digest: &Digest) -> u64 {
        let id = request_id.0.as_bytes();
        let mac = hmac_sha256(
            &self.0,
            &[&(id.len() as u64).to_be_bytes(), id, request_digest],
        );
        let mut head = [0u8; 8];
        head.copy_from_slice(&mac[..8]);
        u64::from_be_bytes(head) >> 11
    }
}

impl fmt::Debug for SamplingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SamplingKey(..)")
    }
}

/// HMAC-SHA256 (RFC 2104) over the concatenation of `parts`.
fn hmac_sha256(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|byte| byte ^ 0x36));
    for part in parts {
        inner.update(part);
    }
    let inner: [u8; 32] = inner.finalize().into();
    let mut outer = Sha256::new();
    outer.update(block.map(|byte| byte ^ 0x5c));
    outer.update(inner);
    outer.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn hmac_matches_rfc_4231_vectors() {
        // Test case 2.
        assert_eq!(
            hex(&hmac_sha256(
                b"Jefe",
                &[b"what do ya want ", b"for nothing?"]
            )),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 3.
        assert_eq!(
            hex(&hmac_sha256(&[0xaa; 20], &[&[0xdd; 50]])),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
        );
        // Test case 6: a key longer than the block size is hashed first.
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                &[b"Test Using Larger Than Block-Size Key - Hash Key First"]
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn debug_hides_the_key() {
        let key = SamplingKey::from_bytes([0x41; 32]);
        assert_eq!(format!("{key:?}"), "SamplingKey(..)");
        assert_eq!(format!("{key:#?}"), "SamplingKey(..)");
    }

    #[test]
    fn draw_is_below_the_resolution() {
        let key = SamplingKey::from_bytes([1; 32]);
        for n in 0..100 {
            let id = RequestId::from(format!("r{n}").as_str());
            assert!(key.draw(&id, &[0; 32]) < 1 << 53);
        }
    }
}
