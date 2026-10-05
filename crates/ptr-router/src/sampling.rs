//! The secret sampling key and the keyed draw used by the trusted routing path.

use ptr_types::{Digest, RequestId};
use sha2::{Digest as ShaDigest, Sha256};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

const BLOCK: usize = 64;

/// A per-decision value mixed into the sampling draw so that routing the same request twice
/// yields independent draws.
///
/// The draw is `HMAC-SHA256(key, nonce || len(request_id) || request_id || request_digest)`. A
/// request may legitimately be routed again (a retry or a supported replay); if the draw
/// depended only on the key and the request, a stateful policy that saw the earlier record
/// (which contains [`crate::RoutedChoice::draw`]) could place its larger weight around the
/// known draw, force the action and still log the smaller-looking propensity. Every fresh
/// decision therefore uses a nonce that was never used before, and the nonce is recorded in the
/// [`crate::RoutedChoice`].
///
/// The nonce is not part of a [`crate::RouteRequest`], so a policy cannot learn it before it
/// chooses its distribution, and it does not need to be secret: without the [`SamplingKey`] a
/// known nonce and a known past draw say nothing about the next draw. Replay is
/// `(key, nonce, request)`: [`crate::route_checked`] with the recorded nonce reproduces the
/// recorded draw exactly. The trusted caller must never pass the same nonce for two fresh
/// decisions; [`NonceSequence`] (used by [`crate::CheckedRouter`]) guarantees that.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecisionNonce([u8; 16]);

impl DecisionNonce {
    /// A nonce from its 16 bytes, for example one recorded in a routing log for replay.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The 16 bytes of the nonce, for recording next to the draw and propensity.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Returned by [`NonceSequence::next_nonce`] when the counter cannot advance without wrapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonceExhausted;

impl fmt::Display for NonceExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("decision nonce sequence is exhausted")
    }
}

impl std::error::Error for NonceExhausted {}

/// A strictly increasing source of [`DecisionNonce`] values: an 8-byte instance id followed by
/// an 8-byte big-endian counter.
///
/// The runtime picks the instance id (for example fresh entropy per process or router) and the
/// first counter value; after a restart it must either use a new instance id or continue from a
/// counter above every value it ever handed out, since reusing a nonce would make the draw of
/// the new decision predictable from the old record. The counter never wraps: when it is used
/// up, [`NonceSequence::next_nonce`] fails with [`NonceExhausted`]. It is safe to share between
/// threads.
#[derive(Debug)]
pub struct NonceSequence {
    instance: [u8; 8],
    next: AtomicU64,
}

impl NonceSequence {
    /// A sequence for `instance` whose first nonce carries counter value `first_counter`.
    pub fn starting_at(instance: [u8; 8], first_counter: u64) -> Self {
        Self {
            instance,
            next: AtomicU64::new(first_counter),
        }
    }

    /// The nonce with the next counter value; every call returns a nonce never returned before.
    pub fn next_nonce(&self) -> Result<DecisionNonce, NonceExhausted> {
        let counter = self
            .next
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                value.checked_add(1)
            })
            .map_err(|_| NonceExhausted)?;
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.instance);
        bytes[8..].copy_from_slice(&counter.to_be_bytes());
        Ok(DecisionNonce(bytes))
    }
}

/// Secret key of the trusted routing path that determines every sampling draw.
///
/// The draw for a decision is
/// `HMAC-SHA256(key, nonce || len(request_id) || request_id || request_digest)` reduced to its
/// top 53 bits, so nobody without the key (in particular a [`crate::Router`] policy, which sees
/// the request id and digest) can predict the draw and correlate its weights with it, and a draw
/// revealed for one decision does not predict the draw of a later decision of the same request
/// (see [`DecisionNonce`]). Replaying a routing decision requires the same key and nonce: the
/// runtime keeps the key in trusted configuration (for example next to the ledger) and passes it
/// to [`crate::route_checked`] or [`crate::CheckedRouter::new`]. It is never part of a
/// [`crate::RouteRequest`].
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

    /// A non-secret identifier of this key: the first 8 bytes of
    /// `SHA-256("ptr-router/sampling-key-id/v1" || key)`. The runtime records it next to every
    /// draw and propensity so an audit can tell which key a replay needs, without the key
    /// itself ever leaving this crate. It is a one-way hash of a 256-bit secret, so it does not
    /// help to recover the key or predict a draw.
    pub fn key_id(&self) -> [u8; 8] {
        let mut hasher = Sha256::new();
        hasher.update(b"ptr-router/sampling-key-id/v1");
        hasher.update(self.0);
        let digest = hasher.finalize();
        let mut id = [0u8; 8];
        id.copy_from_slice(&digest[..8]);
        id
    }

    /// The uniform draw in `[0, 2^53)` for a decision: the top 53 bits of the first eight bytes
    /// (big-endian) of the keyed hash.
    pub(crate) fn draw(
        &self,
        nonce: &DecisionNonce,
        request_id: &RequestId,
        request_digest: &Digest,
    ) -> u64 {
        let id = request_id.0.as_bytes();
        let mac = hmac_sha256(
            &self.0,
            &[
                nonce.as_bytes(),
                &(id.len() as u64).to_be_bytes(),
                id,
                request_digest,
            ],
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
    fn key_id_is_stable_distinct_and_not_a_key_prefix() {
        let a = SamplingKey::from_bytes([7; 32]);
        let b = SamplingKey::from_bytes([8; 32]);
        assert_eq!(a.key_id(), SamplingKey::from_bytes([7; 32]).key_id());
        assert_ne!(a.key_id(), b.key_id());
        assert_ne!(a.key_id(), [7u8; 8]);
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
            assert!(key.draw(&DecisionNonce::from_bytes([n as u8; 16]), &id, &[0; 32]) < 1 << 53);
        }
    }

    #[test]
    fn nonce_sequence_is_strictly_increasing_and_never_wraps() {
        let sequence = NonceSequence::starting_at([7; 8], u64::MAX - 2);
        let first = sequence.next_nonce().unwrap();
        let second = sequence.next_nonce().unwrap();
        assert_eq!(&first.as_bytes()[..8], &[7; 8]);
        assert_eq!(&first.as_bytes()[8..], &(u64::MAX - 2).to_be_bytes());
        assert_eq!(&second.as_bytes()[8..], &(u64::MAX - 1).to_be_bytes());
        assert!(first.as_bytes() < second.as_bytes());
        assert_eq!(sequence.next_nonce(), Err(NonceExhausted));
        assert_eq!(sequence.next_nonce(), Err(NonceExhausted));
    }

    #[test]
    fn nonce_changes_the_draw() {
        let key = SamplingKey::from_bytes([1; 32]);
        let id = RequestId::from("r");
        let a = key.draw(&DecisionNonce::from_bytes([0; 16]), &id, &[0; 32]);
        let b = key.draw(&DecisionNonce::from_bytes([1; 16]), &id, &[0; 32]);
        assert_ne!(a, b);
    }
}
