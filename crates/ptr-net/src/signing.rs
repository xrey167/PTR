//! Statements a node signs with its endpoint key, and their verification.
//!
//! An iroh endpoint's id **is** an Ed25519 public key, and the QUIC handshake proves
//! possession of its secret half. That makes the key a node already has the only one
//! it needs to sign with: a signature from it is bound to the very identity every
//! authority table is keyed by, and there is no second key to issue, rotate, revoke
//! or keep in step with the first. That is the decision recorded as D2 in issue #23.
//! A rotated key is a new identity, by construction rather than by policy.
//!
//! **What this module exposes is deliberately narrow.** [`NodeSigner`] has no public
//! constructor outside tests, no accessor for the secret, and signs only under a
//! [`SigningDomain`]: the message a key actually signs is always
//! `domain ‖ message`, never bytes a caller chose in full. A key that signs anything
//! it is handed is an oracle for every protocol that trusts it — the TLS handshake
//! that authenticates this same key included — and a domain that is required by the
//! type rather than by convention cannot be forgotten at one call site.
//!
//! ```compile_fail
//! // A signer cannot be built from a key a caller holds: the field is private and
//! // there is no public constructor.
//! let signer = ptr_net::NodeSigner { key: todo!() };
//! ```
//!
//! **Verification needs no transport.** [`verify`] takes the textual key every frame
//! names its author by and is available without the `iroh-backend` feature, so a
//! stored receipt can be checked by a process that has never opened a socket.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use std::fmt;

/// Bytes in a signature.
pub const SIGNATURE_BYTES: usize = 64;

/// Bytes in a public key, before its textual encoding.
const PUBLIC_KEY_BYTES: usize = 32;

/// Longest domain tag this module accepts.
const MAX_DOMAIN_BYTES: usize = 64;

/// The context a signature is made in.
///
/// Every tag starts with `ptr-` and ends with its only NUL byte. The terminator
/// makes the set of tags prefix-free, so no `domain ‖ message` under one tag is also
/// a `domain ‖ message` under another: a receipt can never be re-read as a Pod
/// answer, or as anything a later protocol signs. The `ptr-` prefix keeps every
/// signed message distinct from the TLS 1.3 handshake transcript the same key
/// signs, which begins with sixty-four spaces.
///
/// Checked when the constant is built, so an invalid tag is a build failure rather
/// than a runtime one:
///
/// ```compile_fail
/// // No terminator: the tag would be a prefix of a longer one.
/// const BAD: ptr_net::SigningDomain = ptr_net::SigningDomain::new(b"ptr-receipt");
/// ```
///
/// ```
/// // The control: a well-formed tag builds.
/// const GOOD: ptr_net::SigningDomain = ptr_net::SigningDomain::new(b"ptr-receipt/v1\0");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SigningDomain(&'static [u8]);

impl SigningDomain {
    /// A domain tag, or a panic — at compile time when used in a `const`.
    pub const fn new(tag: &'static [u8]) -> Self {
        assert!(
            tag.len() > 5 && tag.len() <= MAX_DOMAIN_BYTES,
            "a signing domain is between six and sixty-four bytes"
        );
        assert!(
            tag[0] == b'p' && tag[1] == b't' && tag[2] == b'r' && tag[3] == b'-',
            "a signing domain starts with `ptr-`"
        );
        assert!(
            tag[tag.len() - 1] == 0,
            "a signing domain ends with a NUL byte"
        );
        let mut index = 0;
        while index < tag.len() - 1 {
            assert!(tag[index] != 0, "a signing domain has exactly one NUL byte");
            index += 1;
        }
        Self(tag)
    }

    /// The tag, terminator included.
    pub fn as_bytes(self) -> &'static [u8] {
        self.0
    }
}

/// The message a key signs under `domain`: the tag, then the caller's bytes.
fn domain_separated(domain: SigningDomain, message: &[u8]) -> Vec<u8> {
    let mut signed = Vec::with_capacity(domain.0.len() + message.len());
    signed.extend_from_slice(domain.0);
    signed.extend_from_slice(message);
    signed
}

/// A node's endpoint key, able to sign and unable to be read.
///
/// Built by the transport from the key its endpoint was bound with
/// (`IrohTransport::signer`), so what it signs is attributable to exactly the
/// identity a connection to that endpoint authenticates. Not `Clone`: a holder asks
/// the transport for another rather than copying key material around a process.
pub struct NodeSigner {
    key: SigningKey,
    public_key: String,
}

impl NodeSigner {
    /// Built only where the transport holds the secret. Crate-private on purpose.
    #[cfg_attr(not(feature = "iroh-backend"), allow(dead_code))]
    pub(crate) fn from_secret(secret: &[u8; 32]) -> Self {
        let key = SigningKey::from_bytes(secret);
        let public_key = encode_public_key(key.verifying_key().as_bytes());
        Self { key, public_key }
    }

    /// A signer from a fixed seed, for tests that have no network.
    ///
    /// Only with the `test-support` feature, which `scripts/check_repo.py` allows in
    /// `[dev-dependencies]` alone. A test signer forges nothing: a signature is
    /// checked against the key a frame must be *expected* to come from, and a test
    /// key is nobody's identity.
    #[cfg(feature = "test-support")]
    pub fn from_test_seed(seed: [u8; 32]) -> Self {
        Self::from_secret(&seed)
    }

    /// The public key, in the textual form the transport renders an endpoint id in
    /// (64 lowercase hex digits), so it compares equal to an authenticated peer's.
    pub fn public_key(&self) -> &str {
        &self.public_key
    }

    /// Sign `domain ‖ message`.
    pub fn sign(&self, domain: SigningDomain, message: &[u8]) -> [u8; SIGNATURE_BYTES] {
        self.key.sign(&domain_separated(domain, message)).to_bytes()
    }
}

impl fmt::Debug for NodeSigner {
    /// Names the public key and nothing else.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeSigner")
            .field("public_key", &self.public_key)
            .finish_non_exhaustive()
    }
}

/// Whether `signature` is `public_key`'s signature over `domain ‖ message`.
///
/// `public_key` must be in exactly the textual form the transport renders an
/// endpoint id in: 64 **lowercase** hex digits. iroh's own parser also accepts
/// base32, but a key with two spellings is two strings for one identity, and every
/// comparison in this system is a string comparison — so only the canonical one
/// verifies. The check is `verify_strict`, which refuses the small-order keys and
/// non-canonical signatures plain Ed25519 verification lets through.
pub fn verify(
    public_key: &str,
    domain: SigningDomain,
    message: &[u8],
    signature: &[u8; SIGNATURE_BYTES],
) -> bool {
    let Some(bytes) = decode_public_key(public_key) else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(&bytes) else {
        return false;
    };
    key.verify_strict(
        &domain_separated(domain, message),
        &Signature::from_bytes(signature),
    )
    .is_ok()
}

/// Render a key as 64 lowercase hex digits.
fn encode_public_key(bytes: &[u8; PUBLIC_KEY_BYTES]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(PUBLIC_KEY_BYTES * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    text
}

/// Read 64 lowercase hex digits, or nothing.
fn decode_public_key(text: &str) -> Option<[u8; PUBLIC_KEY_BYTES]> {
    let digits = text.as_bytes();
    if digits.len() != PUBLIC_KEY_BYTES * 2 {
        return None;
    }
    let (pairs, rest) = digits.as_chunks::<2>();
    if !rest.is_empty() {
        return None;
    }
    let mut bytes = [0_u8; PUBLIC_KEY_BYTES];
    for (byte, [high, low]) in bytes.iter_mut().zip(pairs) {
        *byte = (nibble(*high)? << 4) | nibble(*low)?;
    }
    Some(bytes)
}

/// One lowercase hex digit.
fn nibble(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: SigningDomain = SigningDomain::new(b"ptr-test-one/v1\0");
    const TWO: SigningDomain = SigningDomain::new(b"ptr-test-two/v1\0");

    fn signer(seed: u8) -> NodeSigner {
        NodeSigner::from_secret(&[seed; 32])
    }

    #[test]
    fn a_signature_verifies_under_its_key_and_domain_and_under_nothing_else() {
        let alice = signer(1);
        let bob = signer(2);
        let signature = alice.sign(ONE, b"statement");

        // The control: the untouched statement verifies.
        assert!(verify(alice.public_key(), ONE, b"statement", &signature));

        assert!(
            !verify(bob.public_key(), ONE, b"statement", &signature),
            "another key"
        );
        assert!(
            !verify(alice.public_key(), TWO, b"statement", &signature),
            "another domain: a signature made for one protocol is not one for another"
        );
        assert!(
            !verify(alice.public_key(), ONE, b"statemenT", &signature),
            "another message"
        );
        for index in 0..SIGNATURE_BYTES {
            let mut damaged = signature;
            damaged[index] ^= 1;
            assert!(
                !verify(alice.public_key(), ONE, b"statement", &damaged),
                "signature byte {index}"
            );
        }
    }

    #[test]
    fn only_the_canonical_spelling_of_a_key_verifies() {
        let alice = signer(3);
        let signature = alice.sign(ONE, b"statement");
        let canonical = alice.public_key().to_owned();
        assert_eq!(canonical.len(), 64);
        assert!(verify(&canonical, ONE, b"statement", &signature));

        // The same key in upper case is the same bytes and a different string. Every
        // comparison in this system is a string comparison, so it does not verify.
        assert!(!verify(
            &canonical.to_uppercase(),
            ONE,
            b"statement",
            &signature
        ));
        assert!(!verify(&canonical[..62], ONE, b"statement", &signature));
        assert!(!verify("", ONE, b"statement", &signature));
        assert!(!verify(
            &format!("{canonical}00"),
            ONE,
            b"statement",
            &signature
        ));
    }

    #[test]
    fn the_textual_key_round_trips_through_the_codec() {
        let bytes: [u8; 32] = core::array::from_fn(|index| u8::try_from(index * 7).unwrap());
        assert_eq!(
            decode_public_key(&encode_public_key(&bytes)),
            Some(bytes),
            "every byte value position survives"
        );
        assert_eq!(decode_public_key(&"g".repeat(64)), None);
    }

    #[test]
    fn a_debug_rendering_names_the_public_key_and_not_the_secret() {
        let alice = signer(4);
        let rendered = format!("{alice:?}");
        assert!(rendered.contains(alice.public_key()));
        assert!(!rendered.contains(&encode_public_key(&[4; 32])));
    }

    #[test]
    #[should_panic(expected = "ends with a NUL byte")]
    fn a_domain_without_its_terminator_is_refused() {
        let _ = SigningDomain::new(b"ptr-unterminated");
    }

    #[test]
    #[should_panic(expected = "exactly one NUL byte")]
    fn a_domain_with_an_inner_terminator_is_refused() {
        let _ = SigningDomain::new(b"ptr-a\0b\0");
    }

    #[test]
    #[should_panic(expected = "starts with `ptr-`")]
    fn a_domain_outside_this_system_s_prefix_is_refused() {
        let _ = SigningDomain::new(b"tls-handshake\0");
    }
}
