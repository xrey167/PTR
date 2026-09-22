//! How a committed semantic value becomes the vector a model sees in a slot.
//!
//! **Read this first: it encodes identity, not similarity.** Two payloads that
//! mean almost the same thing get unrelated vectors, and that is not a defect of
//! the construction — it is the whole of what the construction claims. What a slot
//! vector establishes is *this payload and not another one*, deterministically,
//! everywhere. Making nearby meanings land near each other is a learned encoder,
//! which is a modelling decision and is recorded as open in
//! `docs/architecture/26-cognitive-codebook.md` rather than guessed at here.
//!
//! ## Why anything is here at all
//!
//! `PtrA0::forward` has always taken a `slot_values` tensor, and no call site in
//! the repository has ever passed one: the doctests, the unit test and the router
//! training test all pass zeros. The socket was built, wired into
//! `slots = slot_values + typed_metadata`, and left empty, and nothing could fill
//! it because nothing turned a committed value into a vector.
//!
//! It also could not be filled *safely*. Every other input to that function went
//! through the discipline `codebook` describes — a [`crate::TypeCode`] only exists
//! as the output of a versioned assignment — while `slot_values` was a bare
//! tensor: no type, no version, no provenance. A caller could have put anything
//! there. So this module gives a slot's value the treatment slot *identity*
//! already had.
//!
//! ## Why the mixer is written out here
//!
//! This crate has no external dependencies, so it has no hash to reach for. That
//! turns out to be the right constraint rather than an obstacle: the encoding has
//! to be byte-identical in the PTR workspace and in the isolated model workspace,
//! and stable for as long as any checkpoint trained under it exists. A borrowed
//! hash would make that a property of somebody else's version pin. So the mixing
//! function is specified in full below, in fixed-width wrapping arithmetic that
//! cannot differ by platform, and the tests assert **exact** output values — the
//! same discipline `codebook` applies to codes, for the same reason: a change to
//! the arithmetic must break a build rather than a checkpoint.
//!
//! Being obviously not a learned model is a feature. Nobody can mistake twenty
//! lines of integer mixing for a semantic representation.
//!
//! ## Why the output is normalised
//!
//! A slot vector is *summed* into the same space as the learned metadata
//! embeddings. Unbounded values would swamp them, and the model would be reading
//! the payload and little else; values that were too small would be a channel
//! nothing could learn from. So each vector is scaled to unit L2 norm, which puts
//! it on the same footing as a freshly initialised embedding row. The choice is
//! stated rather than tuned: nothing here claims it is the best scale, only that
//! it is a defined one.
use crate::TypeId;
use std::fmt;

/// Version of a whole slot-encoding definition.
///
/// A slot vector is only interpretable against the version that produced it —
/// weights trained on vectors from one definition are being fed a differently
/// shaped input by another — so this travels with them rather than being implied.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EncodingVersion(pub u32);

impl EncodingVersion {
    /// Initial frozen definition.
    pub const V1: Self = Self(1);
}

impl fmt::Display for EncodingVersion {
    /// Render the version in manifest-friendly form.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The widest slot vector this crate will produce.
///
/// A bound rather than a shape: a model asks for its own `d_model` and gets
/// exactly that. The bound exists so a caller cannot ask for an allocation the
/// size of its input.
pub const MAX_SLOT_WIDTH: usize = 4096;

/// The largest payload a slot vector may be taken over.
///
/// Every byte is mixed, so this is a bound on work rather than on meaning. A
/// payload past it is refused rather than truncated: a truncated payload would
/// encode as a *different* payload that happens to share a prefix, which is
/// exactly the confusion this module exists to prevent.
pub const MAX_PAYLOAD_BYTES: usize = 1 << 20;

/// Why a slot vector could not be produced.
///
/// Every variant is a refusal. None of them substitutes a vector, because a
/// defaulted slot value is a payload the model believes it has seen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EncodingError {
    /// A width of zero, or one past [`MAX_SLOT_WIDTH`].
    Width { requested: usize, limit: usize },
    /// A payload past [`MAX_PAYLOAD_BYTES`].
    PayloadTooLarge { bytes: usize, limit: usize },
}

impl EncodingError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Width { .. } => "PTR_ENCODING_WIDTH",
            Self::PayloadTooLarge { .. } => "PTR_ENCODING_PAYLOAD_TOO_LARGE",
        }
    }
}

impl fmt::Display for EncodingError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for EncodingError {}

/// One slot's value, as the vector a model reads.
///
/// Deliberately not constructible from an arbitrary `Vec<f32>`: a slot vector only
/// means anything as the output of [`SlotEncoding::encode`], which names a version.
/// That is the same rule [`crate::TypeCode`] follows, and it is here for the same
/// reason — a tensor full of numbers nobody can account for is how a model comes to
/// be confidently wrong about what it is looking at.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotVector {
    encoding: EncodingVersion,
    values: Vec<f32>,
}

impl SlotVector {
    /// The definition that produced these values.
    pub fn encoding(&self) -> EncodingVersion {
        self.encoding
    }

    /// The values, for building a tensor.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// How many values, which is the `d_model` this was encoded for.
    pub fn width(&self) -> usize {
        self.values.len()
    }
}

/// A frozen definition of how a payload becomes a slot vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotEncoding {
    /// The initial definition. See the module documentation for what it claims.
    V1,
}

impl SlotEncoding {
    /// The definition a version denotes, or nothing for a version this build has
    /// no definition for.
    ///
    /// Refused rather than approximated: running weights against an encoding this
    /// build had to guess at is the failure the version exists to prevent.
    pub fn at(version: EncodingVersion) -> Option<Self> {
        match version {
            EncodingVersion::V1 => Some(Self::V1),
            _ => None,
        }
    }

    /// This definition's version.
    pub fn version(self) -> EncodingVersion {
        match self {
            Self::V1 => EncodingVersion::V1,
        }
    }

    /// Encode one payload into a slot vector of `width` values.
    ///
    /// The type participates, so the same bytes under two different types encode
    /// differently. That is not a nicety: a `Document` and a `Summary` holding
    /// identical bytes are different claims about the world, and a model that saw
    /// one vector for both would have no way to tell them apart.
    ///
    /// ```
    /// use ptr_types::{SlotEncoding, TypeId};
    ///
    /// let book = SlotEncoding::V1;
    /// let a = book.encode(&TypeId::from("Document"), b"the same bytes", 8).unwrap();
    /// let b = book.encode(&TypeId::from("Summary"), b"the same bytes", 8).unwrap();
    /// assert_ne!(a.values(), b.values(), "the type is part of the payload");
    ///
    /// let again = book.encode(&TypeId::from("Document"), b"the same bytes", 8).unwrap();
    /// assert_eq!(a.values(), again.values(), "and the same payload encodes the same way");
    /// ```
    pub fn encode(
        self,
        type_id: &TypeId,
        bytes: &[u8],
        width: usize,
    ) -> Result<SlotVector, EncodingError> {
        if width == 0 || width > MAX_SLOT_WIDTH {
            return Err(EncodingError::Width {
                requested: width,
                limit: MAX_SLOT_WIDTH,
            });
        }
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(EncodingError::PayloadTooLarge {
                bytes: bytes.len(),
                limit: MAX_PAYLOAD_BYTES,
            });
        }
        match self {
            Self::V1 => Ok(SlotVector {
                encoding: EncodingVersion::V1,
                values: encode_v1(type_id, bytes, width),
            }),
        }
    }
}

/// Domain separation, so a slot vector's mixing state cannot coincide with any
/// other use of the same arithmetic.
const DOMAIN_V1: &[u8] = b"PTR-SLOT-ENCODING-V1";

/// FNV-1a's 64-bit offset basis and prime, written out rather than named, because
/// this file is the specification of the encoding and a reader should not have to
/// look them up to reproduce it.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The V1 definition.
///
/// Three steps, each stated so the whole function can be reimplemented from this
/// comment alone:
///
/// 1. Absorb, into one 64-bit FNV-1a state: the domain string, the type's length as
///    eight little-endian bytes, the type's bytes, then the payload. The type's
///    length is what keeps `("ab", "c")` and `("a", "bc")` apart; the payload needs
///    no length because nothing follows it.
/// 2. Expand that state into `width` values by mixing it with the slot index
///    through splitmix64, so every position is a different function of the same
///    payload rather than a repetition of it.
/// 3. Map each 64-bit word into `[-1, 1]` and scale the whole vector to unit L2
///    norm.
///
/// All arithmetic is explicitly wrapping and fixed-width, so the result does not
/// depend on the platform or on the optimiser.
fn encode_v1(type_id: &TypeId, bytes: &[u8], width: usize) -> Vec<f32> {
    let mut state = FNV_OFFSET;
    let mut absorb = |value: u8| {
        state ^= u64::from(value);
        state = state.wrapping_mul(FNV_PRIME);
    };
    for byte in DOMAIN_V1 {
        absorb(*byte);
    }
    // The type's length before its bytes, so the boundary between the type and the
    // payload is unambiguous: without it `("ab", "c")` and `("a", "bc")` absorb the
    // same byte sequence and collide. The payload needs no length of its own,
    // because it is last and nothing follows it to be confused with.
    for byte in (type_id.0.len() as u64).to_le_bytes() {
        absorb(byte);
    }
    for byte in type_id.0.as_bytes() {
        absorb(*byte);
    }
    for byte in bytes {
        absorb(*byte);
    }

    let mut values: Vec<f32> = Vec::with_capacity(width);
    for index in 0..width {
        let word = splitmix64(state ^ splitmix64(index as u64));
        // The top 24 bits, divided by 2^23, give a value in [0, 2); subtracting one
        // centres it on [-1, 1). Twenty-four bits is deliberate: an f32 carries 24
        // bits of mantissa, so every distinct word maps to a distinct float and no
        // resolution is quietly discarded.
        const SPAN: f32 = (1_u32 << 23) as f32;
        values.push((word >> 40) as f32 / SPAN - 1.0);
    }
    normalise(&mut values);
    values
}

/// Scale a vector to unit L2 norm, leaving an all-zero vector alone.
///
/// An all-zero vector cannot arise from [`encode_v1`] — every value is a distinct
/// function of the payload — but a definition that divided by a norm without
/// saying what it does at zero would be a definition with a hole in it.
fn normalise(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return;
    }
    for value in values.iter_mut() {
        *value /= norm;
    }
}

/// splitmix64, written out for the same reason the FNV constants are.
fn splitmix64(input: u64) -> u64 {
    let mut z = input.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(bytes: &[u8], width: usize) -> SlotVector {
        SlotEncoding::V1
            .encode(&TypeId::from("Document"), bytes, width)
            .expect("a small payload at a small width")
    }

    #[test]
    fn the_v1_definition_produces_exactly_these_values() {
        // Pinned, like the codebook's codes, and for the same reason: weights
        // trained against these numbers are only interpretable against them. A
        // change to the mixing arithmetic must break this test rather than a
        // checkpoint somebody loads a year from now.
        //
        // If this test ever needs updating, the encoding version needs incrementing
        // in the same commit.
        let vector = document(b"a committed value", 4);
        assert_eq!(vector.encoding(), EncodingVersion::V1);
        let expected = [0.6904566_f32, 0.6022163, -0.3978477, -0.04819055];
        assert_eq!(
            vector.values(),
            expected,
            "the V1 definition changed; increment the version"
        );
    }

    #[test]
    fn the_same_payload_always_encodes_the_same_way() {
        assert_eq!(document(b"x", 16).values(), document(b"x", 16).values());
        // And across widths the prefix is not assumed to be stable — each position
        // is its own function of the payload, which is stated rather than relied on.
        assert_eq!(document(b"x", 4).values().len(), 4);
        assert_eq!(document(b"x", 64).values().len(), 64);
    }

    #[test]
    fn different_payloads_encode_differently() {
        assert_ne!(
            document(b"one claim", 32).values(),
            document(b"another claim", 32).values()
        );
        // One byte is enough. A construction that only separated wildly different
        // payloads would let two near-identical claims share a slot vector.
        assert_ne!(
            document(b"claim", 32).values(),
            document(b"claiM", 32).values()
        );
        // Including at the very end, where a weak mixer would lose it.
        assert_ne!(
            document(b"claim0", 32).values(),
            document(b"claim1", 32).values()
        );
        // And an empty payload is a payload, distinct from a one-byte one.
        assert_ne!(document(b"", 32).values(), document(b"\x00", 32).values());
    }

    #[test]
    fn the_type_is_part_of_the_payload() {
        let book = SlotEncoding::V1;
        let bytes = b"identical bytes";
        let as_document = book.encode(&TypeId::from("Document"), bytes, 32).unwrap();
        let as_summary = book.encode(&TypeId::from("Summary"), bytes, 32).unwrap();
        assert_ne!(as_document.values(), as_summary.values());

        // The type's length is absorbed before its bytes, so a pair that
        // concatenates to the same string does not collide. Without the length this
        // assertion fails: ("ab", "c") and ("a", "bc") absorb the same byte sequence.
        let left = book.encode(&TypeId::from("ab"), b"c", 32).unwrap();
        let right = book.encode(&TypeId::from("a"), b"bc", 32).unwrap();
        assert_ne!(left.values(), right.values());
    }

    #[test]
    fn every_vector_is_bounded_and_of_unit_norm() {
        // Both properties matter for one reason: a slot vector is summed into the
        // same space as the learned metadata embeddings. Unbounded values would
        // swamp them.
        for payload in [
            b"".as_slice(),
            b"a",
            b"a longer committed value",
            &[0xff; 300],
        ] {
            for width in [1, 2, 7, 64] {
                let vector = document(payload, width);
                let values = vector.values();
                assert_eq!(values.len(), width);
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "every value is finite"
                );
                assert!(
                    values.iter().all(|value| value.abs() <= 1.0),
                    "unit norm bounds every component: {values:?}"
                );
                let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "width {width} gave norm {norm}, not 1"
                );
            }
        }
    }

    #[test]
    fn a_width_of_zero_or_past_the_bound_is_refused_rather_than_clamped() {
        let book = SlotEncoding::V1;
        let id = TypeId::from("Document");
        assert_eq!(
            book.encode(&id, b"x", 0)
                .expect_err("a slot with no values"),
            EncodingError::Width {
                requested: 0,
                limit: MAX_SLOT_WIDTH
            }
        );
        assert_eq!(
            book.encode(&id, b"x", MAX_SLOT_WIDTH + 1)
                .expect_err("past the bound"),
            EncodingError::Width {
                requested: MAX_SLOT_WIDTH + 1,
                limit: MAX_SLOT_WIDTH
            }
        );
        // The control: the bound itself is allowed, so the refusal is about being
        // past it rather than about the bound being unreachable.
        assert_eq!(
            book.encode(&id, b"x", MAX_SLOT_WIDTH).unwrap().width(),
            MAX_SLOT_WIDTH
        );
    }

    #[test]
    fn a_payload_past_the_bound_is_refused_rather_than_truncated() {
        // Truncating would encode a *different* payload that happens to share a
        // prefix, which is exactly the confusion this module exists to prevent.
        let book = SlotEncoding::V1;
        let id = TypeId::from("Document");
        let huge = vec![0_u8; MAX_PAYLOAD_BYTES + 1];
        assert_eq!(
            book.encode(&id, &huge, 8).expect_err("past the bound"),
            EncodingError::PayloadTooLarge {
                bytes: MAX_PAYLOAD_BYTES + 1,
                limit: MAX_PAYLOAD_BYTES
            }
        );
        assert!(book.encode(&id, &huge[..MAX_PAYLOAD_BYTES], 8).is_ok());
    }

    #[test]
    fn a_version_this_build_has_no_definition_for_is_refused() {
        assert_eq!(
            SlotEncoding::at(EncodingVersion::V1),
            Some(SlotEncoding::V1)
        );
        assert_eq!(SlotEncoding::V1.version(), EncodingVersion::V1);
        assert_eq!(SlotEncoding::at(EncodingVersion(0)), None);
        assert_eq!(
            SlotEncoding::at(EncodingVersion(2)),
            None,
            "a later definition"
        );
    }

    #[test]
    fn every_position_is_a_different_function_of_the_payload() {
        // Without mixing the slot index into the expansion, every position of a
        // vector would be the same word: a width-64 vector carrying one word's worth
        // of information, and a model with sixty-four copies of one feature.
        let vector = document(b"a committed value", 64);
        let first = vector.values()[0];
        assert!(
            vector.values().iter().any(|value| *value != first),
            "the expansion does not vary by position"
        );
        // Stronger: no two positions coincide, for this payload.
        let mut seen: Vec<f32> = vector.values().to_vec();
        seen.sort_by(f32::total_cmp);
        let total = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), total, "two positions produced the same value");
    }

    /// The V1 definition with the domain string as a parameter, so a test can
    /// produce the un-domained variant without editing the real function.
    ///
    /// A reimplementation, deliberately: what is under test is that dropping one
    /// step of the specification changes the output. Passing the real domain must
    /// reproduce `encode_v1` exactly, which the assertion below checks first — so
    /// this helper cannot drift into testing itself.
    fn with_domain(domain: &[u8], type_id: &TypeId, bytes: &[u8], width: usize) -> Vec<f32> {
        let mut state = FNV_OFFSET;
        let mut absorb = |value: u8| {
            state ^= u64::from(value);
            state = state.wrapping_mul(FNV_PRIME);
        };
        for byte in domain {
            absorb(*byte);
        }
        for byte in (type_id.0.len() as u64).to_le_bytes() {
            absorb(byte);
        }
        for byte in type_id.0.as_bytes() {
            absorb(*byte);
        }
        for byte in bytes {
            absorb(*byte);
        }
        const SPAN: f32 = (1_u32 << 23) as f32;
        let mut values: Vec<f32> = (0..width)
            .map(|index| {
                let word = splitmix64(state ^ splitmix64(index as u64));
                (word >> 40) as f32 / SPAN - 1.0
            })
            .collect();
        normalise(&mut values);
        values
    }

    #[test]
    fn the_domain_separates_this_encoding_from_the_undomained_arithmetic() {
        // The domain string is absorbed first so a slot vector's mixing state cannot
        // coincide with any other use of the same arithmetic over the same material.
        let type_id = TypeId::from("Document");
        let payload = b"a committed value";

        // First: the helper with the real domain reproduces the real function, so
        // what follows is about the domain and not about the reimplementation.
        assert_eq!(
            with_domain(DOMAIN_V1, &type_id, payload, 4),
            document(payload, 4).values(),
            "the helper does not reproduce encode_v1"
        );

        // Then: without it, the output differs. Width four, so normalisation does not
        // collapse the comparison to a sign the way width one would — which is how a
        // first version of this test managed to compare nothing at all.
        assert_ne!(
            with_domain(b"", &type_id, payload, 4),
            document(payload, 4).values(),
            "dropping the domain does not change the encoding"
        );
    }

    #[test]
    fn each_refusal_carries_its_own_diagnostic_code() {
        let errors = [
            EncodingError::Width {
                requested: 0,
                limit: 1,
            },
            EncodingError::PayloadTooLarge { bytes: 1, limit: 0 },
        ];
        let mut codes: Vec<&str> = errors.iter().map(EncodingError::code).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "two refusals share a diagnostic code");
    }
}
