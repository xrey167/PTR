//! The harness-supplied derivation of S001: `ptr-semdb` has no query engine, so what a derived key
//! "recomputes to" is defined here. A derived key has a [`Kind`]; its value is a pure function of
//! the values of its inputs, taken in ascending key order. Both arms use this function (arm A
//! only for the keys SemDB marks affected, arm G for every derived key); the independent oracle
//! (`oracle.rs`) carries its own implementation of the same three functions, and the harness
//! fails with status 2 if the two ever disagree.
//!
//! Values are `u64` printed in decimal as `SemanticValue::Text`.

/// How a derived key combines the values of its inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Lossless in practice: any change of any input changes the result (a salted hash chain).
    Mix(u64),
    /// Lossy: the minimum of the inputs; a change of a non-minimal input is absorbed.
    Min,
    /// Lossy: the parity of the sum of the inputs; half of the single-input changes are absorbed.
    Parity,
}

impl Kind {
    /// Whether a change of an input can leave the result unchanged (the lossy class).
    pub fn is_lossy(self) -> bool {
        matches!(self, Self::Min | Self::Parity)
    }
}

/// The value of a derived key of `kind` from its inputs' values, in ascending input-key order.
/// `values` must not be empty.
pub fn eval(kind: Kind, values: &[u64]) -> u64 {
    match kind {
        Kind::Mix(salt) => {
            let mut hash = salt;
            for value in values {
                hash = (hash.rotate_left(23) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            }
            hash ^ (hash >> 32)
        }
        Kind::Min => values.iter().copied().min().unwrap_or(0),
        Kind::Parity => {
            values
                .iter()
                .fold(0u64, |sum, value| sum.wrapping_add(*value))
                & 1
        }
    }
}

/// The text a value is stored as.
pub fn text(value: u64) -> String {
    value.to_string()
}

/// The value a stored text denotes, or `None` for anything that is not a decimal `u64`.
pub fn parse(stored: &str) -> Option<u64> {
    stored.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kinds_have_the_documented_loss_behaviour() {
        assert_eq!(eval(Kind::Min, &[5, 3, 9]), 3);
        assert_eq!(
            eval(Kind::Min, &[5, 3, 90]),
            3,
            "a non-minimal change is absorbed"
        );
        assert_eq!(eval(Kind::Parity, &[1, 2, 4]), 1);
        assert_eq!(
            eval(Kind::Parity, &[1, 2, 6]),
            1,
            "an even change is absorbed"
        );
        assert_ne!(
            eval(Kind::Mix(1), &[1, 2, 4]),
            eval(Kind::Mix(1), &[1, 2, 6])
        );
        assert_ne!(eval(Kind::Mix(1), &[1, 2]), eval(Kind::Mix(2), &[1, 2]));
        assert!(Kind::Min.is_lossy() && Kind::Parity.is_lossy() && !Kind::Mix(0).is_lossy());
    }

    #[test]
    fn values_round_trip_through_their_text_and_nothing_else_parses() {
        for value in [0, 1, 17, u64::MAX] {
            assert_eq!(parse(&text(value)), Some(value));
        }
        for bad in ["", "-1", "1.5", "x", " 1"] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
    }
}
