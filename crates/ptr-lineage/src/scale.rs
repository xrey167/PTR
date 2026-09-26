//! Arithmetic on finite values whose intermediate results would leave the
//! range of `f64` although the answer does not.
//!
//! [`Wide`] carries an `f64` significand with an exponent of its own, and
//! rounds every sum, product and square root exactly as `f64` rounds it:
//! scaling by a power of two is exact, so a computation carried out on
//! `Wide` values is the direct computation with an unbounded exponent range.
//! It equals the direct one wherever no intermediate result of that
//! overflows or underflows, and nothing is lost for being small beside a
//! value it is never combined with: `1e-30` read through a weight of one
//! keeps its effect beside an input of `1e300` that nothing reads, and
//! `1e-20` still decides `MAX + MAX - MAX - MAX + 1e-20`.
//!
//! [`exponent`] and [`power_of_two`] give the power of two at or below the
//! largest magnitude of a set of values; dividing the whole set by it keeps
//! a span, a sum of squares or a ratio of norms of that one set in range.
//! The division is exact except for entries more than `2^1022` times smaller
//! than the largest, so it only suits results that such entries cannot
//! move: a sum of squares, or a basis taken with a rank tolerance relative
//! to the largest column. A signed sum, whose large terms may cancel, and a
//! product, which may read only the small entries, are computed on [`Wide`]
//! values instead.

/// The exponent `e` of the power of two at or below the largest magnitude in
/// `values` (`2^e <= max |v| < 2^(e + 1)`), raised to `-1022` for a subnormal
/// magnitude so that `2^e` is a normal number, or `None` when every value is
/// zero. The values must be finite.
pub(crate) fn exponent(values: impl IntoIterator<Item = f64>) -> Option<i32> {
    let largest = values
        .into_iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    (largest > 0.0).then(|| {
        // The biased exponent field of a finite value is below 0x7ff.
        let biased = ((largest.to_bits() >> 52) & 0x7ff) as i32;
        biased.max(1) - 1023
    })
}

/// `2^exponent` for an exponent in `[-1022, 1023]`, the normal range.
pub(crate) fn power_of_two(exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// `value * 2^exponent` for any `exponent`, applied in steps that are each a
/// normal power of two. Scaling up, every step is at most the result, so it
/// overflows only when the exact result does.
pub(crate) fn times_power_of_two(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        value *= power_of_two(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= power_of_two(-1022);
        exponent += 1022;
    }
    value * power_of_two(exponent)
}

/// A finite value `significand * 2^exponent`, the significand zero or of
/// magnitude in `[1, 2)`, whose exponent `f64` does not bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Wide {
    significand: f64,
    exponent: i32,
}

impl Wide {
    pub(crate) const ZERO: Self = Self {
        significand: 0.0,
        exponent: 0,
    };

    /// `value`, exactly. The value must be finite.
    pub(crate) fn new(value: f64) -> Self {
        Self::normalised(value, 0)
    }

    /// `significand * 2^exponent` with the significand brought into
    /// `[1, 2)`, exactly: only powers of two are applied to it.
    fn normalised(significand: f64, exponent: i32) -> Self {
        if significand == 0.0 {
            return Self::ZERO;
        }
        let shift = exact_exponent(significand);
        Self {
            significand: times_power_of_two(significand, -shift),
            exponent: exponent + shift,
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        self.significand == 0.0
    }

    /// `1.0` or `-1.0` for a nonzero value, `0.0` for zero.
    pub(crate) fn signum(self) -> f64 {
        if self.is_zero() {
            0.0
        } else {
            self.significand.signum()
        }
    }

    /// The nearest `f64`: infinite beyond `f64::MAX`, subnormal or zero below
    /// the normal range, and otherwise exact.
    pub(crate) fn to_f64(self) -> f64 {
        times_power_of_two(self.significand, self.exponent)
    }

    /// `self / other` as the nearest `f64`, rounded once for the significands
    /// and once more only when the quotient leaves the normal range.
    ///
    /// # Panics
    /// Panics in debug builds if `other` is zero.
    pub(crate) fn ratio(self, other: Self) -> f64 {
        debug_assert!(!other.is_zero(), "a ratio to zero is undefined");
        times_power_of_two(
            self.significand / other.significand,
            self.exponent - other.exponent,
        )
    }

    /// `self / count`, rounded as `f64` division rounds.
    pub(crate) fn divided_by(self, count: usize) -> Self {
        Self::normalised(self.significand / count as f64, self.exponent)
    }

    /// The square root of a value that is not negative, rounded as
    /// `f64::sqrt` rounds.
    pub(crate) fn sqrt(self) -> Self {
        debug_assert!(self.significand >= 0.0, "the root of a negative value");
        let half = self.exponent.div_euclid(2);
        let odd = self.exponent.rem_euclid(2);
        Self::normalised((self.significand * f64::from(1 + odd)).sqrt(), half)
    }
}

impl std::ops::Mul for Wide {
    type Output = Self;

    /// Rounded as the `f64` product of the two values rounds, the
    /// significands' product being below four.
    fn mul(self, other: Self) -> Self {
        Self::normalised(
            self.significand * other.significand,
            self.exponent + other.exponent,
        )
    }
}

impl std::ops::Add for Wide {
    type Output = Self;

    /// Rounded as the `f64` sum of the two values rounds. The smaller is
    /// aligned to the larger's exponent, which is exact for a gap of up to
    /// 1000; a value more than `2^999` times smaller than the other is below
    /// half a unit in its last place and leaves it unchanged, as `f64`
    /// addition does.
    fn add(self, other: Self) -> Self {
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let (large, small) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let gap = large.exponent - small.exponent;
        if gap > 1000 {
            return large;
        }
        Self::normalised(
            large.significand + small.significand * power_of_two(-gap),
            large.exponent,
        )
    }
}

/// `floor(log2 |value|)` for a finite nonzero value, subnormals included.
pub(crate) fn exact_exponent(value: f64) -> i32 {
    let biased = ((value.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scaling up by 2^64 is exact and makes it normal.
        exact_exponent(value * power_of_two(64)) - 64
    } else {
        biased - 1023
    }
}

/// The in-order sum of finite `values`, rounded step by step as `f64`
/// addition rounds but with an unbounded exponent range. When no partial sum
/// overflows this is the direct sum: `f64` addition loses nothing to
/// underflow, and a running sum that has overflowed stays infinite (or turns
/// NaN), so a finite direct sum is one in which nothing left the range.
/// Otherwise the values are summed again as [`Wide`] values, so the large
/// entries of `MAX + MAX - MAX - MAX + 1e-20` cancel and `1e-20` remains.
pub(crate) fn sum(values: &[f64]) -> Wide {
    let direct: f64 = values.iter().sum();
    if direct.is_finite() {
        return Wide::new(direct);
    }
    values
        .iter()
        .fold(Wide::ZERO, |total, &value| total + Wide::new(value))
}

/// The mean of non-empty finite `values`, finite whatever their magnitudes:
/// the [`sum`] divided by the count, and clamped to the range of the values,
/// where the exact mean lies, so rounding cannot carry it past them (or past
/// `f64::MAX`).
///
/// # Panics
/// Panics if `values` is empty.
pub(crate) fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "the mean of no values is undefined");
    let mean = sum(values).divided_by(values.len()).to_f64();
    let (low, high) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    mean.clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exponent_brackets_the_largest_magnitude() {
        assert_eq!(exponent([0.0, -0.0]), None);
        assert_eq!(exponent([1.0]), Some(0));
        assert_eq!(exponent([0.75, -3.0]), Some(1));
        assert_eq!(exponent([f64::MAX]), Some(1023));
        assert_eq!(exponent([f64::MIN_POSITIVE]), Some(-1022));
        // A subnormal magnitude still yields a normal power of two.
        assert_eq!(exponent([5e-324]), Some(-1022));
        assert_eq!(power_of_two(1023), 2f64.powi(1023));
        assert_eq!(power_of_two(-1022), f64::MIN_POSITIVE);
    }

    #[test]
    fn scaling_by_a_power_of_two_overflows_only_when_the_result_does() {
        // 2^2000 overflows on its own, but 2^-1000 * 2^2000 = 2^1000 does not.
        assert_eq!(times_power_of_two(2f64.powi(-1000), 2000), 2f64.powi(1000));
        assert_eq!(times_power_of_two(2f64.powi(1000), -2000), 2f64.powi(-1000));
        assert_eq!(times_power_of_two(2.0, 1023), f64::INFINITY);
    }

    #[test]
    fn wide_arithmetic_rounds_as_f64_does_with_an_unbounded_exponent_range() {
        let values = [
            1.0,
            -3.5,
            0.1,
            1e-300,
            -7e-310,
            5e-324,
            f64::MAX,
            -1e308,
            f64::MIN_POSITIVE,
            123456.789,
        ];
        for &a in &values {
            assert_eq!(Wide::new(a).to_f64(), a, "{a}");
            for &b in &values {
                // In range, every operation is the f64 one.
                if (a + b).is_finite() {
                    assert_eq!((Wide::new(a) + Wide::new(b)).to_f64(), a + b, "{a} + {b}");
                }
                let product = a * b;
                if product.is_finite() && (product == 0.0 || product.abs() >= f64::MIN_POSITIVE) {
                    assert_eq!((Wide::new(a) * Wide::new(b)).to_f64(), product, "{a} * {b}");
                }
            }
        }
        // Beyond the range, the value is kept rather than overflowed or
        // flushed, and comes back when it is in range again.
        let huge = Wide::new(f64::MAX) * Wide::new(f64::MAX);
        assert_eq!(huge.to_f64(), f64::INFINITY);
        assert_eq!(huge.sqrt().to_f64(), f64::MAX);
        assert_eq!(huge.ratio(Wide::new(f64::MAX)), f64::MAX);
        let tiny = Wide::new(1e-200) * Wide::new(1e-200);
        assert_eq!(tiny.to_f64(), 0.0);
        // sqrt(x * x) is |x| in f64, and so it is beyond its range.
        assert_eq!(tiny.sqrt().to_f64(), 1e-200);
        assert_eq!(Wide::new(4.0).sqrt().to_f64(), 2.0);
        assert_eq!(Wide::new(8.0).sqrt().to_f64(), 8f64.sqrt());
        // A sum whose large terms cancel keeps the small one.
        assert_eq!(
            sum(&[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1e-20]).to_f64(),
            1e-20
        );
        assert_eq!(sum(&[f64::MAX, f64::MAX, -f64::MAX]).to_f64(), f64::MAX);
        assert_eq!(sum(&[1.0, -1.0]), Wide::ZERO);
        assert_eq!(sum(&[1e308, 1e308]).signum(), 1.0);
        assert_eq!(sum(&[-1e308, -1e308]).signum(), -1.0);
    }

    #[test]
    fn a_mean_of_values_near_the_largest_finite_value_is_finite() {
        assert_eq!(mean(&[f64::MAX, f64::MAX, f64::MAX]), f64::MAX);
        assert_eq!(mean(&[-f64::MAX, -f64::MAX]), -f64::MAX);
        assert_eq!(mean(&[1e308, 1e308, -1e308]), 1e308 / 3.0);
        // In range, it is the direct mean.
        assert_eq!(mean(&[0.1, 0.2, 0.4]), (0.1 + 0.2 + 0.4) / 3.0);
    }
}
