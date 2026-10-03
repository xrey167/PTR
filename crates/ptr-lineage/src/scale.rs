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
//! values instead; a product of factors, whose entries may cancel to far
//! less than any factor entry, is brought to scale only once it is formed.
//! A direct `f64` result is taken for the wide one only where it provably is
//! that result, which finiteness alone shows for a sum (`f64` addition loses
//! nothing below the normal range) but not for a product, whose terms may
//! each round to zero although their sum does not; and a mean, a ratio or
//! an entry of a product ([`Rounded`]) is rounded to `f64` once, not once to
//! 53 bits and again to the subnormal grid.
//! Bases of such a product are taken from its factors, each brought to
//! scale on its own, only where no entry of a factor is more than `2^400`
//! below the largest of that factor and the product's largest column and
//! row are at most `2^8` below the size their terms would give them without
//! cancellation.

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

    /// `floor(log2 |self|)` for a nonzero value, `None` for zero.
    pub(crate) fn exponent(self) -> Option<i32> {
        (!self.is_zero()).then_some(self.exponent)
    }

    /// `self * 2^shift`, exactly: only the exponent changes.
    pub(crate) fn times_power_of_two(self, shift: i32) -> Self {
        if self.is_zero() {
            return self;
        }
        Self {
            significand: self.significand,
            exponent: self.exponent + shift,
        }
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

    /// `self / other` as the nearest `f64`, rounded once: infinite beyond
    /// `f64::MAX`, and on the subnormal grid below the normal range, as the
    /// `f64` quotient of the two values would be if both were `f64` values.
    ///
    /// The quotient of the significands lies in `(1/2, 2)`. Where the ratio
    /// is certainly normal it is rounded there and scaled exactly. Below
    /// that, the significands are first scaled to normal values whose
    /// quotient is the ratio itself, which `f64` division then rounds once to
    /// the subnormal grid; rounding the quotient of the significands first
    /// would round twice.
    ///
    /// # Panics
    /// Panics in debug builds if `other` is zero.
    pub(crate) fn ratio(self, other: Self) -> f64 {
        debug_assert!(!other.is_zero(), "a ratio to zero is undefined");
        let shift = self.exponent - other.exponent;
        if (-1076..=-1022).contains(&shift) {
            // Both scaled values are normal and exact: shift + 1023 lies in
            // [-53, 1], and the divisor is below 2^1024.
            (self.significand * power_of_two(shift + 1023))
                / (other.significand * power_of_two(1023))
        } else {
            // Normal above -1022 (the quotient exceeds 1/2); below -1076 the
            // ratio is under a quarter of the smallest subnormal, and zero
            // however it is rounded.
            times_power_of_two(self.significand / other.significand, shift)
        }
    }

    /// `self / count`, rounded to 53 bits as `f64` division rounds a normal
    /// quotient. Rounding the result to `f64` again below the normal range
    /// would round twice, which [`quotient`] avoids.
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
        self.sum(other).0
    }
}

impl Wide {
    /// `self + other` as [`Wide`] addition rounds it, and the sign of what
    /// that rounding discarded (see [`Rounded`]).
    fn sum(self, other: Self) -> (Self, f64) {
        if self.is_zero() {
            return (other, 0.0);
        }
        if other.is_zero() {
            return (self, 0.0);
        }
        let (large, small) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let gap = large.exponent - small.exponent;
        if gap > 1000 {
            // The whole of the smaller value is discarded.
            return (large, small.significand.signum());
        }
        let (a, b) = (large.significand, small.significand * power_of_two(-gap));
        let sum = a + b;
        // The error of an f64 sum is an f64 value, and Knuth's TwoSum gives
        // it exactly, subnormal or not; nothing here is near overflow.
        let b_part = sum - a;
        let a_part = sum - b_part;
        let error = (a - a_part) + (b - b_part);
        (Self::normalised(sum, large.exponent), sign(error))
    }
}

/// `1.0`, `-1.0` or `0.0` as `value` is positive, negative or zero, of
/// either sign.
fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// A [`Wide`] value `v` as an operation rounded it to 53 bits from its exact
/// result `v + e`, with the sign of `e`, what the rounding discarded: `|e|`
/// is at most half a unit in the last place of `v`, and zero when the
/// result was exact.
///
/// [`Rounded::to_f64`] rounds the exact result `v + e` once to `f64`, as
/// `f64` itself rounds the result of an operation. Rounding `v` instead
/// ([`Wide::to_f64`]) rounds twice below the normal range, where `v` can lie
/// exactly halfway between two neighbouring `f64` values that `v + e` does
/// not: `(1 + 2^-52) 2^-537 * (1 - 2^-53) 2^-538` is
/// `(1 + 2^-53 - 2^-105) 2^-1075`, just above half the smallest subnormal,
/// which `f64` rounds to `2^-1074`; rounded first to 53 bits it is
/// `2^-1075`, exactly half, which then rounds to even, zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rounded {
    value: Wide,
    /// The sign of `e`: `1.0`, `-1.0`, or `0.0` when `value` is exact.
    discarded: f64,
}

impl Rounded {
    pub(crate) const ZERO: Self = Self::exact(Wide::ZERO);

    /// `value`, which no rounding produced.
    pub(crate) const fn exact(value: Wide) -> Self {
        Self {
            value,
            discarded: 0.0,
        }
    }

    /// The value, rounded to 53 bits.
    pub(crate) fn value(self) -> Wide {
        self.value
    }

    /// `left * right`, rounded as [`Wide`] multiplication rounds it.
    pub(crate) fn product(left: Wide, right: Wide) -> Self {
        let value = left * right;
        // The error of an f64 product is an f64 value, which a fused
        // multiply-add gives exactly: the significands, and so their
        // product, lie in [1, 4), far from underflow.
        let rounded = left.significand * right.significand;
        let error = left.significand.mul_add(right.significand, -rounded);
        Self {
            value,
            discarded: sign(error),
        }
    }

    /// The nearest `f64` to the exact result `v + e`, rounded once:
    /// infinite beyond `f64::MAX`, on the subnormal grid below the normal
    /// range, and otherwise exact.
    ///
    /// It is [`Wide::to_f64`] of `v` except where `v` is exactly halfway
    /// between two neighbouring `f64` values below the normal range (an odd
    /// multiple of `2^-1075` there) and `e` is not zero: `v + e` then lies on
    /// one side of that halfway point, and rounds to the neighbour on that
    /// side. Everywhere else `v` and `v + e` round alike: below the normal
    /// range every halfway point is itself a 53-bit value, so none lies
    /// strictly between `v` and `v + e`, where it would be nearer to `v + e`
    /// than `v` is; and at or above it `f64` has `v`'s 53 bits.
    pub(crate) fn to_f64(self) -> f64 {
        let Self { value, discarded } = self;
        let rounded = value.to_f64();
        if discarded == 0.0 || value.is_zero() || !(-1075..=-1023).contains(&value.exponent) {
            return rounded;
        }
        // |v| in units of 2^-1075: exact, in [1, 2^53), and an integer
        // exactly when v lies on that grid.
        let halves = value.significand.abs() * power_of_two(value.exponent + 1075);
        if halves.fract() != 0.0 || halves % 2.0 == 0.0 {
            return rounded;
        }
        // One unit of 2^-1075 toward v + e is a multiple of 2^-1074 of at
        // most 2^-1022, which f64 holds exactly.
        let toward = if discarded == value.significand.signum() {
            halves + 1.0
        } else {
            halves - 1.0
        };
        times_power_of_two(toward, -1075).copysign(value.significand)
    }
}

impl std::ops::Add for Rounded {
    type Output = Self;

    /// The sum of the two values, rounded as [`Wide`] addition rounds it.
    /// Adding zero is exact and leaves the other operand as it is, what its
    /// own rounding discarded included, as `f64` addition leaves an `f64`
    /// value to which zero is added: the sum of one nonzero term is that
    /// term, rounded once.
    fn add(self, other: Self) -> Self {
        if other.value.is_zero() {
            return self;
        }
        if self.value.is_zero() {
            return other;
        }
        let (value, discarded) = self.value.sum(other.value);
        Self { value, discarded }
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
/// the [`sum`] divided by the count and rounded once ([`quotient`]), and
/// clamped to the range of the values, where the exact mean lies, so
/// rounding cannot carry it past them (or past `f64::MAX`).
///
/// # Panics
/// Panics if `values` is empty.
pub(crate) fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "the mean of no values is undefined");
    let (low, high) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
    quotient(sum(values), values.len()).clamp(low, high)
}

/// `a - b` for finite values, rounded as `f64` subtraction rounds but with
/// an unbounded exponent range: the direct difference wherever that is
/// finite, and never infinite or NaN.
pub(crate) fn difference(a: f64, b: f64) -> Wide {
    Wide::new(a) + Wide::new(-b)
}

/// The mean of non-empty `values`, each a [`difference`] or an `f64`
/// value: their in-order sum, rounded step by step as `f64` addition rounds
/// but with an unbounded exponent range, divided by the count and rounded
/// once ([`quotient`]), and clamped to the range of the values rounded to
/// `f64`. It is infinite only when that mean exceeds `f64::MAX`.
///
/// # Panics
/// Panics if `values` is empty.
pub(crate) fn wide_mean(values: &[Wide]) -> f64 {
    assert!(!values.is_empty(), "the mean of no values is undefined");
    let (low, high) = values
        .iter()
        .map(|value| value.to_f64())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let total = values
        .iter()
        .fold(Wide::ZERO, |total, &value| total + value);
    quotient(total, values.len()).clamp(low, high)
}

/// `total / count` rounded once to the nearest `f64`.
///
/// Where `total` is an `f64` value, as an in-order sum of `f64` values or of
/// their [`difference`]s is whenever it does not exceed `f64::MAX` (each is
/// a multiple of the smallest subnormal, and so is every rounded sum of
/// them), this is the `f64` quotient, rounded on the subnormal grid below
/// the normal range. Rounding the quotient of the significands first and
/// then to that grid would round twice: `(5 * 2^50 + 7) * 2^-1074` over
/// five is `2^50 + 1.4` units, which rounds to `2^50 + 1` once but to
/// `2^50 + 1.5` and then `2^50 + 2` twice. A `total` beyond `f64::MAX` is
/// divided as a [`Wide`] value. Its quotient over any count never falls
/// below the normal range, where rounding it to `f64` cannot round twice,
/// but it can stay beyond `f64::MAX` and is then infinite: a total of
/// differences over a small count may, as `-MAX - MAX` over one does.
fn quotient(total: Wide, count: usize) -> f64 {
    let direct = total.to_f64();
    if direct.is_finite() && Wide::new(direct) == total {
        direct / count as f64
    } else {
        total.divided_by(count).to_f64()
    }
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
    fn a_finite_direct_sum_loses_nothing_to_underflow() {
        // A sum of f64 values below MIN_POSITIVE is exact, so the direct sum
        // is the wide one wherever it is finite.
        let unit = 5e-324;
        let values = [
            unit,
            3.0 * unit,
            -unit,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE / 2.0,
            -f64::MIN_POSITIVE,
        ];
        let wide = values
            .iter()
            .fold(Wide::ZERO, |total, &value| total + Wide::new(value));
        assert_eq!(sum(&values), wide);
        assert_eq!(sum(&values).to_f64(), 3.0 * unit - f64::MIN_POSITIVE / 2.0);
    }

    #[test]
    fn a_mean_below_the_normal_range_is_rounded_once() {
        let unit = 5e-324;
        // (5 * 2^50 + 7) units over five is 2^50 + 1.4 units: 2^50 + 1 once
        // rounded, but 2^50 + 2 when the quotient of the significands is
        // rounded first (to 2^50 + 1.5) and then again to the subnormal grid.
        let total = (5.0 * 2f64.powi(50) + 7.0) * unit;
        let values = [total - 4.0 * unit, unit, unit, unit, unit];
        assert_eq!(mean(&values), total / 5.0);
        assert_eq!(mean(&values), (2f64.powi(50) + 1.0) * unit);
        // The same for a sum that overflows on its way and cancels to it.
        let mut cancelling = vec![f64::MAX, f64::MAX, -f64::MAX, -f64::MAX];
        cancelling.extend(values);
        assert_eq!(mean(&cancelling), total / 9.0);
        // And for a mean of differences.
        let changes = [difference(total, 4.0 * unit), Wide::new(4.0 * unit)];
        assert_eq!(wide_mean(&changes), total / 2.0);
        // A ratio is rounded once as well, wherever it lands.
        assert_eq!(Wide::new(total).ratio(Wide::new(5.0)), total / 5.0);
        for (numerator, denominator) in [
            (total, 3.0),
            (total, 7.0),
            (3.0 * unit, 2.0),
            (5.0 * unit, 4.0),
            (1.0, 3.0),
            (f64::MIN_POSITIVE, 3.0),
            (f64::MIN_POSITIVE, 2f64.powi(52) * 1.5),
            (f64::MIN_POSITIVE, 2f64.powi(53) * 1.5),
            (unit, 1.9),
            (-unit, 2.1),
            (f64::MAX, 0.5),
        ] {
            assert_eq!(
                Wide::new(numerator).ratio(Wide::new(denominator)),
                numerator / denominator,
                "{numerator} / {denominator}"
            );
        }
    }

    /// Deterministic nonzero values of both signs and every magnitude, one in
    /// eight subnormal.
    #[allow(clippy::manual_is_multiple_of)]
    fn values_of_every_magnitude() -> Vec<f64> {
        let mut state = 7u64;
        (0..400)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = state & !(0x7ffu64 << 52);
                let exponent = if state % 8 == 0 {
                    0
                } else {
                    (state >> 20) % 0x7ff
                };
                f64::from_bits(bits | (exponent << 52))
            })
            .filter(|value| value.is_finite() && *value != 0.0)
            .collect()
    }

    #[test]
    fn a_ratio_of_f64_values_is_their_f64_quotient_across_the_whole_range() {
        let values = values_of_every_magnitude();
        for &a in &values {
            for &b in &values {
                let ratio = Wide::new(a).ratio(Wide::new(b));
                assert_eq!(ratio.to_bits(), (a / b).to_bits(), "{a:e} / {b:e}");
            }
        }
    }

    #[test]
    fn a_product_or_sum_rounded_once_is_the_f64_one_across_the_whole_range() {
        let values = values_of_every_magnitude();
        let mut below_the_normal_range = 0;
        for &a in &values {
            for &b in &values {
                let product = Rounded::product(Wide::new(a), Wide::new(b));
                assert_eq!(
                    product.to_f64().to_bits(),
                    (a * b).to_bits(),
                    "{a:e} * {b:e}"
                );
                assert_eq!(product.value(), Wide::new(a) * Wide::new(b));
                if a * b != 0.0 && (a * b).abs() < f64::MIN_POSITIVE {
                    below_the_normal_range += 1;
                }
                let sum = Rounded::exact(Wide::new(a)) + Rounded::exact(Wide::new(b));
                assert_eq!(sum.to_f64().to_bits(), (a + b).to_bits(), "{a:e} + {b:e}");
                assert_eq!(sum.value(), Wide::new(a) + Wide::new(b));
            }
        }
        // Some of the products are subnormal.
        assert!(below_the_normal_range > 500);
        // Zero added to a rounded value leaves it, and what its rounding
        // discarded, as it is.
        let unit = 5e-324;
        let (b, a) = (
            (1.0 + f64::EPSILON) * 2f64.powi(-537),
            (1.0 - f64::EPSILON / 2.0) * 2f64.powi(-538),
        );
        let product = Rounded::product(Wide::new(b), Wide::new(a));
        assert_eq!(product.value().to_f64(), 0.0);
        assert_eq!(product.to_f64(), unit);
        assert_eq!((Rounded::ZERO + product).to_f64(), unit);
        assert_eq!((product + Rounded::ZERO).to_f64(), unit);
        // A value too small to move the other's last place still decides
        // which way a sum halfway between two subnormal values rounds.
        let half = Rounded::exact(Wide::new(1.0).times_power_of_two(-1075));
        let far = |sign: f64| Rounded::exact(Wide::new(sign).times_power_of_two(-2100));
        assert_eq!((half + far(1.0)).value(), half.value());
        assert_eq!((half + far(1.0)).to_f64(), unit);
        assert_eq!((half + far(-1.0)).to_f64().to_bits(), 0.0f64.to_bits());
        assert_eq!(half.to_f64(), 0.0);
        let three_halves = Rounded::exact(Wide::new(-3.0).times_power_of_two(-1075));
        assert_eq!(three_halves.to_f64(), -2.0 * unit);
        assert_eq!((three_halves + far(1.0)).to_f64(), -unit);
        assert_eq!((three_halves + far(-1.0)).to_f64(), -2.0 * unit);
        let negative_half = Rounded::exact(Wide::new(-1.0).times_power_of_two(-1075));
        assert_eq!(
            (negative_half + far(1.0)).to_f64().to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn a_mean_of_differences_beyond_the_largest_finite_value_is_infinite() {
        // -MAX - MAX over one or two is beyond f64::MAX, and over three it is
        // back in the normal range, rounded once.
        let change = difference(-f64::MAX, f64::MAX);
        assert_eq!(change.to_f64(), f64::NEG_INFINITY);
        assert_eq!(wide_mean(&[change]), f64::NEG_INFINITY);
        assert_eq!(wide_mean(&[change, change]), f64::NEG_INFINITY);
        assert_eq!(
            wide_mean(&[change, Wide::ZERO, Wide::ZERO]),
            -2.0 * (f64::MAX / 3.0)
        );
        // A total beyond f64::MAX never falls below the normal range, over
        // any count.
        let total = Wide::new(f64::MAX) * Wide::new(f64::MAX);
        assert_eq!(quotient(total, usize::MAX), f64::INFINITY);
        assert_eq!(
            quotient(Wide::new(2.0).times_power_of_two(1023), usize::MAX),
            2f64.powi(1024 - 64)
        );
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
