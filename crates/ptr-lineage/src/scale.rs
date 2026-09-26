//! Power-of-two rescaling for sums, norms and products of finite values whose
//! intermediate results would leave the range of `f64` although the answer
//! does not.
//!
//! Dividing or multiplying by a power of two is exact unless the result
//! underflows, so a computation carried out on rescaled values rounds exactly
//! as the direct one wherever the direct one stays in range; it differs only
//! where the direct one would have overflowed, or where an entry more than
//! `2^1022` times smaller than the largest loses bits it could not affect.

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

/// The in-order sum of finite `values` as `(total, exponent)`, the sum being
/// `total * 2^exponent`. When no partial sum overflows this is the direct sum
/// with exponent zero. Otherwise it is the sum of the values divided by the
/// power of two at or below the largest magnitude: every term is then below
/// two in magnitude, so no partial sum can overflow. A running sum that has
/// overflowed stays infinite (or turns NaN), so a finite direct sum is one in
/// which nothing overflowed.
pub(crate) fn sum(values: &[f64]) -> (f64, i32) {
    let direct: f64 = values.iter().sum();
    if direct.is_finite() {
        return (direct, 0);
    }
    let exponent = exponent(values.iter().copied()).unwrap_or(0);
    let scale = power_of_two(exponent);
    (values.iter().map(|value| value / scale).sum(), exponent)
}

/// The mean of non-empty finite `values`, finite whatever their magnitudes:
/// the [`sum`] divided by the count, scaled back, and clamped to the range of
/// the values, where the exact mean lies, so rounding cannot carry it past
/// them (or past `f64::MAX`).
///
/// # Panics
/// Panics if `values` is empty.
pub(crate) fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "the mean of no values is undefined");
    let (total, exponent) = sum(values);
    let mean = times_power_of_two(total / values.len() as f64, exponent);
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
    fn a_mean_of_values_near_the_largest_finite_value_is_finite() {
        assert_eq!(mean(&[f64::MAX, f64::MAX, f64::MAX]), f64::MAX);
        assert_eq!(mean(&[-f64::MAX, -f64::MAX]), -f64::MAX);
        assert_eq!(mean(&[1e308, 1e308, -1e308]), 1e308 / 3.0);
        // In range, it is the direct mean.
        assert_eq!(mean(&[0.1, 0.2, 0.4]), (0.1 + 0.2 + 0.4) / 3.0);
    }
}
