//! A small bounded arbitrary-precision signed integer.
//!
//! Sign and magnitude, with the magnitude as little-endian `u32` limbs. Every operation is
//! bounded by [`MAX_LIMBS`] (1280 bits, about 385 decimal digits): a result that would need more
//! is [`BigIntError::TooLarge`], never an allocation of attacker-chosen size. No operation
//! panics: division by zero is [`BigIntError::DivisionByZero`], and every index is derived from
//! lengths the code itself maintains.
//!
//! This is the exact-integer kernel of [`crate::ArithmeticVerifier`]; it is not a general
//! numeric library (no shifts, bit operations or radix other than 10 are exposed).

use std::cmp::Ordering;
use std::fmt;

/// Most limbs of any value this module produces.
pub(crate) const MAX_LIMBS: usize = 40;

const BASE: u64 = 1 << 32;

/// Why an operation on a [`BigInt`] has no value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BigIntError {
    /// The result would exceed [`MAX_LIMBS`] limbs.
    TooLarge,
    DivisionByZero,
    /// The decimal text is empty or has a character other than an ASCII digit.
    InvalidDecimal,
}

/// A signed integer. Invariants: `mag` has no most-significant zero limb, and zero is the empty
/// magnitude and never negative, so equal values have equal representations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BigInt {
    negative: bool,
    mag: Vec<u32>,
}

fn trim(mag: &mut Vec<u32>) {
    while mag.last() == Some(&0) {
        mag.pop();
    }
}

fn cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn add_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry = 0_u64;
    for (i, limb) in long.iter().enumerate() {
        let sum = u64::from(*limb) + u64::from(short.get(i).copied().unwrap_or(0)) + carry;
        out.push(sum as u32);
        carry = sum >> 32;
    }
    if carry != 0 {
        out.push(carry as u32);
    }
    out
}

/// `a - b` for `a >= b` (the caller compares first).
fn sub_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0_i64;
    for (i, limb) in a.iter().enumerate() {
        let diff = i64::from(*limb) - i64::from(b.get(i).copied().unwrap_or(0)) - borrow;
        out.push(diff as u32);
        borrow = i64::from(diff < 0);
    }
    trim(&mut out);
    out
}

fn mul_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0_u32; a.len() + b.len()];
    for (i, x) in a.iter().enumerate() {
        let mut carry = 0_u64;
        for (j, y) in b.iter().enumerate() {
            let cell = u64::from(out[i + j]) + u64::from(*x) * u64::from(*y) + carry;
            out[i + j] = cell as u32;
            carry = cell >> 32;
        }
        out[i + b.len()] = carry as u32;
    }
    trim(&mut out);
    out
}

/// Divides by a single non-zero limb.
fn divrem_limb(u: &[u32], d: u32) -> (Vec<u32>, u32) {
    let mut quotient = vec![0_u32; u.len()];
    let mut rem = 0_u64;
    for (i, limb) in u.iter().enumerate().rev() {
        let cur = (rem << 32) | u64::from(*limb);
        quotient[i] = (cur / u64::from(d)) as u32;
        rem = cur % u64::from(d);
    }
    trim(&mut quotient);
    (quotient, rem as u32)
}

/// Knuth's algorithm D. `v` is non-zero; both inputs are trimmed.
fn divrem_mag(u: &[u32], v: &[u32]) -> (Vec<u32>, Vec<u32>) {
    if cmp_mag(u, v) == Ordering::Less {
        return (Vec::new(), u.to_vec());
    }
    if let [d] = v {
        let (q, r) = divrem_limb(u, *d);
        return (q, if r == 0 { Vec::new() } else { vec![r] });
    }
    let n = v.len();
    let m = u.len() - n;
    let shift = v[n - 1].leading_zeros();
    let shl = |x: &[u32], extra: bool| -> Vec<u32> {
        let mut out = Vec::with_capacity(x.len() + 1);
        let mut carry = 0_u32;
        for limb in x {
            out.push(if shift == 0 {
                *limb
            } else {
                (*limb << shift) | carry
            });
            carry = if shift == 0 { 0 } else { *limb >> (32 - shift) };
        }
        if extra {
            out.push(carry);
        }
        out
    };
    let vn = shl(v, false);
    let mut un = shl(u, true);
    let mut q = vec![0_u32; m + 1];
    let top = u64::from(vn[n - 1]);
    let second = u64::from(vn[n - 2]);
    for j in (0..=m).rev() {
        let num = (u64::from(un[j + n]) << 32) | u64::from(un[j + n - 1]);
        let mut qhat = num / top;
        let mut rhat = num % top;
        while qhat >= BASE || qhat * second > ((rhat << 32) | u64::from(un[j + n - 2])) {
            qhat -= 1;
            rhat += top;
            if rhat >= BASE {
                break;
            }
        }
        let mut borrow = 0_i64;
        let mut carry = 0_u64;
        for i in 0..n {
            let product = qhat * u64::from(vn[i]) + carry;
            carry = product >> 32;
            let diff = i64::from(un[i + j]) - borrow - (product & 0xFFFF_FFFF) as i64;
            un[i + j] = diff as u32;
            borrow = i64::from(diff < 0);
        }
        let diff = i64::from(un[j + n]) - borrow - carry as i64;
        un[j + n] = diff as u32;
        if diff < 0 {
            // qhat was one too large: add the divisor back.
            qhat -= 1;
            let mut carry = 0_u64;
            for i in 0..n {
                let sum = u64::from(un[i + j]) + u64::from(vn[i]) + carry;
                un[i + j] = sum as u32;
                carry = sum >> 32;
            }
            un[j + n] = un[j + n].wrapping_add(carry as u32);
        }
        q[j] = qhat as u32;
    }
    trim(&mut q);
    // Undo the normalisation shift on the remainder.
    let mut rem = Vec::with_capacity(n);
    for i in 0..n {
        let low = un[i] >> shift;
        let high = if shift == 0 {
            0
        } else {
            un[i + 1] << (32 - shift)
        };
        rem.push(low | high);
    }
    trim(&mut rem);
    (q, rem)
}

impl BigInt {
    pub(crate) fn zero() -> Self {
        Self::default()
    }

    pub(crate) fn from_i128(value: i128) -> Self {
        let mut magnitude = value.unsigned_abs();
        let mut mag = Vec::new();
        while magnitude != 0 {
            mag.push(magnitude as u32);
            magnitude >>= 32;
        }
        Self {
            negative: value < 0,
            mag,
        }
    }

    /// The value as an `i128`, if it fits.
    pub(crate) fn to_i128(&self) -> Option<i128> {
        if self.mag.len() > 4 {
            return None;
        }
        let magnitude = self
            .mag
            .iter()
            .rev()
            .fold(0_u128, |acc, limb| (acc << 32) | u128::from(*limb));
        if self.negative {
            if magnitude == 1_u128 << 127 {
                Some(i128::MIN)
            } else {
                i128::try_from(magnitude).ok().map(|v| -v)
            }
        } else {
            i128::try_from(magnitude).ok()
        }
    }

    /// Parses ASCII decimal digits (no sign, no separators).
    pub(crate) fn parse_decimal(digits: &str) -> Result<Self, BigIntError> {
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(BigIntError::InvalidDecimal);
        }
        let mut mag: Vec<u32> = Vec::new();
        for digit in digits.bytes() {
            let mut carry = u64::from(digit - b'0');
            for limb in &mut mag {
                let cell = u64::from(*limb) * 10 + carry;
                *limb = cell as u32;
                carry = cell >> 32;
            }
            if carry != 0 {
                mag.push(carry as u32);
            }
            if mag.len() > MAX_LIMBS {
                return Err(BigIntError::TooLarge);
            }
        }
        trim(&mut mag);
        Ok(Self {
            negative: false,
            mag,
        })
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub(crate) fn is_negative(&self) -> bool {
        self.negative
    }

    pub(crate) fn limbs(&self) -> usize {
        self.mag.len()
    }

    #[cfg(test)]
    pub(crate) fn abs(&self) -> Self {
        Self {
            negative: false,
            mag: self.mag.clone(),
        }
    }

    pub(crate) fn neg(&self) -> Self {
        Self {
            negative: !self.negative && !self.mag.is_empty(),
            mag: self.mag.clone(),
        }
    }

    fn checked(negative: bool, mag: Vec<u32>) -> Result<Self, BigIntError> {
        if mag.len() > MAX_LIMBS {
            return Err(BigIntError::TooLarge);
        }
        Ok(Self {
            negative: negative && !mag.is_empty(),
            mag,
        })
    }

    pub(crate) fn add(&self, other: &Self) -> Result<Self, BigIntError> {
        if self.negative == other.negative {
            return Self::checked(self.negative, add_mag(&self.mag, &other.mag));
        }
        match cmp_mag(&self.mag, &other.mag) {
            Ordering::Equal => Ok(Self::zero()),
            Ordering::Greater => Self::checked(self.negative, sub_mag(&self.mag, &other.mag)),
            Ordering::Less => Self::checked(other.negative, sub_mag(&other.mag, &self.mag)),
        }
    }

    pub(crate) fn sub(&self, other: &Self) -> Result<Self, BigIntError> {
        self.add(&other.neg())
    }

    pub(crate) fn mul(&self, other: &Self) -> Result<Self, BigIntError> {
        Self::checked(
            self.negative != other.negative,
            mul_mag(&self.mag, &other.mag),
        )
    }

    /// Truncating division: the quotient rounds toward zero and the remainder takes the sign of
    /// the dividend, as Rust's `/` and `%` do.
    pub(crate) fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), BigIntError> {
        if divisor.is_zero() {
            return Err(BigIntError::DivisionByZero);
        }
        let (q, r) = divrem_mag(&self.mag, &divisor.mag);
        Ok((
            Self::checked(self.negative != divisor.negative, q)?,
            Self::checked(self.negative, r)?,
        ))
    }

    /// The quotient if `divisor` divides `self` exactly, `Ok(None)` if there is a remainder.
    pub(crate) fn div_exact(&self, divisor: &Self) -> Result<Option<Self>, BigIntError> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        Ok(remainder.is_zero().then_some(quotient))
    }

    /// Greatest common divisor of the magnitudes (never negative; `gcd(0, 0)` is 0).
    pub(crate) fn gcd(&self, other: &Self) -> Self {
        let mut a = self.mag.clone();
        let mut b = other.mag.clone();
        while !b.is_empty() {
            let (_, rem) = divrem_mag(&a, &b);
            a = std::mem::replace(&mut b, rem);
        }
        Self {
            negative: false,
            mag: a,
        }
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.mag, &other.mag),
            (true, true) => cmp_mag(&other.mag, &self.mag),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mag.is_empty() {
            return f.write_str("0");
        }
        let mut chunks = Vec::new();
        let mut rest = self.mag.clone();
        while !rest.is_empty() {
            let (quotient, rem) = divrem_limb(&rest, 1_000_000_000);
            chunks.push(rem);
            rest = quotient;
        }
        let mut text = String::new();
        if self.negative {
            text.push('-');
        }
        for (i, chunk) in chunks.iter().rev().enumerate() {
            if i == 0 {
                text.push_str(&chunk.to_string());
            } else {
                text.push_str(&format!("{chunk:09}"));
            }
        }
        f.write_str(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny xorshift64*: deterministic, seeded, enough for property-style tests.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// A signed value of up to `bits` bits (bits <= 100), biased toward edge shapes.
        fn value(&mut self, bits: u32) -> i128 {
            let wide = (u128::from(self.next()) << 64) | u128::from(self.next());
            let width = 1 + (self.next() % u64::from(bits)) as u32;
            let mut magnitude = (wide & ((1_u128 << width) - 1)) as i128;
            match self.next() % 8 {
                0 => magnitude = (1_i128 << width) - 1,
                1 => magnitude = 1_i128 << (width - 1),
                _ => {}
            }
            if self.next().is_multiple_of(2) {
                -magnitude
            } else {
                magnitude
            }
        }
    }

    fn big(value: i128) -> BigInt {
        BigInt::from_i128(value)
    }

    #[test]
    fn i128_round_trip_including_extremes() {
        for value in [0, 1, -1, 4_294_967_296, i128::MAX, i128::MIN, -i128::MAX] {
            assert_eq!(big(value).to_i128(), Some(value), "{value}");
            assert_eq!(big(value).to_string(), value.to_string());
        }
        let too_big = big(i128::MAX).add(&big(1)).unwrap();
        assert_eq!(too_big.to_i128(), None);
        assert_eq!(too_big.limbs(), 4);
    }

    #[test]
    fn add_sub_mul_div_agree_with_i128() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for _ in 0..4000 {
            let a = rng.value(100);
            let b = rng.value(100);
            assert_eq!(
                big(a).add(&big(b)).unwrap().to_i128(),
                Some(a + b),
                "{a}+{b}"
            );
            assert_eq!(
                big(a).sub(&big(b)).unwrap().to_i128(),
                Some(a - b),
                "{a}-{b}"
            );
            assert_eq!(big(a).cmp(&big(b)), a.cmp(&b), "{a} cmp {b}");
            let (x, y) = (rng.value(60), rng.value(60));
            assert_eq!(
                big(x).mul(&big(y)).unwrap().to_i128(),
                Some(x * y),
                "{x}*{y}"
            );
            if b != 0 {
                let (q, r) = big(a).div_rem(&big(b)).unwrap();
                assert_eq!(q.to_i128(), Some(a / b), "{a}/{b}");
                assert_eq!(r.to_i128(), Some(a % b), "{a}%{b}");
                let exact = big(a).div_exact(&big(b)).unwrap();
                assert_eq!(exact.and_then(|v| v.to_i128()), (a % b == 0).then(|| a / b));
            }
        }
    }

    #[test]
    fn gcd_agrees_with_a_reference() {
        fn reference(mut a: u128, mut b: u128) -> u128 {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        }
        let mut rng = XorShift(7);
        for _ in 0..2000 {
            let (a, b) = (rng.value(100), rng.value(100));
            let got = big(a).gcd(&big(b)).to_i128().unwrap();
            assert_eq!(got as u128, reference(a.unsigned_abs(), b.unsigned_abs()));
        }
        assert!(big(0).gcd(&big(0)).is_zero());
    }

    #[test]
    fn decimal_parse_and_display_round_trip_beyond_i128() {
        let text = "123456789012345678901234567890123456789012345678901234567890";
        let value = BigInt::parse_decimal(text).unwrap();
        assert_eq!(value.to_string(), text);
        assert_eq!(value.to_i128(), None);
        assert_eq!(BigInt::parse_decimal("0007").unwrap().to_string(), "7");
        assert_eq!(BigInt::parse_decimal("000").unwrap().to_string(), "0");
        assert!(value.neg().is_negative());
        assert_eq!(value.neg().to_string(), format!("-{text}"));
        assert_eq!(value.neg().abs(), value);
        assert!(!BigInt::zero().neg().is_negative());
        for bad in ["", "-1", "1.5", "1e3", "٣", " 1"] {
            assert_eq!(BigInt::parse_decimal(bad), Err(BigIntError::InvalidDecimal));
        }
    }

    #[test]
    fn multi_limb_division_matches_multiplication_back() {
        // (a * b + r) / b == a remainder r for operands far beyond i128.
        let mut rng = XorShift(0xDEAD_BEEF);
        for _ in 0..500 {
            let a =
                BigInt::parse_decimal(&format!("{}{}{}", rng.next() | 1, rng.next(), rng.next()))
                    .unwrap();
            let b = BigInt::parse_decimal(&format!("{}{}", rng.next() | 1, rng.next())).unwrap();
            let r = BigInt::from_i128(i128::from(rng.next() % 1000));
            let r = r.div_rem(&b).unwrap().1;
            let n = a.mul(&b).unwrap().add(&r).unwrap();
            let (q, rem) = n.div_rem(&b).unwrap();
            assert_eq!(q, a);
            assert_eq!(rem, r);
        }
        // A divisor with a maximal top limb.
        let u = BigInt::parse_decimal("340282366920938463463374607431768211455").unwrap();
        let v = BigInt::parse_decimal("18446744073709551615").unwrap();
        let (q, r) = u.div_rem(&v).unwrap();
        assert_eq!(q.mul(&v).unwrap().add(&r).unwrap(), u);
    }

    #[test]
    fn typed_errors_for_zero_division_and_oversize() {
        assert_eq!(
            big(1).div_rem(&BigInt::zero()),
            Err(BigIntError::DivisionByZero)
        );
        assert_eq!(
            big(1).div_exact(&BigInt::zero()),
            Err(BigIntError::DivisionByZero)
        );
        let digits = "9".repeat(400);
        assert_eq!(BigInt::parse_decimal(&digits), Err(BigIntError::TooLarge));
        let value = BigInt::parse_decimal(&"9".repeat(300)).unwrap();
        assert!(value.limbs() <= MAX_LIMBS);
        assert_eq!(value.mul(&value), Err(BigIntError::TooLarge));
    }

    #[test]
    fn division_truncates_toward_zero_with_the_dividends_sign() {
        let (q, r) = big(-7).div_rem(&big(2)).unwrap();
        assert_eq!((q.to_i128(), r.to_i128()), (Some(-3), Some(-1)));
        let (q, r) = big(7).div_rem(&big(-2)).unwrap();
        assert_eq!((q.to_i128(), r.to_i128()), (Some(-3), Some(1)));
        assert_eq!(
            big(-6).div_exact(&big(3)).unwrap().unwrap().to_i128(),
            Some(-2)
        );
        assert_eq!(big(7).div_exact(&big(2)).unwrap(), None);
    }
}
