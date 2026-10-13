//! Exact arithmetic verification.
//!
//! The task is an expression over integers with `+ - * /`, unary `+ -` and parentheses. It is
//! parsed and evaluated by an independent bounded recursive-descent evaluator (no `eval`, no
//! floating point): every value is an exact reduced rational of arbitrary-precision integers
//! (the in-crate `bigint` kernel, at most [`MAX_VALUE_LIMBS`] 32-bit limbs per numerator or
//! denominator, about 154 digits), every operation is checked, and the final value must be within
//! [`MAX_MAGNITUDE`]. The candidate is the claimed result, an integer (`-12`) or a rational
//! (`7/3`); it passes only if it equals the exact value.
//!
//! The candidate is judged by value, not by spelling: `007`, `-0` and `6/4` are accepted numerals
//! (equal to 7, 0 and 3/2); only a non-numeral (`+7`, `7.5`, unicode digits, `7/0`) is
//! `arith.format`.
//!
//! Coverage limit: intermediates may be far larger than the result (`10^20 * 10^20 / 10^20` is
//! evaluated exactly and passes for `10^20`), but the FINAL numerator and denominator must stay
//! within [`MAX_MAGNITUDE`] (10^30, the candidate bound), and no reduced intermediate may exceed
//! [`MAX_VALUE_LIMBS`] limbs; beyond either the task is `Unknown` (`arith.task`), never a wrong
//! `Pass`.
//!
//! A task the evaluator cannot decide (syntax error, oversize, nesting beyond
//! [`MAX_NESTING`], an intermediate beyond the bound, division by zero) makes the report
//! `Unknown` with the soft code `arith.task`: the verifier does not guess. A candidate problem is
//! a hard finding: `arith.format`, `arith.oversize`, `arith.value`, `arith.bounds` (see
//! [`crate::codes`]).

use ptr_types::{Probability, VerificationLevel};

use crate::bigint::{BigInt, BigIntError};
use crate::codes::finding;
use crate::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// Longest task expression, in bytes.
pub const MAX_TASK_BYTES: usize = 512;
/// Longest candidate, in bytes.
pub const MAX_CANDIDATE_BYTES: usize = 128;
/// Deepest parenthesis / unary nesting of a task.
pub const MAX_NESTING: usize = 48;
/// Largest absolute numerator or denominator of any value, task intermediate or candidate.
pub const MAX_MAGNITUDE: i128 = 1_000_000_000_000_000_000_000_000_000_000; // 10^30

/// Most 32-bit limbs of a reduced intermediate numerator or denominator (512 bits).
pub const MAX_VALUE_LIMBS: usize = 16;

/// An exact rational of `i128`s, always reduced with a positive denominator, bounded by
/// [`MAX_MAGNITUDE`]. It holds the final task value and the candidate; the evaluator's
/// intermediates are [`BigRational`]s.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rational {
    num: i128,
    den: i128,
}

/// Greatest common divisor of two magnitudes. Unsigned, so it cannot overflow or panic.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    fn new(num: i128, den: i128) -> Option<Self> {
        // `i128::MIN` has no positive counterpart. A checked product or sum can land on exactly
        // that value, so it is refused before anything takes an absolute value or negates.
        if den == 0 || num == i128::MIN || den == i128::MIN {
            return None;
        }
        // Both magnitudes fit in i128 now, so the unsigned gcd (>= 1 since den != 0) fits too.
        let g = i128::try_from(gcd(num.unsigned_abs(), den.unsigned_abs())).ok()?;
        let (mut num, mut den) = (num / g, den / g);
        if den < 0 {
            num = -num;
            den = -den;
        }
        // The bound applies to the reduced value: `2/10^18 + 3/10^18` is representable.
        if num.abs() > MAX_MAGNITUDE || den > MAX_MAGNITUDE {
            return None;
        }
        Some(Self { num, den })
    }

    #[cfg(test)]
    fn add(self, other: Self) -> Option<Self> {
        let num = self
            .num
            .checked_mul(other.den)?
            .checked_add(other.num.checked_mul(self.den)?)?;
        Self::new(num, self.den.checked_mul(other.den)?)
    }

    #[cfg(test)]
    fn mul(self, other: Self) -> Option<Self> {
        Self::new(
            self.num.checked_mul(other.num)?,
            self.den.checked_mul(other.den)?,
        )
    }

    #[cfg(test)]
    fn div(self, other: Self) -> Option<Self> {
        if other.num == 0 {
            return None;
        }
        Self::new(
            self.num.checked_mul(other.den)?,
            self.den.checked_mul(other.num)?,
        )
    }
}

/// An exact reduced rational of [`BigInt`]s with a positive denominator, at most
/// [`MAX_VALUE_LIMBS`] limbs each. Every operation is checked; none panics.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BigRational {
    num: BigInt,
    den: BigInt,
}

impl From<BigIntError> for TaskError {
    fn from(error: BigIntError) -> Self {
        match error {
            BigIntError::TooLarge | BigIntError::InvalidDecimal => Self::Overflow,
            BigIntError::DivisionByZero => Self::DivisionByZero,
        }
    }
}

impl BigRational {
    fn new(num: BigInt, den: BigInt) -> Result<Self, TaskError> {
        if den.is_zero() {
            return Err(TaskError::DivisionByZero);
        }
        // `den != 0`, so the gcd is at least 1 and both divisions are exact.
        let g = num.gcd(&den);
        let mut num = num.div_exact(&g)?.ok_or(TaskError::Overflow)?;
        let mut den = den.div_exact(&g)?.ok_or(TaskError::Overflow)?;
        if den.is_negative() {
            num = num.neg();
            den = den.neg();
        }
        if num.limbs() > MAX_VALUE_LIMBS || den.limbs() > MAX_VALUE_LIMBS {
            return Err(TaskError::Overflow);
        }
        Ok(Self { num, den })
    }

    fn int(value: BigInt) -> Result<Self, TaskError> {
        Self::new(value, BigInt::from_i128(1))
    }

    fn add(&self, other: &Self) -> Result<Self, TaskError> {
        let num = self.num.mul(&other.den)?.add(&other.num.mul(&self.den)?)?;
        Self::new(num, self.den.mul(&other.den)?)
    }

    fn sub(&self, other: &Self) -> Result<Self, TaskError> {
        let num = self.num.mul(&other.den)?.sub(&other.num.mul(&self.den)?)?;
        Self::new(num, self.den.mul(&other.den)?)
    }

    fn neg(&self) -> Self {
        Self {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    fn mul(&self, other: &Self) -> Result<Self, TaskError> {
        Self::new(self.num.mul(&other.num)?, self.den.mul(&other.den)?)
    }

    fn div(&self, other: &Self) -> Result<Self, TaskError> {
        if other.num.is_zero() {
            return Err(TaskError::DivisionByZero);
        }
        Self::new(self.num.mul(&other.den)?, self.den.mul(&other.num)?)
    }

    /// The value as a bounded [`Rational`]; beyond [`MAX_MAGNITUDE`] it is an overflow.
    fn to_small(&self) -> Result<Rational, TaskError> {
        let num = self.num.to_i128().ok_or(TaskError::Overflow)?;
        let den = self.den.to_i128().ok_or(TaskError::Overflow)?;
        Rational::new(num, den).ok_or(TaskError::Overflow)
    }
}

/// Why the task could not be evaluated.
#[derive(Clone, Debug, Eq, PartialEq)]
enum TaskError {
    Oversize,
    Syntax(usize),
    Nesting,
    Overflow,
    DivisionByZero,
}

impl TaskError {
    fn describe(&self) -> String {
        match self {
            Self::Oversize => format!("task is longer than {MAX_TASK_BYTES} bytes"),
            Self::Syntax(at) => format!("task does not parse at byte {at}"),
            Self::Nesting => format!("task is nested deeper than {MAX_NESTING}"),
            Self::Overflow => "a task intermediate exceeds the magnitude bound".into(),
            Self::DivisionByZero => "task divides by zero".into(),
        }
    }
}

struct Evaluator<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Evaluator<'_> {
    fn skip(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.bytes.get(self.pos).copied()
    }

    fn expr(&mut self, depth: usize) -> Result<BigRational, TaskError> {
        let mut value = self.term(depth)?;
        while let Some(op @ (b'+' | b'-')) = self.peek() {
            self.pos += 1;
            let rhs = self.term(depth)?;
            value = if op == b'-' {
                value.sub(&rhs)?
            } else {
                value.add(&rhs)?
            };
        }
        Ok(value)
    }

    fn term(&mut self, depth: usize) -> Result<BigRational, TaskError> {
        let mut value = self.unary(depth)?;
        while let Some(op @ (b'*' | b'/')) = self.peek() {
            self.pos += 1;
            let rhs = self.unary(depth)?;
            value = if op == b'*' {
                value.mul(&rhs)?
            } else {
                value.div(&rhs)?
            };
        }
        Ok(value)
    }

    fn unary(&mut self, depth: usize) -> Result<BigRational, TaskError> {
        if depth >= MAX_NESTING {
            return Err(TaskError::Nesting);
        }
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                self.unary(depth + 1).map(|value| value.neg())
            }
            Some(b'+') => {
                self.pos += 1;
                self.unary(depth + 1)
            }
            Some(b'(') => {
                self.pos += 1;
                let value = self.expr(depth + 1)?;
                if self.peek() != Some(b')') {
                    return Err(TaskError::Syntax(self.pos));
                }
                self.pos += 1;
                Ok(value)
            }
            Some(b'0'..=b'9') => self.literal(),
            _ => Err(TaskError::Syntax(self.pos)),
        }
    }

    fn literal(&mut self) -> Result<BigRational, TaskError> {
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        let digits = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| TaskError::Syntax(start))?;
        let value = BigInt::parse_decimal(digits).map_err(|error| match error {
            BigIntError::InvalidDecimal => TaskError::Syntax(start),
            other => TaskError::from(other),
        })?;
        BigRational::int(value)
    }
}

fn evaluate_task(task: &str) -> Result<Rational, TaskError> {
    if task.len() > MAX_TASK_BYTES {
        return Err(TaskError::Oversize);
    }
    let mut evaluator = Evaluator {
        bytes: task.as_bytes(),
        pos: 0,
    };
    let value = evaluator.expr(0)?;
    if evaluator.peek().is_some() {
        return Err(TaskError::Syntax(evaluator.pos));
    }
    value.to_small()
}

enum CandidateError {
    Format(&'static str),
    Bounds,
}

fn parse_integer(text: &str) -> Result<i128, CandidateError> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(CandidateError::Format("not an integer"));
    }
    let mut value: i128 = 0;
    for digit in digits.bytes() {
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(i128::from(digit - b'0')))
            .filter(|v| *v <= MAX_MAGNITUDE)
            .ok_or(CandidateError::Bounds)?;
    }
    Ok(if negative { -value } else { value })
}

fn parse_candidate(text: &str) -> Result<Rational, CandidateError> {
    let text = text.trim_matches(|c: char| c.is_ascii_whitespace());
    let (num, den) = match text.split_once('/') {
        Some((n, d)) => (
            parse_integer(n.trim_end_matches(|c: char| c.is_ascii_whitespace()))?,
            parse_integer(d.trim_start_matches(|c: char| c.is_ascii_whitespace()))?,
        ),
        None => (parse_integer(text)?, 1),
    };
    if den <= 0 {
        return Err(CandidateError::Format("denominator is not positive"));
    }
    Rational::new(num, den).ok_or(CandidateError::Bounds)
}

/// Verifies a claimed result against one arithmetic task. See the module documentation.
#[derive(Clone, Debug)]
pub struct ArithmeticVerifier {
    task: String,
    expected: Result<Rational, TaskError>,
}

impl ArithmeticVerifier {
    /// A verifier for `task`. The task is evaluated once, here; an undecidable task is reported
    /// as `Unknown` by every [`verify`](Verifier::verify).
    pub fn new(task: &str) -> Self {
        Self {
            task: task.to_owned(),
            expected: evaluate_task(task),
        }
    }

    pub fn task(&self) -> &str {
        &self.task
    }
}

fn report(status: VerificationStatus, findings: Vec<Finding>) -> VerificationReport {
    let score = if status == VerificationStatus::Pass {
        1.0
    } else {
        0.0
    };
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

impl Verifier<String> for ArithmeticVerifier {
    fn verify(&self, candidate: &String) -> VerificationReport {
        let expected = match &self.expected {
            Ok(expected) => *expected,
            Err(error) => {
                return report(
                    VerificationStatus::Unknown,
                    vec![finding("arith.task", error.describe())],
                );
            }
        };
        if candidate.len() > MAX_CANDIDATE_BYTES {
            return report(
                VerificationStatus::Fail,
                vec![finding(
                    "arith.oversize",
                    format!(
                        "candidate is {} bytes, the bound is {MAX_CANDIDATE_BYTES}",
                        candidate.len()
                    ),
                )],
            );
        }
        match parse_candidate(candidate) {
            Err(CandidateError::Format(why)) => report(
                VerificationStatus::Fail,
                vec![finding(
                    "arith.format",
                    format!("candidate must be an integer or a/b: {why}"),
                )],
            ),
            Err(CandidateError::Bounds) => report(
                VerificationStatus::Fail,
                vec![finding(
                    "arith.bounds",
                    "candidate magnitude exceeds 10^30".into(),
                )],
            ),
            Ok(claimed) if claimed == expected => report(VerificationStatus::Pass, Vec::new()),
            Ok(claimed) => {
                // Compare claimed - expected without dividing: both denominators are positive.
                let direction = match (
                    claimed.num.checked_mul(expected.den),
                    expected.num.checked_mul(claimed.den),
                ) {
                    (Some(l), Some(r)) if l > r => "greater than",
                    (Some(_), Some(_)) => "less than",
                    _ => "different from",
                };
                report(
                    VerificationStatus::Fail,
                    vec![finding(
                        "arith.value",
                        format!("claimed value is {direction} the exact value"),
                    )],
                )
            }
        }
    }
}

impl NamedVerifier<String> for ArithmeticVerifier {
    fn name(&self) -> &'static str {
        "arithmetic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(task: &str, candidate: &str) -> VerificationReport {
        ArithmeticVerifier::new(task).verify(&candidate.to_owned())
    }

    fn codes(report: &VerificationReport) -> Vec<&str> {
        report.findings.iter().map(|f| f.code.as_str()).collect()
    }

    #[test]
    fn exact_integer_and_rational_values_pass() {
        for (task, candidate) in [
            ("1 + 2 * 3", "7"),
            ("(1 + 2) * 3", "9"),
            ("7 / 3", "7/3"),
            ("1/2 + 1/3", "5/6"),
            ("6 / 4", " 3 / 2 "),
            ("-(2 - 5) * 2", "6"),
            ("10 - 4 - 3", "3"),
            ("2 * 3 / 4", "3/2"),
            ("0 / 5", "0"),
            ("--3", "3"),
        ] {
            let report = check(task, candidate);
            assert_eq!(
                report.status,
                VerificationStatus::Pass,
                "{task} = {candidate}"
            );
            assert_eq!(report.level, VerificationLevel::Deterministic);
            assert_eq!(report.score.get(), 1.0);
            assert!(report.findings.is_empty());
        }
    }

    #[test]
    fn an_equal_unreduced_rational_is_the_same_value() {
        assert_eq!(check("3/2", "6/4").status, VerificationStatus::Pass);
    }

    #[test]
    fn leading_zeros_and_negative_zero_are_accepted_numerals() {
        // Decision (P2-5): a candidate is judged by value, not by canonical spelling. `007` is
        // 7, `-0` and `-0/5` are 0; none of them is an `arith.format` finding.
        assert_eq!(check("3 + 4", "007").status, VerificationStatus::Pass);
        assert_eq!(check("3 + 4", "-007/001").status, VerificationStatus::Fail);
        assert_eq!(check("1 - 1", "-0").status, VerificationStatus::Pass);
        assert_eq!(check("1 - 1", "-0/5").status, VerificationStatus::Pass);
        assert_eq!(check("1 - 1", "000").status, VerificationStatus::Pass);
        assert_eq!(check("14/2", "0014/02").status, VerificationStatus::Pass);
    }

    #[test]
    fn i128_min_never_panics_in_any_operation() {
        // Run under overflow checks (any `cargo test` build): 2^64 * 2^63 = 2^127 = -i128::MIN.
        let task = "-18446744073709551616 * 9223372036854775808";
        let report = check(task, "1");
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["arith.task"]);
        let report = check("18446744073709551616 * -9223372036854775808", "1");
        assert_eq!(codes(&report), ["arith.task"]);
        // The same value reached by add and div directly on the rational type.
        let half = Rational {
            num: -(1_i128 << 126),
            den: 1,
        };
        assert_eq!(half.add(half), None);
        let a = Rational {
            num: -(1_i128 << 64),
            den: 1,
        };
        let b = Rational {
            num: 1,
            den: 1 << 63,
        };
        assert_eq!(a.div(b), None);
        assert_eq!(
            a.mul(Rational {
                num: 1 << 63,
                den: 1
            }),
            None
        );
        assert_eq!(Rational::new(i128::MIN, 1), None);
        assert_eq!(Rational::new(1, i128::MIN), None);
        assert_eq!(Rational::new(i128::MIN, i128::MIN), None);
        assert_eq!(Rational::new(i128::MAX, 1), None);
        assert_eq!(Rational::new(0, i128::MIN), None);
        assert_eq!(Rational::new(0, -5), Some(Rational { num: 0, den: 1 }));
    }

    #[test]
    fn the_bound_applies_to_the_reduced_value() {
        assert_eq!(
            check(
                "2/1000000000000000000 + 3/1000000000000000000",
                "1/200000000000000000"
            )
            .status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn wrong_value_is_a_hard_value_finding_with_a_direction() {
        let report = check("1 + 2 * 3", "9");
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["arith.value"]);
        assert!(report.findings[0].hard);
        assert!(report.findings[0].message.contains("greater than"));
        assert!(check("1 + 2 * 3", "6").findings[0]
            .message
            .contains("less than"));
        // The message never states the exact value.
        assert!(!check("1 + 2 * 3", "9").findings[0].message.contains('7'));
        assert_eq!(codes(&check("1/2", "1")), ["arith.value"]);
    }

    #[test]
    fn unparsable_candidate_is_a_format_finding() {
        for candidate in [
            "", "abc", "7.5", "1e3", "+7", "7/", "/7", "7/0", "7/-2", "1 2", "--7", "0x10", "٣",
            "7/3/2", "(7)",
        ] {
            let report = check("7", candidate);
            assert_eq!(report.status, VerificationStatus::Fail, "{candidate:?}");
            assert_eq!(codes(&report), ["arith.format"], "{candidate:?}");
            assert!(report.findings[0].hard);
        }
    }

    #[test]
    fn out_of_range_candidate_is_a_bounds_finding() {
        let huge = "9".repeat(31);
        for candidate in [
            huge.clone(),
            format!("-{huge}"),
            format!("1/{huge}"),
            "9".repeat(120),
        ] {
            let report = check("7", &candidate);
            assert_eq!(codes(&report), ["arith.bounds"], "{candidate}");
            assert_eq!(report.status, VerificationStatus::Fail);
        }
        // Exactly at the bound the value is merely wrong.
        assert_eq!(
            codes(&check("7", "1000000000000000000000000000000")),
            ["arith.value"]
        );
    }

    #[test]
    fn oversize_candidate_is_refused_as_a_finding() {
        let report = check("7", &"7".repeat(MAX_CANDIDATE_BYTES + 1));
        assert_eq!(report.status, VerificationStatus::Fail);
        assert_eq!(codes(&report), ["arith.oversize"]);
        assert!(report.findings[0].hard);
        assert_eq!(
            check("7", &"0".repeat(MAX_CANDIDATE_BYTES)).status,
            VerificationStatus::Fail
        );
    }

    #[test]
    fn undecidable_tasks_are_unknown_never_a_guess() {
        let deep_open = "(".repeat(MAX_NESTING + 5);
        let deep = format!("{deep_open}1{}", ")".repeat(MAX_NESTING + 5));
        let deep_unary = format!("{}1", "-".repeat(MAX_NESTING + 5));
        let wide_power = ["1000000"; 6].join("*");
        for task in [
            "",
            "1 +",
            "1 + * 2",
            "(1",
            "1)",
            "1.5 + 1",
            "2 ** 3",
            "1 / 0",
            "1 / (2 - 2)",
            "x + 1",
            &"1+".repeat(MAX_TASK_BYTES),
            &deep,
            &deep_unary,
            &wide_power,
            "99999999999999999999999999999999999999999",
        ] {
            let report = check(task, "1");
            assert_eq!(report.status, VerificationStatus::Unknown, "{task:?}");
            assert_eq!(codes(&report), ["arith.task"], "{task:?}");
            assert!(!report.findings[0].hard);
            assert_eq!(report.score.get(), 0.0);
        }
    }

    #[test]
    fn deep_but_allowed_nesting_and_large_values_evaluate() {
        let task = format!("{}1{}", "(".repeat(40), ")".repeat(40));
        assert_eq!(check(&task, "1").status, VerificationStatus::Pass);
        assert_eq!(
            check(
                "1000000 * 1000000 * 1000000 * 1000000 * 1000000",
                "1000000000000000000000000000000"
            )
            .status,
            VerificationStatus::Pass
        );
    }

    #[test]
    fn intermediates_beyond_i128_are_exact_when_the_result_is_in_range() {
        let e20 = "100000000000000000000";
        let task = "100000000000000000000 * 100000000000000000000 / 100000000000000000000";
        let report = check(task, e20);
        assert_eq!(report.status, VerificationStatus::Pass);
        assert!(report.findings.is_empty());
        // Wrong candidates are hard value findings, not Unknown.
        for wrong in [
            "100000000000000000001",
            "99999999999999999999",
            "1",
            "-100000000000000000000",
        ] {
            let report = check(task, wrong);
            assert_eq!(report.status, VerificationStatus::Fail, "{wrong}");
            assert_eq!(codes(&report), ["arith.value"], "{wrong}");
        }
        // Rational results through huge intermediates reduce exactly.
        let task = "100000000000000000000 * 100000000000000000000 / (3 * 100000000000000000000)";
        assert_eq!(
            check(task, "100000000000000000000/3").status,
            VerificationStatus::Pass
        );
        // (10^25 + 1)^2 - 10^50 - 2 * 10^25 = 1, with intermediates of 10^50.
        let big = "10000000000000000000000001";
        let task = format!("{big} * {big} - 10000000000000000000000000 * 10000000000000000000000000 - 2 * 10000000000000000000000000");
        assert_eq!(check(&task, "1").status, VerificationStatus::Pass);
    }

    #[test]
    fn a_final_value_beyond_the_candidate_bound_is_still_unknown() {
        // 10^40 is computed exactly but cannot be compared with a bounded candidate.
        let task = "100000000000000000000 * 100000000000000000000";
        let report = check(task, "1");
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["arith.task"]);
        assert!(report.findings[0].message.contains("magnitude bound"));
    }

    #[test]
    fn reduced_intermediates_beyond_the_limb_bound_are_unknown() {
        let hundred_digits = format!("1{}", "0".repeat(100));
        let task = format!("{hundred_digits} * {hundred_digits} / {hundred_digits}");
        let report = check(&task, "1");
        assert_eq!(report.status, VerificationStatus::Unknown);
        assert_eq!(codes(&report), ["arith.task"]);
        // A literal beyond the integer kernel's own bound is refused as well, not allocated.
        let huge = "9".repeat(500);
        assert_eq!(check(&huge, "1").status, VerificationStatus::Unknown);
        // Division by a huge zero-valued expression is still a division by zero.
        let report = check(
            &format!("{hundred_digits} / ({hundred_digits} - {hundred_digits})"),
            "1",
        );
        assert_eq!(codes(&report), ["arith.task"]);
        assert!(report.findings[0].message.contains("divides by zero"));
    }

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn small(&mut self) -> i128 {
            i128::from(self.next() % 2001) - 1000
        }
    }

    #[test]
    fn random_integer_expressions_agree_with_i128() {
        let mut rng = XorShift(0x1234_5678_9ABC_DEF1);
        for _ in 0..1500 {
            let (a, b, c, d) = (rng.small(), rng.small(), rng.small(), rng.small());
            // Spelled with parentheses for negatives: the grammar has unary minus.
            let task = format!("(({a}) * ({b}) + ({c})) * ({d}) - ({a}) * ({d}) + ({b})");
            let expected = (a * b + c) * d - a * d + b;
            assert_eq!(
                check(&task, &expected.to_string()).status,
                VerificationStatus::Pass,
                "{task}"
            );
            let wrong = check(&task, &(expected + 1).to_string());
            assert_eq!(codes(&wrong), ["arith.value"], "{task}");
            if b != 0 {
                // a * b / b == a exactly, and a / b matches the reduced rational.
                let task = format!("({a}) * ({b}) / ({b})");
                assert_eq!(
                    check(&task, &a.to_string()).status,
                    VerificationStatus::Pass,
                    "{task}"
                );
                let g = {
                    let (mut x, mut y) = (a.unsigned_abs(), b.unsigned_abs());
                    while y != 0 {
                        (x, y) = (y, x % y);
                    }
                    i128::try_from(x).unwrap().max(1)
                };
                let (mut n, mut d) = (a / g, b / g);
                if d < 0 {
                    (n, d) = (-n, -d);
                }
                let task = format!("({a}) / ({b})");
                assert_eq!(
                    check(&task, &format!("{n}/{d}")).status,
                    VerificationStatus::Pass,
                    "{task}"
                );
            }
        }
    }

    #[test]
    fn verification_is_deterministic() {
        let verifier = ArithmeticVerifier::new("3 * (4 + 5) / 7");
        for candidate in ["27/7", "4", "x", ""] {
            let candidate = candidate.to_owned();
            assert_eq!(verifier.verify(&candidate), verifier.verify(&candidate));
        }
        assert_eq!(NamedVerifier::<String>::name(&verifier), "arithmetic");
        assert_eq!(verifier.task(), "3 * (4 + 5) / 7");
    }
}
