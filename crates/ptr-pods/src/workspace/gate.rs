//! The acceptance gate.

use ptr_types::VerificationLevel;
use ptr_verifier::{VerificationReport, VerificationStatus};

/// Why a verification report does not open the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GateRefusal {
    /// The status was not `Pass`: `Fail`, `Unknown` and `Disputed` never occupy a slot.
    NotPassed(VerificationStatus),
    /// A `Pass` that carries a hard finding contradicts itself and is refused.
    HardFinding,
    /// The report's level is shallower than required.
    BelowRequiredLevel {
        required: VerificationLevel,
        reached: VerificationLevel,
    },
}

/// The acceptance gate: `Ok` only for status `Pass`, no hard finding and a level of at least
/// `required`. This is the rule of `Candidate::passes` in `ptr-feedback`, restated here so that
/// the crate does not depend on it at runtime; `tests/workspace.rs` pins it against
/// `Candidate::passes` itself (a dev-dependency) over the full matrix. The report's score is
/// never read (invariant 11).
pub fn check_verification_gate(
    report: &VerificationReport,
    required: VerificationLevel,
) -> Result<(), GateRefusal> {
    if report.status != VerificationStatus::Pass {
        return Err(GateRefusal::NotPassed(report.status));
    }
    if report.findings.iter().any(|finding| finding.hard) {
        return Err(GateRefusal::HardFinding);
    }
    if level_rank(report.level) < level_rank(required) {
        return Err(GateRefusal::BelowRequiredLevel {
            required,
            reached: report.level,
        });
    }
    Ok(())
}

pub(super) fn level_rank(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

pub(super) fn status_code(status: VerificationStatus) -> u8 {
    match status {
        VerificationStatus::Pass => 0,
        VerificationStatus::Fail => 1,
        VerificationStatus::Disputed => 2,
        VerificationStatus::Unknown => 3,
    }
}
