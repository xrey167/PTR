//! The independent oracle of F001: the contract the task must meet, the defects an action has,
//! the report the verifier fabric must give for it, and what the harness accepts. Nothing here
//! calls or copies `ptr-verifier` or `ptr-feedback`; the contract is restated from the task's
//! definition with `std` alone, and the harness's own ranking of statuses and levels decides
//! what is acceptable.

use ptr_types::{ActionIr, CapabilityId, Effect, TypeId, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

use super::params::{SLOTS, SLOT_CODES};

pub const TARGET: &str = "note";
pub const OPERATION: &str = "create";
pub const CAPABILITY: &str = "note.create";
pub const INPUT_TYPE: &str = "note.v1";
pub const MAX_PAYLOAD_BYTES: usize = 8;

pub const LYING_CODE: &str = "lying-verifier";

/// The bit of a defect slot in a defect mask.
pub const fn bit(slot: usize) -> u8 {
    1 << slot
}

pub const ALL_SLOTS: u8 = (1 << SLOTS) - 1;

/// The defects of `action` against the contract, one bit per slot, by the harness's own check.
pub fn defect_mask(action: &ActionIr) -> u8 {
    let mut mask = 0;
    if action.target != TARGET {
        mask |= bit(0);
    }
    if action.operation != OPERATION {
        mask |= bit(1);
    }
    if action.capability != CapabilityId::from(CAPABILITY) {
        mask |= bit(2);
    }
    if action.input_type != TypeId::from(INPUT_TYPE) {
        mask |= bit(3);
    }
    if action.effect != Effect::Mutation {
        mask |= bit(4);
    }
    if action.payload.len() > MAX_PAYLOAD_BYTES {
        mask |= bit(5);
    }
    if std::str::from_utf8(&action.payload).is_err() {
        mask |= bit(6);
    }
    mask
}

/// The hard-finding codes a mask must produce, in slot order.
pub fn codes_of(mask: u8) -> Vec<&'static str> {
    (0..SLOTS)
        .filter(|slot| mask & bit(*slot) != 0)
        .map(|slot| SLOT_CODES[slot])
        .collect()
}

/// What the scripted stubs of the fabric do on one verifier call. The mode is drawn per call
/// from the seed; the contract verifier answers from the action whatever the mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Clean,
    /// A stub answers `Unknown` at `LatentAgreement`.
    Unknown,
    /// A stub answers `Disputed` at `Deterministic`.
    Disputed,
    /// A stub answers `Pass` at `LatentAgreement`: shallower than required.
    Shallow,
    /// A stub answers `Pass` with a hard finding.
    Lying,
    /// The fabric's `Fail` is rewritten into a `Pass` at `Deterministic` that keeps its hard
    /// findings: a verdict that contradicts itself, past the fabric.
    Forged,
}

impl Mode {
    pub const ALL: [Mode; 6] = [
        Self::Clean,
        Self::Unknown,
        Self::Disputed,
        Self::Shallow,
        Self::Lying,
        Self::Forged,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Unknown => "unknown",
            Self::Disputed => "disputed",
            Self::Shallow => "shallow",
            Self::Lying => "lying",
            Self::Forged => "forged",
        }
    }

    pub fn code(self) -> u8 {
        match self {
            Self::Clean => 0,
            Self::Unknown => 1,
            Self::Disputed => 2,
            Self::Shallow => 3,
            Self::Lying => 4,
            Self::Forged => 5,
        }
    }

    pub fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Unknown,
            2 => Self::Disputed,
            3 => Self::Shallow,
            4 => Self::Lying,
            5 => Self::Forged,
            _ => Self::Clean,
        }
    }
}

/// The report the fabric must give for an action with defects `mask` on a call in `mode`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Expected {
    pub status: VerificationStatus,
    pub level: VerificationLevel,
    /// The hard findings' codes in report order.
    pub hard: Vec<&'static str>,
}

pub fn expected(mask: u8, mode: Mode) -> Expected {
    let mut hard = codes_of(mask);
    if mode == Mode::Lying {
        hard.push(LYING_CODE);
    }
    let level = match mode {
        Mode::Unknown | Mode::Shallow => VerificationLevel::LatentAgreement,
        Mode::Clean | Mode::Disputed | Mode::Lying | Mode::Forged => {
            VerificationLevel::Deterministic
        }
    };
    let status = if !hard.is_empty() {
        if mode == Mode::Forged {
            VerificationStatus::Pass
        } else {
            VerificationStatus::Fail
        }
    } else {
        match mode {
            Mode::Unknown => VerificationStatus::Unknown,
            Mode::Disputed => VerificationStatus::Disputed,
            Mode::Clean | Mode::Shallow | Mode::Lying | Mode::Forged => VerificationStatus::Pass,
        }
    };
    Expected {
        status,
        level,
        hard,
    }
}

/// The hard-finding codes of a report, in report order.
pub fn hard_codes(report: &VerificationReport) -> Vec<String> {
    report
        .findings
        .iter()
        .filter(|finding| finding.hard)
        .map(|finding| finding.code.clone())
        .collect()
}

/// Whether `report` matches what the fabric must give.
pub fn matches_expected(report: &VerificationReport, expected: &Expected) -> bool {
    report.status == expected.status
        && report.level == expected.level
        && hard_codes(report) == expected.hard
}

pub fn level_rank(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

/// What the harness accepts: a `Pass`, with no hard finding, at `Deterministic` depth.
pub fn acceptable(report: &VerificationReport) -> bool {
    report.status == VerificationStatus::Pass
        && report.findings.iter().all(|finding| !finding.hard)
        && level_rank(report.level) >= level_rank(VerificationLevel::Deterministic)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{Generation, Revision};

    fn clean() -> ActionIr {
        ActionIr {
            operation: OPERATION.into(),
            target: TARGET.into(),
            capability: CapabilityId::from(CAPABILITY),
            effect: Effect::Mutation,
            input_type: TypeId::from(INPUT_TYPE),
            generation: Generation(0),
            revision: Revision(0),
            payload: b"hello".to_vec(),
        }
    }

    #[test]
    fn each_slot_has_its_own_defect_bit() {
        assert_eq!(defect_mask(&clean()), 0);
        type Damage = fn(&mut ActionIr);
        let cases: [(usize, Damage); 7] = [
            (0, |a| a.target = "x".into()),
            (1, |a| a.operation = "x".into()),
            (2, |a| a.capability = CapabilityId::from("x")),
            (3, |a| a.input_type = TypeId::from("x")),
            (4, |a| a.effect = Effect::Read),
            (5, |a| a.payload = vec![b'a'; 9]),
            (6, |a| a.payload = vec![0xff]),
        ];
        for (slot, damage) in cases {
            let mut action = clean();
            damage(&mut action);
            assert_eq!(defect_mask(&action), bit(slot), "{}", SLOT_CODES[slot]);
        }
        let mut both = clean();
        both.payload = vec![0xff; 9];
        assert_eq!(defect_mask(&both), bit(5) | bit(6));
        assert_eq!(codes_of(bit(5) | bit(6)), ["payload-size", "payload-utf8"]);
    }

    #[test]
    fn the_expected_reports_follow_the_fabric_rules() {
        use VerificationStatus as S;
        let clean = |mode| expected(0, mode);
        assert_eq!(clean(Mode::Clean).status, S::Pass);
        assert_eq!(clean(Mode::Unknown).status, S::Unknown);
        assert_eq!(clean(Mode::Disputed).status, S::Disputed);
        assert_eq!(
            clean(Mode::Shallow).level,
            VerificationLevel::LatentAgreement
        );
        assert_eq!(clean(Mode::Lying).status, S::Fail);
        assert_eq!(clean(Mode::Lying).hard, [LYING_CODE]);
        assert_eq!(clean(Mode::Forged), clean(Mode::Clean), "nothing to forge");
        let broken = |mode| expected(bit(0) | bit(4), mode);
        for mode in Mode::ALL {
            let want = if mode == Mode::Forged {
                S::Pass
            } else {
                S::Fail
            };
            assert_eq!(broken(mode).status, want, "{mode:?}");
        }
        assert_eq!(broken(Mode::Lying).hard, ["target", "effect", LYING_CODE]);
    }

    #[test]
    fn modes_round_trip_through_their_codes() {
        for mode in Mode::ALL {
            assert_eq!(Mode::from_code(mode.code()), mode);
        }
    }
}
