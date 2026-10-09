//! The stubs of F001: a seeded generator and repairer of `ActionIr` actions with a known skill,
//! the scripted verifier stubs that join the real `ActionContractVerifier` in the fabric, and the
//! boundary that counts every call into the fabric.
//!
//! Every random choice comes from a SplitMix stream named by (seed, profile, task, kind,
//! index), so the arms of a task see the same draws wherever they make the same request.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use ptr_feedback::{Candidate, Critique, Generator, Repairer};
use ptr_types::{
    ActionIr, CapabilityId, Effect, Generation, Probability, Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{
    ActionContract, ActionContractVerifier, Finding, NamedVerifier, PayloadEncoding,
    VerificationReport, VerificationStatus, Verifier, VerifierFabric,
};

use super::oracle::{
    bit, defect_mask, Mode, ALL_SLOTS, CAPABILITY, INPUT_TYPE, LYING_CODE, MAX_PAYLOAD_BYTES,
    OPERATION, TARGET,
};
use super::params::{
    MODE_DISPUTED_PERMILLE, MODE_FORGED_PERMILLE, MODE_LYING_PERMILLE, MODE_SHALLOW_PERMILLE,
    MODE_UNKNOWN_PERMILLE, SLOTS, SLOT_CODES,
};
use crate::experiments::rng::Rng;

/// The kinds of random stream.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Generate = 1,
    Repair = 2,
    Mode = 3,
    Selftest = 4,
}

/// The stream for the `index`-th request of `kind` in `task` of `profile` under `seed`.
pub fn stream(seed: u64, profile: usize, task: usize, kind: Kind, index: u64) -> Rng {
    let mut state = seed;
    for part in [profile as u64, task as u64, kind as u64, index] {
        state = Rng::new(state ^ part.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64();
    }
    Rng::new(state)
}

/// The share of verifier calls in each noise mode.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    pub unknown: u64,
    pub disputed: u64,
    pub shallow: u64,
    pub lying: u64,
    pub forged: u64,
}

impl Noise {
    pub fn pinned() -> Self {
        Self {
            unknown: MODE_UNKNOWN_PERMILLE,
            disputed: MODE_DISPUTED_PERMILLE,
            shallow: MODE_SHALLOW_PERMILLE,
            lying: MODE_LYING_PERMILLE,
            forged: MODE_FORGED_PERMILLE,
        }
    }

    /// The share of calls on which a defect-free candidate is not accepted.
    pub fn rejecting_permille(&self) -> u64 {
        self.unknown + self.disputed + self.shallow + self.lying
    }

    pub fn draw(&self, rng: &mut Rng) -> Mode {
        let mut u = rng.below(1000);
        for (share, mode) in [
            (self.unknown, Mode::Unknown),
            (self.disputed, Mode::Disputed),
            (self.shallow, Mode::Shallow),
            (self.lying, Mode::Lying),
            (self.forged, Mode::Forged),
        ] {
            if u < share {
                return mode;
            }
            u -= share;
        }
        Mode::Clean
    }
}

pub fn contract() -> ActionContract {
    ActionContract {
        target: TARGET.into(),
        operation: OPERATION.into(),
        capability: CapabilityId::from(CAPABILITY),
        input_type: TypeId::from(INPUT_TYPE),
        effect: Effect::Mutation,
        max_payload_bytes: MAX_PAYLOAD_BYTES,
        payload_encoding: PayloadEncoding::Utf8,
    }
}

// ---- the actions ----------------------------------------------------------------------------

const WRONG_TARGETS: [&str; 4] = ["notes", "Note", "", "other-target"];
const WRONG_OPERATIONS: [&str; 3] = ["delete", "update", "Create"];
const WRONG_CAPABILITIES: [&str; 3] = ["note.delete", "x", "note.create "];
const WRONG_INPUT_TYPES: [&str; 2] = ["note.v2", "x"];
const WRONG_EFFECTS: [Effect; 4] = [
    Effect::Pure,
    Effect::Read,
    Effect::External,
    Effect::Irreversible,
];

/// An action with no defect, stamped with `task` and the candidate's `serial`. The generation
/// and the revision are not part of the contract; the harness uses them as the candidate's
/// identity.
fn correct(task: usize, serial: u64) -> ActionIr {
    ActionIr {
        operation: OPERATION.into(),
        target: TARGET.into(),
        capability: CapabilityId::from(CAPABILITY),
        effect: Effect::Mutation,
        input_type: TypeId::from(INPUT_TYPE),
        generation: Generation(task as u64),
        revision: Revision(serial),
        payload: b"hello".to_vec(),
    }
}

/// Make `slot` of `action` right, or wrong with a variant drawn from `rng`. The payload slots
/// are not set here (see [`set_payload`]).
fn set_slot(action: &mut ActionIr, slot: usize, right: bool, rng: &mut Rng) {
    match slot {
        0 => {
            action.target = if right {
                TARGET.into()
            } else {
                (*rng.pick(&WRONG_TARGETS)).into()
            }
        }
        1 => {
            action.operation = if right {
                OPERATION.into()
            } else {
                (*rng.pick(&WRONG_OPERATIONS)).into()
            }
        }
        2 => {
            action.capability = CapabilityId::from(if right {
                CAPABILITY
            } else {
                *rng.pick(&WRONG_CAPABILITIES)
            })
        }
        3 => {
            action.input_type = TypeId::from(if right {
                INPUT_TYPE
            } else {
                *rng.pick(&WRONG_INPUT_TYPES)
            })
        }
        4 => {
            action.effect = if right {
                Effect::Mutation
            } else {
                *rng.pick(&WRONG_EFFECTS)
            }
        }
        other => unreachable!("slot {other} is a payload slot"),
    }
}

/// Write a payload with the requested defects: too long (9 to 16 bytes) or short (1 to 8), and
/// containing an invalid UTF-8 byte or not.
fn set_payload(action: &mut ActionIr, too_long: bool, bad_utf8: bool, rng: &mut Rng) {
    let len = if too_long {
        MAX_PAYLOAD_BYTES as u64 + 1 + rng.below(8)
    } else {
        1 + rng.below(MAX_PAYLOAD_BYTES as u64)
    } as usize;
    let mut bytes: Vec<u8> = (0..len).map(|_| b'a' + rng.below(26) as u8).collect();
    if bad_utf8 {
        let at = rng.index(len);
        bytes[at] = 0xff;
    }
    action.payload = bytes;
}

// ---- the log the stubs keep -------------------------------------------------------------------

/// How a candidate was produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Produced {
    Fresh,
    Repaired,
}

/// What the stubs did, kept for the harness' accounting.
#[derive(Debug, Default)]
pub struct StubLog {
    /// Per produced candidate: its serial, how it was produced and the defect mask the stub
    /// meant it to have.
    pub produced: Vec<(u64, Produced, u8)>,
    /// Every action the stubs handed out, as handed out (before the loop saw it), by serial.
    pub actions: Vec<(u64, ActionIr)>,
    /// Candidates whose labelled defects differ from the oracle's reading of the action at the
    /// moment the stub produced it: a fault of the harness, found before the product can touch
    /// the candidate.
    pub label_faults: u32,
    pub generated: u32,
    pub repairs_returned: u32,
    pub repairs_declined: u32,
    pub slot_draws: u64,
    /// Every repair: the action before and after.
    pub edits: Vec<(ActionIr, ActionIr)>,
}

pub type SharedLog = Rc<RefCell<StubLog>>;

/// A task's candidate numbering, shared by its generator and repairer.
pub type Serial = Rc<Cell<u64>>;

fn next_serial(serial: &Serial) -> u64 {
    let now = serial.get();
    serial.set(now + 1);
    now
}

/// The seeded generator: slot `s` is right with probability `p_s`.
pub struct StubGenerator {
    pub seed: u64,
    pub profile: usize,
    pub task: usize,
    pub p_permille: [u16; SLOTS],
    pub serial: Serial,
    pub log: SharedLog,
}

impl StubGenerator {
    /// A fresh action drawn from `rng`, its serial and the defect mask meant.
    pub fn draw(&self, rng: &mut Rng, serial: u64) -> (ActionIr, u8) {
        let mut action = correct(self.task, serial);
        let mut meant = 0u8;
        for slot in 0..5 {
            let right = rng.below(1000) < u64::from(self.p_permille[slot]);
            if !right {
                meant |= bit(slot);
            }
            set_slot(&mut action, slot, right, rng);
        }
        let size_right = rng.below(1000) < u64::from(self.p_permille[5]);
        let utf8_right = rng.below(1000) < u64::from(self.p_permille[6]);
        if !size_right {
            meant |= bit(5);
        }
        if !utf8_right {
            meant |= bit(6);
        }
        set_payload(&mut action, !size_right, !utf8_right, rng);
        (action, meant)
    }
}

impl Generator<ActionIr> for StubGenerator {
    fn generate(&mut self) -> ActionIr {
        let index = u64::from(self.log.borrow().generated);
        let mut rng = stream(self.seed, self.profile, self.task, Kind::Generate, index);
        let serial = next_serial(&self.serial);
        let (action, meant) = self.draw(&mut rng, serial);
        let mut log = self.log.borrow_mut();
        log.generated += 1;
        log.slot_draws += SLOTS as u64;
        log.produced.push((serial, Produced::Fresh, meant));
        log.actions.push((serial, action.clone()));
        if defect_mask(&action) != meant {
            log.label_faults += 1;
        }
        action
    }
}

/// Which slots a repairer redraws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reads {
    /// The slots the critique names (the repair arm).
    Critique,
    /// Every slot, whatever the critique says (the negative control that must edit
    /// collaterally).
    Everything,
}

/// The seeded repairer: redraws the slots it reads, each right with probability `q`; declines
/// when the critique names no slot of the task.
pub struct StubRepairer {
    pub seed: u64,
    pub profile: usize,
    pub task: usize,
    pub q_permille: u64,
    pub reads: Reads,
    pub serial: Serial,
    pub log: SharedLog,
}

impl StubRepairer {
    fn slots_named(&self, critique: &Critique) -> Vec<usize> {
        match self.reads {
            Reads::Everything => (0..SLOTS).collect(),
            Reads::Critique => (0..SLOTS)
                .filter(|slot| {
                    critique
                        .issues
                        .iter()
                        .any(|issue| issue == SLOT_CODES[*slot])
                })
                .collect(),
        }
    }
}

impl Repairer<ActionIr> for StubRepairer {
    fn repair(&mut self, previous: &Candidate<ActionIr>, critique: &Critique) -> Option<ActionIr> {
        let named = self.slots_named(critique);
        if named.is_empty() {
            self.log.borrow_mut().repairs_declined += 1;
            return None;
        }
        let index = u64::from(self.log.borrow().repairs_returned);
        let mut rng = stream(self.seed, self.profile, self.task, Kind::Repair, index);
        let serial = next_serial(&self.serial);
        let before = &previous.value;
        let mut action = before.clone();
        action.revision = Revision(serial);
        let current = defect_mask(before);
        let mut meant = current;
        for slot in named.iter().copied().filter(|slot| *slot < 5) {
            let right = rng.below(1000) < self.q_permille;
            meant = if right {
                meant & !bit(slot)
            } else {
                meant | bit(slot)
            };
            set_slot(&mut action, slot, right, &mut rng);
        }
        let payload_named = named.iter().any(|slot| *slot >= 5);
        if payload_named {
            let mut too_long = current & bit(5) != 0;
            let mut bad_utf8 = current & bit(6) != 0;
            for slot in named.iter().copied().filter(|slot| *slot >= 5) {
                let right = rng.below(1000) < self.q_permille;
                let flag = !right;
                if slot == 5 {
                    too_long = flag;
                } else {
                    bad_utf8 = flag;
                }
                meant = if right {
                    meant & !bit(slot)
                } else {
                    meant | bit(slot)
                };
            }
            set_payload(&mut action, too_long, bad_utf8, &mut rng);
        }
        let mut log = self.log.borrow_mut();
        log.repairs_returned += 1;
        log.slot_draws += named.len() as u64;
        log.produced
            .push((serial, Produced::Repaired, meant & ALL_SLOTS));
        log.edits.push((before.clone(), action.clone()));
        log.actions.push((serial, action.clone()));
        if defect_mask(&action) != meant & ALL_SLOTS {
            log.label_faults += 1;
        }
        Some(action)
    }
}

/// The rewrite arm's repairer: it never repairs.
pub struct NeverRepairs {
    pub log: SharedLog,
}

impl Repairer<ActionIr> for NeverRepairs {
    fn repair(&mut self, _: &Candidate<ActionIr>, _: &Critique) -> Option<ActionIr> {
        self.log.borrow_mut().repairs_declined += 1;
        None
    }
}

// ---- the verifier stubs and the boundary -------------------------------------------------------

/// The mode of the call in flight, set by the boundary before it calls the fabric.
pub type ModeCell = Arc<AtomicU8>;

fn pass(level: VerificationLevel, score: f32, findings: Vec<Finding>) -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Pass,
        level,
        score: Probability::new(score).expect("bounded score"),
        findings,
    }
}

/// Which stub of the fabric a [`ScriptedVerifier`] is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StubRole {
    Unknown,
    Disputed,
    Shallow,
    Lying,
}

/// A stub verifier that answers cleanly except in its own mode.
pub struct ScriptedVerifier {
    role: StubRole,
    mode: ModeCell,
}

impl ScriptedVerifier {
    pub fn new(role: StubRole, mode: &ModeCell) -> Self {
        Self {
            role,
            mode: Arc::clone(mode),
        }
    }
}

impl Verifier<ActionIr> for ScriptedVerifier {
    fn verify(&self, _: &ActionIr) -> VerificationReport {
        let mode = Mode::from_code(self.mode.load(Ordering::SeqCst));
        let clean = || pass(VerificationLevel::Deterministic, 1.0, vec![]);
        match (self.role, mode) {
            (StubRole::Unknown, Mode::Unknown) => VerificationReport {
                status: VerificationStatus::Unknown,
                level: VerificationLevel::LatentAgreement,
                score: Probability::new(0.5).expect("bounded score"),
                findings: vec![],
            },
            (StubRole::Disputed, Mode::Disputed) => VerificationReport {
                status: VerificationStatus::Disputed,
                level: VerificationLevel::Deterministic,
                score: Probability::new(0.5).expect("bounded score"),
                findings: vec![],
            },
            (StubRole::Shallow, Mode::Shallow) => {
                pass(VerificationLevel::LatentAgreement, 0.9, vec![])
            }
            (StubRole::Lying, Mode::Lying) => pass(
                VerificationLevel::Deterministic,
                1.0,
                vec![Finding {
                    code: LYING_CODE.into(),
                    message: "a verifier that passes and objects at once".into(),
                    hard: true,
                }],
            ),
            _ => clean(),
        }
    }
}

impl NamedVerifier<ActionIr> for ScriptedVerifier {
    fn name(&self) -> &'static str {
        match self.role {
            StubRole::Unknown => "stub-unknown",
            StubRole::Disputed => "stub-disputed",
            StubRole::Shallow => "stub-shallow",
            StubRole::Lying => "stub-lying",
        }
    }
}

/// The fabric of the task: the real contract verifier followed by the four scripted stubs.
pub fn build_fabric(mode: &ModeCell) -> VerifierFabric<ActionIr> {
    let mut fabric = VerifierFabric::default();
    fabric.push(ActionContractVerifier::new(contract()));
    for role in [
        StubRole::Unknown,
        StubRole::Disputed,
        StubRole::Shallow,
        StubRole::Lying,
    ] {
        fabric.push(ScriptedVerifier::new(role, mode));
    }
    fabric
}

/// One call across the boundary, as the harness saw it.
#[derive(Clone, Debug)]
pub struct Logged {
    /// The candidate's serial, read from the action's revision.
    pub serial: u64,
    /// The defects of the action by the harness' oracle.
    pub mask: u8,
    /// The action as the loop handed it to the verifier.
    pub action: ActionIr,
    pub mode: Mode,
    /// Whether the boundary rewrote a `Fail` into a `Pass`.
    pub forged: bool,
    pub report: VerificationReport,
}

/// The boundary around the fabric: it draws the call's mode, runs the fabric, forges the
/// `Forged` mode's contradictory `Pass` and keeps the log of every call. The log is the harness'
/// count of verifier calls.
pub struct Boundary {
    fabric: VerifierFabric<ActionIr>,
    mode: ModeCell,
    noise: Noise,
    seed: u64,
    profile: usize,
    task: usize,
    pub log: RefCell<Vec<Logged>>,
}

impl Boundary {
    pub fn new(seed: u64, profile: usize, task: usize, noise: Noise) -> Self {
        let mode: ModeCell = Arc::new(AtomicU8::new(Mode::Clean.code()));
        Self {
            fabric: build_fabric(&mode),
            mode,
            noise,
            seed,
            profile,
            task,
            log: RefCell::new(Vec::new()),
        }
    }
}

impl Verifier<ActionIr> for Boundary {
    fn verify(&self, action: &ActionIr) -> VerificationReport {
        let call = self.log.borrow().len() as u64;
        let mut rng = stream(self.seed, self.profile, self.task, Kind::Mode, call);
        let drawn = self.noise.draw(&mut rng);
        // The forged mode acts after the fabric; the stubs inside it stay clean.
        let inside = if drawn == Mode::Forged {
            Mode::Clean
        } else {
            drawn
        };
        self.mode.store(inside.code(), Ordering::SeqCst);
        let mut report = self.fabric.verify(action);
        let mut forged = false;
        if drawn == Mode::Forged && report.status == VerificationStatus::Fail {
            report.status = VerificationStatus::Pass;
            report.level = VerificationLevel::Deterministic;
            report.score = Probability::new(1.0).expect("bounded score");
            forged = true;
        }
        self.log.borrow_mut().push(Logged {
            serial: action.revision.0,
            mask: defect_mask(action),
            action: action.clone(),
            mode: drawn,
            forged,
            report: report.clone(),
        });
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator(seed: u64, p: u16) -> (StubGenerator, SharedLog) {
        let log: SharedLog = Rc::default();
        let generator = StubGenerator {
            seed,
            profile: 0,
            task: 3,
            p_permille: [p; SLOTS],
            serial: Rc::default(),
            log: Rc::clone(&log),
        };
        (generator, log)
    }

    #[test]
    fn streams_are_reproducible_and_distinct_by_name() {
        let first = stream(17, 0, 1, Kind::Generate, 0).next_u64();
        assert_eq!(first, stream(17, 0, 1, Kind::Generate, 0).next_u64());
        for other in [
            stream(18, 0, 1, Kind::Generate, 0),
            stream(17, 1, 1, Kind::Generate, 0),
            stream(17, 0, 2, Kind::Generate, 0),
            stream(17, 0, 1, Kind::Repair, 0),
            stream(17, 0, 1, Kind::Generate, 1),
        ] {
            let mut other = other;
            assert_ne!(first, other.next_u64());
        }
    }

    #[test]
    fn the_generator_labels_what_the_oracle_finds_and_is_deterministic() {
        let (mut a, log_a) = generator(17, 600);
        let (mut b, _) = generator(17, 600);
        for _ in 0..200 {
            let action = a.generate();
            assert_eq!(action, b.generate());
        }
        for (i, (serial, how, _)) in log_a.borrow().produced.iter().copied().enumerate() {
            assert_eq!(serial, i as u64);
            assert_eq!(how, Produced::Fresh);
        }
        // Re-draw to compare each label with the oracle.
        let (g, _) = generator(29, 500);
        for i in 0..300u64 {
            let mut rng = stream(29, 0, 3, Kind::Generate, i);
            let (action, meant) = g.draw(&mut rng, i);
            assert_eq!(defect_mask(&action), meant);
            assert_eq!(action.revision, Revision(i));
            assert_eq!(action.generation, Generation(3));
        }
    }

    #[test]
    fn a_repair_edits_only_the_named_slots_and_labels_what_it_did() {
        let log: SharedLog = Rc::default();
        let serial: Serial = Rc::default();
        let mut repairer = StubRepairer {
            seed: 5,
            profile: 0,
            task: 1,
            q_permille: 1000,
            reads: Reads::Critique,
            serial: Rc::clone(&serial),
            log: Rc::clone(&log),
        };
        let mut broken = correct(1, 0);
        broken.target = "other-target".into();
        broken.payload = vec![0xff; 10];
        let previous = Candidate {
            id: ptr_types::CandidateId::from("c0"),
            value: broken.clone(),
            verification: None,
        };
        let critique = Critique {
            candidate: previous.id.clone(),
            issues: vec!["target".into(), "payload-size".into(), LYING_CODE.into()],
            repairable: true,
        };
        let fixed = repairer.repair(&previous, &critique).expect("slots named");
        // The target and the size are repaired; the encoding defect was not named and stays.
        assert_eq!(defect_mask(&fixed), bit(6));
        assert_eq!(fixed.operation, broken.operation);
        assert_eq!(log.borrow().slot_draws, 2);
        assert_eq!(log.borrow().produced[0].2, bit(6));
        let nothing = Critique {
            issues: vec![LYING_CODE.into()],
            ..critique
        };
        assert!(repairer.repair(&previous, &nothing).is_none());
        assert_eq!(log.borrow().repairs_declined, 1);
    }

    #[test]
    fn the_boundary_logs_each_call_and_forges_only_a_fail() {
        let noise = Noise {
            unknown: 0,
            disputed: 0,
            shallow: 0,
            lying: 0,
            forged: 1000,
        };
        let boundary = Boundary::new(17, 0, 0, noise);
        let mut broken = correct(0, 7);
        broken.effect = Effect::Read;
        let report = boundary.verify(&broken);
        assert_eq!(report.status, VerificationStatus::Pass);
        assert_eq!(
            crate::experiments::f001::oracle::hard_codes(&report),
            ["effect"]
        );
        let good = boundary.verify(&correct(0, 8));
        assert_eq!(good.status, VerificationStatus::Pass);
        let log = boundary.log.borrow();
        assert_eq!(log.len(), 2);
        assert!(log[0].forged && !log[1].forged);
        assert_eq!((log[0].serial, log[1].serial), (7, 8));
        assert_eq!(log[0].mask, bit(4));
    }

    #[test]
    fn the_noise_modes_are_drawn_in_their_shares() {
        let noise = Noise::pinned();
        let mut counts = [0u64; 6];
        for i in 0..100_000u64 {
            let mut rng = stream(1, 0, 0, Kind::Selftest, i);
            counts[noise.draw(&mut rng).code() as usize] += 1;
        }
        let expect = [850u64, 40, 30, 30, 30, 20];
        for (got, want) in counts.iter().zip(expect) {
            let want = want * 100;
            assert!(got.abs_diff(want) < 400, "{counts:?}");
        }
    }
}
