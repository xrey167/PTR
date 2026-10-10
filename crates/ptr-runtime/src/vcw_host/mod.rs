//! The host side of the verifier-closed loop: the part that holds the runtime and the
//! admission grant, and so the only part that can change the expert structure.
//!
//! `crate::vcw` composes the loop and is kept free of any handle that could admit, fence or
//! commit (a source scan pins that). This module is where a host does those things, and only
//! through the existing admission path:
//!
//! * [`StructureDriver`] closes the loop's windows into structural requests, completes each with
//!   a specialist from a [`SpecialistTrainer`], measures a [`HeldOutSuite`] before and after
//!   the change, binds a `ForgettingGate` report to the proposal and calls
//!   `PtrRuntime::admit_expert_proposal`;
//! * [`SimulatedTrainer`] is the CPU experiments' stand-in for training: a seeded skill table,
//!   declared as such in every specialist it produces;
//! * [`SimulatedHeldOut`] runs real loops over verified held-out defect tasks. It is told only a
//!   candidate's identity; its simulated world (true skill, collateral) is configuration, and
//!   its digest is the evaluation manifest of the bound gate subject;
//! * [`rebuild_roster`] reproduces the loop's roster from the ledger-derived expert registry and
//!   the host's manifests, which is what a restart or a replay does.

mod driver;
mod holdout;
mod roster;
mod specialist;

pub use driver::{
    AbandonReason, AbandonedInstall, DriverError, DriverHost, DriverReport, InstallRetry,
    PendingInstall, Refusal, StructureDriver, StructureOutcome, StructureResult, MAX_PENDING,
};
pub use holdout::{
    check_coverage, gate_inputs, HeldOutSuite, MeasurementError, RosterChange, RosterView,
    SimulatedHeldOut, SpecialistIdentity, SuiteError,
};
pub use roster::{rebuild_roster, RosterRebuildError};
pub use specialist::{
    repaired_codes, Collateral, SimulatedTrainer, SpecialistTrainer, TrainedSpecialist,
    TrainingError, TrainingRequest, REPAIRS_PREFIX,
};
