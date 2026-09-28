//! Semantic grants for tests: a permissive verifier whose name every fixture
//! record carries, a verifier from a closure, a verifier whose report a test
//! scripts, and a host write under them.
//!
//! Included with `#[path]` by each test file that writes semantic state, so it
//! stays test code: a library feature would be unified across the workspace
//! and let the compile-fail doctests compile.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::SemanticCommit;
use ptr_runtime::{PtrRuntime, RuntimeError, SemanticChange, SemanticGrant};
use ptr_semdb::SemanticDelta;
use ptr_types::{PrincipalId, Probability, Revision, VerificationLevel};
use ptr_verifier::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

/// The name [`AcceptAll`] records, so a permissive verifier is visible in
/// every fixture record it admits.
pub const ACCEPT_ALL: &str = "test-accept-all";

/// A report with `status` at `level`, full score and no finding.
pub fn report(status: VerificationStatus, level: VerificationLevel) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(1.0).unwrap(),
        findings: Vec::new(),
    }
}

/// A passing report at `level`.
pub fn pass(level: VerificationLevel) -> VerificationReport {
    report(VerificationStatus::Pass, level)
}

/// A finding with `code`, hard or soft.
pub fn finding(code: &str, hard: bool) -> Finding {
    Finding {
        code: code.into(),
        message: format!("message for {code}"),
        hard,
    }
}

/// Passes every change at the deterministic level.
pub struct AcceptAll;

impl<'a> Verifier<SemanticChange<'a>> for AcceptAll {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        pass(VerificationLevel::Deterministic)
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for AcceptAll {
    fn name(&self) -> &'static str {
        ACCEPT_ALL
    }
}

/// A verifier that judges with a closure.
pub struct FnVerifier<F> {
    name: &'static str,
    judge: F,
}

impl<F> FnVerifier<F>
where
    F: Fn(&SemanticChange<'_>) -> VerificationReport + Send + Sync,
{
    pub fn new(name: &'static str, judge: F) -> Self {
        Self { name, judge }
    }
}

impl<'a, F> Verifier<SemanticChange<'a>> for FnVerifier<F>
where
    F: Fn(&SemanticChange<'_>) -> VerificationReport + Send + Sync,
{
    fn verify(&self, change: &SemanticChange<'a>) -> VerificationReport {
        (self.judge)(change)
    }
}

impl<'a, F> NamedVerifier<SemanticChange<'a>> for FnVerifier<F>
where
    F: Fn(&SemanticChange<'_>) -> VerificationReport + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }
}

/// A verifier that returns whatever report its test last set and counts how
/// often it was asked; clones share both, so a test keeps one to script the
/// verifier it installed.
#[derive(Clone)]
pub struct Scripted {
    name: &'static str,
    report: Arc<Mutex<VerificationReport>>,
    calls: Arc<AtomicUsize>,
}

impl Scripted {
    pub fn new(name: &'static str, report: VerificationReport) -> Self {
        Self {
            name,
            report: Arc::new(Mutex::new(report)),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn set(&self, report: VerificationReport) {
        *self.report.lock().unwrap() = report;
    }

    /// How many changes it has judged.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl<'a> Verifier<SemanticChange<'a>> for Scripted {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.report.lock().unwrap().clone()
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for Scripted {
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Install a grant that requires the deterministic level of [`AcceptAll`]
/// alone and allows host writes.
pub fn granted(runtime: &mut PtrRuntime) {
    runtime
        .install_semantic_grant(
            SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(AcceptAll)
                .allow_host_writes(),
        )
        .unwrap();
}

/// The principal fixture host writes are recorded as written by.
pub fn operator() -> PrincipalId {
    PrincipalId::from("test-operator")
}

/// A host write by [`operator`] of `delta` against `expected`.
pub fn host_write(
    runtime: &mut PtrRuntime,
    expected: Revision,
    delta: SemanticDelta,
) -> Result<SemanticCommit, RuntimeError> {
    runtime.apply_verified_semantic_delta(expected, delta, &operator())
}

/// A host write by [`operator`] of `delta` against the runtime's current
/// revision.
pub fn host_write_now(
    runtime: &mut PtrRuntime,
    delta: SemanticDelta,
) -> Result<SemanticCommit, RuntimeError> {
    let expected = runtime.revision();
    host_write(runtime, expected, delta)
}

/// A runtime whose grant requires `required` of one [`Scripted`] verifier,
/// named `scripted`, that starts out returning `report`, with host writes on;
/// the verifier is returned so the test can script it.
pub fn scripted_runtime(
    required: RequiredVerification,
    report: VerificationReport,
) -> (PtrRuntime, Scripted) {
    let verifier = Scripted::new("scripted", report);
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            SemanticGrant::new(required)
                .with_verifier(verifier.clone())
                .allow_host_writes(),
        )
        .unwrap();
    (runtime, verifier)
}
