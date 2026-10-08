//! C9: commit-time identity and grant revalidation. Every race is a sequenced interleaving in
//! the single-writer runtime: mint a grant (or take the early identity check), let the
//! authority change, then commit with the grant minted before. The thread race is not modelled.

use super::core_cases::event_type_of;
use super::fixture::{
    readmission_report, report_for, ReportSpec, Rt, Scenario, Setup, D1, EXPIRES_AT, KEY_ID, NOW,
    SESSION,
};
use super::{control, is, is_where, Cx, Verdict};
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{fail, Res};
use ptr_pods::StructuralOp;
use ptr_runtime::expert::{
    ExpertAdmissionError, ExpertGrantBinding, ExpertGrantKind, InputInvalidation,
};
use ptr_runtime::{ManifestAuthorityRegistry, RuntimeError};
use ptr_types::{CommitIndex, Generation, SessionId, Timestamp};

const ROTATED_KEY: &str = "r011-policy-key-2";

/// The commit paths that rest on a grant.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// The proposer's admission of a birth.
    Admission,
    /// The runtime's fence of `x`.
    Fence,
    /// The gate's readmission of the quarantined `x`.
    Gate,
    /// The runtime-wide invalidation of an input.
    Global,
}

const PATHS: [Path; 4] = [Path::Admission, Path::Fence, Path::Gate, Path::Global];

impl Path {
    fn kind(self) -> ExpertGrantKind {
        match self {
            Self::Admission => ExpertGrantKind::Admission,
            Self::Fence => ExpertGrantKind::Fence,
            Self::Gate => ExpertGrantKind::Gate,
            Self::Global => ExpertGrantKind::GlobalRevocation,
        }
    }

    /// A runtime in which this path's commit is one valid step away.
    fn setup(self) -> Res<Rt> {
        let mut rt = Rt::fresh()?;
        match self {
            Self::Admission | Self::Global => {}
            Self::Fence => rt.birth("x")?,
            Self::Gate => {
                rt.birth("x")?;
                rt.fence_quarantine("x")?;
            }
        }
        Ok(rt)
    }

    /// Commit this path's step under the grant `binding` (minted earlier).
    fn commit(self, rt: &mut Rt, binding: ExpertGrantBinding) -> Result<CommitIndex, RuntimeError> {
        let invalid = RuntimeError::InvalidConfig;
        match self {
            Self::Admission => {
                let proposal = rt
                    .proposal(StructuralOp::Birth {
                        result: rt.spec("n"),
                    })
                    .map_err(invalid)?;
                let report =
                    report_for(&proposal, None, &ReportSpec::correct()).map_err(invalid)?;
                let grant = ptr_runtime::expert::ExpertAdmissionGrant::new(
                    rt.project.clone(),
                    rt.proposer(),
                    Setup::PRE.requirement().map_err(invalid)?,
                    binding,
                );
                rt.commit_proposal(&proposal, &report, &grant, None)
            }
            Self::Fence => {
                let request = rt
                    .fence_request("x", ptr_ledger::ExpertFenceCause::AuthorityQuarantine)
                    .map_err(invalid)?;
                let grant =
                    ptr_runtime::expert::ExpertFenceGrant::revocation(rt.project.clone(), binding);
                rt.runtime.fence_expert(&request, &grant)
            }
            Self::Gate => {
                let request = rt
                    .fence_request("x", ptr_ledger::ExpertFenceCause::AuthorityQuarantine)
                    .map_err(invalid)?;
                let manifest = rt.manifest("x").map_err(invalid)?;
                let report = readmission_report(&request, manifest, &ReportSpec::correct())
                    .map_err(invalid)?;
                let grant = ptr_runtime::expert::ExpertFenceGrant::gate(
                    rt.project.clone(),
                    Setup::PRE.requirement().map_err(invalid)?,
                    binding,
                );
                rt.runtime.readmit_expert(&request, &report, &grant)
            }
            Self::Global => {
                let grant = ptr_runtime::expert::ExpertFenceGrant::global_revocation(binding);
                rt.runtime
                    .invalidate_input(
                        &grant,
                        InputInvalidation::revoked("dataset:race", Generation(1), D1),
                    )
                    .map(|outcome| outcome.index())
            }
        }
    }
}

/// The faults of the identity the grants rest on.
#[derive(Clone, Copy, Debug)]
enum Fault {
    SessionRevoked,
    SessionExpired,
    PolicyKeyRotated,
    PolicyKeyRevoked,
    PeerWithdrawn,
    PrincipalWithdrawn,
}

const FAULTS: [Fault; 6] = [
    Fault::SessionRevoked,
    Fault::SessionExpired,
    Fault::PolicyKeyRotated,
    Fault::PolicyKeyRevoked,
    Fault::PeerWithdrawn,
    Fault::PrincipalWithdrawn,
];

impl Fault {
    /// The adversarial case the fault is counted under, the metric that counts its cases and
    /// the commit-after-change violation counter.
    fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::SessionRevoked => (
                "session_revocation_race",
                "session_revocation_race_cases",
                "structural_commits_after_session_end",
            ),
            Self::SessionExpired => (
                "session_expiry_race",
                "session_expiry_race_cases",
                "structural_commits_after_session_end",
            ),
            Self::PolicyKeyRotated => (
                "identity_policy_rotation_race",
                "identity_policy_rotation_race_cases",
                "structural_commits_after_identity_dependency_change",
            ),
            Self::PolicyKeyRevoked => (
                "identity_policy_revocation_race",
                "identity_policy_revocation_race_cases",
                "structural_commits_after_identity_dependency_change",
            ),
            Self::PeerWithdrawn | Self::PrincipalWithdrawn => (
                "peer_deadmission_race",
                "peer_deadmission_race_cases",
                "structural_commits_after_identity_dependency_change",
            ),
        }
    }

    /// Let the authority change; the refusal the grant minted before must now meet.
    fn inflict(self, rt: &mut Rt) -> Res<ExpertAdmissionError> {
        Ok(match self {
            Self::SessionRevoked => {
                rt.runtime
                    .revoke_identity_session(SessionId(SESSION.into()), "operator revoked")
                    .map_err(fail("revoking the session"))?;
                ExpertAdmissionError::GrantSessionRevoked
            }
            Self::SessionExpired => {
                rt.runtime
                    .observe_host_time(Timestamp(EXPIRES_AT))
                    .map_err(fail("observing the host time"))?;
                ExpertAdmissionError::GrantIdentityInvalid
            }
            Self::PolicyKeyRotated => {
                rt.runtime.install_policy_key(
                    ROTATED_KEY,
                    ed25519_dalek::SigningKey::from_bytes(&[8; 32]).verifying_key(),
                );
                rt.runtime.revoke_policy_key(KEY_ID);
                ExpertAdmissionError::GrantPolicyKeyUntrusted
            }
            Self::PolicyKeyRevoked => {
                rt.runtime.revoke_policy_key(KEY_ID);
                ExpertAdmissionError::GrantPolicyKeyUntrusted
            }
            Self::PeerWithdrawn => {
                rt.runtime
                    .withdraw_peer(&super::fixture::peer())
                    .map_err(fail("withdrawing the peer"))?;
                ExpertAdmissionError::GrantPeerNotAdmitted
            }
            Self::PrincipalWithdrawn => {
                *rt.runtime
                    .manifest_authority_registry()
                    .write()
                    .map_err(|_| "a poisoned lock")? = ManifestAuthorityRegistry::default();
                ExpertAdmissionError::GrantPrincipalNotAdmitted
            }
        })
    }
}

pub fn c9_identity_and_grant_races(cx: &mut Cx) -> Res<()> {
    // The controls: a fresh grant commits on every path.
    for path in PATHS {
        let mut rt = path.setup()?;
        let binding = rt.binding(path.kind());
        control(
            &format!("{path:?} with a fresh grant"),
            path.commit(&mut rt, binding),
        )?;
    }
    identity_races(cx)?;
    proposer_grant_races(cx)?;
    gate_grant_races(cx)
}

fn identity_races(cx: &mut Cx) -> Res<()> {
    for fault in cx.shuffled("c9.faults", &FAULTS) {
        let (case, cases, commits) = fault.names();
        for path in PATHS {
            cx.sub(&format!("{case}_{path:?}"));
            let mut rt = path.setup()?;
            // The early identity check: the grant is minted now.
            let minted = rt.binding(path.kind());
            let expected = fault.inflict(&mut rt)?;
            // A session revocation is itself a committed record; count from after it.
            let events = rt.event_count();
            let registry = rt.runtime.expert_registry().clone();
            let verdict = cx.verdict(
                case,
                &format!("{path:?} after {fault:?}"),
                path.commit(&mut rt, minted),
                is(expected),
            )?;
            cx.count(cases, 1);
            cx.count(commits, verdict.violation());
            let unchanged = rt.event_count() == events && rt.runtime.expert_registry() == &registry;
            cx.count("uncommitted_effective_changes", i64::from(!unchanged));
            if let Fault::PolicyKeyRotated = fault {
                // The control: a grant minted under the new key commits.
                let renewed = ExpertGrantBinding::new(
                    super::fixture::identity(),
                    super::fixture::peer(),
                    NOW,
                    ROTATED_KEY,
                    rt.runtime.expert_grant_generation(path.kind()),
                );
                control(
                    &format!("{path:?} under the rotated key"),
                    path.commit(&mut rt, renewed),
                )?;
            }
        }
    }
    Ok(())
}

/// The proposer's grant superseded between mint and commit, per operation kind.
fn proposer_grant_races(cx: &mut Cx) -> Res<()> {
    let kinds = [
        ("Birth", Scenario::Birth),
        ("Merge", Scenario::Merge),
        ("Split", Scenario::Split),
        ("Freeze", Scenario::Freeze),
        ("Reactivate", Scenario::ReactivateFrozen),
        ("Prune", Scenario::Prune),
        ("Retire", Scenario::Retire),
    ];
    for (kind, scenario) in cx.shuffled("c9.proposer_kinds", &kinds) {
        cx.sub(&format!("proposer_grant_revocation_{kind}"));
        let mut rt = Rt::fresh()?;
        rt.seed(scenario)?;
        let op = rt.operation(scenario, &super::fixture::exact)?;
        let proposal = rt.proposal(op)?;
        let report = report_for(&proposal, None, &ReportSpec::correct())?;
        let minted = rt.admission_grant(Setup::PRE)?;
        rt.runtime
            .supersede_expert_grant(ExpertGrantKind::Admission)
            .map_err(fail("superseding the admission grant"))?;
        let (events, registry) = (rt.event_count(), rt.runtime.expert_registry().clone());
        let verdict = cx.verdict(
            "proposer_grant_revocation_race",
            &format!("a {kind} proposal under a superseded grant"),
            rt.commit_proposal(&proposal, &report, &minted, None),
            is_where(|e| {
                matches!(
                    e,
                    ExpertAdmissionError::GrantGenerationStale {
                        kind: ExpertGrantKind::Admission,
                        ..
                    }
                )
            }),
        )?;
        cx.count_in("proposer_grant_revocation_race_cases", kind, 1);
        cx.count("revoked_proposer_commits", verdict.violation());
        let unchanged = rt.event_count() == events && rt.runtime.expert_registry() == &registry;
        cx.count("uncommitted_effective_changes", i64::from(!unchanged));
        // The control: the grant minted at the new generation commits.
        let renewed = rt.admission_grant(Setup::PRE)?;
        control(
            &format!("a {kind} proposal under the renewed grant"),
            rt.commit_proposal(&proposal, &report, &renewed, None),
        )?;
    }
    Ok(())
}

/// The gate-host-issued events: the gated promotion and retirement of a proposal and the
/// gated readmission, each raced against a superseded grant of its kind.
fn gate_grant_races(cx: &mut Cx) -> Res<()> {
    cx.sub("gate_grant_revocation_race");
    // The gated promotion and retirement rest on the admission grant of the gate requirement;
    // the readmission on the grant of the gate authority itself.
    for (kind, scenario, event_type) in [
        (
            "ExpertPromoted_gate_driven",
            Scenario::Birth,
            "ExpertPromoted",
        ),
        (
            "ExpertRetired_gate_driven",
            Scenario::Retire,
            "ExpertRetired",
        ),
    ] {
        let mut rt = Rt::fresh()?;
        rt.seed(scenario)?;
        let op = rt.operation(scenario, &super::fixture::exact)?;
        let proposal = rt.proposal(op)?;
        let report = report_for(&proposal, None, &ReportSpec::correct())?;
        let minted = rt.admission_grant(Setup::PRE)?;
        rt.runtime
            .supersede_expert_grant(ExpertGrantKind::Admission)
            .map_err(fail("superseding the admission grant"))?;
        let verdict = cx.verdict(
            "gate_grant_revocation_race",
            kind,
            rt.commit_proposal(&proposal, &report, &minted, None),
            is_where(|e| matches!(e, ExpertAdmissionError::GrantGenerationStale { .. })),
        )?;
        record_gate_race(cx, kind, verdict);
        let renewed = rt.admission_grant(Setup::PRE)?;
        control(kind, rt.commit_proposal(&proposal, &report, &renewed, None))?;
        host_issued(cx, kind, &rt, event_type)?;

        // The gate's own authority revoked between the gate pass and the commit: the
        // evidence of the old gate configuration epoch is stale at the commit.
        let mut rt = Rt::fresh()?;
        rt.seed(scenario)?;
        let op = rt.operation(scenario, &super::fixture::exact)?;
        let proposal = rt.proposal(op)?;
        let report = report_for(&proposal, None, &ReportSpec::correct())?;
        let minted = rt.admission_grant(Setup::PRE)?;
        rt.change_authority(
            ptr_ledger::ExpertAuthorityKind::GateConfig,
            2,
            ptr_ledger::ExpertAuthorityClass::Revoke,
        )
        .map_err(fail("revoking the gate configuration"))?;
        let verdict = cx.verdict(
            "gate_grant_revocation_race",
            &format!("{kind} after the gate authority was revoked"),
            rt.commit_proposal(&proposal, &report, &minted, None),
            is_where(|e| {
                matches!(
                    e,
                    ExpertAdmissionError::StalePolicyEvidence {
                        kind: ptr_ledger::ExpertAuthorityKind::GateConfig,
                        ..
                    }
                )
            }),
        )?;
        record_gate_race(cx, kind, verdict);
    }
    let mut rt = Rt::fresh()?;
    rt.birth("x")?;
    rt.fence_quarantine("x")?;
    let request = rt.fence_request("x", ptr_ledger::ExpertFenceCause::AuthorityQuarantine)?;
    let manifest = rt.manifest("x")?;
    let report = readmission_report(&request, manifest, &ReportSpec::correct())?;
    let minted = rt.gate_grant(Setup::PRE)?;
    rt.runtime
        .supersede_expert_grant(ExpertGrantKind::Gate)
        .map_err(fail("superseding the gate grant"))?;
    let verdict = cx.verdict(
        "gate_grant_revocation_race",
        "ExpertReadmitted",
        rt.runtime.readmit_expert(&request, &report, &minted),
        is_where(|e| {
            matches!(
                e,
                ExpertAdmissionError::GrantGenerationStale {
                    kind: ExpertGrantKind::Gate,
                    ..
                }
            )
        }),
    )?;
    record_gate_race(cx, "ExpertReadmitted", verdict);
    let renewed = rt.gate_grant(Setup::PRE)?;
    control(
        "ExpertReadmitted",
        rt.runtime.readmit_expert(&request, &report, &renewed),
    )?;
    host_issued(cx, "ExpertReadmitted", &rt, "ExpertReadmitted")?;
    let raced = cx.set_digest("gate_grant_race_kinds");
    let issued = cx.set_digest("gate_host_issued_kinds");
    cx.tally
        .set_json("gate_grant_race_kind_set_digest", Json::str(raced));
    cx.tally
        .set_json("gate_host_issued_kind_set_digest", Json::str(issued));
    Ok(())
}

fn record_gate_race(cx: &mut Cx, kind: &str, verdict: Verdict) {
    cx.count_in("gate_grant_revocation_race_cases", kind, 1);
    cx.count("revoked_gate_grant_commits", verdict.violation());
    cx.set_add("gate_grant_race_kinds", kind);
}

/// The kind of event the renewed gate-side grant issued, read off the ledger.
fn host_issued(cx: &mut Cx, kind: &str, rt: &Rt, expected: &str) -> Res<()> {
    match rt
        .runtime
        .committed_events()
        .last()
        .and_then(|c| event_type_of(&c.event))
    {
        Some(found) if found == expected => {
            cx.set_add("gate_host_issued_kinds", kind);
            Ok(())
        }
        other => Err(format!("the {kind} control committed {other:?}")),
    }
}
