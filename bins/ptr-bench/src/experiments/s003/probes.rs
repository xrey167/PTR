//! The adversarial probes, run in every case on fresh fixture runtimes.
//!
//! Each probe builds one situation a merge, a write or a replay must refuse
//! (or hold) as specified, runs it directly against the runtime, and counts
//! in the hard counter it names when the runtime accepts what it must not.
//! A runtime that refuses for another reason than the one specified is
//! counted in `refusal_kind_mismatches`. Each probe counts its own coverage
//! counter when its decisive step ran, and P22 to P26 also run their
//! situations through the world, so that every hazard class is a trial of
//! certification in every case.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ptr_branch::{
    merge_plan_digest, Branch, BranchError, BranchId, BranchOp, InputsDigest, SealedBranch,
    SealedBranchParts, ValueDigest,
};
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity, Attestation, CommittedEvent, FileLedger, LedgerEvent, MergeAuthorityRecord,
    MergeRecord, SemanticOrigin,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::semantic::{pod_output_key, request_raw_key};
use ptr_runtime::{
    HoldReason, MergeAuthority, MergeOutcome, PtrRuntime, RuntimeError, SemanticChange,
    SemanticGrant,
};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{
    CommitIndex, PodId, PrincipalId, Probability, RequestId, Revision, TypeId, VerificationLevel,
};
use ptr_verifier::{Finding, NamedVerifier, VerificationReport, VerificationStatus, Verifier};

use super::metrics::Metrics;
use super::model::{Delta, Val, OP_SOURCE};
use super::params;
use super::program::{keys, Program};
use super::verifier::Domain;
use super::workload::{Background, Case, Genesis};
use super::world::{grant, policy_record, Authority, GrantKind, HostOutcome, Settled, World};

/// What the probes of one case counted.
pub struct Outcome {
    pub metrics: Metrics,
    pub notes: Vec<String>,
}

/// Run every probe of one case. `Err` only for a failure of the harness
/// itself: a fixture that could not be built.
pub fn run_all(case: &Case, seed: u64) -> Result<Outcome, String> {
    let mut probes = Probes {
        // Probes run on the smallest state of the case's seed, so a fixture
        // is cheap however far up the ladder the case is.
        genesis: Case::new(seed, case.index - case.index % params::GROUPS_LADDER.len()).genesis(),
        seed,
        case: case.index,
        metrics: Metrics::default(),
        notes: Vec::new(),
    };
    probes.merged_twice()?;
    probes.foreign_base()?;
    probes.relied_generation()?;
    probes.approvals()?;
    probes.held_merges()?;
    probes.host_write_rules()?;
    probes.sealing_rules()?;
    probes.grants_and_paths()?;
    probes.forged_histories()?;
    probes.hazards()?;
    Ok(Outcome {
        metrics: probes.metrics,
        notes: probes.notes,
    })
}

/// A verifier that reports a fixed result on every change.
struct Fixed {
    name: &'static str,
    level: VerificationLevel,
    hard_finding: bool,
}

impl<'a> Verifier<SemanticChange<'a>> for Fixed {
    fn verify(&self, _: &SemanticChange<'a>) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: self.level,
            score: Probability::new(1.0).expect("a probability"),
            findings: if self.hard_finding {
                vec![Finding {
                    code: "forced".to_string(),
                    message: "a hard finding beside a pass".to_string(),
                    hard: true,
                }]
            } else {
                Vec::new()
            },
        }
    }
}

impl<'a> NamedVerifier<SemanticChange<'a>> for Fixed {
    fn name(&self) -> &'static str {
        self.name
    }
}

struct Probes {
    genesis: Genesis,
    seed: u64,
    case: usize,
    metrics: Metrics,
    notes: Vec<String>,
}

fn already_merged(error: &RuntimeError) -> bool {
    matches!(error, RuntimeError::BranchAlreadyMerged { .. })
}

fn is_reserved_namespace(error: &BranchError) -> bool {
    matches!(error, BranchError::ReservedNamespace { .. })
}

fn config() -> PtrConfig {
    PtrConfig::default()
}

fn triage(score: f32) -> MergeAuthority {
    MergeAuthority::Triage {
        score: Probability::new(score).expect("a probability"),
    }
}

impl Probes {
    fn fixture(&self, kind: GrantKind) -> Result<World, String> {
        World::with_genesis(&self.genesis, kind)
    }

    fn note(&mut self, text: String) {
        if self.notes.len() < 24 {
            self.notes.push(text);
        }
    }

    /// Fold a fixture's counters and notes into the case's.
    fn absorb(&mut self, world: World) {
        self.metrics.absorb(&world.metrics);
        for note in world.notes {
            self.note(note);
        }
    }

    /// The runtime accepted where it had to refuse: `counter` names why that
    /// is a failure. A refusal of another kind is counted as such.
    fn refused<T: Debug>(
        &mut self,
        what: &str,
        result: Result<T, RuntimeError>,
        expected: impl Fn(&RuntimeError) -> bool,
        accepted: fn(&mut Metrics),
    ) {
        match result {
            Ok(value) => {
                accepted(&mut self.metrics);
                self.note(format!("{what}: accepted, {value:?}"));
            }
            Err(error) if expected(&error) => {}
            Err(error) => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("{what}: refused as {error:?}"));
            }
        }
    }

    /// The runtime had to hold a merge for `reason` and append nothing.
    fn held(
        &mut self,
        what: &str,
        result: Result<MergeOutcome, RuntimeError>,
        reason: HoldReason,
        appended: bool,
        accepted: fn(&mut Metrics),
    ) {
        match result {
            Ok(MergeOutcome::Held(hold)) if hold.reason == reason && !appended => {}
            Ok(MergeOutcome::Committed(_)) => {
                accepted(&mut self.metrics);
                self.note(format!("{what}: committed"));
            }
            other => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!(
                    "{what}: expected a hold for {reason:?}, got {other:?}"
                ));
            }
        }
    }

    fn attempt(
        world: &mut World,
        id: &str,
        program: &Program,
        rely: Option<usize>,
    ) -> Result<super::world::Attempt, String> {
        world.open(id, 0, program, rely)
    }

    fn committed(world: &mut World, attempt: &super::world::Attempt) -> Result<(), String> {
        match world.merge(attempt, &Authority::Triage { score: 1.0 })? {
            Settled::Committed => Ok(()),
            other => Err(format!("a fixture merge was {other:?}")),
        }
    }

    /// A file name of this probe's own, so that two runs in one process, or a
    /// run that died, never meet.
    fn durable_path(&self, label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-s003-{label}-{}-{}-{}-{}.log",
            std::process::id(),
            self.seed,
            self.case,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    // ---- P1 to P4: a branch merges once, however the runtime came back --------------

    fn merged_twice(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let add = Program::CounterAdd {
            counter: 0,
            amount: 3,
        };
        let attempt = Self::attempt(&mut world, "p1", &add, None)?;
        Self::committed(&mut world, &attempt)?;

        self.metrics.probe_p1_exercised += 1;
        let again = world.runtime.merge_branch(&attempt.sealed, triage(1.0));
        self.refused("P1", again, already_merged, |m| m.double_merges += 1);

        let events = world.runtime.committed_events().to_vec();
        self.metrics.probe_p2_exercised += 1;
        let replayed = PtrRuntime::replay(config(), &events)
            .map_err(|error| format!("P2: the journal does not replay: {error:?}"))?;
        self.resubmit("P2", replayed, &attempt.sealed, |m| m.double_merges += 1)?;

        self.metrics.probe_p3_exercised += 1;
        let snapshot = world
            .runtime
            .export_compacted_snapshot()
            .map_err(|error| format!("P3: no compacted snapshot: {error:?}"))?;
        let restored =
            PtrRuntime::restore_compacted(config(), snapshot.bytes(), snapshot.anchor(), &[])
                .map_err(|error| format!("P3: the snapshot does not restore: {error:?}"))?;
        self.resubmit("P3", restored, &attempt.sealed, |m| m.double_merges += 1)?;

        self.metrics.probe_p4_exercised += 1;
        let path = self.durable_path("p4");
        let bytes = integrity::encode_log(&events).map_err(|error| format!("P4: {error}"))?;
        let anchor = integrity::decode_log(&bytes)
            .map_err(|error| format!("P4: {error}"))?
            .anchor();
        drop(
            FileLedger::create_from_log(&path, &bytes, anchor)
                .map_err(|error| format!("P4: {error}"))?,
        );
        let reopened = PtrRuntime::open_durable(config(), &path);
        let _ = std::fs::remove_file(&path);
        let reopened =
            reopened.map_err(|error| format!("P4: the journal does not reopen: {error:?}"))?;
        self.resubmit("P4", reopened, &attempt.sealed, |m| m.double_merges += 1)?;
        self.absorb(world);
        Ok(())
    }

    /// Install the grant on a runtime that came back from the journal and
    /// submit the branch again.
    fn resubmit(
        &mut self,
        what: &str,
        mut runtime: PtrRuntime,
        sealed: &SealedBranch,
        accepted: fn(&mut Metrics),
    ) -> Result<(), String> {
        runtime
            .install_semantic_grant(grant(GrantKind::Auto))
            .map_err(|error| format!("{what}: {error:?}"))?;
        let again = runtime.merge_branch(sealed, triage(1.0));
        self.refused(what, again, already_merged, accepted);
        Ok(())
    }

    // ---- P5: a base that is not the runtime's ---------------------------------------

    fn foreign_base(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let mut foreign = self.fixture(GrantKind::Auto)?;
        let mut moved = Delta::default();
        moved.upserts.insert(keys::item(0, 0), Val::text("999"));
        foreign.host_write(&moved, "s003-probe")?;
        let mut other = Delta::default();
        other.upserts.insert(keys::item(0, 1), Val::text("5"));
        world.host_write(&other, "s003-probe")?;
        // Both runtimes are at one revision, holding different values.
        let program = Program::Rmw {
            group: 0,
            item: 0,
            delta: 3,
        };
        let attempt = Self::attempt(&mut foreign, "p5", &program, None)?;
        self.metrics.probe_p5_exercised += 1;
        let result = world.runtime.merge_branch(&attempt.sealed, triage(1.0));
        match result {
            Err(RuntimeError::Certification(BranchError::Conflict { keys }))
                if keys.contains(&self::keys::item(0, 0)) => {}
            Ok(_) => {
                self.metrics.lost_updates += 1;
                self.note("P5: a branch over another state was merged".into());
            }
            other => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("P5: {other:?}"));
            }
        }
        self.absorb(world);
        self.absorb(foreign);
        Ok(())
    }

    // ---- P6: a generation revoked after the preview ----------------------------------

    fn relied_generation(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let program = Program::Rmw {
            group: 1,
            item: 0,
            delta: 1,
        };
        let attempt = world.open("p6", 0, &program, Some(0))?;
        if attempt.footprint.relied.is_empty() {
            return Err("P6: the branch relied on nothing".into());
        }
        world
            .runtime
            .preview_merge(&attempt.sealed)
            .map_err(|error| format!("P6: the preview is refused: {error:?}"))?;
        world.background(
            1,
            &[Background::Lifecycle {
                policy: 0,
                revoke: true,
            }],
        )?;
        self.metrics.probe_p6_exercised += 1;
        match world.runtime.merge_branch(&attempt.sealed, triage(1.0)) {
            Err(RuntimeError::Certification(BranchError::LifecycleChanged { targets }))
                if targets.contains("policy-0") => {}
            Ok(_) => {
                self.metrics.stale_reliance_merges += 1;
                self.note("P6: a revoked generation was relied on".into());
            }
            other => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("P6: {other:?}"));
            }
        }
        self.absorb(world);
        Ok(())
    }

    // ---- P7 to P9: approval and escalation --------------------------------------------

    fn approvals(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Review)?;
        let program = Program::Rmw {
            group: 2,
            item: 0,
            delta: 2,
        };

        let stale = Self::attempt(&mut world, "p7", &program, None)?;
        let preview = world
            .runtime
            .preview_merge(&stale.sealed)
            .map_err(|error| format!("P7: the preview is refused: {error:?}"))?;
        let mut unrelated = Delta::default();
        unrelated.upserts.insert(keys::item(5, 5), Val::text("9"));
        world.host_write(&unrelated, "s003-probe")?;
        self.metrics.probe_p7_exercised += 1;
        let result = world.runtime.merge_branch(
            &stale.sealed,
            MergeAuthority::Reviewed {
                plan_digest: preview.plan_digest,
                reviewer: PrincipalId::from(params::REVIEWER),
            },
        );
        self.refused(
            "P7",
            result,
            |error| matches!(error, RuntimeError::MergePlanChanged { .. }),
            |m| m.approval_bypasses += 1,
        );

        let branch = Self::attempt(&mut world, "p8", &program, None)?;
        let preview = world
            .runtime
            .preview_merge(&branch.sealed)
            .map_err(|error| format!("P8: the preview is refused: {error:?}"))?;
        self.metrics.probe_p8_exercised += 1;
        let result = world.runtime.merge_branch(
            &branch.sealed,
            MergeAuthority::Reviewed {
                plan_digest: preview.plan_digest,
                reviewer: PrincipalId::from("intruder"),
            },
        );
        self.refused(
            "P8",
            result,
            |error| matches!(error, RuntimeError::UnknownReviewer { .. }),
            |m| m.approval_bypasses += 1,
        );

        let low = Self::attempt(&mut world, "p9", &program, None)?;
        let before = world.runtime.committed_events().len();
        self.metrics.probe_p9_exercised += 1;
        let result = world.runtime.merge_branch(&low.sealed, triage(0.1));
        let appended = world.runtime.committed_events().len() != before;
        self.held("P9", result, HoldReason::Escalated, appended, |m| {
            m.ungated_commits += 1
        });
        self.absorb(world);
        Ok(())
    }

    // ---- P10 to P13: what verification holds ------------------------------------------

    fn held_merges(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let negative = Program::CounterAdd {
            counter: 0,
            amount: -(params::COUNTER_START + 1),
        };
        let attempt = Self::attempt(&mut world, "p10", &negative, None)?;
        let before = world.runtime.committed_events().len();
        self.metrics.probe_p10_exercised += 1;
        let result = world.runtime.merge_branch(&attempt.sealed, triage(1.0));
        let appended = world.runtime.committed_events().len() != before;
        self.held(
            "P10",
            result,
            HoldReason::VerificationRejected,
            appended,
            |m| m.ungated_commits += 1,
        );

        self.metrics.probe_p13_exercised += 1;
        let mut jump = Delta::default();
        jump.upserts.insert(
            keys::counter(0),
            Val::counter(params::COUNTER_START + 1_000, "s003-probe"),
        );
        let result = world.runtime.apply_verified_semantic_delta(
            world.runtime.revision(),
            super::program::to_semantic_delta(&jump),
            &PrincipalId::from("s003-probe"),
        );
        self.refused(
            "P13",
            result,
            |error| matches!(error, RuntimeError::SemanticVerificationRejected(_)),
            |m| m.ungated_commits += 1,
        );
        self.absorb(world);

        // P11: two verifiers whose weakest level is below the requirement.
        // P12: a verifier that passes with a hard finding.
        for (probe, verifiers) in [
            (
                11,
                vec![
                    Fixed {
                        name: "probe-deterministic",
                        level: VerificationLevel::Deterministic,
                        hard_finding: false,
                    },
                    Fixed {
                        name: "probe-full-semantic",
                        level: VerificationLevel::FullSemantic,
                        hard_finding: false,
                    },
                ],
            ),
            (
                12,
                vec![Fixed {
                    name: "probe-hard-finding",
                    level: VerificationLevel::Deterministic,
                    hard_finding: true,
                }],
            ),
        ] {
            let mut runtime =
                PtrRuntime::new(config()).map_err(|error| format!("P{probe}: {error:?}"))?;
            let mut custom = SemanticGrant::new(RequiredVerification::Deterministic)
                .with_merge_policy(policy_record(GrantKind::Auto), params::CALIBRATION_SEED);
            for verifier in verifiers {
                custom = custom.with_verifier(verifier);
            }
            runtime
                .install_semantic_grant(custom)
                .map_err(|error| format!("P{probe}: {error:?}"))?;
            let sealed = counter_branch(&runtime, &format!("p{probe}"))?;
            if probe == 11 {
                self.metrics.probe_p11_exercised += 1;
            } else {
                self.metrics.probe_p12_exercised += 1;
            }
            let before = runtime.committed_events().len();
            let result = runtime.merge_branch(&sealed, triage(1.0));
            let appended = runtime.committed_events().len() != before;
            self.held(
                &format!("P{probe}"),
                result,
                HoldReason::VerificationRejected,
                appended,
                |m| m.ungated_commits += 1,
            );
        }
        Ok(())
    }

    // ---- P14: what ingress alone writes -------------------------------------------------

    fn host_write_rules(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let snapshot = world.runtime.snapshot();
        let key = "request:probe:raw".to_string();
        let absent = ValueDigest::of(&key, None).map_err(|error| format!("P14: {error:?}"))?;
        let parts = SealedBranchParts {
            id: BranchId("p14".into()),
            author: PrincipalId::from("s003-probe"),
            base_revision: snapshot.revision,
            reads: [(key.clone(), absent)].into(),
            scans: BTreeMap::new(),
            relied: BTreeMap::new(),
            touched_base: [(key.clone(), absent)].into(),
            touched_inputs: [(
                key.clone(),
                InputsDigest::of(&key, std::iter::empty::<&str>()),
            )]
            .into(),
            ops: vec![BranchOp::Put {
                key: key.clone(),
                value: SemanticValue::Text("forged".into()),
            }],
        };
        self.metrics.probe_p14_exercised += 1;
        match SealedBranch::from_parts(parts) {
            Err(error) if is_reserved_namespace(&error) => {}
            Ok(_) => {
                self.metrics.seal_invariant_failures += 1;
                self.note("P14: a branch that writes an ingress key was sealed".into());
            }
            Err(other) => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("P14: sealing refused with {other:?}"));
            }
        }
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(key, SemanticValue::Text("forged".into()));
        let result = world.runtime.apply_verified_semantic_delta(
            world.runtime.revision(),
            delta,
            &PrincipalId::from("s003-probe"),
        );
        self.refused(
            "P14 host write",
            result,
            |error| matches!(error, RuntimeError::ReservedSemanticNamespace { .. }),
            |m| m.reserved_writes_accepted += 1,
        );
        self.absorb(world);
        Ok(())
    }

    // ---- P15, P16: what sealing and certification insist on --------------------------------

    fn sealing_rules(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        let snapshot = world.runtime.snapshot();
        let digest = |key: &str| {
            ValueDigest::of(key, snapshot.value(key)).map_err(|error| format!("{key}: {error:?}"))
        };
        let put = |key: &str, text: &str| BranchOp::Put {
            key: key.to_string(),
            value: SemanticValue::Text(text.to_string()),
        };
        let none = std::iter::empty::<&str>();

        // P15: an overwrite of a key the branch never read.
        let target = keys::item(0, 0);
        let parts = SealedBranchParts {
            id: BranchId("p15".into()),
            author: PrincipalId::from("s003-probe"),
            base_revision: snapshot.revision,
            reads: BTreeMap::new(),
            scans: BTreeMap::new(),
            relied: BTreeMap::new(),
            touched_base: [(target.clone(), digest(&target)?)].into(),
            touched_inputs: [(target.clone(), InputsDigest::of(&target, none.clone()))].into(),
            ops: vec![put(&target, "1")],
        };
        self.metrics.probe_p15_exercised += 1;
        match SealedBranch::from_parts(parts) {
            Err(BranchError::UnreadTarget { .. }) => {}
            Ok(_) => {
                self.metrics.seal_invariant_failures += 1;
                self.note("P15: a branch that overwrites an unread key was sealed".into());
            }
            Err(other) => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("P15: sealing refused with {other:?}"));
            }
        }

        // P16: a derived key put with the true digest of its input set, but
        // with one of those inputs missing from what the branch read.
        let total = keys::total(0);
        let inputs: BTreeSet<String> = world.model.inputs(&total);
        let missing = keys::item(0, params::ITEMS_PER_GROUP - 1);
        let mut reads: BTreeMap<String, ValueDigest> = BTreeMap::new();
        reads.insert(total.clone(), digest(&total)?);
        for input in inputs.iter().filter(|input| **input != missing) {
            reads.insert(input.clone(), digest(input)?);
        }
        let parts = SealedBranchParts {
            id: BranchId("p16".into()),
            author: PrincipalId::from("s003-probe"),
            base_revision: snapshot.revision,
            reads,
            scans: BTreeMap::new(),
            relied: BTreeMap::new(),
            touched_base: [(total.clone(), digest(&total)?)].into(),
            touched_inputs: [(
                total.clone(),
                InputsDigest::of(&total, inputs.iter().map(String::as_str)),
            )]
            .into(),
            ops: vec![put(&total, "0")],
        };
        let sealed = SealedBranch::from_parts(parts)
            .map_err(|error| format!("P16: the fixture does not seal: {error:?}"))?;
        self.metrics.probe_p16_exercised += 1;
        match world.runtime.merge_branch(&sealed, triage(1.0)) {
            Err(RuntimeError::Certification(BranchError::Conflict { keys }))
                if keys.contains(&missing) => {}
            Ok(_) => {
                self.metrics.undeclared_input_merges += 1;
                self.note("P16: a derived key was merged with an input unread".into());
            }
            other => {
                self.metrics.refusal_kind_mismatches += 1;
                self.note(format!("P16: {other:?}"));
            }
        }
        self.absorb(world);
        Ok(())
    }

    // ---- P17 to P20: the paths that are not the semantic path ---------------------------------

    fn grants_and_paths(&mut self) -> Result<(), String> {
        // P17: a semantic record through `commit`, whatever its origin.
        let mut runtime = PtrRuntime::new(config()).map_err(|error| format!("P17: {error:?}"))?;
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(keys::item(0, 0), SemanticValue::Text("1".into()));
        let encoded = delta.encode().map_err(|error| format!("P17: {error:?}"))?;
        let record = |origin| LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(0),
            revision: Revision(1),
            encoded_delta: encoded.clone(),
            origin,
        };
        self.metrics.probe_p17_exercised += 1;
        let legacy = runtime.commit(record(SemanticOrigin::Legacy));
        self.refused(
            "P17 legacy",
            legacy,
            |error| matches!(error, RuntimeError::SemanticRecordOutsideSemanticPath),
            |m| m.bypass_commits_accepted += 1,
        );
        let host = runtime.commit(record(SemanticOrigin::Host {
            principal: "s003-probe".into(),
            verification: attestation(),
        }));
        self.refused(
            "P17 host",
            host,
            |error| matches!(error, RuntimeError::SemanticRecordOutsideSemanticPath),
            |m| m.bypass_commits_accepted += 1,
        );

        // P18: a host write under a grant that does not allow one.
        let mut closed = PtrRuntime::new(config()).map_err(|error| format!("P18: {error:?}"))?;
        closed
            .install_semantic_grant(
                SemanticGrant::new(RequiredVerification::Deterministic).with_verifier(Domain),
            )
            .map_err(|error| format!("P18: {error:?}"))?;
        self.metrics.probe_p18_exercised += 1;
        let result = closed.apply_verified_semantic_delta(
            closed.revision(),
            delta.clone(),
            &PrincipalId::from("s003-probe"),
        );
        self.refused(
            "P18",
            result,
            |error| matches!(error, RuntimeError::HostWritesNotGranted),
            |m| m.bypass_commits_accepted += 1,
        );

        // P19: a second grant.
        let mut world = self.fixture(GrantKind::Auto)?;
        self.metrics.probe_p19_exercised += 1;
        let second = world.runtime.install_semantic_grant(grant(GrantKind::Auto));
        self.refused(
            "P19",
            second,
            |error| matches!(error, RuntimeError::SemanticGrantInstalled),
            |m| m.bypass_commits_accepted += 1,
        );

        // P20: a merge with no grant at all.
        let add = Program::CounterAdd {
            counter: 0,
            amount: 1,
        };
        let attempt = Self::attempt(&mut world, "p20", &add, None)?;
        let mut bare = PtrRuntime::new(config()).map_err(|error| format!("P20: {error:?}"))?;
        self.metrics.probe_p20_exercised += 1;
        let result = bare.merge_branch(&attempt.sealed, triage(1.0));
        self.refused(
            "P20",
            result,
            |error| matches!(error, RuntimeError::NoSemanticGrant),
            |m| m.bypass_commits_accepted += 1,
        );
        self.absorb(world);
        Ok(())
    }

    // ---- P21: histories no writer could have written ----------------------------------------------

    fn forged_histories(&mut self) -> Result<(), String> {
        let mut world = self.fixture(GrantKind::Auto)?;
        world.background(
            1,
            &[Background::Ingress {
                request: "rp".into(),
                text: "a request".into(),
            }],
        )?;
        let add = Program::CounterAdd {
            counter: 0,
            amount: 2,
        };
        let attempt = Self::attempt(&mut world, "p21-real", &add, None)?;
        Self::committed(&mut world, &attempt)?;
        let history = History::of(&world)?;
        self.metrics.probe_p21_exercised += 1;

        // (a) a record with no origin after one with an origin: replayed, and
        // restored above a compaction floor.
        let legacy = history.forge(
            SemanticOrigin::Legacy,
            &history.single(&keys::item(0, 0), "legacy"),
        )?;
        self.replayed("P21a", &history, &legacy);
        let snapshot = world
            .runtime
            .export_compacted_snapshot()
            .map_err(|error| format!("P21a: {error:?}"))?;
        let restored = PtrRuntime::restore_compacted(
            config(),
            snapshot.bytes(),
            snapshot.anchor(),
            std::slice::from_ref(&legacy),
        );
        self.forged("P21a compacted", restored.map(|_| ()));

        // (b) a merge whose plan digest is not the digest of its delta.
        let delta = history.single(&keys::item(0, 0), "merged");
        let forged = history.merge("p21-b", &delta, true)?;
        self.replayed("P21b", &history, &forged);
        // (c) a merge that carries a dependency entry.
        let mut with_entry = history.single(&keys::item(0, 0), "merged");
        with_entry
            .dependencies
            .insert(keys::total(0), world.model.inputs(&keys::total(0)));
        let forged = history.merge("p21-c", &with_entry, false)?;
        self.replayed("P21c", &history, &forged);
        // (d) a second merge of one branch id.
        let forged = history.merge(&history.merged_branch()?, &delta, false)?;
        self.replayed("P21d", &history, &forged);
        // (e) a request that writes two keys.
        let mut two = SemanticDelta::default();
        two.upserts.insert(
            request_raw_key(&RequestId::from("rq")),
            SemanticValue::Text("a".into()),
        );
        two.upserts.insert(
            request_raw_key(&RequestId::from("rq2")),
            SemanticValue::Text("b".into()),
        );
        let forged = history.forge(
            SemanticOrigin::Request {
                request: "rq".into(),
            },
            &two,
        )?;
        self.replayed("P21e", &history, &forged);
        // (f) a host write that touches an ingress key.
        let touching = history.single(&request_raw_key(&RequestId::from("rz")), "x");
        let forged = history.forge(history.host(), &touching)?;
        self.replayed("P21f", &history, &forged);
        // (g) a Pod's output whose source is not the Pod.
        let forged = history.forge(
            SemanticOrigin::PodOutput {
                request: "rp".into(),
                pod: "pod-1".into(),
                level: VerificationLevel::Deterministic,
            },
            &history.pod_output("rp", "pod-1", "someone-else"),
        )?;
        self.replayed("P21g", &history, &forged);
        self.absorb(world);
        Ok(())
    }

    /// Replay a history and its forged next record: it must be refused.
    fn replayed(&mut self, what: &str, history: &History, forged: &CommittedEvent) {
        let mut events = history.events.clone();
        events.push(forged.clone());
        self.forged(what, PtrRuntime::replay(config(), &events).map(|_| ()));
    }

    fn forged(&mut self, what: &str, result: Result<(), RuntimeError>) {
        self.refused(
            what,
            result,
            |error| {
                matches!(
                    error,
                    RuntimeError::LegacySemanticRecord { .. }
                        | RuntimeError::InvalidSemanticOrigin { .. }
                )
            },
            |m| m.forged_histories_accepted += 1,
        );
    }

    // ---- P22 to P26: one deterministic case of every hazard class -------------------------------------

    fn hazards(&mut self) -> Result<(), String> {
        let triage = Authority::Triage { score: 1.0 };

        // P22: write skew.
        let mut world = self.fixture(GrantKind::Auto)?;
        for item in [0, 1] {
            let mut delta = Delta::default();
            delta.upserts.insert(keys::item(0, item), Val::text("40"));
            world.host_write(&delta, "s003-probe")?;
        }
        let first = Self::attempt(
            &mut world,
            "p22-a",
            &Program::WriteSkew {
                group: 0,
                first: 0,
                second: 1,
            },
            None,
        )?;
        let second = Self::attempt(
            &mut world,
            "p22-b",
            &Program::WriteSkew {
                group: 0,
                first: 1,
                second: 0,
            },
            None,
        )?;
        Self::committed(&mut world, &first)?;
        world.merge(&second, &triage)?;
        self.metrics.probe_p22_exercised += 1;
        self.absorb(world);

        // P23: a phantom insert.
        let mut world = self.fixture(GrantKind::Auto)?;
        let first = Self::attempt(
            &mut world,
            "p23-a",
            &Program::InsertCapped { group: 0, task: 1 },
            None,
        )?;
        let second = Self::attempt(
            &mut world,
            "p23-b",
            &Program::InsertCapped { group: 0, task: 2 },
            None,
        )?;
        Self::committed(&mut world, &first)?;
        world.merge(&second, &triage)?;
        self.metrics.probe_p23_exercised += 1;
        self.absorb(world);

        // P24: a total whose input set was swapped without a value moving.
        let mut world = self.fixture(GrantKind::Auto)?;
        let branch = Self::attempt(
            &mut world,
            "p24",
            &Program::GroupTotal {
                group: 0,
                item: 0,
                delta: 1,
            },
            None,
        )?;
        world.background(
            1,
            &[Background::Rewire {
                group: 0,
                swap_after: 1,
            }],
        )?;
        world.background(2, &[])?;
        if world
            .model
            .inputs(&keys::total(0))
            .contains(&keys::spare(0))
        {
            world.merge(&branch, &triage)?;
            self.metrics.probe_p24_exercised += 1;
        } else {
            self.note("P24: the input set was not swapped".into());
        }
        self.absorb(world);

        // P25: an increment that is negative once rebased.
        let mut world = self.fixture(GrantKind::Auto)?;
        let branch = Self::attempt(
            &mut world,
            "p25",
            &Program::CounterAdd {
                counter: 1,
                amount: -30,
            },
            None,
        )?;
        let mut delta = Delta::default();
        delta
            .upserts
            .insert(keys::counter(1), Val::counter(10, "s003-probe"));
        if world.host_write(&delta, "s003-probe")? != HostOutcome::Committed {
            return Err("P25: the counter was not lowered".into());
        }
        world.merge(&branch, &triage)?;
        self.metrics.probe_p25_exercised += 1;
        self.absorb(world);

        // P26: a set operation that would undo a concurrent one.
        let mut world = self.fixture(GrantKind::Auto)?;
        let mut members = world
            .model
            .value(&keys::set(0))
            .and_then(Val::as_set)
            .ok_or("P26: no set")?;
        let absent = (0..params::SET_MEMBERS)
            .find(|member| !members.contains(&keys::member(*member)))
            .ok_or("P26: the set holds every member")?;
        let branch = Self::attempt(
            &mut world,
            "p26",
            &Program::SetOp {
                set: 0,
                member: absent,
                insert: false,
            },
            None,
        )?;
        members.insert(keys::member(absent));
        let mut delta = Delta::default();
        delta
            .upserts
            .insert(keys::set(0), Val::set(&members, OP_SOURCE));
        world.host_write(&delta, "s003-probe")?;
        world.merge(&branch, &triage)?;
        self.metrics.probe_p26_exercised += 1;
        self.absorb(world);
        Ok(())
    }
}

/// The attestation a grant with the harness's verifiers records.
fn attestation() -> Attestation {
    Attestation {
        required: VerificationLevel::Deterministic,
        level: VerificationLevel::Deterministic,
        verifiers: params::VERIFIERS
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        findings: Vec::new(),
    }
}

/// A branch that adds one to a counter on an empty state.
fn counter_branch(runtime: &PtrRuntime, id: &str) -> Result<SealedBranch, String> {
    let mut branch = Branch::open(
        BranchId(id.to_string()),
        PrincipalId::from("s003-probe"),
        runtime.snapshot(),
    );
    branch
        .stage_commutative(BranchOp::Add {
            key: keys::counter(0),
            amount: 1,
        })
        .map_err(|error| format!("{id}: {error:?}"))?;
    branch.seal().map_err(|error| format!("{id}: {error:?}"))
}

/// A valid history and what a forged next record has to continue.
struct History {
    events: Vec<CommittedEvent>,
    revision: Revision,
    next_index: CommitIndex,
}

impl History {
    fn of(world: &World) -> Result<Self, String> {
        let events = world.runtime.committed_events().to_vec();
        let last = events.last().ok_or("an empty history")?.index;
        Ok(Self {
            events,
            revision: world.runtime.revision(),
            next_index: CommitIndex(last.0 + 1),
        })
    }

    /// A delta writing one text value.
    fn single(&self, key: &str, text: &str) -> SemanticDelta {
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(key.to_string(), SemanticValue::Text(text.to_string()));
        delta
    }

    /// A Pod's output for `request`, whose payload names `source` as its source.
    fn pod_output(&self, request: &str, pod: &str, source: &str) -> SemanticDelta {
        let key = pod_output_key(&RequestId::from(request), &PodId::from(pod));
        let mut delta = SemanticDelta::default();
        delta.upserts.insert(
            key.clone(),
            SemanticValue::Payload(SemanticPayload {
                type_id: TypeId::from("s003.probe"),
                source: source.to_string(),
                bytes: vec![1],
            }),
        );
        delta
            .dependencies
            .insert(key, [request_raw_key(&RequestId::from(request))].into());
        delta
    }

    fn host(&self) -> SemanticOrigin {
        SemanticOrigin::Host {
            principal: "s003-probe".into(),
            verification: attestation(),
        }
    }

    /// The branch id of the last merge the history holds.
    fn merged_branch(&self) -> Result<String, String> {
        self.events
            .iter()
            .rev()
            .find_map(|committed| match &committed.event {
                LedgerEvent::SemanticDeltaCommitted {
                    origin: SemanticOrigin::Merge(record),
                    ..
                } => Some(record.branch.clone()),
                _ => None,
            })
            .ok_or_else(|| "the history holds no merge".to_string())
    }

    /// The next record of the history, with `origin` and `delta`, moving the
    /// revision by one.
    fn forge(
        &self,
        origin: SemanticOrigin,
        delta: &SemanticDelta,
    ) -> Result<CommittedEvent, String> {
        Ok(CommittedEvent {
            index: self.next_index,
            event: LedgerEvent::SemanticDeltaCommitted {
                base_revision: self.revision,
                revision: self.revision.next(),
                encoded_delta: delta.encode().map_err(|error| format!("{error:?}"))?,
                origin,
            },
        })
    }

    /// A merge record of `branch` carrying `delta`, whose plan digest is the
    /// digest of what it carries, or (with `wrong_plan`) something else.
    fn merge(
        &self,
        branch: &str,
        delta: &SemanticDelta,
        wrong_plan: bool,
    ) -> Result<CommittedEvent, String> {
        let encoded = delta.encode().map_err(|error| format!("{error:?}"))?;
        let dependencies = [7u8; 32];
        let rebased = BTreeSet::new();
        let plan = if wrong_plan {
            [9u8; 32]
        } else {
            merge_plan_digest(branch, self.revision, &encoded, &dependencies, &rebased)
        };
        self.forge(
            SemanticOrigin::Merge(MergeRecord {
                branch: branch.to_string(),
                author: "s003-probe".into(),
                seal: [1u8; 32],
                plan,
                dependencies,
                rebased,
                verification: attestation(),
                authority: MergeAuthorityRecord::Triage {
                    policy_version: params::AUTO_POLICY.into(),
                    score_bits: 1.0f32.to_bits(),
                },
            }),
            delta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(case_index: usize) -> Outcome {
        run_all(&Case::new(17, case_index), 17).expect("the probes run")
    }

    #[test]
    fn every_probe_runs_in_a_case_and_the_runtime_passes_them_all() {
        let outcome = outcome(0);
        assert_eq!(outcome.notes, Vec::<String>::new());
        assert_eq!(outcome.metrics.hard(), Metrics::default().hard());
        for probe in 1..=26 {
            let name = format!("probe_p{probe}_exercised");
            let count = outcome
                .metrics
                .coverage()
                .into_iter()
                .find(|(field, _)| *field == name)
                .map(|(_, count)| count);
            assert_eq!(count, Some(1), "{name}");
        }
    }

    #[test]
    fn the_hazard_probes_are_trials_of_every_class_of_certification() {
        let metrics = outcome(0).metrics;
        assert!(metrics.hazard_write >= 1 || metrics.hazard_read >= 1);
        assert!(metrics.hazard_read >= 1, "write skew is a stale read");
        assert!(metrics.hazard_scan_keys >= 1, "an insert is a phantom");
        assert!(metrics.hazard_inputs >= 1, "a swapped input set is stale");
        assert!(
            metrics.hazard_negative >= 1,
            "a rebased increment goes negative"
        );
        assert!(
            metrics.hazard_set_member >= 1,
            "a set operation undoes another"
        );
        assert!(
            metrics.hazard_lifecycle == 0,
            "no probe relies on a generation and merges it"
        );
        assert_eq!(
            metrics.conflicts, 4,
            "write skew, phantom, swap and set undo, and no more"
        );
    }

    #[test]
    fn a_probe_run_is_the_same_in_every_case_of_a_level() {
        let first = outcome(0).metrics;
        let second = outcome(6).metrics;
        assert_eq!(first.coverage(), second.coverage());
    }

    #[test]
    fn the_forged_histories_are_well_formed_but_for_their_one_defect() {
        // Each forgery without its defect is a record replay accepts, so a
        // refusal is the defect's.
        let mut world =
            World::with_genesis(&Case::new(17, 0).genesis(), GrantKind::Auto).expect("a world");
        world
            .background(
                1,
                &[Background::Ingress {
                    request: "rp".into(),
                    text: "a request".into(),
                }],
            )
            .expect("ingress");
        let add = Program::CounterAdd {
            counter: 0,
            amount: 2,
        };
        let attempt = world.open("p21-real", 0, &add, None).expect("an attempt");
        Probes::committed(&mut world, &attempt).expect("a merge");
        let history = History::of(&world).expect("a history");
        let accepted = |forged: CommittedEvent| {
            let mut events = history.events.clone();
            events.push(forged);
            PtrRuntime::replay(config(), &events).map(|_| ())
        };
        let delta = history.single(&keys::item(0, 0), "merged");
        assert_eq!(
            accepted(history.merge("p21-fresh", &delta, false).unwrap()),
            Ok(())
        );
        assert_eq!(
            accepted(history.forge(history.host(), &delta).unwrap()),
            Ok(())
        );
        assert_eq!(
            accepted(
                history
                    .forge(
                        SemanticOrigin::Request {
                            request: "rq".into()
                        },
                        &history.single(&request_raw_key(&RequestId::from("rq")), "a")
                    )
                    .unwrap()
            ),
            Ok(())
        );
        assert_eq!(
            accepted(
                history
                    .forge(
                        SemanticOrigin::PodOutput {
                            request: "rp".into(),
                            pod: "pod-1".into(),
                            level: VerificationLevel::Deterministic,
                        },
                        &history.pod_output("rp", "pod-1", "pod-1"),
                    )
                    .unwrap()
            ),
            Ok(())
        );
        // A record with no origin is accepted where no attributed one came first.
        let bare = Vec::<CommittedEvent>::new();
        let legacy = CommittedEvent {
            index: CommitIndex(1),
            event: LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(0),
                revision: Revision(1),
                encoded_delta: delta.encode().unwrap(),
                origin: SemanticOrigin::Legacy,
            },
        };
        let mut events = bare;
        events.push(legacy);
        assert!(PtrRuntime::replay(config(), &events).is_ok());
    }
}
