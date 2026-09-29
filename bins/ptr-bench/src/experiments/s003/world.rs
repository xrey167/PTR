//! The system under test and the reference model, run side by side.
//!
//! A [`World`] owns a real [`PtrRuntime`] with the harness's grant and the
//! independent [`Model`] of what it must hold. Every write, whether a host
//! write, ingress, a lifecycle change or a merge, is predicted from the model
//! first, run against the runtime, compared with the prediction, and only
//! then does the model follow what the runtime committed, read back from its
//! own journal. After every committed event the whole state and the lifecycle
//! authority are compared with the model.
//!
//! What a merge must do comes from [`oracle::judge`], which shares no code
//! with certification. Anything the runtime does that the oracle did not
//! predict is counted in one of the hard counters, and a baseline arm that
//! commits what the oracle names a hazard is counted as evidence, not as a
//! failure.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use ptr_branch::{
    merge_plan_digest, AutoThreshold, BranchError, BranchId, CertificationKind, PolicyRecord,
    TriageDecision, TriagePolicy,
};
use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity, CommittedEvent, FileLedger, LedgerEvent, MergeAuthorityRecord, SemanticOrigin,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{
    HoldReason, MergeAuthority, MergeHold, MergeOutcome, MergeReceipt, PtrRuntime, RuntimeError,
    SemanticGrant,
};
use ptr_semdb::{SemanticDelta, SemanticSnapshot, SemanticValue};
use ptr_types::{
    CapsuleId, Generation, PrincipalId, Probability, ProjectId, RequestId, Revision,
    Validity as RuntimeValidity, VerificationLevel,
};

use super::metrics::Metrics;
use super::model::{Delta, Model, Net, OpRefusal, Val, Validity};
use super::oracle::{self, Predicted};
use super::params;
use super::program::{
    from_semantic_delta, keys, sealed_differences, to_semantic_delta, BranchView, Footprint,
    Program, RefView,
};
use super::verifier::{Diff, Domain};
use super::workload::{Background, Case, Genesis, PROJECT};

/// The principal of the genesis write.
const GENESIS_PRINCIPAL: &str = "s003-genesis";
/// The principal of the external writer.
pub const EXTERNAL_PRINCIPAL: &str = "s003-external";
/// The principal of the rewirer.
const REWIRER_PRINCIPAL: &str = "s003-rewirer";
/// The most notes a world keeps to explain a divergence.
const MAX_NOTES: usize = 24;

/// Which merge policy the grant carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrantKind {
    /// `s003-auto/v1`: everything the verifiers admit is auto-proposed.
    Auto,
    /// `s003-review/v1` and one listed reviewer: a person sees what the
    /// policy does not auto-propose.
    Review,
}

/// The merge policy of a grant kind, as the grant carries it.
pub fn policy_record(kind: GrantKind) -> PolicyRecord {
    let (version, threshold, calibration) = match kind {
        GrantKind::Auto => (
            params::AUTO_POLICY,
            params::AUTO_THRESHOLD_PERMILLE,
            params::AUTO_CALIBRATION_PERMILLE,
        ),
        GrantKind::Review => (
            params::REVIEW_POLICY,
            params::REVIEW_THRESHOLD_PERMILLE,
            params::REVIEW_CALIBRATION_PERMILLE,
        ),
    };
    let policy = TriagePolicy::new(
        AutoThreshold::AtLeast(threshold as f32 / 1000.0),
        calibration as f64 / 1000.0,
    )
    .expect("the preregistered policy is valid");
    PolicyRecord::manual(version, policy).expect("the preregistered policy record is valid")
}

/// The grant of a kind: the two verifiers, host writes, the merge policy and,
/// for review, the reviewer.
pub fn grant(kind: GrantKind) -> SemanticGrant {
    let grant = SemanticGrant::new(RequiredVerification::Deterministic)
        .with_verifier(Domain)
        .with_verifier(Diff)
        .allow_host_writes()
        .with_merge_policy(policy_record(kind), params::CALIBRATION_SEED);
    match kind {
        GrantKind::Auto => grant,
        GrantKind::Review => grant.with_reviewer(PrincipalId::from(params::REVIEWER)),
    }
}

/// A branch an agent opened: sealed, with the footprint the harness derived
/// from what the program did.
pub struct Attempt {
    pub id: BranchId,
    pub author: String,
    pub program: Program,
    pub rely: Option<usize>,
    pub sealed: ptr_branch::SealedBranch,
    pub footprint: Footprint,
}

/// What a plan is, as a digest tells two plans apart: the revision it was
/// certified against, the keys it rebased and the delta it commits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanKey {
    pub revision: u64,
    pub rebased: BTreeSet<String>,
    pub delta: Delta,
}

/// Who lets a merge commit.
#[derive(Clone, Debug)]
pub enum Authority {
    /// The runtime triages the branch under the grant's policy.
    Triage { score: f32 },
    /// A reviewer approved the plan a hold showed.
    Reviewed { digest: [u8; 32], approved: PlanKey },
}

/// What became of a merge, in the terms an arm acts on.
#[derive(Clone, Debug)]
pub enum Settled {
    Committed,
    NoChange,
    /// Certification refused: a value read, a prefix scanned or an input set
    /// changed.
    Conflict,
    /// A generation the branch relied on is not live.
    LifecycleChanged,
    /// The verifiers did not admit it: dropped.
    Rejected,
    /// Admitted, and the policy sent it to a person.
    Escalated {
        digest: [u8; 32],
        plan: PlanKey,
    },
    /// The approved plan is not the plan now.
    PlanChanged,
    /// Refused for a reason no retry changes: dropped.
    Refused,
}

/// What a host write did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostOutcome {
    Committed,
    NoChange,
    /// The delta did not apply or a verifier did not admit it.
    Refused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Predicted3 {
    Committed,
    NoChange,
    Refused,
}

/// A rewire waiting for its swap.
#[derive(Clone, Copy, Debug)]
struct Swap {
    due: u64,
    group: usize,
    wait: u64,
}

/// The runtime, the model, and the counters.
pub struct World {
    pub runtime: PtrRuntime,
    pub model: Model,
    pub metrics: Metrics,
    /// Why a counter moved, for the first few times.
    pub notes: Vec<String>,
    policy: PolicyRecord,
    swaps: Vec<Swap>,
    recommits: Vec<(u64, usize)>,
}

impl World {
    /// A runtime with the grant of `kind` and the genesis of `case`, and a
    /// model that agrees with it.
    pub fn new(case: &Case, kind: GrantKind) -> Result<Self, String> {
        let genesis = case.genesis();
        Self::with_genesis(&genesis, kind)
    }

    /// A runtime with the grant of `kind` and `genesis`.
    pub fn with_genesis(genesis: &Genesis, kind: GrantKind) -> Result<Self, String> {
        let mut runtime =
            PtrRuntime::new(PtrConfig::default()).map_err(|error| format!("runtime: {error:?}"))?;
        runtime
            .install_semantic_grant(grant(kind))
            .map_err(|error| format!("grant: {error:?}"))?;
        let mut world = Self {
            runtime,
            model: Model::default(),
            metrics: Metrics::default(),
            notes: Vec::new(),
            policy: policy_record(kind),
            swaps: Vec::new(),
            recommits: Vec::new(),
        };
        for policy in &genesis.policies {
            world.commit_lifecycle(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from(PROJECT),
                capsule: CapsuleId::from(policy.as_str()),
                generation: Generation(1),
            })?;
        }
        match world.host_write(&genesis.delta, GENESIS_PRINCIPAL)? {
            HostOutcome::Committed => Ok(world),
            other => Err(format!("the genesis write was not committed: {other:?}")),
        }
    }

    /// Remember why something diverged, for the first few times.
    pub fn note(&mut self, text: String) {
        if self.notes.len() < MAX_NOTES {
            self.notes.push(text);
        }
    }

    fn ledger_len(&self) -> usize {
        self.runtime.committed_events().len()
    }

    // ---- following the runtime ----------------------------------------------------

    /// Bring the model up to what the runtime committed from index `from` on,
    /// reading it back from the journal, then compare everything.
    fn follow(&mut self, from: usize) -> Result<(), String> {
        let events: Vec<LedgerEvent> = self.runtime.committed_events()[from..]
            .iter()
            .map(|committed| committed.event.clone())
            .collect();
        for event in events {
            match event {
                LedgerEvent::SemanticDeltaCommitted { encoded_delta, .. } => {
                    let delta = SemanticDelta::decode(&encoded_delta)
                        .map_err(|error| format!("a committed delta does not decode: {error:?}"))?;
                    if let Err(refusal) = self.model.apply(&from_semantic_delta(&delta)) {
                        self.metrics.model_divergences += 1;
                        self.note(format!("the model refuses a committed delta: {refusal:?}"));
                    }
                }
                LedgerEvent::CapsuleCommitted {
                    capsule,
                    generation,
                    ..
                } => self.model.lifecycle.set_live(&capsule.0, generation.0),
                LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                    self.model.lifecycle.set_live(&capsule.0, new.0);
                }
                LedgerEvent::Revoked {
                    subject,
                    generation,
                } => self.model.lifecycle.revoke(&subject, generation.0),
                _ => {}
            }
        }
        self.cross_check();
        Ok(())
    }

    /// Compare the runtime's state and lifecycle authority with the model.
    fn cross_check(&mut self) {
        if let Some(difference) = state_difference(&self.runtime, &self.model) {
            self.metrics.model_divergences += 1;
            self.note(format!("runtime and model differ: {difference}"));
        }
    }

    // ---- lifecycle and ingress ----------------------------------------------------

    fn commit_lifecycle(&mut self, event: LedgerEvent) -> Result<(), String> {
        let from = self.ledger_len();
        self.runtime
            .commit(event.clone())
            .map_err(|error| format!("lifecycle {event:?}: {error:?}"))?;
        self.check_lifecycle_record(from, &event);
        self.follow(from)
    }

    fn check_lifecycle_record(&mut self, from: usize, requested: &LedgerEvent) {
        if self.ledger_len() != from + 1
            || self.runtime.committed_events().get(from).map(|c| &c.event) != Some(requested)
        {
            self.metrics.provenance_mismatches += 1;
            self.note(format!(
                "the lifecycle journal does not record exactly {requested:?}"
            ));
        }
    }

    /// Supersede a policy, or revoke its live generation and commit the next
    /// one on the following tick.
    fn lifecycle(&mut self, tick: u64, policy: usize, revoke: bool) -> Result<(), String> {
        let target = keys::policy(policy);
        let live = self
            .model
            .lifecycle
            .live(&target)
            .ok_or_else(|| format!("{target} has no live generation"))?;
        if revoke {
            self.commit_lifecycle(LedgerEvent::Revoked {
                subject: target,
                generation: Generation(live),
            })?;
            self.recommits.push((tick + 1, policy));
            Ok(())
        } else {
            self.commit_lifecycle(LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from(target.as_str()),
                old: Generation(live),
                new: Generation(live + 1),
            })
        }
    }

    fn ingest(&mut self, request: &str, text: &str) -> Result<(), String> {
        let from = self.ledger_len();
        self.runtime
            .ingest_text(RequestId::from(request), text)
            .map_err(|error| format!("ingest {request}: {error:?}"))?;
        self.check_ingress_record(from, request, text)?;
        self.follow(from)
    }

    fn check_ingress_record(
        &mut self,
        from: usize,
        request: &str,
        text: &str,
    ) -> Result<(), String> {
        let mut requested = Delta::default();
        requested
            .upserts
            .insert(format!("request:{request}:raw"), Val::text(text));
        let plan = self
            .model
            .plan(&requested)
            .map_err(|error| format!("predict ingress {request}: {error:?}"))?;
        if !plan.moved() {
            if self.ledger_len() != from {
                self.metrics.provenance_mismatches += 1;
                self.note(format!("unchanged ingress {request} appended a record"));
            }
            return Ok(());
        }
        let good_origin = matches!(
            self.runtime.committed_events().get(from).map(|c| &c.event),
            Some(LedgerEvent::SemanticDeltaCommitted {
                base_revision, revision, origin: SemanticOrigin::Request { request: recorded }, ..
            }) if recorded == request
                && base_revision.0 == self.model.revision()
                && revision.0 == self.model.revision() + 1
        );
        if self.ledger_len() != from + 1 || !good_origin {
            self.metrics.provenance_mismatches += 1;
            self.note(format!(
                "the ingress journal does not record exactly request {request}"
            ));
        }
        if self.committed_net(from) != Some(plan.net) {
            self.metrics.model_divergences += 1;
            self.note(format!(
                "the ingress journal changes something else than request {request}"
            ));
        }
        Ok(())
    }

    // ---- host writes --------------------------------------------------------------

    /// What a host write of `delta` must do, from the model and the rules of
    /// the two verifiers alone.
    fn predict_host(&self, delta: &Delta) -> Predicted3 {
        let families = ["item:", "total:", "audit:", "ctr:", "set:"];
        let stray = delta
            .upserts
            .keys()
            .chain(delta.removals.iter())
            .chain(delta.dependencies.keys())
            .any(|key| !families.iter().any(|family| key.starts_with(family)));
        if stray {
            return Predicted3::Refused;
        }
        let Ok(plan) = self.model.plan(delta) else {
            return Predicted3::Refused;
        };
        if self.model.negative_counter_after(&plan.net) {
            return Predicted3::Refused;
        }
        let jumps = delta
            .upserts
            .keys()
            .filter(|key| key.starts_with("ctr:"))
            .any(|key| {
                let held = self.model.value(key);
                let before = held.and_then(Val::as_counter);
                let now = plan
                    .net
                    .value_after(key, held)
                    .and_then(|value| value.as_counter());
                matches!((before, now), (Some(before), Some(now))
                if before.abs_diff(now) > params::HOST_COUNTER_JUMP_MAX as u64)
            });
        if jumps {
            Predicted3::Refused
        } else if plan.moved() {
            Predicted3::Committed
        } else {
            Predicted3::NoChange
        }
    }

    /// A host write of `delta` under `principal`, compared with what the
    /// model predicts. An outcome that differs is a model divergence.
    pub fn host_write(&mut self, delta: &Delta, principal: &str) -> Result<HostOutcome, String> {
        let predicted = self.predict_host(delta);
        let from = self.ledger_len();
        let result = self.runtime.apply_verified_semantic_delta(
            self.runtime.revision(),
            to_semantic_delta(delta),
            &PrincipalId::from(principal),
        );
        let outcome = match result {
            Ok(commit) if commit.commit_index.is_some() => HostOutcome::Committed,
            Ok(_) => HostOutcome::NoChange,
            Err(RuntimeError::Semantic(_) | RuntimeError::SemanticVerificationRejected(_)) => {
                HostOutcome::Refused
            }
            Err(error) => return Err(format!("host write by {principal}: {error:?}")),
        };
        let expected = match predicted {
            Predicted3::Committed => HostOutcome::Committed,
            Predicted3::NoChange => HostOutcome::NoChange,
            Predicted3::Refused => HostOutcome::Refused,
        };
        if outcome != expected {
            self.metrics.model_divergences += 1;
            self.note(format!(
                "a host write by {principal} was {outcome:?}, predicted {expected:?}"
            ));
        }
        if outcome == HostOutcome::Committed {
            if self.ledger_len() != from + 1 {
                self.metrics.provenance_mismatches += 1;
                self.note("a committed host write did not append exactly one record".into());
            }
            // What the record changes, against the state before it, must be
            // what the request changes: the model follows the journal next, so
            // this is the one place a record of another change would show.
            let requested = self.model.plan(delta).ok().map(|plan| plan.net);
            let recorded = self.committed_net(from);
            if requested.is_none() || recorded != requested {
                self.metrics.model_divergences += 1;
                self.note(format!(
                    "the record of a host write by {principal} changes something else than was requested"
                ));
            }
            self.follow(from)?;
            self.check_host_record(from, principal);
            if oracle::negative_counter(&self.model) {
                self.metrics.invariant_violations += 1;
            }
        } else if self.ledger_len() != from {
            self.metrics.provenance_mismatches += 1;
            self.note(format!(
                "a host write that was {outcome:?} appended a record"
            ));
        }
        Ok(outcome)
    }

    /// The record of a committed host write names the principal and the
    /// verifiers that admitted it.
    fn check_host_record(&mut self, from: usize, principal: &str) {
        let expected_verifiers: Vec<String> = params::VERIFIERS
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        let good = match self.runtime.committed_events().get(from).map(|c| &c.event) {
            Some(LedgerEvent::SemanticDeltaCommitted {
                origin:
                    SemanticOrigin::Host {
                        principal: recorded,
                        verification,
                    },
                ..
            }) => {
                recorded == principal
                    && verification.required == VerificationLevel::Deterministic
                    && verification.level == VerificationLevel::Deterministic
                    && verification.verifiers == expected_verifiers
                    && verification.findings.is_empty()
            }
            _ => false,
        };
        if !good {
            self.metrics.provenance_mismatches += 1;
            self.note(format!(
                "the record of a host write by {principal} is not as admitted"
            ));
        }
    }

    // ---- background actors --------------------------------------------------------

    /// Complete what earlier ticks left pending, then apply `events`.
    pub fn background(&mut self, tick: u64, events: &[Background]) -> Result<(), String> {
        let recommits: Vec<(u64, usize)> = self
            .recommits
            .iter()
            .copied()
            .filter(|(due, _)| *due <= tick)
            .collect();
        self.recommits.retain(|(due, _)| *due > tick);
        for (_, policy) in recommits {
            self.recommit(policy)?;
        }
        let due: Vec<Swap> = self
            .swaps
            .iter()
            .copied()
            .filter(|swap| swap.due <= tick)
            .collect();
        self.swaps.retain(|swap| swap.due > tick);
        for swap in due {
            self.swap(tick, swap)?;
        }
        for event in events {
            match event {
                Background::Ingress { request, text } => self.ingest(request, text)?,
                Background::External { group, item, value } => {
                    let mut delta = Delta::default();
                    delta
                        .upserts
                        .insert(keys::item(*group, *item), Val::text(value.to_string()));
                    self.host_write(&delta, EXTERNAL_PRINCIPAL)?;
                }
                Background::Lifecycle { policy, revoke } => {
                    self.lifecycle(tick, *policy, *revoke)?
                }
                Background::Rewire { group, swap_after } => {
                    self.prime(*group)?;
                    self.swaps.push(Swap {
                        due: tick + swap_after,
                        group: *group,
                        wait: *swap_after,
                    });
                }
            }
        }
        Ok(())
    }

    /// Commit the generation after the one a revocation left live.
    fn recommit(&mut self, policy: usize) -> Result<(), String> {
        let target = keys::policy(policy);
        let live = self.model.lifecycle.live(&target).unwrap_or(0);
        self.commit_lifecycle(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from(PROJECT),
            capsule: CapsuleId::from(target.as_str()),
            generation: Generation(live + 1),
        })
    }

    /// The key that leaves and the key that enters the input set of a
    /// group's total when it is next rewired.
    fn rewire_keys(&self, group: usize) -> (String, String) {
        let last = keys::item(group, params::ITEMS_PER_GROUP - 1);
        if self.model.inputs(&keys::total(group)).contains(&last) {
            (last, keys::spare(group))
        } else {
            (keys::spare(group), last)
        }
    }

    fn number(&self, key: &str) -> Option<i64> {
        self.model.value(key)?.as_text()?.parse().ok()
    }

    /// The sum of the values `inputs` hold in the model, with `overrides`
    /// standing in for what the write that carries the sum changes.
    fn sum(&self, inputs: &BTreeSet<String>, overrides: &BTreeMap<String, i64>) -> Option<i64> {
        inputs
            .iter()
            .map(|input| overrides.get(input).copied().or_else(|| self.number(input)))
            .sum()
    }

    /// Make the key that will leave a group's input set hold what the key
    /// that will enter holds, keeping the total valid.
    fn prime(&mut self, group: usize) -> Result<(), String> {
        let (leaving, entering) = self.rewire_keys(group);
        let total = keys::total(group);
        let (Some(value), inputs) = (self.number(&entering), self.model.inputs(&total)) else {
            return Ok(());
        };
        let overrides = BTreeMap::from([(leaving.clone(), value)]);
        let Some(sum) = self.sum(&inputs, &overrides) else {
            return Ok(());
        };
        let mut delta = Delta::default();
        delta.upserts.insert(leaving, Val::text(value.to_string()));
        delta.upserts.insert(total, Val::text(sum.to_string()));
        self.host_write(&delta, REWIRER_PRINCIPAL)?;
        Ok(())
    }

    /// Swap a group's input set if the two keys still hold the same value,
    /// leaving every value as it was; otherwise prime again and wait.
    fn swap(&mut self, tick: u64, swap: Swap) -> Result<(), String> {
        let (leaving, entering) = self.rewire_keys(swap.group);
        let total = keys::total(swap.group);
        if self.number(&leaving).is_none() || self.number(&leaving) != self.number(&entering) {
            self.prime(swap.group)?;
            self.swaps.push(Swap {
                due: tick + swap.wait,
                ..swap
            });
            return Ok(());
        }
        let mut inputs = self.model.inputs(&total);
        inputs.remove(&leaving);
        inputs.insert(entering);
        let Some(sum) = self.sum(&inputs, &BTreeMap::new()) else {
            return Ok(());
        };
        let mut delta = Delta::default();
        delta
            .upserts
            .insert(total.clone(), Val::text(sum.to_string()));
        delta.dependencies.insert(total, inputs);
        self.host_write(&delta, REWIRER_PRINCIPAL)?;
        Ok(())
    }

    // ---- opening a branch ---------------------------------------------------------

    /// Open a branch on the runtime's current snapshot, run `program` on it,
    /// seal it, and require what it declares to be what the reference model
    /// says the program does.
    pub fn open(
        &mut self,
        id: &str,
        agent: usize,
        program: &Program,
        rely: Option<usize>,
    ) -> Result<Attempt, String> {
        let snapshot = self.runtime.snapshot();
        let mut lifecycle = BTreeMap::new();
        for policy in 0..params::POLICIES {
            let target = keys::policy(policy);
            let live = self
                .runtime
                .live_generation(&target)
                .and_then(|generation| {
                    (self.runtime.generation_validity(&target, generation)
                        == Some(RuntimeValidity::Live))
                    .then_some(generation.0)
                });
            lifecycle.insert(target, live);
        }
        let author = format!("s003-agent-{agent}");
        let mut view = BranchView::open(
            BranchId(id.to_string()),
            PrincipalId::from(author.as_str()),
            snapshot,
            lifecycle,
        );
        program.run(&mut view, rely);
        let opened = view
            .finish()
            .map_err(|error| format!("sealing {id}: {error:?}"))?;
        let mut mirror = RefView::open(&self.model);
        program.run(&mut mirror, rely);
        let (log, footprint, error) = mirror.finish();
        if opened.log != log || opened.footprint != footprint || opened.error != error {
            self.metrics.program_mirror_divergences += 1;
            self.note(format!(
                "{id}: the branch and the model saw {program:?} differently"
            ));
        }
        let differences = sealed_differences(&opened.sealed, &opened.footprint);
        if !differences.is_empty() {
            self.metrics.program_mirror_divergences += 1;
            self.note(format!(
                "{id}: the sealed branch is not what it did: {differences:?}"
            ));
        }
        Ok(Attempt {
            id: BranchId(id.to_string()),
            author,
            program: program.clone(),
            rely,
            sealed: opened.sealed,
            footprint: opened.footprint,
        })
    }

    /// What changes when the program is run again on the model as it is and
    /// its operations are merged at once: the serial execution the merge must
    /// be equivalent to.
    fn serial_net(&self, attempt: &Attempt) -> Option<Net> {
        let mut view = RefView::open(&self.model);
        attempt.program.run(&mut view, attempt.rely);
        let (_, footprint, _) = view.finish();
        let delta = oracle::merge_delta(&footprint.ops, &self.model).ok()?;
        Some(self.model.plan(&delta).ok()?.net)
    }

    fn plan_key(&self, attempt: &Attempt) -> Option<PlanKey> {
        let (rebased, delta) = oracle::plan(&attempt.footprint, &self.model)?;
        Some(PlanKey {
            revision: self.model.revision(),
            rebased,
            delta,
        })
    }

    // ---- merging ------------------------------------------------------------------

    /// Count the hazards a branch runs into, as trials of certification. A
    /// hazard is a trial of the check that looks for it only where that check
    /// ran, and certification stops early: a stale reliance is refused before
    /// any digest is compared, reads, scans, input sets and set members are
    /// compared together, and the rebased keys are worked out only past every
    /// conflict.
    fn count_trials(&mut self, judgement: &oracle::Judgement, rebased: bool) {
        let hazards = &judgement.hazards;
        if hazards.stale_reliance {
            self.metrics.hazard_lifecycle += 1;
            return;
        }
        self.metrics.hazard_write += u64::from(hazards.lost_update);
        self.metrics.hazard_read += u64::from(hazards.stale_read);
        self.metrics.hazard_scan_keys += u64::from(hazards.phantom);
        self.metrics.hazard_scan_values += u64::from(hazards.stale_scan);
        self.metrics.hazard_inputs += u64::from(hazards.stale_input);
        self.metrics.hazard_set_member += u64::from(judgement.set_member_undo);
        if matches!(judgement.predicted, Predicted::Conflict(_)) {
            return;
        }
        self.metrics.hazard_rebase += u64::from(rebased);
        self.metrics.hazard_negative += u64::from(matches!(
            judgement.predicted,
            Predicted::VerificationRejected { .. }
        ));
    }

    /// Merge `attempt` through the runtime under `authority`, compared with
    /// the oracle's judgement of it against the model as it is now.
    pub fn merge(&mut self, attempt: &Attempt, authority: &Authority) -> Result<Settled, String> {
        let judgement = oracle::judge(&attempt.footprint, &self.model);
        let now = self.plan_key(attempt);
        let rebased = oracle::rebased_keys(&attempt.footprint, &self.model);
        if matches!(authority, Authority::Triage { .. }) {
            self.count_trials(&judgement, !rebased.is_empty());
        }
        let before_revision = self.model.revision();
        let from = self.ledger_len();
        let runtime_authority = match authority {
            Authority::Triage { score } => MergeAuthority::Triage {
                score: Probability::new(*score).ok_or("a score outside [0, 1]")?,
            },
            Authority::Reviewed { digest, .. } => MergeAuthority::Reviewed {
                plan_digest: *digest,
                reviewer: PrincipalId::from(params::REVIEWER),
            },
        };
        let started = Instant::now();
        let result = self
            .runtime
            .merge_branch(&attempt.sealed, runtime_authority);
        self.metrics.record_merge(started.elapsed());
        let settled = match result {
            Ok(MergeOutcome::Committed(receipt)) => self.committed(
                attempt,
                authority,
                &judgement,
                &rebased,
                &now,
                before_revision,
                from,
                &receipt,
            )?,
            Ok(MergeOutcome::NoChange(_)) => self.no_change(&judgement, from),
            Ok(MergeOutcome::Held(hold)) => {
                self.held(attempt, authority, &judgement, now, from, &hold)
            }
            Err(error) => self.refused(attempt, authority, &judgement, &now, from, error)?,
        };
        Ok(settled)
    }

    fn require_unchanged(&mut self, from: usize, what: &str) {
        if self.ledger_len() != from {
            self.metrics.provenance_mismatches += 1;
            self.note(format!("{what} appended a record"));
        }
    }

    /// A refusal or hold where the oracle predicted something else.
    fn mismatch(&mut self, judgement: &oracle::Judgement, what: &str) {
        let admitted = matches!(
            judgement.predicted,
            Predicted::Merge { .. } | Predicted::NoChange { .. }
        );
        if admitted {
            self.metrics.spurious_refusals += 1;
        } else {
            self.metrics.refusal_kind_mismatches += 1;
        }
        self.note(format!("{what}, predicted {:?}", judgement.predicted));
    }

    #[allow(clippy::too_many_arguments)]
    fn committed(
        &mut self,
        attempt: &Attempt,
        authority: &Authority,
        judgement: &oracle::Judgement,
        rebased: &BTreeSet<String>,
        now: &Option<PlanKey>,
        before_revision: u64,
        from: usize,
        receipt: &MergeReceipt,
    ) -> Result<Settled, String> {
        let hazards = &judgement.hazards;
        self.metrics.lost_updates += u64::from(hazards.lost_update);
        self.metrics.stale_read_merges += u64::from(hazards.stale_read);
        self.metrics.undetected_phantoms += u64::from(hazards.phantom);
        self.metrics.stale_scan_merges += u64::from(hazards.stale_scan);
        self.metrics.stale_input_merges += u64::from(hazards.stale_input);
        self.metrics.stale_reliance_merges += u64::from(hazards.stale_reliance);
        if !matches!(judgement.predicted, Predicted::Merge { .. }) {
            self.metrics.refusal_kind_mismatches += 1;
            self.note(format!(
                "a merge was committed, predicted {:?}",
                judgement.predicted
            ));
        }
        if let Authority::Reviewed { approved, .. } = authority {
            self.metrics.reviewed_merges += 1;
            if now.as_ref() != Some(approved) {
                self.metrics.approval_bypasses += 1;
                self.note("a merge committed a plan no reviewer approved".into());
            }
        }
        if self.ledger_len() != from + 1 {
            self.metrics.provenance_mismatches += 1;
            self.note("a committed merge did not append exactly one record".into());
        }
        // What the record changes, and what serial execution would change,
        // both against the state before the merge.
        let committed = self.committed_net(from);
        let serial = self.serial_net(attempt);
        if let Some(net) = &committed {
            if oracle::lost_increments_net(&attempt.footprint, &self.model, net) {
                self.metrics.lost_increments += 1;
                self.note(format!("{}: an increment was lost", attempt.id.0));
            }
        }
        if committed.is_none() || committed != serial {
            self.metrics.serialization_divergences += 1;
            self.note(format!(
                "{}: the merged state is not the serial state",
                attempt.id.0
            ));
        }
        self.follow(from)?;
        self.check_merge_record(attempt, authority, rebased, before_revision, from, receipt);
        if oracle::negative_counter(&self.model) {
            self.metrics.invariant_violations += 1;
        }
        let expected_kind = if rebased.is_empty() {
            CertificationKind::Clean
        } else {
            CertificationKind::Rebased
        };
        if receipt.kind != expected_kind || &receipt.rebased != rebased {
            self.metrics.rebase_classification_errors += 1;
            self.note(format!(
                "{}: rebased {:?} as {:?}, predicted {rebased:?}",
                attempt.id.0, receipt.rebased, receipt.kind
            ));
        }
        if rebased.is_empty() {
            self.metrics.merges_clean += 1;
        } else {
            self.metrics.merges_rebased += 1;
        }
        Ok(Settled::Committed)
    }

    /// What the record appended at `from` changes in the model as it is now,
    /// read back from the journal.
    fn committed_net(&self, from: usize) -> Option<Net> {
        let Some(CommittedEvent {
            event: LedgerEvent::SemanticDeltaCommitted { encoded_delta, .. },
            ..
        }) = self.runtime.committed_events().get(from)
        else {
            return None;
        };
        let delta = from_semantic_delta(&SemanticDelta::decode(encoded_delta).ok()?);
        Some(self.model.plan(&delta).ok()?.net)
    }

    fn check_merge_record(
        &mut self,
        attempt: &Attempt,
        authority: &Authority,
        rebased: &BTreeSet<String>,
        before_revision: u64,
        from: usize,
        receipt: &MergeReceipt,
    ) {
        let mut faults: Vec<String> = Vec::new();
        let seal = attempt.sealed.seal_digest();
        match self.runtime.committed_events().get(from) {
            Some(CommittedEvent {
                index,
                event:
                    LedgerEvent::SemanticDeltaCommitted {
                        base_revision,
                        revision,
                        encoded_delta,
                        origin: SemanticOrigin::Merge(record),
                    },
            }) => {
                if base_revision.0 != before_revision || revision.0 != self.model.revision() {
                    faults.push("the record's revisions are not the model's".into());
                }
                if record.branch != attempt.id.0 || record.author != attempt.author {
                    faults.push("the record names another branch or author".into());
                }
                if seal.as_ref().ok() != Some(&record.seal) || record.seal != receipt.seal_digest {
                    faults.push("the record's seal is not the sealed branch's".into());
                }
                let recomputed = merge_plan_digest(
                    &record.branch,
                    Revision(base_revision.0),
                    encoded_delta,
                    &record.dependencies,
                    &record.rebased,
                );
                if record.plan != recomputed || record.plan != receipt.plan_digest {
                    faults.push("the record's plan is not the digest of what it holds".into());
                }
                if &record.rebased != rebased {
                    faults.push("the record's rebased keys are not the predicted ones".into());
                }
                let verification = &record.verification;
                let verifiers: Vec<String> = params::VERIFIERS
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect();
                if verification.required != VerificationLevel::Deterministic
                    || verification.level != VerificationLevel::Deterministic
                    || verification.verifiers != verifiers
                    || !verification.findings.is_empty()
                {
                    faults.push("the record's verification is not what the grant requires".into());
                }
                let authority_ok = match (authority, &record.authority) {
                    (
                        Authority::Triage { score },
                        MergeAuthorityRecord::Triage {
                            policy_version,
                            score_bits,
                        },
                    ) => policy_version == self.policy.version() && *score_bits == score.to_bits(),
                    (Authority::Reviewed { .. }, MergeAuthorityRecord::Reviewed { reviewer }) => {
                        reviewer == params::REVIEWER
                    }
                    _ => false,
                };
                if !authority_ok {
                    faults.push("the record's authority is not the one the merge ran under".into());
                }
                if self.runtime.merged_at(&attempt.id) != Some(*index)
                    || receipt.commit.commit_index != Some(*index)
                {
                    faults.push("the branch is not marked merged at the record's index".into());
                }
            }
            _ => faults.push("the record is not a merge record".into()),
        }
        match (authority, &receipt.triage) {
            (Authority::Triage { .. }, Some(triage)) => {
                if triage.decision() != TriageDecision::AutoPropose
                    || self.policy.policy().explains(triage).is_err()
                    || receipt.policy_version.as_deref() != Some(self.policy.version())
                {
                    faults.push("the triage does not explain the commit".into());
                }
            }
            (Authority::Triage { .. }, None) => faults.push("a triaged merge has no triage".into()),
            (Authority::Reviewed { .. }, triage) => {
                if triage.is_some() {
                    faults.push("a reviewed merge carries a triage".into());
                }
            }
        }
        for fault in faults {
            self.metrics.provenance_mismatches += 1;
            self.note(format!("{}: {fault}", attempt.id.0));
        }
    }

    fn no_change(&mut self, judgement: &oracle::Judgement, from: usize) -> Settled {
        self.require_unchanged(from, "a merge that changes nothing");
        if !matches!(judgement.predicted, Predicted::NoChange { .. }) {
            self.mismatch(judgement, "a merge changed nothing");
        }
        self.metrics.no_change_merges += 1;
        Settled::NoChange
    }

    fn held(
        &mut self,
        attempt: &Attempt,
        authority: &Authority,
        judgement: &oracle::Judgement,
        now: Option<PlanKey>,
        from: usize,
        hold: &MergeHold,
    ) -> Settled {
        self.require_unchanged(from, "a held merge");
        match hold.reason {
            HoldReason::VerificationRejected => {
                if !matches!(judgement.predicted, Predicted::VerificationRejected { .. }) {
                    self.mismatch(judgement, "a merge was held by verification");
                }
                let findings = hold.preview.verdict.hard_findings();
                if !findings
                    .iter()
                    .any(|code| code == "s003-domain-v1/counter-negative")
                {
                    self.metrics.refusal_kind_mismatches += 1;
                    self.note(format!(
                        "{}: held by {findings:?}, not a negative counter",
                        attempt.id.0
                    ));
                }
                self.metrics.verification_holds += 1;
                Settled::Rejected
            }
            HoldReason::Escalated => {
                let admitted = matches!(
                    judgement.predicted,
                    Predicted::Merge { .. } | Predicted::NoChange { .. }
                );
                let escalated = hold.triage.as_ref().is_some_and(|triage| {
                    triage.decision() != TriageDecision::AutoPropose
                        && self.policy.policy().explains(triage).is_ok()
                });
                if !admitted || !escalated || !matches!(authority, Authority::Triage { .. }) {
                    self.mismatch(judgement, "a merge was escalated");
                    self.metrics.provenance_mismatches += u64::from(!escalated);
                }
                self.metrics.escalations += 1;
                match now {
                    Some(plan) => Settled::Escalated {
                        digest: hold.preview.plan_digest,
                        plan,
                    },
                    None => Settled::Refused,
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refused(
        &mut self,
        attempt: &Attempt,
        authority: &Authority,
        judgement: &oracle::Judgement,
        now: &Option<PlanKey>,
        from: usize,
        error: RuntimeError,
    ) -> Result<Settled, String> {
        self.require_unchanged(from, "a refused merge");
        match error {
            RuntimeError::Certification(BranchError::Conflict { keys }) => {
                match &judgement.predicted {
                    Predicted::Conflict(predicted) => {
                        if keys.is_empty() || !keys.is_subset(predicted) {
                            self.metrics.conflict_key_errors += 1;
                            self.note(format!("conflict on {keys:?}, predicted {predicted:?}"));
                        } else if &keys != predicted {
                            self.metrics.conflict_key_set_differs += 1;
                        }
                    }
                    _ => self.mismatch(judgement, "a merge conflicted"),
                }
                self.metrics.conflicts += 1;
                self.describe_refusal(attempt, &keys);
                Ok(Settled::Conflict)
            }
            RuntimeError::Certification(BranchError::LifecycleChanged { targets }) => {
                match &judgement.predicted {
                    Predicted::LifecycleChanged(predicted) => {
                        if &targets != predicted {
                            self.metrics.conflict_key_errors += 1;
                            self.note(format!(
                                "lifecycle change of {targets:?}, predicted {predicted:?}"
                            ));
                        }
                    }
                    _ => self.mismatch(judgement, "a merge found a lifecycle change"),
                }
                self.metrics.lifecycle_refusals += 1;
                Ok(Settled::LifecycleChanged)
            }
            RuntimeError::Certification(error) => {
                let matches_prediction = matches!(
                    (&judgement.predicted, &error),
                    (
                        Predicted::OperationRefused(OpRefusal::NotACounter),
                        BranchError::NotACounter { .. }
                    ) | (
                        Predicted::OperationRefused(OpRefusal::NotASet),
                        BranchError::NotASet { .. }
                    ) | (
                        Predicted::OperationRefused(OpRefusal::EmptyMember),
                        BranchError::InvalidMember { .. }
                    ) | (
                        Predicted::OperationRefused(OpRefusal::Overflow),
                        BranchError::CounterOverflow { .. }
                    )
                );
                if !matches_prediction {
                    self.mismatch(judgement, &format!("certification refused with {error:?}"));
                }
                Ok(Settled::Refused)
            }
            RuntimeError::MergePlanChanged { .. } => {
                let reviewed = match authority {
                    Authority::Reviewed { approved, .. } => Some(approved),
                    Authority::Triage { .. } => None,
                };
                let plan_differs = match (reviewed, now) {
                    (Some(approved), Some(now)) => approved != now,
                    _ => false,
                };
                let reachable = matches!(
                    judgement.predicted,
                    Predicted::Merge { .. }
                        | Predicted::NoChange { .. }
                        | Predicted::VerificationRejected { .. }
                        | Predicted::DeltaRefused(_)
                );
                if !plan_differs || !reachable {
                    self.mismatch(judgement, "a reviewed merge found its plan changed");
                }
                Ok(Settled::PlanChanged)
            }
            RuntimeError::Semantic(_) => {
                if !matches!(judgement.predicted, Predicted::DeltaRefused(_)) {
                    self.mismatch(judgement, "a merge's delta was refused");
                }
                Ok(Settled::Refused)
            }
            RuntimeError::BranchAlreadyMerged { .. } => {
                self.metrics.double_merges += 1;
                self.note(format!("{}: merged twice", attempt.id.0));
                Ok(Settled::Refused)
            }
            other => Err(format!("merging {}: {other:?}", attempt.id.0)),
        }
    }

    /// Descriptive counters of a certification refusal: whether merging it
    /// anyway would have given the serial state, and whether it was a
    /// refused increment written as a value.
    fn describe_refusal(&mut self, attempt: &Attempt, conflicting: &BTreeSet<String>) {
        let merged = oracle::merge_delta(&attempt.footprint.ops, &self.model)
            .ok()
            .and_then(|delta| Some(self.model.plan(&delta).ok()?.net));
        if let (Some(merged), Some(serial)) = (merged, self.serial_net(attempt)) {
            if merged == serial {
                self.metrics.unnecessary_refusals += 1;
            }
        }
        if let Program::Rmw { group, item, .. } = &attempt.program {
            let key = keys::item(*group, *item);
            if conflicting.len() == 1 && conflicting.contains(&key) {
                self.metrics.put_increment_conflicts += 1;
            }
        }
    }

    // ---- baseline arms ------------------------------------------------------------

    /// The delta of a last-writer-wins commit of `attempt`: each key's
    /// operations folded over the value the branch's base held, written
    /// whatever the target holds now.
    pub fn overlay(&self, attempt: &Attempt) -> Option<Delta> {
        let mut finals: BTreeMap<&str, Option<Val>> = BTreeMap::new();
        for op in &attempt.footprint.ops {
            let current = match finals.remove(op.key()) {
                Some(value) => value,
                None => attempt.footprint.touched.get(op.key())?.base.clone(),
            };
            finals.insert(op.key(), op.apply(current.as_ref()).ok()?);
        }
        let mut delta = Delta::default();
        for (key, value) in finals {
            match value {
                Some(value) => {
                    delta.upserts.insert(key.to_string(), value);
                }
                None => {
                    delta.removals.insert(key.to_string());
                }
            }
        }
        Some(delta)
    }

    /// Commit `delta` for a baseline arm and count the hazards it ran into
    /// as evidence, when the commit was not refused.
    pub fn baseline_commit(
        &mut self,
        attempt: &Attempt,
        delta: &Delta,
        principal: &str,
        occ: bool,
    ) -> Result<HostOutcome, String> {
        let hazards = oracle::hazards(&attempt.footprint, &self.model);
        let before: BTreeMap<&str, Option<Val>> = attempt
            .footprint
            .commutative
            .iter()
            .map(|key| (key.as_str(), self.model.value(key).cloned()))
            .collect();
        let outcome = self.host_write(delta, principal)?;
        if outcome == HostOutcome::Committed {
            if occ {
                self.metrics.occ_lost_updates += u64::from(hazards.lost_update);
                self.metrics.occ_stale_scan_commits += u64::from(hazards.stale_scan);
                self.metrics.occ_undetected_phantoms += u64::from(hazards.phantom);
                self.metrics.occ_stale_input_commits += u64::from(hazards.stale_input);
                self.metrics.occ_stale_reliance_commits += u64::from(hazards.stale_reliance);
            } else {
                self.metrics.lww_lost_updates += u64::from(hazards.lost_update);
                let lost = oracle::lost_increments_with(
                    &attempt.footprint,
                    |key| before.get(key).cloned().flatten(),
                    |key| self.model.value(key).cloned(),
                );
                self.metrics.lww_lost_increments += u64::from(lost);
            }
        }
        Ok(outcome)
    }

    // ---- end of a run -------------------------------------------------------------

    /// Rebuild the runtime from its journal, from a compacted snapshot and
    /// (when `durable` names a path) from a file, and compare each with the
    /// live runtime.
    pub fn round_trips(&mut self, durable: Option<&Path>) -> Result<(), String> {
        let events: Vec<CommittedEvent> = self.runtime.committed_events().to_vec();
        self.metrics.replays += 1;
        match PtrRuntime::replay(PtrConfig::default(), &events) {
            Ok(replayed) => self.compare_rebuilt(&replayed, "replay"),
            Err(error) => {
                self.metrics.replay_divergences += 1;
                self.note(format!("the journal does not replay: {error:?}"));
            }
        }
        self.metrics.compaction_roundtrips += 1;
        match self.runtime.export_compacted_snapshot() {
            Ok(snapshot) => match PtrRuntime::restore_compacted(
                PtrConfig::default(),
                snapshot.bytes(),
                snapshot.anchor(),
                &[],
            ) {
                Ok(restored) => self.compare_rebuilt(&restored, "compacted restore"),
                Err(error) => {
                    self.metrics.replay_divergences += 1;
                    self.note(format!("a compacted snapshot does not restore: {error:?}"));
                }
            },
            Err(error) => {
                self.metrics.replay_divergences += 1;
                self.note(format!(
                    "the runtime cannot export a compacted snapshot: {error:?}"
                ));
            }
        }
        if let Some(path) = durable {
            self.metrics.durable_roundtrips += 1;
            self.durable_round_trip(path, &events)?;
        }
        Ok(())
    }

    fn durable_round_trip(&mut self, path: &Path, events: &[CommittedEvent]) -> Result<(), String> {
        let bytes =
            integrity::encode_log(events).map_err(|error| format!("encoding the log: {error}"))?;
        let anchor = integrity::decode_log(&bytes)
            .map_err(|error| format!("decoding the log: {error}"))?
            .anchor();
        drop(
            FileLedger::create_from_log(path, &bytes, anchor)
                .map_err(|error| format!("durable ledger: {error}"))?,
        );
        // The file is this call's from here on, and only then is it removed:
        // a path that was already taken made the creation above fail.
        let _created = Unlink(path);
        match PtrRuntime::open_durable(PtrConfig::default(), path) {
            Ok(reopened) => self.compare_rebuilt(&reopened, "durable reopen"),
            Err(error) => {
                self.metrics.replay_divergences += 1;
                self.note(format!("a durable journal does not reopen: {error:?}"));
            }
        }
        Ok(())
    }

    fn compare_rebuilt(&mut self, rebuilt: &PtrRuntime, how: &str) {
        let mut differences: Vec<String> = Vec::new();
        if let Some(difference) = state_difference(rebuilt, &self.model) {
            differences.push(difference);
        }
        let live = self.runtime.materialized_state();
        let other = rebuilt.materialized_state();
        if how != "compacted restore"
            && (live.values != other.values || live.last_applied != other.last_applied)
        {
            differences.push("the materialized entries differ".into());
        }
        if how == "compacted restore" && live.last_applied != other.last_applied {
            differences.push("the restored runtime applied another index".into());
        }
        if !differences.is_empty() {
            self.metrics.replay_divergences += 1;
            self.note(format!(
                "{how} differs from the live runtime: {differences:?}"
            ));
        }
    }
}

/// Whether the runtime holds `value` where the model holds `expected`.
fn same_value(value: &SemanticValue, expected: &Val) -> bool {
    match (value, expected) {
        (SemanticValue::Text(text), Val::Text(expected)) => text == expected,
        (
            SemanticValue::Payload(payload),
            Val::Payload {
                type_id,
                source,
                bytes,
            },
        ) => payload.type_id.0 == *type_id && payload.source == *source && payload.bytes == *bytes,
        _ => false,
    }
}

/// How a runtime's state differs from a model's: its revision, every value
/// and input set, and the validity of every generation up to one past each
/// live one.
pub fn state_difference(runtime: &PtrRuntime, model: &Model) -> Option<String> {
    if runtime.revision().0 != model.revision() {
        return Some(format!(
            "revision {} against {}",
            runtime.revision().0,
            model.revision()
        ));
    }
    let snapshot: SemanticSnapshot = runtime.snapshot();
    let mut expected = model.values().iter();
    for key in snapshot.keys() {
        let Some((model_key, model_value)) = expected.next() else {
            return Some(format!("the runtime holds {key}, the model does not"));
        };
        if model_key != key {
            return Some(format!("the runtime holds {key}, the model {model_key}"));
        }
        let held = snapshot.value(key).expect("a listed key holds a value");
        if !same_value(held, model_value) {
            return Some(format!("value of {key}"));
        }
    }
    if let Some((model_key, _)) = expected.next() {
        return Some(format!("the model holds {model_key}, the runtime does not"));
    }
    for (derived, inputs) in model.dependencies() {
        if !snapshot
            .inputs(derived)
            .eq(inputs.iter().map(String::as_str))
        {
            return Some(format!("inputs of {derived}"));
        }
    }
    // The other direction, for the families that hold derived keys: the
    // runtime records no input set that the model does not.
    let families: BTreeSet<&str> = model
        .dependencies()
        .keys()
        .filter_map(|key| key.split_once(':').map(|(family, _)| family))
        .collect();
    for family in families {
        let prefix = format!("{family}:");
        for key in snapshot.keys().filter(|key| key.starts_with(&prefix)) {
            if !model.dependencies().contains_key(key) && snapshot.inputs(key).next().is_some() {
                return Some(format!("inputs of {key}"));
            }
        }
    }
    for policy in 0..params::POLICIES {
        let target = keys::policy(policy);
        let live = model.lifecycle.live(&target);
        if runtime
            .live_generation(&target)
            .map(|generation| generation.0)
            != live
        {
            return Some(format!("the live generation of {target}"));
        }
        for generation in 1..=live.unwrap_or(0) + 1 {
            let expected = model
                .lifecycle
                .validity(&target, generation)
                .map(|validity| match validity {
                    Validity::Live => RuntimeValidity::Live,
                    Validity::Superseded => RuntimeValidity::Superseded,
                    Validity::Revoked => RuntimeValidity::Revoked,
                });
            if runtime.generation_validity(&target, Generation(generation)) != expected {
                return Some(format!("the validity of {target} at {generation}"));
            }
        }
    }
    None
}

/// Removes the file at its path when dropped: made only for a file this run created.
struct Unlink<'a>(&'a Path);

impl Drop for Unlink<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(kind: GrantKind) -> World {
        World::new(&Case::new(17, 0), kind).expect("a world")
    }

    fn clean(world: &World) {
        assert_eq!(world.notes, Vec::<String>::new());
        assert_eq!(world.metrics.hard(), Metrics::default().hard());
    }

    fn auto() -> Authority {
        Authority::Triage { score: 1.0 }
    }

    fn open(world: &mut World, id: &str, program: Program) -> Attempt {
        world.open(id, 0, &program, None).expect("an attempt")
    }

    fn rmw(item: usize, delta: i64) -> Program {
        Program::Rmw {
            group: 0,
            item,
            delta,
        }
    }

    fn host(world: &mut World, key: &str, value: Val) -> HostOutcome {
        let mut delta = Delta::default();
        delta.upserts.insert(key.to_string(), value);
        world.host_write(&delta, "s003-test").expect("a host write")
    }

    #[test]
    fn a_new_world_agrees_with_its_model_and_has_counted_nothing_wrong() {
        let world = world(GrantKind::Auto);
        assert_eq!(state_difference(&world.runtime, &world.model), None);
        clean(&world);
        // The four policy capsules and the genesis write.
        assert_eq!(world.runtime.committed_events().len(), params::POLICIES + 1);
        assert_eq!(world.model.lifecycle.live("policy-3"), Some(1));
    }

    #[test]
    fn lifecycle_record_is_checked_before_the_model_learns_it() {
        let mut world = world(GrantKind::Auto);
        let from = world.ledger_len();
        world
            .runtime
            .commit(LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from("policy-0"),
                old: Generation(1),
                new: Generation(3),
            })
            .expect("a well-formed but wrong event");
        world.check_lifecycle_record(
            from,
            &LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from("policy-0"),
                old: Generation(1),
                new: Generation(2),
            },
        );
        assert_eq!(world.metrics.provenance_mismatches, 1);
        world.follow(from).expect("follow recorded lifecycle");
        assert_eq!(state_difference(&world.runtime, &world.model), None);
        assert!(world.metrics.hard().iter().any(|(_, count)| *count > 0));
    }

    #[test]
    fn ingress_record_is_checked_against_requested_identity_and_content() {
        for (recorded_request, recorded_text) in [("other", "expected"), ("r1", "wrong")] {
            let mut world = world(GrantKind::Auto);
            let from = world.ledger_len();
            world
                .runtime
                .ingest_text(RequestId::from(recorded_request), recorded_text)
                .expect("a well-formed but wrong ingress");
            world
                .check_ingress_record(from, "r1", "expected")
                .expect("prediction");
            assert_eq!(world.metrics.model_divergences, 1);
            assert_eq!(
                world.metrics.provenance_mismatches,
                u64::from(recorded_request != "r1")
            );
            world.follow(from).expect("follow recorded ingress");
            assert_eq!(state_difference(&world.runtime, &world.model), None);
        }
    }

    #[test]
    fn identical_ingress_is_a_clean_no_change_and_unexpected_records_are_counted() {
        let mut world = world(GrantKind::Auto);
        world.ingest("r1", "same").expect("first ingress");
        let from = world.ledger_len();
        world.ingest("r1", "same").expect("unchanged ingress");
        assert_eq!(world.ledger_len(), from);
        clean(&world);
        world
            .runtime
            .ingest_text(RequestId::from("r2"), "unexpected")
            .expect("extra record");
        world
            .check_ingress_record(from, "r1", "same")
            .expect("prediction");
        assert_eq!(world.metrics.provenance_mismatches, 1);
    }

    #[test]
    fn a_host_write_is_predicted_and_its_record_names_the_principal() {
        let mut world = world(GrantKind::Auto);
        assert_eq!(
            host(&mut world, "item:0:0", Val::text("77")),
            HostOutcome::Committed
        );
        assert_eq!(
            host(&mut world, "item:0:0", Val::text("77")),
            HostOutcome::NoChange
        );
        assert_eq!(
            host(&mut world, "ctr:0", Val::counter(-1, "s003-test")),
            HostOutcome::Refused
        );
        assert_eq!(
            host(&mut world, "ctr:0", Val::counter(500, "s003-test")),
            HostOutcome::Refused
        );
        assert_eq!(
            host(&mut world, "stray:0", Val::text("1")),
            HostOutcome::Refused
        );
        clean(&world);
        assert_eq!(world.model.value("item:0:0"), Some(&Val::text("77")));
    }

    #[test]
    fn a_clean_merge_commits_and_every_check_accepts_its_record() {
        let mut world = world(GrantKind::Auto);
        let attempt = open(&mut world, "b-1", rmw(0, 3));
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::Committed)
        ));
        clean(&world);
        assert_eq!(world.metrics.merges_clean, 1);
        assert_eq!(world.metrics.merge_calls, 1);
        assert!(world.runtime.merged_at(&attempt.id).is_some());
    }

    #[test]
    fn an_overwritten_key_that_moved_is_a_conflict_the_oracle_predicted() {
        let mut world = world(GrantKind::Auto);
        let first = open(&mut world, "b-1", rmw(0, 3));
        let second = open(&mut world, "b-2", rmw(0, 4));
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&second, &auto()),
            Ok(Settled::Conflict)
        ));
        clean(&world);
        assert_eq!(world.metrics.conflicts, 1);
        assert_eq!(world.metrics.hazard_write, 1);
        assert_eq!(world.metrics.put_increment_conflicts, 1);
        assert_eq!(world.metrics.unnecessary_refusals, 0);
        assert!(world.runtime.merged_at(&second.id).is_none());
    }

    #[test]
    fn a_scanned_prefix_that_gained_a_key_is_a_phantom_and_refused() {
        let mut world = world(GrantKind::Auto);
        let first = open(
            &mut world,
            "b-1",
            Program::InsertCapped { group: 0, task: 1 },
        );
        let second = open(
            &mut world,
            "b-2",
            Program::InsertCapped { group: 0, task: 2 },
        );
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&second, &auto()),
            Ok(Settled::Conflict)
        ));
        clean(&world);
        assert_eq!(world.metrics.hazard_scan_keys, 1);
    }

    #[test]
    fn a_scan_whose_values_moved_is_stale_and_refused() {
        let mut world = world(GrantKind::Auto);
        let audit = open(&mut world, "b-1", Program::Audit { group: 0 });
        let bump = open(&mut world, "b-2", rmw(1, 2));
        assert!(matches!(
            world.merge(&bump, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&audit, &auto()),
            Ok(Settled::Conflict)
        ));
        clean(&world);
        assert_eq!(world.metrics.hazard_scan_values, 1);
    }

    #[test]
    fn a_write_skew_is_a_stale_read_and_refused() {
        let mut world = world(GrantKind::Auto);
        host(&mut world, "item:0:0", Val::text("40"));
        host(&mut world, "item:0:1", Val::text("40"));
        let skew_a = open(
            &mut world,
            "b-1",
            Program::WriteSkew {
                group: 0,
                first: 0,
                second: 1,
            },
        );
        let skew_b = open(
            &mut world,
            "b-2",
            Program::WriteSkew {
                group: 0,
                first: 1,
                second: 0,
            },
        );
        assert!(matches!(
            world.merge(&skew_a, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&skew_b, &auto()),
            Ok(Settled::Conflict)
        ));
        clean(&world);
        assert_eq!(world.metrics.hazard_read, 1);
    }

    #[test]
    fn concurrent_increments_are_rebased_and_neither_is_lost() {
        let mut world = world(GrantKind::Auto);
        let first = open(
            &mut world,
            "b-1",
            Program::CounterAdd {
                counter: 0,
                amount: 3,
            },
        );
        let second = open(
            &mut world,
            "b-2",
            Program::CounterAdd {
                counter: 0,
                amount: 4,
            },
        );
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&second, &auto()),
            Ok(Settled::Committed)
        ));
        clean(&world);
        assert_eq!(
            world.model.value("ctr:0").and_then(Val::as_counter),
            Some(57)
        );
        assert_eq!(
            (world.metrics.merges_clean, world.metrics.merges_rebased),
            (1, 1)
        );
        assert_eq!(world.metrics.hazard_rebase, 1);
    }

    #[test]
    fn a_merge_that_would_drive_a_counter_negative_is_held_by_verification() {
        let mut world = world(GrantKind::Auto);
        let attempt = open(
            &mut world,
            "b-1",
            Program::CounterAdd {
                counter: 0,
                amount: -7,
            },
        );
        host(&mut world, "ctr:0", Val::counter(5, "s003-test"));
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::Rejected)
        ));
        clean(&world);
        assert_eq!(world.metrics.verification_holds, 1);
        assert_eq!(world.metrics.hazard_negative, 1);
        assert!(world.runtime.merged_at(&attempt.id).is_none());
    }

    #[test]
    fn a_merge_that_changes_nothing_is_reported_and_leaves_the_branch_unmerged() {
        let mut world = world(GrantKind::Auto);
        let mut members = world
            .model
            .value("set:0")
            .and_then(Val::as_set)
            .expect("a set");
        let absent = (0..params::SET_MEMBERS)
            .find(|member| !members.contains(&keys::member(*member)))
            .expect("a member the set does not hold");
        let attempt = open(
            &mut world,
            "b-1",
            Program::SetOp {
                set: 0,
                member: absent,
                insert: true,
            },
        );
        // Meanwhile the state came to hold the member, written exactly as a merge writes a set.
        members.insert(keys::member(absent));
        host(
            &mut world,
            "set:0",
            Val::set(&members, super::super::model::OP_SOURCE),
        );
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::NoChange)
        ));
        clean(&world);
        assert_eq!(world.metrics.no_change_merges, 1);
        assert!(world.runtime.merged_at(&attempt.id).is_none());
    }

    #[test]
    fn a_relied_generation_that_is_superseded_refuses_the_merge() {
        let mut world = world(GrantKind::Auto);
        let attempt = world
            .open("b-1", 0, &rmw(0, 1), Some(0))
            .expect("an attempt");
        assert_eq!(attempt.footprint.relied.get("policy-0"), Some(&1));
        world
            .background(
                5,
                &[Background::Lifecycle {
                    policy: 0,
                    revoke: false,
                }],
            )
            .expect("a supersession");
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::LifecycleChanged)
        ));
        clean(&world);
        assert_eq!(world.metrics.lifecycle_refusals, 1);
        assert_eq!(world.metrics.hazard_lifecycle, 1);
    }

    #[test]
    fn a_revoked_generation_is_recommitted_a_tick_later_and_a_new_branch_relies_on_it() {
        let mut world = world(GrantKind::Auto);
        world
            .background(
                5,
                &[Background::Lifecycle {
                    policy: 1,
                    revoke: true,
                }],
            )
            .expect("a revocation");
        let during = world
            .open("b-1", 0, &rmw(0, 1), Some(1))
            .expect("an attempt");
        assert!(
            during.footprint.relied.is_empty(),
            "a revoked generation is not relied on"
        );
        world.background(6, &[]).expect("the recommit");
        assert_eq!(world.model.lifecycle.live("policy-1"), Some(2));
        let after = world
            .open("b-2", 0, &rmw(0, 1), Some(1))
            .expect("an attempt");
        assert_eq!(after.footprint.relied.get("policy-1"), Some(&2));
        clean(&world);
    }

    #[test]
    fn an_escalated_merge_commits_only_the_plan_a_reviewer_approved() {
        let mut world = world(GrantKind::Review);
        let approved = open(&mut world, "b-1", rmw(0, 3));
        let Ok(Settled::Escalated { digest, plan }) =
            world.merge(&approved, &Authority::Triage { score: 0.1 })
        else {
            panic!("a score below the threshold is escalated");
        };
        assert_eq!(world.metrics.escalations, 1);
        let review = Authority::Reviewed {
            digest,
            approved: plan,
        };
        assert!(matches!(
            world.merge(&approved, &review),
            Ok(Settled::Committed)
        ));
        assert_eq!(world.metrics.reviewed_merges, 1);

        let voided = open(&mut world, "b-2", rmw(1, 3));
        let Ok(Settled::Escalated { digest, plan }) =
            world.merge(&voided, &Authority::Triage { score: 0.1 })
        else {
            panic!("a score below the threshold is escalated");
        };
        host(&mut world, "item:5:5", Val::text("9"));
        let review = Authority::Reviewed {
            digest,
            approved: plan,
        };
        assert!(matches!(
            world.merge(&voided, &review),
            Ok(Settled::PlanChanged)
        ));
        clean(&world);
    }

    #[test]
    fn the_rewirer_swaps_a_totals_input_set_without_moving_a_value_and_that_conflicts_a_group_total(
    ) {
        let mut world = world(GrantKind::Auto);
        let attempt = open(
            &mut world,
            "b-1",
            Program::GroupTotal {
                group: 3,
                item: 0,
                delta: 2,
            },
        );
        assert_eq!(
            attempt
                .footprint
                .touched
                .get("total:3")
                .map(|touched| touched.inputs.len()),
            Some(8)
        );
        let before = world.model.value("total:3").cloned();
        world
            .background(
                10,
                &[Background::Rewire {
                    group: 3,
                    swap_after: 5,
                }],
            )
            .expect("a prime");
        world.background(15, &[]).expect("the swap");
        assert_eq!(world.model.value("total:3").cloned(), before);
        assert!(world.model.inputs("total:3").contains("item:3:s"));
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::Conflict)
        ));
        clean(&world);
        assert_eq!(world.metrics.hazard_inputs, 1);
    }

    #[test]
    fn round_trips_of_the_journal_agree_with_the_live_runtime() {
        let mut world = world(GrantKind::Auto);
        for (index, delta) in [1, 2, 3].into_iter().enumerate() {
            let attempt = open(&mut world, &format!("b-{index}"), rmw(index, delta));
            assert!(matches!(
                world.merge(&attempt, &auto()),
                Ok(Settled::Committed)
            ));
        }
        world
            .background(
                4,
                &[Background::Ingress {
                    request: "r4".into(),
                    text: "hello".into(),
                }],
            )
            .expect("ingress");
        let path =
            std::env::temp_dir().join(format!("ptr-s003-world-test-{}.log", std::process::id()));
        world.round_trips(Some(&path)).expect("round trips");
        clean(&world);
        assert_eq!(
            (
                world.metrics.replays,
                world.metrics.compaction_roundtrips,
                world.metrics.durable_roundtrips
            ),
            (1, 1, 1)
        );
        assert!(!path.exists());
    }

    #[test]
    fn a_stale_reliance_is_refused_before_any_digest_so_no_other_hazard_is_a_trial() {
        // A branch that relies on a superseded generation is refused before
        // any digest is compared, so the read it also has stale is no trial.
        let mut world = world(GrantKind::Auto);
        let relying = world
            .open("b-1", 0, &rmw(0, 1), Some(1))
            .expect("an attempt");
        world
            .background(
                5,
                &[Background::Lifecycle {
                    policy: 1,
                    revoke: false,
                }],
            )
            .expect("a supersession");
        assert_eq!(
            host(&mut world, "item:0:0", Val::text("77")),
            HostOutcome::Committed
        );
        assert!(matches!(
            world.merge(&relying, &auto()),
            Ok(Settled::LifecycleChanged)
        ));
        assert_eq!(world.metrics.hazard_lifecycle, 1);
        assert_eq!(world.metrics.hazard_read + world.metrics.hazard_write, 0);
        clean(&world);
    }

    #[test]
    fn a_conflict_is_returned_before_the_rebased_keys_are_worked_out() {
        // The decrement's stale read is a trial, its rebased counter is not.
        let mut world = world(GrantKind::Auto);
        let decrement = open(&mut world, "b-1", Program::GuardedDecrement { counter: 3 });
        let add = open(
            &mut world,
            "b-2",
            Program::CounterAdd {
                counter: 3,
                amount: 5,
            },
        );
        assert!(matches!(world.merge(&add, &auto()), Ok(Settled::Committed)));
        assert!(matches!(
            world.merge(&decrement, &auto()),
            Ok(Settled::Conflict)
        ));
        assert_eq!(world.metrics.hazard_read, 1);
        assert_eq!(world.metrics.hazard_rebase, 0);
        clean(&world);
    }

    #[test]
    fn past_every_conflict_a_rebased_counter_is_a_trial() {
        let mut world = world(GrantKind::Auto);
        let first = open(
            &mut world,
            "b-1",
            Program::CounterAdd {
                counter: 4,
                amount: 2,
            },
        );
        let second = open(
            &mut world,
            "b-2",
            Program::CounterAdd {
                counter: 4,
                amount: 3,
            },
        );
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        assert!(matches!(
            world.merge(&second, &auto()),
            Ok(Settled::Committed)
        ));
        assert_eq!(world.metrics.hazard_rebase, 1);
        clean(&world);
    }

    #[test]
    fn last_writer_wins_commits_a_stale_overlay_and_the_lost_update_is_evidence_not_failure() {
        let mut world = world(GrantKind::Auto);
        let first = open(&mut world, "b-1", rmw(0, 3));
        let second = open(&mut world, "b-2", rmw(0, 4));
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        let delta = world.overlay(&second).expect("an overlay");
        let outcome = world
            .baseline_commit(&second, &delta, "s003-lww", false)
            .expect("a commit");
        assert_eq!(outcome, HostOutcome::Committed);
        assert_eq!(world.metrics.lww_lost_updates, 1);
        clean(&world);
    }

    #[test]
    fn key_level_occ_commits_a_stale_scan_and_counts_it_beside_phantoms() {
        let mut world = world(GrantKind::Auto);
        // A first writer moves an item, so an audit opened now records a sum
        // that differs from the audit key's value and has something to write.
        let first = open(&mut world, "b-1", rmw(0, 3));
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        let audit = open(&mut world, "b-2", Program::Audit { group: 0 });
        let writer = open(&mut world, "b-3", rmw(1, 4));
        assert!(matches!(
            world.merge(&writer, &auto()),
            Ok(Settled::Committed)
        ));
        // The audit read no value that moved and the scanned keys are the
        // same, so key-level OCC has nothing to refuse: it commits a branch
        // whose scanned values are stale.
        let delta = oracle::merge_delta(&audit.footprint.ops, &world.model).expect("a delta");
        let outcome = world
            .baseline_commit(&audit, &delta, "s003-occ", true)
            .expect("a commit");
        assert_eq!(outcome, HostOutcome::Committed);
        assert_eq!(world.metrics.occ_stale_scan_commits, 1);
        assert_eq!(world.metrics.occ_undetected_phantoms, 0);
        assert_eq!(world.metrics.occ_lost_updates, 0);
        clean(&world);
    }

    #[test]
    fn last_writer_wins_loses_a_concurrent_increment() {
        let mut world = world(GrantKind::Auto);
        let first = open(
            &mut world,
            "b-1",
            Program::CounterAdd {
                counter: 2,
                amount: 6,
            },
        );
        let second = open(
            &mut world,
            "b-2",
            Program::CounterAdd {
                counter: 2,
                amount: 5,
            },
        );
        assert!(matches!(
            world.merge(&first, &auto()),
            Ok(Settled::Committed)
        ));
        let delta = world.overlay(&second).expect("an overlay");
        world
            .baseline_commit(&second, &delta, "s003-lww", false)
            .expect("a commit");
        assert_eq!(world.metrics.lww_lost_increments, 1);
        assert_eq!(
            world.model.value("ctr:2").and_then(Val::as_counter),
            Some(55)
        );
        clean(&world);
    }

    #[test]
    fn a_model_that_differs_from_the_runtime_is_a_divergence() {
        let mut world = world(GrantKind::Auto);
        let mut delta = Delta::default();
        delta.upserts.insert("item:0:0".into(), Val::text("999"));
        world.model.apply(&delta).expect("the model alone moves");
        world.cross_check();
        assert_eq!(world.metrics.model_divergences, 1);
        assert_eq!(world.notes.len(), 1);
    }

    #[test]
    fn a_merge_the_oracle_judged_a_hazard_is_counted_when_the_runtime_commits_it() {
        let mut world = world(GrantKind::Auto);
        let attempt = open(&mut world, "b-1", rmw(0, 3));
        // Only the model learns that the key the branch read has moved, so the
        // oracle sees a lost update the runtime does not.
        let mut delta = Delta::default();
        delta.upserts.insert("item:0:0".into(), Val::text("999"));
        world.model.apply(&delta).expect("the model alone moves");
        assert!(matches!(
            world.merge(&attempt, &auto()),
            Ok(Settled::Committed)
        ));
        assert_eq!(world.metrics.lost_updates, 1);
        assert_eq!(world.metrics.refusal_kind_mismatches, 1);
        assert!(world.metrics.model_divergences >= 1);
        assert!(world.metrics.hard_failures() >= 3);
    }
}
