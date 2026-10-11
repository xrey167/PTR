//! The cases of R013: dormant seeds (`dormant`), lineage and transitive authority (`lineage`),
//! in-flight calls racing a base-model or adapter invalidation (`inflight`) and the minimal
//! attestor mechanism (`attestor`), each driving the PUBLIC runtime API on R011's
//! [`Rt`] (a `PtrRuntime` over a durable `FileLedger` in a temporary directory).
//!
//! Counting rules (R011's, restated):
//!
//! * An **attempt** counts for its case only after the call returned `Ok` (the runtime let it
//!   through: a violation the caller counts) or the expected typed error ([`Cx::verdict`]). Any
//!   other error is a harness error (`Err`, exit status 2), never a pass, and so is a positive
//!   control that fails.
//! * A **route probe** of an expert that must be refused is kind-checked.
//! * A **violation** counter (`*_admitted`, `routable_*`, `*_after_invalidation`) counts what
//!   the runtime did wrong; the sub-case that produced it is recorded under `subcases`.
//! * A metric family that no case filled stays unseeded, so its coverage rule fails (a case that
//!   never ran is an unmet prerequisite, not a silent zero).

pub mod attestor;
pub mod dormant;
pub mod inflight;
pub mod lineage;

use super::{metrics, params};
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::r011::cases::fixture::{report_for, ReportSpec, Rt, Setup};
use crate::experiments::r011::cases::{classify, Verdict};
use ptr_pods::{ExpertInputs, ExpertProposal, StructuralOp};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, ExpertAdmissionError, ExpertAttestation,
};
use ptr_runtime::RuntimeError;
use ptr_types::{CommitIndex, Generation};
use std::collections::{BTreeMap, BTreeSet};

/// An FNV-1a hash of a label, to derive one shuffle stream per label from the pinned domain.
fn label_hash(label: &str) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in label.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// The state of a run: the tally, the case being counted and the fixture choices the seed made.
pub struct Cx {
    pub tally: Tally,
    pub seed: u64,
    case: String,
    counts: BTreeMap<String, i64>,
    sub: String,
    subs: BTreeMap<String, i64>,
    /// The base names of the metrics a rule bounds above by zero: a positive increment of one
    /// of them is a violation of the current sub-case.
    violating: BTreeSet<String>,
    pub cases: Vec<(String, Json)>,
    /// The fixture choices the seed made, recorded for the reader.
    pub fixture: Vec<(String, Json)>,
}

impl Cx {
    pub fn new(seed: u64) -> Self {
        let mut tally = Tally::new(metrics::LIST);
        let mut violating = BTreeSet::new();
        // Every counter whose rule bounds it above is seeded at zero, so a run in which the
        // violation never happens prints a zero; the prerequisites are never seeded.
        for rule in metrics::falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            if let [left, "<=", "0"] = tokens[..] {
                tally.add(left, 0);
                violating.insert(left.split('.').next().unwrap_or(left).to_string());
            }
        }
        Self {
            tally,
            seed,
            case: String::new(),
            counts: BTreeMap::new(),
            sub: String::new(),
            subs: BTreeMap::new(),
            violating,
            cases: Vec::new(),
            fixture: Vec::new(),
        }
    }

    pub fn begin(&mut self, case: &str) {
        self.case = case.to_string();
        self.counts.clear();
        self.sub = case.to_string();
    }

    /// Name the sub-case whose violations the next counts belong to.
    pub fn sub(&mut self, label: &str) {
        self.sub = label.to_string();
        self.subs.entry(label.to_string()).or_insert(0);
    }

    pub fn end(&mut self) {
        let counts = std::mem::take(&mut self.counts);
        self.cases.push((
            self.case.clone(),
            Json::Obj(
                counts
                    .into_iter()
                    .map(|(name, value)| (name, Json::Int(value)))
                    .collect(),
            ),
        ));
    }

    /// Add `amount` to `name` in the tally and in the current case.
    pub fn count(&mut self, name: &str, amount: i64) {
        self.tally.add(name, amount);
        *self.counts.entry(name.to_string()).or_insert(0) += amount;
        let base = name.split('.').next().unwrap_or(name);
        if amount > 0 && self.violating.contains(base) {
            let sub = self.sub.clone();
            *self.subs.entry(sub).or_insert(0) += amount;
        }
    }

    /// A member of a family, `family.member`.
    pub fn count_in(&mut self, family: &str, member: &str, amount: i64) {
        self.count(&format!("{family}.{member}"), amount);
    }

    /// One internal counter (not a metric), shown under `internal` and in the case.
    pub fn note(&mut self, name: &str) {
        self.count(name, 1);
    }

    /// Count the attempt of the adversarial case `case`. The name must be one of the pinned
    /// suite: a name the config does not list is a bug of the harness.
    pub fn attempt(&mut self, case: &str) -> Res<()> {
        if !params::ADVERSARIAL_CASES.contains(&case) {
            return Err(format!("{case} is not a case of the pinned suite"));
        }
        self.count_in("adversarial_attempts_per_case", case, 1);
        Ok(())
    }

    /// Classify the result of an attempt of the adversarial case `case` and count the attempt
    /// (only an `Ok` or an expected refusal reached the boundary the case names).
    pub fn verdict<T: std::fmt::Debug>(
        &mut self,
        case: &str,
        what: &str,
        result: Result<T, RuntimeError>,
        expected: impl Fn(&RuntimeError) -> bool,
    ) -> Res<Verdict> {
        let verdict = classify(what, result, expected)?;
        self.attempt(case)?;
        Ok(verdict)
    }

    /// The items of `items` in the order the seed draws for `label`, recorded as a fixture
    /// choice. The order never changes what is measured, only the order it is run in.
    pub fn shuffled<T: Clone>(&mut self, label: &str, items: &[T]) -> Vec<T> {
        let order = corpus::order(
            self.seed,
            params::SCENARIO_ORDER_LABEL ^ label_hash(label),
            items.len(),
        );
        self.fixture.push((
            label.to_string(),
            Json::ints(order.iter().map(|index| *index as i64)),
        ));
        order
            .into_iter()
            .map(|index| items[index].clone())
            .collect()
    }

    /// The registry of `rt` against a fresh replay of its committed history: a difference is a
    /// finding (`registry_replay_mismatches`), never a harness error.
    pub fn check_replay(&mut self, rt: &Rt) -> Res<()> {
        let replayed = rt.replayed()?;
        let equal = replayed.expert_registry() == rt.runtime.expert_registry();
        self.count("registry_replay_mismatches", i64::from(!equal));
        Ok(())
    }

    /// The violations of each sub-case, for the result.
    pub fn subcases_json(&self) -> Json {
        Json::Obj(
            self.subs
                .iter()
                .map(|(label, violations)| {
                    (
                        label.clone(),
                        Json::obj([("violations", Json::Int(*violations))]),
                    )
                })
                .collect(),
        )
    }
}

/// Run every case family once, in the seed's order. A panic of the code under test inside a
/// family is contained, counted as `product_panics` and ends that family only.
pub fn run_all(cx: &mut Cx) -> Res<Vec<Json>> {
    type Case = (&'static str, fn(&mut Cx) -> Res<()>);
    let cases: [Case; 6] = [
        ("F1", dormant::revoke_while_retired),
        ("F2", lineage::transitive_authority),
        ("F3", lineage::superseded_sources),
        ("F4", inflight::structural_races),
        ("F5", attestor::attestation_binding),
        ("F6", attestor::attestor_revocation),
    ];
    let mut order = Vec::new();
    for index in corpus::order(cx.seed, params::CASE_ORDER_LABEL, cases.len()) {
        let (name, case) = cases[index];
        cx.begin(name);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| case(cx)));
        match outcome {
            Ok(result) => result.map_err(|error| format!("{name}: {error}"))?,
            Err(payload) => {
                eprintln!(
                    "{}: {name}: the code under test panicked: {}",
                    params::BENCHMARK,
                    panic_text(payload.as_ref())
                );
                cx.count("product_panics", 1);
            }
        }
        cx.end();
        order.push(Json::str(name));
    }
    Ok(order)
}

/// The text a panic carried.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no text".into())
}

// ---- the regime the authority cases start from ------------------------------------------------

/// The gate standing of the regime the cases start from: gate epoch 2 (configuration version
/// 2) and evaluator epoch 1.
pub const GATE_START: u64 = 2;
pub const EVALUATOR_START: u64 = 1;

/// The setup of the evidence the regime's own experts were admitted under.
pub fn regime() -> Setup {
    Setup {
        epoch: GATE_START,
        version: GATE_START as u8,
        evaluator: Some(EVALUATOR_START as u8),
    }
}

/// A runtime at the regime: both authority kinds have a standing (set by non-tightening
/// rotations that fence nothing).
pub fn regime_rt() -> Res<Rt> {
    use ptr_ledger::{ExpertAuthorityClass, ExpertAuthorityKind};
    let mut rt = Rt::fresh()?;
    for (kind, epoch) in [
        (ExpertAuthorityKind::GateConfig, GATE_START),
        (ExpertAuthorityKind::Evaluator, EVALUATOR_START),
    ] {
        let outcome = rt
            .change_authority(kind, epoch, ExpertAuthorityClass::NonTightening)
            .map_err(|error| format!("starting the regime: {error:?}"))?;
        if outcome.fence_count() != 0 {
            return Err("starting the regime fenced experts".to_string());
        }
    }
    Ok(rt)
}

/// Admit `op` with the dependency block `inputs` (when given) under `setup`, with a correct
/// report: a setup step, a harness error when refused.
pub fn must_under(
    rt: &mut Rt,
    what: &str,
    op: StructuralOp,
    inputs: Option<ExpertInputs>,
    setup: Setup,
) -> Res<()> {
    rt.admit_as(op, inputs, &ReportSpec::correct_under(setup), setup)
        .map(|_| ())
        .map_err(|error| format!("{what}: {error:?}"))
}

// ---- attestors --------------------------------------------------------------------------------

/// A trusted training attestor of the harness: a name and an epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Attestor {
    pub name: &'static str,
    pub epoch: u64,
}

impl Attestor {
    pub const fn new(name: &'static str, epoch: u64) -> Self {
        Self { name, epoch }
    }

    /// The digest the attestor's input identity carries at its epoch.
    pub fn digest(&self) -> [u8; 32] {
        let mut digest = [0xA0; 32];
        digest[0] = self.name.as_bytes()[0];
        digest[1] = self.epoch as u8;
        digest
    }

    /// The identity of the attestor as an input.
    pub fn identity(&self) -> String {
        format!("attestor:{}", self.name)
    }

    /// What the attestor asserts of the run `run` that served `proposal` and emitted the
    /// artifact the proposal's operation determines.
    pub fn attest(&self, proposal: &ExpertProposal, run: u8) -> Res<ExpertAttestation> {
        self.attest_with(
            run,
            *proposal.proposal_id().as_bytes(),
            expert_evidence_artifact_digest(proposal.op()),
        )
    }

    /// An attestation with the proposal and artifact it binds named by the caller.
    pub fn attest_with(
        &self,
        run: u8,
        proposal_id: [u8; 32],
        artifact_digest: [u8; 32],
    ) -> Res<ExpertAttestation> {
        ExpertAttestation::new(
            self.name,
            Generation(self.epoch),
            self.digest(),
            run_id(run),
            proposal_id,
            artifact_digest,
        )
        .map_err(|error| format!("an attestation: {error:?}"))
    }
}

/// The id of the training run numbered `n`.
pub fn run_id(n: u8) -> [u8; 32] {
    let mut id = [0xC0; 32];
    id[1] = n;
    id
}

/// Admit `proposal` with `attestation` over the dependency block `inputs` (the block the
/// attestation extends), on a report bound to the attested block, under a grant minted now.
pub fn admit_with_attestation(
    rt: &mut Rt,
    proposal: &ExpertProposal,
    attestation: &ExpertAttestation,
    inputs: ExpertInputs,
    spec: &ReportSpec,
    setup: Setup,
) -> Result<CommitIndex, RuntimeError> {
    let block = attestation
        .attested_inputs(&inputs)
        .map_err(RuntimeError::Expert)?;
    let report = report_for(proposal, Some(&block), spec).map_err(RuntimeError::InvalidConfig)?;
    let grant = rt
        .admission_grant(setup)
        .map_err(RuntimeError::InvalidConfig)?;
    rt.runtime
        .admit_attested_expert_proposal(proposal, &report, &grant, inputs, attestation)
}

/// Admit `op` attested by `who` for the run `run`, with the correct report under `setup`.
pub fn admit_attested(
    rt: &mut Rt,
    op: StructuralOp,
    inputs: ExpertInputs,
    who: Attestor,
    run: u8,
    setup: Setup,
) -> Result<CommitIndex, RuntimeError> {
    let proposal = rt.proposal(op).map_err(RuntimeError::InvalidConfig)?;
    let attestation = who
        .attest(&proposal, run)
        .map_err(RuntimeError::InvalidConfig)?;
    admit_with_attestation(
        rt,
        &proposal,
        &attestation,
        inputs,
        &ReportSpec::correct_under(setup),
        setup,
    )
}

/// A setup step: [`admit_attested`] or fail the harness.
pub fn must_attested(
    rt: &mut Rt,
    what: &str,
    op: StructuralOp,
    inputs: ExpertInputs,
    who: Attestor,
    run: u8,
    setup: Setup,
) -> Res<()> {
    admit_attested(rt, op, inputs, who, run, setup)
        .map(|_| ())
        .map_err(|error| format!("{what}: {error:?}"))
}

/// The predicate "the runtime refused with an expert error satisfying `pred`".
pub fn refused_with(
    pred: impl Fn(&ExpertAdmissionError) -> bool,
) -> impl Fn(&RuntimeError) -> bool {
    move |error| matches!(error, RuntimeError::Expert(inner) if pred(inner))
}
