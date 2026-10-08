//! The cases C1 to C13 of the draft preregistration, each driving the PUBLIC runtime API on a
//! [`fixture::Rt`] (a `PtrRuntime` over a durable `FileLedger` in a temporary directory; C11
//! drives the R010 `KvSession` over the same runtime), and the counting they share.
//!
//! Counting rules:
//!
//! * An **attempt** of an adversarial case counts only when the attempt really reached the
//!   runtime boundary it names: [`Cx::verdict`] counts it after the call returned `Ok` (the
//!   runtime let it through: a violation) or an `Err` of the expected typed kind. Any other
//!   error is a harness error (`Err`, exit status 2), never a pass, and so is an unexpected
//!   `Ok` of a negative control's counterpart or a failing positive control. A refusal by the
//!   proposal constructor is not a runtime attempt: it is noted under `internal` only.
//! * A **route probe** of an expert that must be refused is kind-checked ([`classify_route`]):
//!   `Fenced` after a fence, `DependencyNotLive` after a plain revocation; any other refusal is
//!   a harness error.
//! * A **fence finding** (the event fenced more or fewer experts than the topology owes, a
//!   dependent refused without a fence, a bystander closed) is a violation counted as
//!   `fence_set_digest_mismatches` and as a stale hit of its cell, and the run goes on; exit 2
//!   stays for harness and fixture errors.
//! * A **violation** counter (`*_admitted`, `*_accepted`, `routable_after_*`, ...) counts what
//!   the runtime did wrong; the sub-case that produced it is recorded under `subcases` with
//!   its violation count.
//! * Every metric family is filled by the case that measures it; a family that no case filled
//!   stays unseeded, so its coverage rule fails (a case that never ran is an unmet
//!   prerequisite, not a silent zero).

pub mod authority;
#[path = "core.rs"]
pub mod core_cases;
pub mod fence;
pub mod fixture;
pub mod identity;
pub mod inflight;
pub mod invalidation;
pub mod pins;
pub mod results;

use super::metrics;
use super::params;
use crate::experiments::layered_kv::corpus;
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::lab::{self, id_set_digest, Res};
use crate::experiments::layered_kv::Scale;
use crate::experiments::r010::arms::World;
use ptr_runtime::expert::{ExpertAdmissionError, RouteRefusal};
use ptr_runtime::{ReferenceKvReuseBackend, RuntimeError};
use std::collections::{BTreeMap, BTreeSet};

/// The sequences of the corpus selection, in listed order (what the selection digest covers):
/// each case name of `adversarial_cases` as the sequence of its bytes.
pub fn selection_sequences() -> Vec<Vec<u32>> {
    params::ADVERSARIAL_CASES
        .iter()
        .map(|name| name.bytes().map(u32::from).collect())
        .collect()
}

pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &selection_sequences())
}

/// An FNV-1a hash of a label, to derive one shuffle stream per label from the pinned domain.
fn label_hash(label: &str) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in label.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// What the runtime did with an attempt that it may only refuse with one typed error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// Refused with the expected typed error.
    Refused,
    /// The runtime let it through: a violation the caller counts.
    Admitted,
}

impl Verdict {
    pub fn admitted(self) -> bool {
        self == Self::Admitted
    }

    /// 1 when the runtime let the attempt through, else 0 (the violation increment).
    pub fn violation(self) -> i64 {
        i64::from(self.admitted())
    }
}

/// `Admitted` for an `Ok`, `Refused` for the expected error, a harness error for any other.
pub fn classify<T: std::fmt::Debug>(
    what: &str,
    result: Result<T, RuntimeError>,
    expected: impl Fn(&RuntimeError) -> bool,
) -> Res<Verdict> {
    match result {
        Ok(_) => Ok(Verdict::Admitted),
        Err(error) if expected(&error) => Ok(Verdict::Refused),
        Err(error) => Err(format!(
            "{what}: refused with an unexpected error: {error:?}"
        )),
    }
}

/// A positive control: the call must succeed, else the harness cannot run the case.
pub fn control<T: std::fmt::Debug>(what: &str, result: Result<T, RuntimeError>) -> Res<T> {
    result.map_err(|error| format!("{what}: a positive control failed: {error:?}"))
}

/// The expert refusal inside a runtime error, if it is one.
pub fn expert(error: &RuntimeError) -> Option<&ExpertAdmissionError> {
    match error {
        RuntimeError::Expert(inner) => Some(inner),
        _ => None,
    }
}

/// The refusal kind a route probe must meet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteExpect {
    /// The registry fences the expert: `Err(RouteRefusal::Fenced)`.
    Fenced,
    /// The expert is not fenced (a plain revocation, no fence written) but a recorded
    /// dependency is dead: `Err(RouteRefusal::DependencyNotLive)`.
    DependencyNotLive,
}

/// What a route probe of an expert that must be refused found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteFinding {
    /// The route was admitted: the violation every "served" counter counts.
    Served,
    /// Refused with the expected kind.
    Refused,
    /// A fence was expected but only the dependency index refused: the registry does not
    /// fence the expert (under-fencing), though the route is closed.
    Unfenced,
}

impl RouteFinding {
    /// 1 when the route was admitted, else 0.
    pub fn served(self) -> i64 {
        i64::from(self == Self::Served)
    }

    /// 1 when the expert was refused without the fence the event owed, else 0.
    pub fn unfenced(self) -> i64 {
        i64::from(self == Self::Unfenced)
    }
}

/// Classify a route probe of an expert that must be refused: `Ok` is `Served`, the expected
/// refusal kind is `Refused`, a dependency refusal where a fence was expected is `Unfenced`,
/// and any other refusal (an unknown expert, a generation mismatch, a state) is a harness error
/// and never a pass.
pub fn classify_route(
    what: &str,
    result: Result<(), RouteRefusal>,
    expect: RouteExpect,
) -> Res<RouteFinding> {
    match (result, expect) {
        (Ok(()), _) => Ok(RouteFinding::Served),
        (Err(RouteRefusal::Fenced), RouteExpect::Fenced)
        | (Err(RouteRefusal::DependencyNotLive { .. }), RouteExpect::DependencyNotLive) => {
            Ok(RouteFinding::Refused)
        }
        (Err(RouteRefusal::DependencyNotLive { .. }), RouteExpect::Fenced) => {
            Ok(RouteFinding::Unfenced)
        }
        (Err(other), expect) => Err(format!(
            "{what}: the route probe was refused with {other:?}, expected {expect:?}"
        )),
    }
}

/// Classify a route probe of a bystander that must stay routable: `true` when the runtime
/// closed it (over-fencing: a fence, or a dependency refusal, on an expert that rests on
/// nothing invalidated), a harness error for any other refusal.
pub fn classify_bystander(what: &str, result: Result<(), RouteRefusal>) -> Res<bool> {
    match result {
        Ok(()) => Ok(false),
        Err(RouteRefusal::Fenced | RouteRefusal::DependencyNotLive { .. }) => Ok(true),
        Err(other) => Err(format!(
            "{what}: the bystander was refused with {other:?}, which is neither a fence nor a dead dependency"
        )),
    }
}

/// A positive route control: the expert must be routable now, else the harness cannot run the
/// case.
pub fn route_control(what: &str, result: Result<(), RouteRefusal>) -> Res<()> {
    result.map_err(|refusal| format!("{what}: a route control was refused with {refusal:?}"))
}

/// Probe each of `pods` (at `generation` 1 unless the registry says otherwise) of `rt` and sum
/// what the probes found.
pub fn probe_routes(rt: &fixture::Rt, pods: &[&str], expect: RouteExpect) -> Res<RouteTally> {
    let mut tally = RouteTally::default();
    for pod in pods {
        let generation = rt.init(pod).map(|init| init.generation.0).unwrap_or(1);
        tally.add(classify_route(pod, rt.route(pod, generation), expect)?);
    }
    Ok(tally)
}

/// The findings of a set of route probes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RouteTally {
    /// Experts that must be refused and were routed.
    pub served: i64,
    /// Experts that must be fenced and were refused without a fence.
    pub unfenced: i64,
    /// Bystanders that must stay routable and were closed.
    pub overfenced: i64,
}

impl RouteTally {
    pub fn add(&mut self, finding: RouteFinding) {
        self.served += finding.served();
        self.unfenced += finding.unfenced();
    }

    pub fn merge(&mut self, other: RouteTally) {
        self.served += other.served;
        self.unfenced += other.unfenced;
        self.overfenced += other.overfenced;
    }
}

/// Probe each of `pods`, bystanders of the event that must stay routable, on `rt`; the closed
/// ones are counted as `overfenced`.
pub fn probe_bystanders(rt: &fixture::Rt, pods: &[&str]) -> Res<RouteTally> {
    let mut tally = RouteTally::default();
    for pod in pods {
        let generation = rt.init(pod).map(|init| init.generation.0).unwrap_or(1);
        tally.overfenced += i64::from(classify_bystander(pod, rt.route(pod, generation))?);
    }
    Ok(tally)
}

/// The experts a fence event owed and the experts it fenced, compared as sets: the experts it
/// missed (under-fencing) and the experts it fenced without owing it (over-fencing).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FenceDiff {
    pub missed: Vec<String>,
    pub extra: Vec<String>,
}

impl FenceDiff {
    pub fn of<'a>(
        owed: impl IntoIterator<Item = &'a str>,
        fenced: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let owed: BTreeSet<&str> = owed.into_iter().collect();
        let fenced: BTreeSet<&str> = fenced.into_iter().collect();
        Self {
            missed: owed
                .difference(&fenced)
                .map(|pod| (*pod).to_string())
                .collect(),
            extra: fenced
                .difference(&owed)
                .map(|pod| (*pod).to_string())
                .collect(),
        }
    }

    /// The number of experts on which the two sets differ.
    pub fn mismatches(&self) -> i64 {
        (self.missed.len() + self.extra.len()) as i64
    }
}

/// The predicate "the runtime refused with exactly `expected`".
pub fn is(expected: ExpertAdmissionError) -> impl Fn(&RuntimeError) -> bool {
    move |error| expert(error) == Some(&expected)
}

/// The predicate "the runtime refused with an expert error satisfying `pred`".
pub fn is_where(pred: impl Fn(&ExpertAdmissionError) -> bool) -> impl Fn(&RuntimeError) -> bool {
    move |error| expert(error).is_some_and(&pred)
}

/// The predicate "the public commit path refused the event as a structural record".
pub fn outside_admission_path(error: &RuntimeError) -> bool {
    matches!(error, RuntimeError::StructuralRecordOutsideAdmissionPath)
}

/// The state of a run: the backend and the MoE the in-flight cases share, the tally, the case
/// being counted and the identity sets the digest metrics compare.
pub struct Cx {
    pub backend: ReferenceKvReuseBackend,
    pub world: World,
    pub tally: Tally,
    pub seed: u64,
    pub scale: Scale,
    case: String,
    counts: BTreeMap<String, i64>,
    sub: String,
    subs: BTreeMap<String, i64>,
    /// The base names of the metrics a rule bounds above by zero: a positive increment of one
    /// of them is a violation of the current sub-case.
    violating: BTreeSet<String>,
    pub cases: Vec<(String, Json)>,
    /// Named sets of identities (event types, op kinds, cells) the run filled; digests and the
    /// procedure checks read them.
    pub sets: BTreeMap<String, BTreeSet<String>>,
    /// The fixture choices the seed made, recorded for the reader.
    pub fixture: Vec<(String, Json)>,
}

impl Cx {
    pub fn new(seed: u64, scale: Scale) -> Res<Self> {
        let mut tally = Tally::new(metrics::LIST);
        let mut violating = BTreeSet::new();
        // Every counter whose rule bounds it above is seeded at zero, so a run in which the
        // violation never happens prints a zero; the prerequisites are never seeded.
        for rule in metrics::falsifiers().iter().chain(&metrics::coverage()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            if let [left, "<=", "0"] = tokens[..] {
                tally.add(left, 0);
                let base = left.split('.').next().unwrap_or(left);
                violating.insert(base.to_string());
            }
        }
        let backend = lab::backend()?;
        let world = World::new()?;
        Ok(Self {
            backend,
            world,
            tally,
            seed,
            scale,
            case: String::new(),
            counts: BTreeMap::new(),
            sub: String::new(),
            subs: BTreeMap::new(),
            violating,
            cases: Vec::new(),
            sets: BTreeMap::new(),
            fixture: Vec::new(),
        })
    }

    pub fn reduced(&self) -> bool {
        self.scale == Scale::Reduced
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

    /// Add `member` to the named identity set.
    pub fn set_add(&mut self, name: &str, member: &str) {
        self.sets
            .entry(name.to_string())
            .or_default()
            .insert(member.to_string());
    }

    /// The digest of the named set (the digest of an empty set when nothing was added).
    pub fn set_digest(&self, name: &str) -> String {
        let empty = BTreeSet::new();
        let set = self.sets.get(name).unwrap_or(&empty);
        id_set_digest(set.iter().map(String::as_str))
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

    /// Count what a fence event and the probes after it found wrong as
    /// `fence_set_digest_mismatches`: experts the event missed or fenced without owing it
    /// (`diff`), experts refused by the dependency index but not fenced and bystanders closed
    /// (`probes`). A finding is a violation of the sub-case, never a harness error. Returns the
    /// number counted.
    pub fn count_fence_findings(&mut self, diff: &FenceDiff, probes: &RouteTally) -> i64 {
        let found = diff.mismatches() + probes.unfenced + probes.overfenced;
        self.count("fence_set_digest_mismatches", found);
        found
    }

    /// A pinned procedure parameter observed in this case differs from the pin.
    pub fn procedure_mismatch(&mut self, observed_ok: bool) {
        if !observed_ok {
            self.count("procedure_mismatches", 1);
        }
    }

    /// The shape of the backend's stack against the pinned values (checked once per case).
    pub fn check_shape(&mut self) {
        let config = self.backend.stack_config().clone();
        let ok = config.num_layers as u64 == params::TEST_MODEL_LAYERS
            && config.layer.hidden_size as u64 == params::HIDDEN_SIZE
            && config.layer.num_attention_heads as u64 == params::NUM_HEADS
            && config.layer.num_key_value_heads as u64 == params::NUM_KV_HEADS
            && (config.layer.hidden_size / config.layer.num_attention_heads) as u64
                == params::HEAD_DIM
            && config.vocab_size as u64 == params::VOCAB_SIZE;
        self.procedure_mismatch(ok);
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

/// Run every case once, in the seed's order.
pub fn run_all(cx: &mut Cx) -> Res<Vec<Json>> {
    type Case = (&'static str, fn(&mut Cx) -> Res<()>);
    let cases: [Case; 13] = [
        ("C1", core_cases::c1_control_immutability),
        ("C2", core_cases::c2_bypass_and_direct_append),
        ("C3", core_cases::c3_stale_operands),
        ("C4", core_cases::c4_evidence_binding),
        ("C5", core_cases::c5_impersonation),
        ("C6", core_cases::c6_fence_authority),
        ("C7", fence::c7_fence_causes),
        ("C8", results::c8_results_and_succession),
        ("C9", identity::c9_identity_and_grant_races),
        ("C10", invalidation::c10_compound_invalidation),
        ("C11", inflight::c11_inflight_calls),
        ("C12", authority::c12_authority_epoch),
        ("C13", pins::c13_pins),
    ];
    let mut order = Vec::new();
    // C13 reads what the others recorded, so it runs last; the rest run in the seed's order.
    let first = cases.len() - 1;
    for index in corpus::order(cx.seed, params::CASE_ORDER_LABEL, first) {
        let (name, case) = cases[index];
        cx.begin(name);
        cx.check_shape();
        case(cx).map_err(|error| format!("{name}: {error}"))?;
        cx.end();
        order.push(Json::str(name));
    }
    let (name, case) = cases[first];
    cx.begin(name);
    cx.check_shape();
    case(cx).map_err(|error| format!("{name}: {error}"))?;
    cx.end();
    order.push(Json::str(name));
    Ok(order)
}

/// How each weak or definitional metric is measured, and what the cases do not claim, for the
/// reader of a result. The same sentences are the stated limits of the preregistration.
pub fn measurement_notes() -> Json {
    let notes: [(&str, &str); 29] = [
        (
            "attempts",
            "an adversarial attempt counts only after the runtime returned either Ok (a violation) or the expected typed error; any other error is a harness error and ends the run with exit 2, and so does a failing positive control; a refusal by the proposal constructor is not a runtime attempt and is only noted under internal (fenced_gate_retire and reactivate_from_revoked count the calls that reached the runtime: a proposal the constructor let through and the same operation from a stale view)",
        ),
        (
            "route_probes",
            "every route probe of an expert that must be refused is kind-checked: Err(Fenced) after a fence, Err(DependencyNotLive) after a plain revocation without a fence, and a refusal of another kind (an unknown expert, a generation mismatch, a state) is a harness error; a live Ok-control precedes each fence; a bystander is re-probed after every reopen, replay and restore",
        ),
        (
            "fence_findings",
            "an event that fenced a set other than the one the topology owes (under-fencing or over-fencing), a dependent refused by the dependency index without a fence and a bystander closed are findings: they count as fence_set_digest_mismatches (and as stale hits of the matrix cell) and the run goes on; exit 2 stays for harness and fixture errors",
        ),
        (
            "set_digest_equalities",
            "weak: impersonation_attempted_op_kind_set_digest = admitted_op_kind_set_digest and gate_grant_race_kind_set_digest = gate_host_issued_kind_set_digest cannot differ in a run that ends with a status other than 2, because the positive control that fills the admitted side is a harness error when it fails; they are bookkeeping that every kind ran, not a check of the runtime. direct_append_attempted_event_type_set_digest = admitted_event_type_set_digest is read from the ledger of the gated path and is not vacuous",
        ),
        (
            "replacement_visible_without_fence",
            "weak: its first increment (a crash before the event) never calls the runtime beyond a reopen, so it cannot be non-zero there; only the second (a replacement visible after the event and a crash without the fence) measures the runtime",
        ),
        (
            "evidence_reactivate_forms",
            "Reactivate is run in two forms under one name in C4 (the frozen-to-serving form and the retired-target form that creates a new generation), so each of its cells of evidence_binding_negative_cases_per_event_type, evidence_binding_controls_accepted_by_case and control_accepts_by_path counts both; internal notes say how many cells each form filled",
        ),
        (
            "inflight_control",
            "each C11 cell first runs an un-invalidated control session on a backend of its own that must complete with the cold output, then runs the cell on a fresh backend; the cancellation after the event must be InputNotLive and the dependencies the check failed on must be exactly the fenced expert; the race point is drawn in 1..=total steps so a step runs before the event, and a session that ended before it is noted and its finish is the next use",
        ),
        (
            "route",
            "a route is a probe of PtrRuntime::expert_route_admission (registry state, no fence, every recorded dependency live); no production router consumes it and no end-to-end MoE routing is claimed",
        ),
        (
            "direct_append",
            "the public PtrRuntime API only; the grant is not recorded in the ledger and the report digest is unkeyed, so a raw ledger writer is outside the claim; each type is offered the exact event its gated step committed, to a durable runtime standing where that step started",
        ),
        (
            "verifier",
            "the gate evaluation embedded in the commit record (BoundGateReport); verifier_skip is a record without an embedded report, verifier_override a failing or foreign-configuration verdict as evidence; hard-deny verifiers and verifier-set epochs do not exist",
        ),
        (
            "uncommitted_effective_changes",
            "weak: structural and type-level evidence, not an adversary measurement: refused attempts leave the event count and the registry unchanged and the registry equals a fresh replay of the ledger; the proposal types have no field for the gate, a threshold or a report",
        ),
        (
            "direct_serving_mutation_bypass_count",
            "weak: the registry is readable only through a shared borrow (type-level); the measured part is that the public commit of a well-formed structural event is refused and leaves the expert unroutable",
        ),
        (
            "forbidden_control_mutation_count",
            "the embedded configuration digest, gate epoch and verdict of every committed gated record equal what the host required, no authority standing exists after the proposals, only the runtime-wide revocation grant moves a standing, and a mutated embedded report is refused on replay",
        ),
        (
            "l3_fence_attempts_admitted",
            "weak: no L3 policy object exists in the runtime; the L3 path is modelled by the three routes a non-runtime principal has: the public commit of the fence event, a fence under a Gate grant and a fence under the runtime-wide grant",
        ),
        (
            "gate_grant_revocation_race_cases",
            "weak for ExpertPromoted_gate_driven and ExpertRetired_gate_driven: the gated promotion and retirement are admitted under the admission grant, so their grant-generation race supersedes the same grant kind as the proposer race; each of them also races the revocation of the gate configuration authority (stale evidence at the commit); only ExpertReadmitted races the Gate grant kind",
        ),
        (
            "impersonation_op_kinds",
            "Birth_targetless is a Birth onto a pod of its own: since the narrowing a Birth has no source expert, so the two kinds run the same operation; text, config.toml and the harness all count the eight kinds, Birth_targetless kept as a separate attempt",
        ),
        (
            "races",
            "every race is a sequenced interleaving in a single-writer runtime (mint a grant, change the authority, commit); no concurrent-writer claim is made",
        ),
        (
            "crash",
            "a crash between the append and the apply inside one call cannot be injected without a failpoint; it is modelled as drop after the append, open_durable, probe; the same model covers fault_between_revocation_and_fence and supersession_atomicity_crash",
        ),
        (
            "adapter",
            "weak: the runtime has no adapter role; an adapter is an input identity (adapter:lora-1) in the training block, so the adapter cells exercise the same index as a training input under another identity",
        ),
        (
            "source_expert",
            "the dependent rests on the revoked input only through its source expert: a Merge result with no block of its own, fenced through the source edge",
        ),
        (
            "large_fanout_revocation_cases",
            "at the pinned scale the explicit fence list of the event (the member encoding of the fence-set digest) is checked to exceed MAX_RECORD_BYTES; at the reduced scale of the smoke test it does not, and the case says so under internal",
        ),
        (
            "dependency_set_digests",
            "derived_dependency_set_* is derived from the topology each scenario builds (which expert rests on which input, which result descends from which source, which states serve), not from the runtime's index; fenced_and_probed_dependents_* are the experts of the scenario that the registry shows Fenced and the route probe refuses, over the probes now, after a reopen and after a replay; set members carry the scenario name",
        ),
        (
            "inflight_levels",
            "weak: L1 is an expert registered at ExpertLevel::Token, L2 one registered at ExpertLevel::Segment; the KvSession depends on an expert by its lifecycle identity whatever the level, so the L2 cells bind the MoE slot to a Segment-level pod: the mechanism under test is the same, the registered level differs",
        ),
        (
            "inflight_fallback",
            "the replacement path of a fenced expert is the restart of the request on the router with the expert masked; no retrained successor of the same pod exists after an input invalidation (Reactivate from Retired(InputRevoked) is refused), so the reference output is the cold run with the expert masked",
        ),
        (
            "grant_generation",
            "the grant generation and the host-time floor are in-memory host authority: not replayed, restart at their initial values, and a replay cannot know a grant was revoked",
        ),
        (
            "evaluator_identity",
            "a host assertion carried by the grant; the report and the record do not carry an evaluator, so replay does not re-check it",
        ),
        (
            "eval_manifest",
            "checked at admission against the host grant only; replay can re-check the artifact digest and the base-model digest of the evidence binding but not the eval-manifest digest",
        ),
        (
            "input_digest",
            "the digest of an input is the proposer's assertion; the first committed use of an (identity, generation) fixes it in the dependency index, and a proposer can fix a wrong digest for an identity the lifecycle authority does not know",
        ),
        (
            "model_pin_mismatches",
            "weak: base checkpoint, tokenizer revision and topology are constants here, so it checks the tiny-model config digest (the shape plus num_experts and top_k, the table config.toml pins), the tokenizer, the shape, the chain id, the cold run of the backend against an independent stack and the MoE topology and cold run against an independent MoE stack",
        ),
    ];
    Json::Obj(
        notes
            .iter()
            .map(|(name, text)| ((*name).to_string(), Json::str(*text)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::fixture::{inputs, Rt, D1, TRAIN};
    use super::*;
    use crate::experiments::layered_kv::emit;
    use ptr_types::Generation;

    fn dead() -> RouteRefusal {
        RouteRefusal::DependencyNotLive {
            identity: "dataset:train".into(),
            generation: Generation(1),
        }
    }

    #[test]
    fn a_route_probe_is_kind_checked() {
        use RouteExpect::{DependencyNotLive, Fenced};
        use RouteFinding::{Refused, Served, Unfenced};
        let table: [(Result<(), RouteRefusal>, RouteExpect, Option<RouteFinding>); 10] = [
            (Ok(()), Fenced, Some(Served)),
            (Ok(()), DependencyNotLive, Some(Served)),
            (Err(RouteRefusal::Fenced), Fenced, Some(Refused)),
            (Err(dead()), DependencyNotLive, Some(Refused)),
            // The index refuses an expert the registry should have fenced: under-fencing.
            (Err(dead()), Fenced, Some(Unfenced)),
            // A fence where only a dependency refusal was expected is not the model's.
            (Err(RouteRefusal::Fenced), DependencyNotLive, None),
            (Err(RouteRefusal::UnknownExpert), Fenced, None),
            (Err(RouteRefusal::UnknownExpert), DependencyNotLive, None),
            (
                Err(RouteRefusal::GenerationMismatch {
                    current: Generation(2),
                }),
                Fenced,
                None,
            ),
            (
                Err(RouteRefusal::NotServing {
                    state: ptr_pods::ExpertState::Frozen,
                }),
                Fenced,
                None,
            ),
        ];
        for (result, expect, want) in table {
            let found = classify_route("probe", result.clone(), expect);
            match want {
                Some(finding) => assert_eq!(found, Ok(finding), "{result:?} under {expect:?}"),
                None => assert!(found.is_err(), "{result:?} under {expect:?}"),
            }
        }
        assert_eq!(Served.served(), 1);
        assert_eq!(Unfenced.unfenced(), 1);
        assert_eq!(Refused.served() + Refused.unfenced(), 0);
    }

    #[test]
    fn a_bystander_probe_finds_a_closed_route_and_rejects_other_refusals() {
        assert_eq!(classify_bystander("z", Ok(())), Ok(false));
        assert_eq!(classify_bystander("z", Err(RouteRefusal::Fenced)), Ok(true));
        assert_eq!(classify_bystander("z", Err(dead())), Ok(true));
        assert!(classify_bystander("z", Err(RouteRefusal::UnknownExpert)).is_err());
        assert!(route_control("z", Ok(())).is_ok());
        assert!(route_control("z", Err(RouteRefusal::Fenced)).is_err());
    }

    #[test]
    fn the_fence_diff_finds_the_missed_and_the_extra_expert() {
        let exact = FenceDiff::of(["x", "y"], ["y", "x"]);
        assert_eq!(exact.mismatches(), 0);
        let under = FenceDiff::of(["x", "y"], ["x"]);
        assert_eq!(
            (under.missed.clone(), under.extra.len()),
            (vec!["y".to_string()], 0)
        );
        assert_eq!(under.mismatches(), 1);
        let over = FenceDiff::of(["x"], ["x", "w"]);
        assert_eq!(
            (over.missed.len(), over.extra.clone()),
            (0, vec!["w".to_string()])
        );
        assert_eq!(over.mismatches(), 1);
    }

    /// Under-fencing and over-fencing are findings: they are counted, the sub-case records a
    /// violation, the falsifier's bound is passed, and nothing is a harness error.
    #[test]
    fn under_and_over_fencing_are_counted_as_violations_and_do_not_end_the_run() {
        for (owed, fenced) in [(["x", "y"], vec!["x"]), (["x", "y"], vec!["x", "y", "w"])] {
            let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
            cx.begin("T");
            cx.sub("t");
            let diff = FenceDiff::of(owed, fenced.iter().copied());
            let counted = cx.count_fence_findings(&diff, &RouteTally::default());
            assert_eq!(counted, 1);
            assert_eq!(cx.tally.get("fence_set_digest_mismatches"), 1);
            assert!(
                cx.tally.get("fence_set_digest_mismatches")
                    > params::MAX_FENCE_SET_DIGEST_MISMATCHES as i64
            );
            assert_eq!(
                emit::resolve(&cx.subcases_json(), "t.violations"),
                Some(&Json::Int(1))
            );
        }
        // The probes count too: a dependent refused without a fence, a bystander closed.
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        cx.sub("t");
        let probes = RouteTally {
            served: 3,
            unfenced: 2,
            overfenced: 1,
        };
        assert_eq!(cx.count_fence_findings(&FenceDiff::default(), &probes), 3);
    }

    /// The probes against a real runtime: a plain revocation leaves the dependent unfenced, so a
    /// probe that expects a fence finds the missing fence, and a probe that expects the
    /// dependency refusal is satisfied; a ghost expert is a harness error either way.
    #[test]
    fn the_probes_against_a_runtime_find_a_missing_and_an_extra_fence() {
        let mut rt = Rt::fresh().expect("a runtime");
        rt.birth_with("x", inputs(&[(TRAIN, 1, D1)], &[])).unwrap();
        rt.birth("w").unwrap();
        rt.revoke(TRAIN, 1).unwrap();
        let expecting_a_fence = probe_routes(&rt, &["x"], RouteExpect::Fenced).unwrap();
        assert_eq!(
            expecting_a_fence,
            RouteTally {
                served: 0,
                unfenced: 1,
                overfenced: 0
            }
        );
        let expecting_the_dependency = probe_routes(&rt, &["x"], RouteExpect::DependencyNotLive);
        assert_eq!(expecting_the_dependency, Ok(RouteTally::default()));
        assert!(probe_routes(&rt, &["ghost"], RouteExpect::Fenced).is_err());
        assert!(probe_routes(&rt, &["ghost"], RouteExpect::DependencyNotLive).is_err());
        assert!(probe_bystanders(&rt, &["ghost"]).is_err());
        // The bystander w is routable; fenced for no reason it is over-fencing.
        assert_eq!(probe_bystanders(&rt, &["w"]).unwrap().overfenced, 0);
        rt.fence_quarantine("w").unwrap();
        assert_eq!(probe_bystanders(&rt, &["w"]).unwrap().overfenced, 1);
        // An expert the fence did close is refused as `Fenced`.
        assert_eq!(
            probe_routes(&rt, &["w"], RouteExpect::Fenced).unwrap(),
            RouteTally::default()
        );
    }
}
