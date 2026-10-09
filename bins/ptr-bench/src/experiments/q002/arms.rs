//! The four arms of Q002 and the adversarial stage-skip attempts.
//!
//! Every arm sees the same ranked hits of each question. B0 takes the top hit as Known (the
//! manifest's baseline, "direct retrieval-to-answer"); B1 first drops the hits the lifecycle
//! authority does not hold live (`retain_live`) and takes the top survivor; B2 runs the staged
//! API with the caller's word for the source and the verification (`observe(.., true)`,
//! `promote_verified(true)`, today's API fed `true`); M is the mechanism, the typed path:
//! `ExactSourceResolver`, `observe_resolved` against the lifecycle authority, `verify_claim`
//! and `promote_verified_with`. B0 and B2 are the negative controls: if neither produces a
//! false Known on this corpus the corpus cannot show the hallucination it is made to show.
//!
//! A Known is false when the hit's generator label is not eligible (`Class::eligible`). The
//! baselines make one Known assertion per question (their top hit); the mechanism is judged on
//! every hit it promotes, which is stricter. Every call into the code under test is guarded:
//! a panic is a counted `product_panics`, never an abort of the harness.

use super::authority::{Authority, Source};
use super::world::{Class, Hit, Refusal, World};
use crate::experiments::layered_kv::emit::Tally;
use ptr_search::{
    retain_live, verify_claim, Claim, ClaimError, EvidencePromotionError, EvidenceStage,
    ExactSourceResolver, ResolvedSource, SearchHit, SourceError,
};
use ptr_types::{CapsuleId, Generation};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The kinds of adversarial attempt on the API boundary, in the order of `STAGE_SKIP_KINDS`.
pub const STAGE_SKIP_KINDS: [&str; 10] =
    ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"];

/// What the mechanism did with one hit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Verdict {
    Promoted(Claim),
    Refused(Refusal),
    Panicked,
}

pub fn from_source(error: &SourceError) -> Refusal {
    match error {
        SourceError::UnknownRawEvent { .. } => Refusal::UnknownRawEvent,
        SourceError::DigestMismatch { .. } => Refusal::DigestMismatch,
        SourceError::EmptySpan { .. } => Refusal::EmptySpan,
        SourceError::SpanOutOfRange { .. } => Refusal::SpanOutOfRange,
        SourceError::QuoteMismatch { .. } => Refusal::QuoteMismatch,
    }
}

pub fn from_claim(error: &ClaimError) -> Refusal {
    match error {
        ClaimError::MalformedClaim => Refusal::MalformedClaim,
        ClaimError::MalformedStatement => Refusal::MalformedStatement,
        ClaimError::SubjectMismatch => Refusal::SubjectMismatch,
        ClaimError::PredicateMismatch => Refusal::PredicateMismatch,
        ClaimError::ValueMismatch => Refusal::ValueMismatch,
        ClaimError::NegatedStatement => Refusal::NegatedStatement,
    }
}

pub fn from_promotion(error: &EvidencePromotionError) -> Refusal {
    match error {
        EvidencePromotionError::WrongStage => Refusal::WrongStage,
        EvidencePromotionError::StaleGeneration => Refusal::StaleGeneration,
        EvidencePromotionError::RevokedGeneration => Refusal::RevokedGeneration,
        EvidencePromotionError::DisputedGeneration => Refusal::DisputedGeneration,
        EvidencePromotionError::UnknownGeneration => Refusal::UnknownGeneration,
        EvidencePromotionError::SourceNotResolved => Refusal::SourceNotResolved,
        EvidencePromotionError::VerificationFailed => Refusal::VerificationFailed,
        EvidencePromotionError::ClaimProofMismatch => Refusal::ClaimProofMismatch,
    }
}

/// Run `call`, counting a panic of the code under test instead of aborting.
fn guarded<T>(tally: &mut Tally, call: impl FnOnce() -> T) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(value) => Some(value),
        Err(_) => {
            tally.add("product_panics", 1);
            None
        }
    }
}

/// What the harness reads of a hit before and after an attempt.
#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    stage: EvidenceStage,
    source: Option<ResolvedSource>,
    claim: Option<Claim>,
}

fn snapshot(hit: &SearchHit) -> Snapshot {
    Snapshot {
        stage: hit.stage(),
        source: hit.resolved_source().cloned(),
        claim: hit.verified_claim().cloned(),
    }
}

fn search_hit(spec: &Hit) -> SearchHit {
    SearchHit::new(
        CapsuleId::from(spec.capsule.as_str()),
        Generation(spec.generation),
        spec.score_milli as f32 / 1000.0,
        format!("h{}", spec.id),
    )
}

/// One run of the arms over a world.
pub struct Arms<'a> {
    world: &'a World,
    authority: &'a Authority,
    source: &'a Source<'a>,
    pub tally: Tally,
}

/// What an arm answered for a question: a value as Known, or an abstention.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Answer {
    Known(String),
    Abstain,
}

impl<'a> Arms<'a> {
    pub fn new(
        world: &'a World,
        authority: &'a Authority,
        source: &'a Source<'a>,
        tally: Tally,
    ) -> Arms<'a> {
        Arms {
            world,
            authority,
            source,
            tally,
        }
    }

    fn succeeded(&self, question: usize, answer: &Answer) -> bool {
        match (&self.world.questions[question].expected, answer) {
            (Some(expected), Answer::Known(value)) => expected == value,
            (None, Answer::Abstain) => true,
            (Some(_), Answer::Abstain) | (None, Answer::Known(_)) => false,
        }
    }

    /// Run every arm over every question, then the stage-skip attempts.
    pub fn run(&mut self) {
        for question in 0..self.world.questions.len() {
            self.baseline_direct(question);
            self.gen_filter_only(question);
            self.bool_trust(question);
            self.mechanism(question);
        }
        self.stage_skip();
    }

    fn count_known(&mut self, arm: &str, hit: &Hit) {
        if !hit.class.eligible() {
            self.tally.add(&format!("false_known_{arm}"), 1);
            if hit.claim_untrue {
                self.tally.add(&format!("false_known_untrue_{arm}"), 1);
            }
        }
    }

    fn score(&mut self, arm: &str, question: usize, answer: &Answer) {
        if self.succeeded(question, answer) {
            self.tally.add(&format!("task_success_{arm}"), 1);
            if self.world.questions[question].expected.is_some() {
                self.tally.add(&format!("answerable_success_{arm}"), 1);
            }
        }
    }

    /// B0: the top hit is the answer.
    fn baseline_direct(&mut self, question: usize) {
        let world = self.world;
        let answer = match world.by_question[question].first() {
            Some(id) => {
                let hit = &world.hits[*id];
                self.count_known("baseline_direct", hit);
                Answer::Known(hit.claim.value.clone())
            }
            None => Answer::Abstain,
        };
        self.score("baseline", question, &answer);
    }

    /// B1: the lifecycle filter alone, then the top survivor.
    fn gen_filter_only(&mut self, question: usize) {
        let world = self.world;
        let authority = self.authority;
        let candidates: Vec<SearchHit> = world.by_question[question]
            .iter()
            .map(|id| search_hit(&world.hits[*id]))
            .collect();
        let live = guarded(&mut self.tally, || {
            retain_live(candidates, |capsule, generation| {
                authority.validity(capsule, generation)
            })
        })
        .unwrap_or_default();
        let answer = match live.first() {
            Some(top) => {
                let id: usize = top.backend[1..]
                    .parse()
                    .expect("a hit backend names its id");
                let hit = &world.hits[id];
                self.count_known("gen_filter_only", hit);
                Answer::Known(hit.claim.value.clone())
            }
            None => Answer::Abstain,
        };
        self.score("gen_filter_only", question, &answer);
    }

    /// B2: the staged API fed `true` for the source and the verification; the first hit that
    /// the lifecycle lets through is Known.
    fn bool_trust(&mut self, question: usize) {
        let world = self.world;
        let authority = self.authority;
        let mut answer = Answer::Abstain;
        for id in &world.by_question[question] {
            let hit = &world.hits[*id];
            let mut search = search_hit(hit);
            let promoted = guarded(&mut self.tally, || {
                search.mark_possible()?;
                search.observe(|c, g| authority.validity(c, g), true)?;
                search.promote_verified(true)
            });
            if matches!(promoted, Some(Ok(()))) {
                self.count_known("bool_trust", hit);
                answer = Answer::Known(hit.claim.value.clone());
                break;
            }
        }
        self.score("bool_trust", question, &answer);
    }

    /// M for one hit: the typed path, stage by stage.
    fn mechanism_verdict(&mut self, spec: &Hit) -> Verdict {
        let authority = self.authority;
        let source = self.source;
        let mut hit = search_hit(spec);
        match guarded(&mut self.tally, || hit.mark_possible()) {
            None => return Verdict::Panicked,
            Some(Err(error)) => return Verdict::Refused(from_promotion(&error)),
            Some(Ok(())) => {}
        }
        self.tally.add("resolver_calls", 1);
        let resolved = match guarded(&mut self.tally, || {
            ExactSourceResolver::new(source).resolve(&spec.citation)
        }) {
            None => return Verdict::Panicked,
            Some(Err(error)) => return Verdict::Refused(from_source(&error)),
            Some(Ok(resolved)) => resolved,
        };
        match guarded(&mut self.tally, || {
            hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
        }) {
            None => return Verdict::Panicked,
            Some(Err(error)) => return Verdict::Refused(from_promotion(&error)),
            Some(Ok(())) => {}
        }
        let Some(held) = hit.resolved_source().cloned() else {
            return Verdict::Refused(Refusal::SourceNotResolved);
        };
        let proof = match guarded(&mut self.tally, || verify_claim(&spec.claim, &held)) {
            None => return Verdict::Panicked,
            Some(Err(error)) => return Verdict::Refused(from_claim(&error)),
            Some(Ok(proof)) => proof,
        };
        match guarded(&mut self.tally, || hit.promote_verified_with(proof)) {
            None => Verdict::Panicked,
            Some(Err(error)) => Verdict::Refused(from_promotion(&error)),
            Some(Ok(())) => match hit.verified_claim() {
                Some(claim) => Verdict::Promoted(claim.clone()),
                None => Verdict::Refused(Refusal::VerificationFailed),
            },
        }
    }

    /// M for one question: every hit goes through the pipeline; the best-ranked promoted hit
    /// is the answer.
    fn mechanism(&mut self, question: usize) {
        let world = self.world;
        let mut answer = Answer::Abstain;
        for id in &world.by_question[question] {
            let hit = &world.hits[*id];
            let verdict = self.mechanism_verdict(hit);
            self.tally
                .add(&format!("hits_by_class.{}", hit.class.name()), 1);
            self.judge_verdict(hit, &verdict);
            if let Verdict::Promoted(claim) = &verdict {
                if answer == Answer::Abstain {
                    answer = Answer::Known(claim.value.clone());
                }
            }
        }
        if answer == Answer::Abstain && world.questions[question].expected.is_none() {
            self.tally.add("abstentions_correct", 1);
        }
        self.score("mechanism", question, &answer);
    }

    fn judge_verdict(&mut self, hit: &Hit, verdict: &Verdict) {
        let class = hit.class;
        if class.eligible() {
            self.tally.add("true_claims", 1);
        }
        match verdict {
            Verdict::Panicked => {}
            Verdict::Promoted(_) => {
                if class.eligible() {
                    self.tally.add("true_known_promoted", 1);
                } else {
                    self.tally.add("false_known_mechanism", 1);
                    self.tally
                        .add(&format!("poison_promoted.{}", class.name()), 1);
                    if hit.claim_untrue {
                        self.tally.add("false_known_untrue_mechanism", 1);
                    }
                    if class.is_lifecycle() {
                        self.tally.add("stale_or_revoked_promoted", 1);
                    }
                }
            }
            Verdict::Refused(refusal) => {
                if class.eligible() {
                    self.tally
                        .add(&format!("honest_refused.{}", refusal.name()), 1);
                } else if hit.expected == Some(*refusal) {
                    self.tally
                        .add(&format!("poison_attempts.{}", class.name()), 1);
                } else {
                    self.tally.add("wrong_refusal_reason", 1);
                    self.tally.add(
                        &format!("wrong_refusal.{}.{}", class.name(), refusal.name()),
                        1,
                    );
                }
            }
        }
    }

    // ---- the adversarial stage-skip attempts ----

    /// The resolved source and the claim proof of an honest hit, or a counted setup failure.
    fn proofs(&mut self, spec: &Hit) -> Option<(ResolvedSource, ptr_search::ClaimProof)> {
        let source = self.source;
        let outcome = guarded(&mut self.tally, || {
            let resolved = ExactSourceResolver::new(source)
                .resolve(&spec.citation)
                .ok()?;
            let proof = verify_claim(&spec.claim, &resolved).ok()?;
            Some((resolved, proof))
        });
        match outcome {
            Some(Some(pair)) => Some(pair),
            Some(None) => {
                self.setup_failed();
                None
            }
            None => None,
        }
    }

    fn setup_failed(&mut self) {
        self.tally.add("stage_skip_setup_failures", 1);
        self.tally.add("stage_skip_unexpected", 1);
    }

    /// An honest hit set to `stage` through the typed path.
    fn hit_at(&mut self, spec: &Hit, stage: EvidenceStage) -> Option<SearchHit> {
        let authority = self.authority;
        let mut hit = search_hit(spec);
        if stage == EvidenceStage::SearchCandidate {
            return Some(hit);
        }
        let (resolved, proof) = self.proofs(spec)?;
        let ready = guarded(&mut self.tally, || {
            hit.mark_possible().ok()?;
            if stage == EvidenceStage::PossibleEvidence {
                return Some(());
            }
            hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
                .ok()?;
            if stage == EvidenceStage::Observed {
                return Some(());
            }
            hit.promote_verified_with(proof).ok()
        });
        match ready {
            Some(Some(())) => Some(hit),
            Some(None) => {
                self.setup_failed();
                None
            }
            None => None,
        }
    }

    /// Record one attempt: it counts only when the call returned the expected refusal and left
    /// the hit as it was; an accepted call is a stage skip, anything else is unexpected.
    fn record(
        &mut self,
        kind: &str,
        outcome: Option<Result<(), EvidencePromotionError>>,
        expected: Refusal,
        before: &Snapshot,
        after: &SearchHit,
    ) {
        match outcome {
            None => {}
            Some(Ok(())) => {
                self.tally.add("stage_skip_accepted", 1);
                self.tally
                    .add(&format!("stage_skip_accepted_by_kind.{kind}"), 1);
            }
            Some(Err(error)) => {
                if from_promotion(&error) == expected && &snapshot(after) == before {
                    self.tally.add("stage_skip_attempts", 1);
                    self.tally.add(&format!("stage_skip_by_kind.{kind}"), 1);
                } else {
                    self.tally.add("stage_skip_unexpected", 1);
                    self.tally.add(
                        &format!(
                            "stage_skip_unexpected_by_kind.{kind}.{}",
                            from_promotion(&error).name()
                        ),
                        1,
                    );
                }
            }
        }
    }

    fn stage_skip(&mut self) {
        let world = self.world;
        let honest: Vec<&Hit> = world
            .hits
            .iter()
            .filter(|h| h.class == Class::Honest)
            .collect();
        let lifecycle: Vec<&Hit> = world
            .hits
            .iter()
            .filter(|h| h.class.is_lifecycle())
            .collect();
        if honest.len() < 2 || lifecycle.is_empty() {
            self.tally.add("stage_skip_unexpected", 1);
            return;
        }
        for (k, kind) in STAGE_SKIP_KINDS.iter().enumerate() {
            for i in 0..super::params::STAGE_SKIP_ATTEMPTS_PER_KIND {
                let a_index = (i * 5 + k * 11) % honest.len();
                let b_index = (a_index + 1 + i % 7) % honest.len();
                let (a, b) = (honest[a_index], honest[b_index]);
                let l = lifecycle[(i * 3 + k) % lifecycle.len()];
                self.attempt(kind, a, b, l, i);
            }
        }
    }

    fn attempt(&mut self, kind: &str, a: &Hit, b: &Hit, l: &Hit, i: usize) {
        let authority = self.authority;
        match kind {
            // Promote a retrieval candidate, or a hit never observed, with a valid proof.
            "K1" | "K2" => {
                let stage = if kind == "K1" {
                    EvidenceStage::SearchCandidate
                } else {
                    EvidenceStage::PossibleEvidence
                };
                let Some(mut hit) = self.hit_at(a, stage) else {
                    return;
                };
                let Some((_, proof)) = self.proofs(b) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || hit.promote_verified_with(proof));
                self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
            }
            // Observe a candidate that was never marked possible.
            "K3" => {
                let Some(mut hit) = self.hit_at(a, EvidenceStage::SearchCandidate) else {
                    return;
                };
                let Some((resolved, _)) = self.proofs(b) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || {
                    hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
                });
                self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
            }
            // Swap the resolved source of a hit that is already observed.
            "K4" => {
                let Some(mut hit) = self.hit_at(a, EvidenceStage::Observed) else {
                    return;
                };
                let Some((resolved, _)) = self.proofs(b) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || {
                    hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
                });
                self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
            }
            // Step an observed or a known hit back to possible evidence.
            "K5" => {
                let stage = if i.is_multiple_of(2) {
                    EvidenceStage::Observed
                } else {
                    EvidenceStage::VerifiedKnown
                };
                let Some(mut hit) = self.hit_at(a, stage) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || hit.mark_possible());
                self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
            }
            // Promote an observed hit with a valid proof made for another source.
            "K6" => {
                let Some(mut hit) = self.hit_at(a, EvidenceStage::Observed) else {
                    return;
                };
                let Some((_, foreign)) = self.proofs(b) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || hit.promote_verified_with(foreign));
                self.record(kind, outcome, Refusal::ClaimProofMismatch, &before, &hit);
            }
            // Replay a promotion on a hit that is already known.
            "K7" => {
                let Some(mut hit) = self.hit_at(a, EvidenceStage::VerifiedKnown) else {
                    return;
                };
                let Some((_, proof)) = self.proofs(a) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || hit.promote_verified_with(proof));
                self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
            }
            // Observe on the caller's word, then promote with a proof no source backs.
            "K8" => {
                let Some(mut hit) = self.hit_at(a, EvidenceStage::PossibleEvidence) else {
                    return;
                };
                let observed = guarded(&mut self.tally, || {
                    hit.observe(|c, g| authority.validity(c, g), true)
                });
                if !matches!(observed, Some(Ok(()))) {
                    self.setup_failed();
                    return;
                }
                let Some((_, proof)) = self.proofs(a) else {
                    return;
                };
                let before = snapshot(&hit);
                let outcome = guarded(&mut self.tally, || hit.promote_verified_with(proof));
                self.record(kind, outcome, Refusal::SourceNotResolved, &before, &hit);
            }
            // Launder a hit of a generation that is not live with the valid resolution of an
            // honest one; then try to promote it anyway.
            "K9" | "K10" => {
                let Some(expected) = l.expected else { return };
                let mut hit = search_hit(l);
                if !matches!(
                    guarded(&mut self.tally, || hit.mark_possible()),
                    Some(Ok(()))
                ) {
                    self.setup_failed();
                    return;
                }
                let Some((resolved, proof)) = self.proofs(b) else {
                    return;
                };
                let before = snapshot(&hit);
                if kind == "K9" {
                    let outcome = guarded(&mut self.tally, || {
                        hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
                    });
                    self.record(kind, outcome, expected, &before, &hit);
                } else {
                    let refused = guarded(&mut self.tally, || {
                        hit.observe_resolved(resolved, |c, g| authority.validity(c, g))
                    });
                    if !matches!(refused, Some(Err(_))) {
                        self.tally.add("stage_skip_unexpected", 1);
                        return;
                    }
                    let outcome = guarded(&mut self.tally, || hit.promote_verified_with(proof));
                    self.record(kind, outcome, Refusal::WrongStage, &before, &hit);
                }
            }
            other => unreachable!("no stage-skip kind {other}"),
        }
    }
}
