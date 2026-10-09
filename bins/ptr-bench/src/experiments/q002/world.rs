//! The seeded world of Q002 and the truth table the generator writes down while it builds it.
//!
//! A world is capsules, each with one to three generations; every generation is one raw event
//! holding one-line statements in the format `ptr_search::verify_claim` reads
//! (`subject: predicate is [not ]value`). A capsule is healthy (its live generation is
//! authoritative), revoked (a ledger tombstone on its live generation) or disputed (the live
//! generation is contested). Every fact (subject, predicate) is owned by one capsule; the
//! facts of healthy capsules have an authoritative value, the facts of revoked and disputed
//! capsules have none (a question about one is unanswerable), and some questions are about
//! facts no capsule states at all.
//!
//! The retriever's hits are generated per question: an honest hit cites the live statement of
//! the owner of a fact, a poison hit is one of the classes P1..P8 below. The generator knows
//! what each hit is and what a correct mechanism says to it (`Hit::expected`); nothing here
//! calls the code under test.

use super::params::{self, Shape};
use crate::experiments::rng::Rng;
use ptr_memory::RawEvent;
use ptr_search::{Claim, SourceCitation};
use ptr_types::{EventRole, RawEventId, SessionId, Timestamp};
use std::collections::BTreeMap;

/// The lifecycle state the generator gave a capsule.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Status {
    Healthy,
    Revoked,
    Disputed,
}

/// Why a statement is in an event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Origin {
    /// A fact the capsule owns, with the value of this generation.
    Own,
    /// A statement in a revoked or disputed capsule about a fact a healthy capsule owns, with
    /// a value that is not the authoritative one.
    Mirror,
    /// A negated statement about a fact the (healthy) capsule owns.
    Negation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Statement {
    pub subject: String,
    pub predicate: String,
    pub negated: bool,
    pub value: String,
    pub origin: Origin,
    /// The span of the statement's line in the event text.
    pub start: usize,
    pub end: usize,
}

impl Statement {
    pub fn text(&self) -> String {
        render(&self.subject, &self.predicate, self.negated, &self.value)
    }
}

/// `subject: predicate is [not ]value`.
pub fn render(subject: &str, predicate: &str, negated: bool, value: &str) -> String {
    let not = if negated { "not " } else { "" };
    format!("{subject}: {predicate} is {not}{value}")
}

/// One raw event: one generation of one capsule.
#[derive(Clone, Debug)]
pub struct GenerationEvent {
    pub generation: u64,
    pub raw_id: String,
    pub text: String,
    pub lines: Vec<Statement>,
}

#[derive(Clone, Debug)]
pub struct Capsule {
    pub id: String,
    pub status: Status,
    /// Generation `k` is `generations[k - 1]`; the last one is the live generation.
    pub generations: Vec<GenerationEvent>,
}

impl Capsule {
    pub fn live_generation(&self) -> u64 {
        self.generations.len() as u64
    }

    pub fn live(&self) -> &GenerationEvent {
        self.generations.last().expect("a capsule has a generation")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuestionKind {
    /// A fact of a healthy capsule: the authoritative answer is its live value.
    Answerable,
    /// A fact of a revoked or disputed capsule: no authoritative answer.
    Quarantined,
    /// A fact no capsule states.
    Absent,
}

#[derive(Clone, Debug)]
pub struct Question {
    pub subject: String,
    pub predicate: String,
    pub kind: QuestionKind,
    /// The authoritative value; `None` for an unanswerable question (abstain).
    pub expected: Option<String>,
}

/// The poison classes and the honest hit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Class {
    Honest,
    /// A source id no store holds.
    P1,
    /// A span that is not the quote: shifted, past the end, or a doctored quote.
    P2,
    /// A raw event whose bytes no longer match its digest.
    P3,
    /// A statement about a subject that only looks like the claimed one.
    P4,
    /// A statement that contradicts the claim: another value, or the same one negated.
    P5,
    /// A hit of a superseded generation.
    P6,
    /// A hit of a revoked generation that is still the capsule's live one.
    P7,
    /// A hit of a disputed generation.
    P8,
}

impl Class {
    pub const POISON: [Class; 8] = [
        Self::P1,
        Self::P2,
        Self::P3,
        Self::P4,
        Self::P5,
        Self::P6,
        Self::P7,
        Self::P8,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Honest => "H",
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::P3 => "P3",
            Self::P4 => "P4",
            Self::P5 => "P5",
            Self::P6 => "P6",
            Self::P7 => "P7",
            Self::P8 => "P8",
        }
    }

    /// The generator's label: whether a correct mechanism may promote a hit of this class.
    pub fn eligible(self) -> bool {
        matches!(self, Self::Honest)
    }

    /// Whether the class is refused by the lifecycle authority (the other poison classes
    /// carry a live generation and are refused by the source or the claim).
    pub fn is_lifecycle(self) -> bool {
        matches!(self, Self::P6 | Self::P7 | Self::P8)
    }
}

/// The variant of a class (P2 and P5 have several).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    Plain,
    ShiftedSpan,
    SpanPastEnd,
    ForgedQuote,
    ContradictingValue,
    Negation,
}

/// What a correct mechanism says to a hit: the refusal it must give, or `None` to promote.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Refusal {
    UnknownRawEvent,
    DigestMismatch,
    EmptySpan,
    SpanOutOfRange,
    QuoteMismatch,
    MalformedClaim,
    MalformedStatement,
    SubjectMismatch,
    PredicateMismatch,
    ValueMismatch,
    NegatedStatement,
    StaleGeneration,
    RevokedGeneration,
    DisputedGeneration,
    UnknownGeneration,
    WrongStage,
    SourceNotResolved,
    VerificationFailed,
    ClaimProofMismatch,
}

impl Refusal {
    pub fn name(self) -> &'static str {
        match self {
            Self::UnknownRawEvent => "unknown_raw_event",
            Self::DigestMismatch => "digest_mismatch",
            Self::EmptySpan => "empty_span",
            Self::SpanOutOfRange => "span_out_of_range",
            Self::QuoteMismatch => "quote_mismatch",
            Self::MalformedClaim => "malformed_claim",
            Self::MalformedStatement => "malformed_statement",
            Self::SubjectMismatch => "subject_mismatch",
            Self::PredicateMismatch => "predicate_mismatch",
            Self::ValueMismatch => "value_mismatch",
            Self::NegatedStatement => "negated_statement",
            Self::StaleGeneration => "stale_generation",
            Self::RevokedGeneration => "revoked_generation",
            Self::DisputedGeneration => "disputed_generation",
            Self::UnknownGeneration => "unknown_generation",
            Self::WrongStage => "wrong_stage",
            Self::SourceNotResolved => "source_not_resolved",
            Self::VerificationFailed => "verification_failed",
            Self::ClaimProofMismatch => "claim_proof_mismatch",
        }
    }
}

/// One retrieved hit, with what the generator knows about it.
#[derive(Clone, Debug)]
pub struct Hit {
    pub id: usize,
    pub question: usize,
    pub class: Class,
    pub variant: Variant,
    pub capsule: String,
    pub generation: u64,
    pub citation: SourceCitation,
    pub claim: Claim,
    pub score_milli: u64,
    /// The refusal a correct mechanism gives; `None` for an honest hit.
    pub expected: Option<Refusal>,
    /// Whether the claim is false in the world (it is not the authoritative value of the
    /// question's fact, or the fact has none).
    pub claim_untrue: bool,
}

/// The generated world.
pub struct World {
    pub seed: u64,
    pub shape: Shape,
    pub capsules: Vec<Capsule>,
    pub questions: Vec<Question>,
    /// Every hit, by id; `by_question[q]` lists the ids returned for question `q`, ranked by
    /// descending score (ties by id).
    pub hits: Vec<Hit>,
    pub by_question: Vec<Vec<usize>>,
    /// Corrupt replicas of raw events: the bytes changed, the digest of the original kept.
    pub replicas: Vec<RawEvent>,
}

pub fn entity(index: usize) -> String {
    format!("ent-{index:04}")
}

/// A subject that only looks like `entity(index)`: an ASCII letter for a digit, or a Cyrillic
/// letter for a Latin one.
pub fn lookalike(index: usize) -> String {
    let name = entity(index);
    if index.is_multiple_of(2) {
        name.replacen('0', "O", 1)
    } else {
        name.replacen('e', "\u{435}", 1)
    }
}

fn value(predicate: usize, k: usize) -> String {
    format!("v{predicate}{k:02}")
}

fn pick_other(rng: &mut Rng, predicate: usize, current: &str) -> String {
    loop {
        let candidate = value(predicate, rng.index(params::VALUES_PER_PREDICATE));
        if candidate != current {
            return candidate;
        }
    }
}

type FactKey = (String, String);

/// A fact before it is stated: subject, predicate index.
#[derive(Clone, Debug)]
struct Fact {
    subject: String,
    predicate: usize,
}

struct Pools {
    own_live: Vec<(usize, usize)>,
    negations: Vec<(usize, usize)>,
    older: Vec<(usize, usize, usize)>,
}

impl World {
    pub fn generate(seed: u64, shape: Shape) -> Result<World, String> {
        let mut root = Rng::new(seed ^ 0x0051_0002_0000_0001);
        let healthy = shape.healthy;
        let twin_pairs = params::TWIN_PAIRS.min(healthy * 3);
        let mut statuses: Vec<Status> = std::iter::repeat_n(Status::Healthy, shape.healthy)
            .chain(std::iter::repeat_n(Status::Revoked, shape.revoked))
            .chain(std::iter::repeat_n(Status::Disputed, shape.disputed))
            .collect();
        shuffle(&mut root.fork(1), &mut statuses);
        let mut generation_counts: Vec<usize> = (0..shape.capsules())
            .map(|i| 1 + i % params::MAX_GENERATIONS)
            .collect();
        shuffle(&mut root.fork(2), &mut generation_counts);

        // Facts: the twin pairs first, so that every one lands in a healthy capsule.
        let mut twin_facts: Vec<Fact> = Vec::new();
        for i in 0..twin_pairs {
            let predicate = i % params::PREDICATES.len();
            twin_facts.push(Fact {
                subject: entity(i),
                predicate,
            });
            twin_facts.push(Fact {
                subject: lookalike(i),
                predicate,
            });
        }
        let mut other_facts: Vec<Fact> = Vec::new();
        for i in params::TWIN_PAIRS..params::TWIN_PAIRS + 80 {
            for predicate in 0..params::PREDICATES.len() {
                other_facts.push(Fact {
                    subject: entity(i),
                    predicate,
                });
            }
        }
        shuffle(&mut root.fork(3), &mut twin_facts);
        shuffle(&mut root.fork(4), &mut other_facts);
        let needed = shape.capsules() * params::FACTS_PER_CAPSULE;
        let facts: Vec<Fact> = twin_facts.into_iter().chain(other_facts).collect();
        if facts.len() < needed {
            return Err(format!("{needed} facts needed, {} available", facts.len()));
        }

        // Capsules in index order, but the facts go to the healthy ones first.
        let mut order: Vec<usize> = (0..shape.capsules()).collect();
        order.sort_by_key(|i| (statuses[*i], *i));
        let mut owned: Vec<Vec<Fact>> = vec![Vec::new(); shape.capsules()];
        for (slot, capsule) in order.iter().enumerate() {
            let from = slot * params::FACTS_PER_CAPSULE;
            owned[*capsule] = facts[from..from + params::FACTS_PER_CAPSULE].to_vec();
        }

        // Authoritative values: those of healthy capsules.
        let mut value_rng = root.fork(5);
        let mut live_values: Vec<Vec<String>> = Vec::new();
        for capsule_facts in &owned {
            live_values.push(
                capsule_facts
                    .iter()
                    .map(|fact| {
                        value(
                            fact.predicate,
                            value_rng.index(params::VALUES_PER_PREDICATE),
                        )
                    })
                    .collect(),
            );
        }
        let mut truth: BTreeMap<FactKey, String> = BTreeMap::new();
        for (capsule, status) in statuses.iter().enumerate() {
            if *status == Status::Healthy {
                for (fact, v) in owned[capsule].iter().zip(&live_values[capsule]) {
                    truth.insert(
                        (
                            fact.subject.clone(),
                            params::PREDICATES[fact.predicate].to_string(),
                        ),
                        v.clone(),
                    );
                }
            }
        }
        let healthy_facts: Vec<(FactKey, usize)> = truth
            .keys()
            .map(|key| {
                let predicate = params::PREDICATES
                    .iter()
                    .position(|p| *p == key.1)
                    .expect("a known predicate");
                (key.clone(), predicate)
            })
            .collect();

        // The events.
        let mut event_rng = root.fork(6);
        let mut capsules: Vec<Capsule> = Vec::new();
        for (index, status) in statuses.iter().enumerate() {
            let id = format!("cap-{index:02}");
            let count = generation_counts[index];
            let mut generations = Vec::new();
            for generation in 1..=count {
                let live = generation == count;
                let mut lines: Vec<(String, String, bool, String, Origin)> = Vec::new();
                for (fact, live_value) in owned[index].iter().zip(&live_values[index]) {
                    let shown =
                        if live || event_rng.below(1000) >= params::STALE_VALUE_CHANGE_PERMILLE {
                            live_value.clone()
                        } else {
                            pick_other(&mut event_rng, fact.predicate, live_value)
                        };
                    lines.push((
                        fact.subject.clone(),
                        params::PREDICATES[fact.predicate].to_string(),
                        false,
                        shown,
                        Origin::Own,
                    ));
                }
                if *status != Status::Healthy {
                    for _ in 0..params::MIRRORS_PER_QUARANTINED_EVENT {
                        let ((subject, predicate), predicate_index) =
                            healthy_facts[event_rng.index(healthy_facts.len())].clone();
                        let authoritative = &truth[&(subject.clone(), predicate.clone())];
                        lines.push((
                            subject,
                            predicate,
                            false,
                            pick_other(&mut event_rng, predicate_index, authoritative),
                            Origin::Mirror,
                        ));
                    }
                } else if live {
                    for fact in owned[index]
                        .iter()
                        .take(params::NEGATIONS_PER_HEALTHY_EVENT)
                    {
                        let predicate = params::PREDICATES[fact.predicate].to_string();
                        let authoritative = &truth[&(fact.subject.clone(), predicate.clone())];
                        lines.push((
                            fact.subject.clone(),
                            predicate,
                            true,
                            pick_other(&mut event_rng, fact.predicate, authoritative),
                            Origin::Negation,
                        ));
                    }
                }
                shuffle(&mut event_rng, &mut lines);
                let mut text = String::new();
                let mut statements = Vec::new();
                for (subject, predicate, negated, shown, origin) in lines {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    let start = text.len();
                    text.push_str(&render(&subject, &predicate, negated, &shown));
                    statements.push(Statement {
                        subject,
                        predicate,
                        negated,
                        value: shown,
                        origin,
                        start,
                        end: text.len(),
                    });
                }
                generations.push(GenerationEvent {
                    generation: generation as u64,
                    raw_id: format!("raw-{id}-g{generation}"),
                    text,
                    lines: statements,
                });
            }
            capsules.push(Capsule {
                id,
                status: *status,
                generations,
            });
        }

        // The questions.
        let mut questions: Vec<Question> = Vec::new();
        let mut by_fact: BTreeMap<FactKey, usize> = BTreeMap::new();
        for kind in [QuestionKind::Answerable, QuestionKind::Quarantined] {
            for (index, capsule) in capsules.iter().enumerate() {
                let wanted =
                    (capsule.status == Status::Healthy) == (kind == QuestionKind::Answerable);
                if !wanted {
                    continue;
                }
                for fact in &owned[index] {
                    let predicate = params::PREDICATES[fact.predicate].to_string();
                    let key = (fact.subject.clone(), predicate.clone());
                    by_fact.insert(key.clone(), questions.len());
                    questions.push(Question {
                        subject: fact.subject.clone(),
                        predicate,
                        kind,
                        expected: truth.get(&key).cloned(),
                    });
                }
            }
        }
        for j in 0..shape.absent {
            let predicate = params::PREDICATES[j % params::PREDICATES.len()].to_string();
            let subject = entity(params::TWIN_PAIRS + 80 + j / params::PREDICATES.len());
            by_fact.insert((subject.clone(), predicate.clone()), questions.len());
            questions.push(Question {
                subject,
                predicate,
                kind: QuestionKind::Absent,
                expected: None,
            });
        }

        let mut world = World {
            seed,
            shape,
            capsules,
            questions,
            hits: Vec::new(),
            by_question: Vec::new(),
            replicas: Vec::new(),
        };
        world.by_question = vec![Vec::new(); world.questions.len()];
        world.generate_hits(&mut root.fork(7), &by_fact, &truth, twin_pairs)?;
        Ok(world)
    }

    fn pools(&self) -> Pools {
        let mut own_live = Vec::new();
        let mut negations = Vec::new();
        let mut older = Vec::new();
        for (c, capsule) in self.capsules.iter().enumerate() {
            let last = capsule.generations.len() - 1;
            for (g, event) in capsule.generations.iter().enumerate() {
                for (l, line) in event.lines.iter().enumerate() {
                    if g < last {
                        older.push((c, g, l));
                    } else if capsule.status == Status::Healthy {
                        match line.origin {
                            Origin::Own => own_live.push((c, l)),
                            Origin::Negation => negations.push((c, l)),
                            Origin::Mirror => {}
                        }
                    }
                }
            }
        }
        Pools {
            own_live,
            negations,
            older,
        }
    }

    fn generate_hits(
        &mut self,
        rng: &mut Rng,
        by_fact: &BTreeMap<FactKey, usize>,
        truth: &BTreeMap<FactKey, String>,
        twin_pairs: usize,
    ) -> Result<(), String> {
        let pools = self.pools();
        let question_of = |subject: &str, predicate: &str| -> Result<usize, String> {
            by_fact
                .get(&(subject.to_string(), predicate.to_string()))
                .copied()
                .ok_or_else(|| format!("no question for {subject} {predicate}"))
        };
        let untrue = |claim: &Claim| {
            truth
                .get(&(claim.subject.clone(), claim.predicate.clone()))
                .map(|authoritative| authoritative != &claim.value)
                .unwrap_or(true)
        };
        let mut hits: Vec<Hit> = Vec::new();
        let mut replicas: Vec<RawEvent> = Vec::new();

        // Honest: one hit per answerable question, citing the owner's live statement.
        let mut honest_rng = rng.fork(10);
        for (c, l) in &pools.own_live {
            let capsule = &self.capsules[*c];
            let event = capsule.live();
            let line = &event.lines[*l];
            let claim = Claim::new(&line.subject, &line.predicate, &line.value);
            hits.push(Hit {
                id: 0,
                question: question_of(&line.subject, &line.predicate)?,
                class: Class::Honest,
                variant: Variant::Plain,
                capsule: capsule.id.clone(),
                generation: capsule.live_generation(),
                citation: self.citation(event, line),
                claim: claim.clone(),
                score_milli: honest_rng.range(params::HONEST_SCORE_MIN, params::HONEST_SCORE_MAX),
                expected: None,
                claim_untrue: untrue(&claim),
            });
        }

        // P1: a source id no store holds. Every absent question gets one, then random ones.
        let mut p1_rng = rng.fork(11);
        let healthy: Vec<usize> = (0..self.capsules.len())
            .filter(|c| self.capsules[*c].status == Status::Healthy)
            .collect();
        let absent: Vec<usize> = (0..self.questions.len())
            .filter(|q| self.questions[*q].kind == QuestionKind::Absent)
            .collect();
        let mut p1_questions: Vec<usize> = absent.clone();
        let mut others: Vec<usize> = (0..self.questions.len())
            .filter(|q| self.questions[*q].kind != QuestionKind::Absent)
            .collect();
        shuffle(&mut p1_rng, &mut others);
        p1_questions.extend(others);
        p1_questions.truncate(params::QUOTA_P1_FABRICATED_SOURCE.max(absent.len()));
        for (n, q) in p1_questions.into_iter().enumerate() {
            let question = &self.questions[q];
            let predicate = params::PREDICATES
                .iter()
                .position(|p| *p == question.predicate)
                .expect("a known predicate");
            let fabricated_value = value(predicate, p1_rng.index(params::VALUES_PER_PREDICATE));
            let claim = Claim::new(&question.subject, &question.predicate, fabricated_value);
            let quote = render(&claim.subject, &claim.predicate, false, &claim.value);
            let capsule = &self.capsules[healthy[p1_rng.index(healthy.len())]];
            hits.push(Hit {
                id: 0,
                question: q,
                class: Class::P1,
                variant: Variant::Plain,
                capsule: capsule.id.clone(),
                generation: capsule.live_generation(),
                citation: SourceCitation {
                    raw_id: RawEventId::from(format!("raw-fab-{n:04}").as_str()),
                    start: 0,
                    end: quote.len(),
                    quote,
                },
                claim: claim.clone(),
                score_milli: p1_rng.range(params::POISON_SCORE_MIN, params::POISON_SCORE_MAX),
                expected: Some(Refusal::UnknownRawEvent),
                claim_untrue: untrue(&claim),
            });
        }

        // P2: shifted span, span past the end, doctored quote; P3: a corrupt replica.
        let mut p2_rng = rng.fork(12);
        let mut live_pool = pools.own_live.clone();
        shuffle(&mut p2_rng, &mut live_pool);
        let mut cursor = 0usize;
        let mut take = |count: usize| -> Vec<(usize, usize)> {
            let picked: Vec<(usize, usize)> = (0..count)
                .map(|i| live_pool[(cursor + i) % live_pool.len()])
                .collect();
            cursor += count;
            picked
        };
        for (c, l) in take(params::QUOTA_P2_SHIFTED_SPAN) {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let width = line.end - line.start;
            let shift = p2_rng.range(1, 3) as usize;
            // Forwards when the span still fits in the event, else backwards.
            let start = if line.end + shift <= event.text.len() {
                line.start + shift
            } else {
                line.start - shift.min(line.start)
            };
            hits.push(self.poison(
                &mut p2_rng,
                capsule,
                Class::P2,
                Variant::ShiftedSpan,
                question_of(&line.subject, &line.predicate)?,
                SourceCitation {
                    raw_id: RawEventId::from(event.raw_id.as_str()),
                    start,
                    end: start + width,
                    quote: line.text(),
                },
                Claim::new(&line.subject, &line.predicate, &line.value),
                Refusal::QuoteMismatch,
                &untrue,
            ));
        }
        for (c, l) in take(params::QUOTA_P2_SPAN_PAST_END) {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let overshoot = p2_rng.range(1, 4) as usize;
            hits.push(self.poison(
                &mut p2_rng,
                capsule,
                Class::P2,
                Variant::SpanPastEnd,
                question_of(&line.subject, &line.predicate)?,
                SourceCitation {
                    raw_id: RawEventId::from(event.raw_id.as_str()),
                    start: line.start,
                    end: event.text.len() + overshoot,
                    quote: line.text(),
                },
                Claim::new(&line.subject, &line.predicate, &line.value),
                Refusal::SpanOutOfRange,
                &untrue,
            ));
        }
        for (c, l) in take(params::QUOTA_P2_FORGED_QUOTE) {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let predicate = params::PREDICATES
                .iter()
                .position(|p| *p == line.predicate)
                .expect("a known predicate");
            let forged = pick_other(&mut p2_rng, predicate, &line.value);
            hits.push(self.poison(
                &mut p2_rng,
                capsule,
                Class::P2,
                Variant::ForgedQuote,
                question_of(&line.subject, &line.predicate)?,
                SourceCitation {
                    raw_id: RawEventId::from(event.raw_id.as_str()),
                    start: line.start,
                    end: line.end,
                    quote: render(&line.subject, &line.predicate, false, &forged),
                },
                Claim::new(&line.subject, &line.predicate, &forged),
                Refusal::QuoteMismatch,
                &untrue,
            ));
        }
        for (n, (c, l)) in take(params::QUOTA_P3_TAMPERED_DIGEST)
            .into_iter()
            .enumerate()
        {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let predicate = params::PREDICATES
                .iter()
                .position(|p| *p == line.predicate)
                .expect("a known predicate");
            let tampered_value = pick_other(&mut p2_rng, predicate, &line.value);
            let mut text = event.text.clone();
            let value_start = line.end - line.value.len();
            text.replace_range(value_start..line.end, &tampered_value);
            let replica_id = format!("raw-t-{n:04}");
            let mut replica = RawEvent::new(
                RawEventId::from(replica_id.as_str()),
                SessionId::from("q002"),
                EventRole::User,
                text.as_bytes().to_vec(),
                Timestamp(1),
            );
            // The bytes changed, the digest is the original's.
            replica.digest = RawEvent::digest_for(event.text.as_bytes());
            replicas.push(replica);
            let quote = render(&line.subject, &line.predicate, false, &tampered_value);
            hits.push(self.poison(
                &mut p2_rng,
                capsule,
                Class::P3,
                Variant::Plain,
                question_of(&line.subject, &line.predicate)?,
                SourceCitation {
                    raw_id: RawEventId::from(replica_id.as_str()),
                    start: line.start,
                    end: line.end,
                    quote,
                },
                Claim::new(&line.subject, &line.predicate, &tampered_value),
                Refusal::DigestMismatch,
                &untrue,
            ));
        }

        // P4: a hit that cites the statement about the twin of the question's subject.
        let mut p4_rng = rng.fork(13);
        let mut p4_pool: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..twin_pairs {
            let predicate = params::PREDICATES[i % params::PREDICATES.len()];
            for (asked, cited) in [(entity(i), lookalike(i)), (lookalike(i), entity(i))] {
                let q = question_of(&asked, predicate)?;
                let source = pools
                    .own_live
                    .iter()
                    .find(|(c, l)| {
                        let line = &self.capsules[*c].live().lines[*l];
                        line.subject == cited && line.predicate == predicate
                    })
                    .copied()
                    .ok_or_else(|| format!("no live statement about {cited} {predicate}"))?;
                p4_pool.push((q, source.0, source.1));
            }
        }
        shuffle(&mut p4_rng, &mut p4_pool);
        p4_pool.truncate(params::QUOTA_P4_LOOKALIKE_SUBJECT);
        for (q, c, l) in p4_pool {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let question = &self.questions[q];
            hits.push(self.poison(
                &mut p4_rng,
                capsule,
                Class::P4,
                Variant::Plain,
                q,
                self.citation(event, line),
                Claim::new(&question.subject, &line.predicate, &line.value),
                Refusal::SubjectMismatch,
                &untrue,
            ));
        }

        // P5: a contradicting value, a negated statement.
        let mut p5_rng = rng.fork(14);
        let mut value_pool = pools.own_live.clone();
        shuffle(&mut p5_rng, &mut value_pool);
        value_pool.truncate(params::QUOTA_P5_CONTRADICTING_VALUE);
        for (c, l) in value_pool {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            let predicate = params::PREDICATES
                .iter()
                .position(|p| *p == line.predicate)
                .expect("a known predicate");
            let claimed = pick_other(&mut p5_rng, predicate, &line.value);
            hits.push(self.poison(
                &mut p5_rng,
                capsule,
                Class::P5,
                Variant::ContradictingValue,
                question_of(&line.subject, &line.predicate)?,
                self.citation(event, line),
                Claim::new(&line.subject, &line.predicate, &claimed),
                Refusal::ValueMismatch,
                &untrue,
            ));
        }
        let mut negation_pool = pools.negations.clone();
        shuffle(&mut p5_rng, &mut negation_pool);
        negation_pool.truncate(params::QUOTA_P5_NEGATION);
        for (c, l) in negation_pool {
            let capsule = &self.capsules[c];
            let event = capsule.live();
            let line = &event.lines[l];
            hits.push(self.poison(
                &mut p5_rng,
                capsule,
                Class::P5,
                Variant::Negation,
                question_of(&line.subject, &line.predicate)?,
                self.citation(event, line),
                Claim::new(&line.subject, &line.predicate, &line.value),
                Refusal::NegatedStatement,
                &untrue,
            ));
        }

        // P6: statements of superseded generations.
        let mut p6_rng = rng.fork(15);
        let mut older = pools.older.clone();
        shuffle(&mut p6_rng, &mut older);
        older.truncate(params::QUOTA_P6_SUPERSEDED);
        for (c, g, l) in older {
            let capsule = &self.capsules[c];
            let event = &capsule.generations[g];
            let line = &event.lines[l];
            hits.push(self.lifecycle_hit(
                &mut p6_rng,
                capsule,
                event,
                line,
                Class::P6,
                Refusal::StaleGeneration,
                question_of(&line.subject, &line.predicate)?,
                &untrue,
            ));
        }

        // P7 and P8: every statement of the live generation of each revoked or disputed capsule.
        let mut p78_rng = rng.fork(16);
        for capsule in &self.capsules {
            let (class, refusal) = match capsule.status {
                Status::Healthy => continue,
                Status::Revoked => (Class::P7, Refusal::RevokedGeneration),
                Status::Disputed => (Class::P8, Refusal::DisputedGeneration),
            };
            let event = capsule.live();
            for line in &event.lines {
                hits.push(self.lifecycle_hit(
                    &mut p78_rng,
                    capsule,
                    event,
                    line,
                    class,
                    refusal,
                    question_of(&line.subject, &line.predicate)?,
                    &untrue,
                ));
            }
        }

        for (id, hit) in hits.iter_mut().enumerate() {
            hit.id = id;
        }
        let mut by_question: Vec<Vec<usize>> = vec![Vec::new(); self.questions.len()];
        for hit in &hits {
            by_question[hit.question].push(hit.id);
        }
        for ids in &mut by_question {
            ids.sort_by_key(|id| (std::cmp::Reverse(hits[*id].score_milli), *id));
        }
        self.hits = hits;
        self.by_question = by_question;
        self.replicas = replicas;
        Ok(())
    }

    fn citation(&self, event: &GenerationEvent, line: &Statement) -> SourceCitation {
        SourceCitation {
            raw_id: RawEventId::from(event.raw_id.as_str()),
            start: line.start,
            end: line.end,
            quote: event.text[line.start..line.end].to_owned(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn poison(
        &self,
        rng: &mut Rng,
        capsule: &Capsule,
        class: Class,
        variant: Variant,
        question: usize,
        citation: SourceCitation,
        claim: Claim,
        expected: Refusal,
        untrue: &dyn Fn(&Claim) -> bool,
    ) -> Hit {
        Hit {
            id: 0,
            question,
            class,
            variant,
            capsule: capsule.id.clone(),
            generation: capsule.live_generation(),
            citation,
            claim: claim.clone(),
            score_milli: rng.range(params::POISON_SCORE_MIN, params::POISON_SCORE_MAX),
            expected: Some(expected),
            claim_untrue: untrue(&claim),
        }
    }

    /// A hit whose source and claim are exact and whose generation is not live.
    #[allow(clippy::too_many_arguments)]
    fn lifecycle_hit(
        &self,
        rng: &mut Rng,
        capsule: &Capsule,
        event: &GenerationEvent,
        line: &Statement,
        class: Class,
        expected: Refusal,
        question: usize,
        untrue: &dyn Fn(&Claim) -> bool,
    ) -> Hit {
        let claim = Claim::new(&line.subject, &line.predicate, &line.value);
        Hit {
            id: 0,
            question,
            class,
            variant: Variant::Plain,
            capsule: capsule.id.clone(),
            generation: event.generation,
            citation: self.citation(event, line),
            claim: claim.clone(),
            score_milli: rng.range(params::POISON_SCORE_MIN, params::POISON_SCORE_MAX),
            expected: Some(expected),
            claim_untrue: untrue(&claim),
        }
    }

    /// The generator's lifecycle table: whether generation `generation` of `capsule` is the
    /// live generation of a healthy capsule.
    pub fn generator_says_live(&self, capsule: &str, generation: u64) -> bool {
        self.capsules.iter().any(|c| {
            c.id == capsule && c.status == Status::Healthy && c.live_generation() == generation
        })
    }

    /// The raw events as the store of record holds them: id and text.
    pub fn events(&self) -> impl Iterator<Item = (&str, &str)> {
        self.capsules.iter().flat_map(|capsule| {
            capsule
                .generations
                .iter()
                .map(|event| (event.raw_id.as_str(), event.text.as_str()))
        })
    }

    /// The raw event of the store of record as a store entry.
    pub fn raw_event(id: &str, text: &str) -> RawEvent {
        RawEvent::new(
            RawEventId::from(id),
            SessionId::from("q002"),
            EventRole::User,
            text.as_bytes().to_vec(),
            Timestamp(1),
        )
    }

    pub fn answerable(&self) -> usize {
        self.questions
            .iter()
            .filter(|q| q.expected.is_some())
            .count()
    }

    /// The canonical text of the corpus: the shape, every raw event, the replicas, the
    /// questions and the hits with everything the generator decided. Its SHA-256 is pinned.
    pub fn canonical(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "seed {}", self.seed);
        let _ = writeln!(
            out,
            "shape {} {} {} {}",
            self.shape.healthy, self.shape.revoked, self.shape.disputed, self.shape.absent
        );
        for capsule in &self.capsules {
            let _ = writeln!(out, "capsule {} {:?}", capsule.id, capsule.status);
            for event in &capsule.generations {
                let _ = writeln!(
                    out,
                    "event {} g{} {}",
                    event.raw_id,
                    event.generation,
                    event.text.replace('\n', "|")
                );
            }
        }
        for replica in &self.replicas {
            let _ = writeln!(
                out,
                "replica {} {}",
                replica.id,
                String::from_utf8_lossy(&replica.content).replace('\n', "|")
            );
        }
        for (index, question) in self.questions.iter().enumerate() {
            let _ = writeln!(
                out,
                "question {index} {} {} {:?} {:?}",
                question.subject, question.predicate, question.kind, question.expected
            );
        }
        for hit in &self.hits {
            let _ = writeln!(
                out,
                "hit {} q{} {} {:?} {} g{} {} {}..{} {:?} {} {} {} {} {:?} {}",
                hit.id,
                hit.question,
                hit.class.name(),
                hit.variant,
                hit.capsule,
                hit.generation,
                hit.citation.raw_id,
                hit.citation.start,
                hit.citation.end,
                hit.citation.quote,
                hit.claim.subject,
                hit.claim.predicate,
                hit.claim.value,
                hit.score_milli,
                hit.expected,
                hit.claim_untrue
            );
        }
        out
    }

    /// The SHA-256 of [`World::canonical`], in hex.
    pub fn corpus_digest(&self) -> String {
        RawEvent::digest_for(self.canonical().as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// Fisher-Yates with the experiment's generator.
pub fn shuffle<T>(rng: &mut Rng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = rng.index(i + 1);
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned(seed: u64) -> World {
        World::generate(seed, Shape::PINNED).expect("the pinned world generates")
    }

    #[test]
    fn the_same_seed_gives_the_same_corpus_and_another_seed_another() {
        let first = pinned(17);
        assert_eq!(first.corpus_digest(), pinned(17).corpus_digest());
        assert_ne!(first.corpus_digest(), pinned(29).corpus_digest());
    }

    #[test]
    fn the_pinned_world_has_the_shape_and_the_questions_the_design_names() {
        for seed in params::SEEDS {
            let world = pinned(seed);
            assert_eq!(world.capsules.len(), params::CAPSULES);
            let by_status = |status| world.capsules.iter().filter(|c| c.status == status).count();
            assert_eq!(by_status(Status::Healthy), params::HEALTHY_CAPSULES);
            assert_eq!(by_status(Status::Revoked), params::REVOKED_CAPSULES);
            assert_eq!(by_status(Status::Disputed), params::DISPUTED_CAPSULES);
            assert_eq!(
                world.answerable(),
                params::HEALTHY_CAPSULES * params::FACTS_PER_CAPSULE
            );
            assert_eq!(
                world.questions.len(),
                params::CAPSULES * params::FACTS_PER_CAPSULE + params::ABSENT_QUESTIONS
            );
            assert!(world
                .capsules
                .iter()
                .all(|c| (1..=3).contains(&c.generations.len())));
            assert!(world.capsules.iter().any(|c| c.generations.len() == 3));
        }
    }

    #[test]
    fn every_statement_span_is_its_line_and_events_hold_the_declared_statements() {
        let world = pinned(17);
        for capsule in &world.capsules {
            for event in &capsule.generations {
                for line in &event.lines {
                    assert_eq!(&event.text[line.start..line.end], line.text());
                }
                let own = event
                    .lines
                    .iter()
                    .filter(|l| l.origin == Origin::Own)
                    .count();
                assert_eq!(own, params::FACTS_PER_CAPSULE);
            }
        }
    }

    #[test]
    fn every_poison_class_has_at_least_the_floor_of_hits_for_every_seed() {
        for seed in params::SEEDS {
            let world = pinned(seed);
            for class in Class::POISON {
                let count = world.hits.iter().filter(|h| h.class == class).count();
                assert!(
                    count as i64 >= params::MIN_POISON_ATTEMPTS_PER_CLASS,
                    "seed {seed} {}: {count}",
                    class.name()
                );
            }
            let honest = world
                .hits
                .iter()
                .filter(|h| h.class == Class::Honest)
                .count();
            assert_eq!(honest, world.answerable());
            assert!(honest as i64 >= params::MIN_TRUE_CLAIMS);
        }
    }

    #[test]
    fn hits_are_ranked_by_descending_score_and_every_hit_is_returned_once() {
        let world = pinned(17);
        let mut seen = vec![false; world.hits.len()];
        for ids in &world.by_question {
            let scores: Vec<u64> = ids.iter().map(|id| world.hits[*id].score_milli).collect();
            assert!(scores.windows(2).all(|pair| pair[0] >= pair[1]));
            for id in ids {
                assert!(!seen[*id]);
                seen[*id] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn poison_hits_are_what_their_class_says() {
        let world = pinned(43);
        for hit in &world.hits {
            let capsule = world
                .capsules
                .iter()
                .find(|c| c.id == hit.capsule)
                .expect("a capsule");
            match hit.class {
                Class::Honest => {
                    assert_eq!(hit.expected, None);
                    assert_eq!(capsule.status, Status::Healthy);
                    assert_eq!(hit.generation, capsule.live_generation());
                    assert!(!hit.claim_untrue);
                }
                Class::P1 => assert!(world.events().all(|(id, _)| id != hit.citation.raw_id.0)),
                Class::P3 => {
                    assert!(world.events().all(|(id, _)| id != hit.citation.raw_id.0));
                    assert!(world
                        .replicas
                        .iter()
                        .any(|r| r.id == hit.citation.raw_id && r.validate_digest().is_err()));
                    assert!(hit.claim_untrue);
                }
                Class::P6 => assert!(hit.generation < capsule.live_generation()),
                Class::P7 => {
                    assert_eq!(capsule.status, Status::Revoked);
                    assert_eq!(hit.generation, capsule.live_generation());
                }
                Class::P8 => {
                    assert_eq!(capsule.status, Status::Disputed);
                    assert_eq!(hit.generation, capsule.live_generation());
                }
                Class::P2 | Class::P4 | Class::P5 => {}
            }
            assert_eq!(hit.expected.is_none(), hit.class.eligible());
        }
    }

    #[test]
    fn p2_variants_really_miss_the_source_and_p5_negations_cite_a_negated_line() {
        let world = pinned(29);
        for hit in world.hits.iter().filter(|h| h.class == Class::P2) {
            let event = world
                .capsules
                .iter()
                .flat_map(|c| &c.generations)
                .find(|e| e.raw_id == hit.citation.raw_id.0)
                .expect("a real event");
            let bytes = event.text.as_bytes();
            match hit.variant {
                Variant::SpanPastEnd => assert!(hit.citation.end > bytes.len()),
                Variant::ShiftedSpan => {
                    assert!(hit.citation.end <= bytes.len());
                    assert_ne!(
                        &bytes[hit.citation.start..hit.citation.end],
                        hit.citation.quote.as_bytes()
                    );
                    assert!(event.text.contains(&hit.citation.quote));
                }
                Variant::ForgedQuote => {
                    assert_ne!(
                        &bytes[hit.citation.start..hit.citation.end],
                        hit.citation.quote.as_bytes()
                    );
                    assert!(hit.claim_untrue);
                }
                other => panic!("{other:?}"),
            }
        }
        for hit in world.hits.iter().filter(|h| h.variant == Variant::Negation) {
            assert!(hit.citation.quote.contains(" is not "));
            assert!(hit.claim_untrue);
        }
    }

    #[test]
    fn lookalike_subjects_differ_from_their_originals_in_bytes() {
        for i in 0..params::TWIN_PAIRS {
            assert_ne!(entity(i), lookalike(i));
            assert!(!lookalike(i).contains(char::is_whitespace));
        }
        assert!(lookalike(1).contains('\u{435}'));
        assert!(lookalike(2).contains('O'));
    }

    #[test]
    fn a_reduced_world_generates_for_every_seed() {
        for seed in params::SEEDS {
            let world =
                World::generate(seed, Shape::scaled(9).expect("a shape")).expect("generates");
            assert_eq!(world.capsules.len(), 9);
            assert!(!world.hits.is_empty());
        }
    }
}
