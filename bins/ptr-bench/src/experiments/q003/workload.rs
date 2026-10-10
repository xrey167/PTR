//! The seeded workload: which capsule lives which life, which capsules the queries ask for, which
//! stale rows are planted by raw SQL, which late writes are attempted and which probes are asked.
//!
//! Everything is a function of the shape and the seed (the pool only supplies texts), and the
//! SHA-256 of its canonical rendering is the workload digest the harness compares with the pin.
//! The live table the oracle keeps (`oracle.rs`) is built from the same event list, but by its own
//! code: it never asks the product.

use std::collections::HashSet;

use ptr_ledger::integrity::sha256;

use super::data::hex;
use super::params::{self, Shape};
use crate::experiments::rng::Rng;

/// The life of one capsule: how many generations it was committed with and whether it was then
/// revoked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fate {
    /// Live at generation 1.
    Stay,
    /// Superseded once: live at generation 2.
    Once,
    /// Superseded twice: live at generation 3.
    Twice,
    /// Revoked at generation 1, 2 or 3.
    Revoked(u64),
}

impl Fate {
    /// The newest generation the capsule was ever committed with.
    pub fn history(self) -> u64 {
        match self {
            Fate::Stay => 1,
            Fate::Once => 2,
            Fate::Twice => 3,
            Fate::Revoked(generation) => generation,
        }
    }

    pub fn revoked(self) -> bool {
        matches!(self, Fate::Revoked(_))
    }

    pub fn name(self) -> &'static str {
        match self {
            Fate::Stay => "stay",
            Fate::Once => "once",
            Fate::Twice => "twice",
            Fate::Revoked(1) => "revoked1",
            Fate::Revoked(2) => "revoked2",
            Fate::Revoked(_) => "revoked3",
        }
    }
}

pub fn capsule_id(cue: usize) -> String {
    format!("cap-{cue:04}")
}

/// The id of a document in the reference and strong arms.
pub fn doc_id(capsule: &str, generation: u64) -> String {
    format!("{capsule}@{generation}")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind {
    Commit,
    Supersede { old: u64, new: u64 },
    Revoke { generation: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Event {
    pub kind: EventKind,
    pub cue: usize,
}

/// The lifecycle phases: commits of generation 1, then the supersessions to generation 2 and to
/// generation 3 (after which the first checkpoint is taken), then the revocations (after which the
/// second is).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Commits,
    FirstSupersessions,
    SecondSupersessions,
    Revocations,
}

pub const PHASES: [Phase; 4] = [
    Phase::Commits,
    Phase::FirstSupersessions,
    Phase::SecondSupersessions,
    Phase::Revocations,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Query {
    pub cue: usize,
    pub paraphrase: u8,
    /// 1 or 2.
    pub checkpoint: u8,
    /// Asks for a capsule that is revoked at the time of asking: unanswerable.
    pub revoked_target: bool,
    /// The generation live (or, for a revoked target, revoked) at the time of asking.
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Plant {
    pub cue: usize,
    pub generation: u64,
    /// Planted before the queries of this checkpoint.
    pub checkpoint: u8,
    /// A revoked generation (the capsule's live generation row still names it) rather than a
    /// superseded one.
    pub revoked: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LateKind {
    Superseded,
    Revoked,
    Future,
    Ghost,
}

impl LateKind {
    pub fn name(self) -> &'static str {
        match self {
            LateKind::Superseded => "superseded",
            LateKind::Revoked => "revoked",
            LateKind::Future => "future",
            LateKind::Ghost => "ghost",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LateWrite {
    pub kind: LateKind,
    /// The cue whose capsule id is written (for a ghost, the cue its text comes from).
    pub cue: usize,
    pub generation: u64,
}

impl LateWrite {
    pub fn capsule(&self) -> String {
        match self.kind {
            LateKind::Ghost => format!("ghost-{:04}", self.cue),
            _ => capsule_id(self.cue),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Workload {
    pub shape: Shape,
    pub fates: Vec<Fate>,
    pub queries: Vec<Query>,
    pub plants: Vec<Plant>,
    pub late_writes: Vec<LateWrite>,
    /// Indexes into `plants`: the probes whose text is the planted row's own body.
    pub stale_probes: Vec<usize>,
    /// Cues whose live document's body is a probe's text (second checkpoint).
    pub live_probes: Vec<usize>,
    /// Indexes into `queries` that are repeated on the unchanged snapshot.
    pub repeats: Vec<usize>,
    /// Per query: whether the strong arm answers it (a seeded subset stratified like the whole).
    pub strong: Vec<bool>,
}

fn percent(value: usize, percent: usize) -> usize {
    value * percent / 100
}

fn shuffle<T>(rng: &mut Rng, items: &mut [T]) {
    for index in (1..items.len()).rev() {
        items.swap(index, rng.index(index + 1));
    }
}

/// `total` split as evenly as possible over `parts`, the earlier parts taking the remainder.
fn split_evenly(total: usize, parts: usize) -> Vec<usize> {
    (0..parts)
        .map(|part| total / parts + usize::from(part < total % parts))
        .collect()
}

impl Workload {
    pub fn generate(seed: u64, shape: Shape) -> Result<Self, String> {
        shape.check()?;
        let mut root = Rng::new(seed ^ 0x5133_0003_0000_0000);
        let fates = assign_fates(&mut root.fork(1), shape.capsules);
        let queries = pick_queries(&mut root.fork(2), &fates, shape)?;
        let strong = pick_strong(&queries, shape.strong_queries());
        let mut used: HashSet<(usize, u64)> = HashSet::new();
        let mut plants = Vec::new();
        for query in &queries {
            if query.revoked_target {
                plants.push(Plant {
                    cue: query.cue,
                    generation: query.generation,
                    checkpoint: query.checkpoint,
                    revoked: true,
                });
            } else if query.generation >= 2 {
                plants.push(Plant {
                    cue: query.cue,
                    generation: query.generation - 1,
                    checkpoint: query.checkpoint,
                    revoked: false,
                });
            }
        }
        used.extend(plants.iter().map(|plant| (plant.cue, plant.generation)));
        let late_writes = pick_late_writes(&mut root.fork(3), &fates, shape, &mut used)?;
        let extra = percent(shape.queries, params::EXTRA_PLANT_PERCENT);
        for checkpoint in [1u8, 2] {
            let mut rng = root.fork(10 + u64::from(checkpoint));
            let mut candidates: Vec<(usize, u64, bool)> = Vec::new();
            for (cue, fate) in fates.iter().enumerate() {
                for generation in 1..fate.history() {
                    candidates.push((cue, generation, false));
                }
                // A revoked generation exists only after the revocations (second checkpoint).
                if checkpoint == 2 && fate.revoked() {
                    candidates.push((cue, fate.history(), true));
                }
            }
            candidates.retain(|(cue, generation, _)| !used.contains(&(*cue, *generation)));
            shuffle(&mut rng, &mut candidates);
            if candidates.len() < extra {
                return Err(format!(
                    "only {} candidates for {extra} extra plants at checkpoint {checkpoint}",
                    candidates.len()
                ));
            }
            for (cue, generation, revoked) in candidates.into_iter().take(extra) {
                used.insert((cue, generation));
                plants.push(Plant {
                    cue,
                    generation,
                    checkpoint,
                    revoked,
                });
            }
        }
        let mut rng = root.fork(20);
        let stale_count = percent(shape.queries, params::STALE_PROBE_PERCENT);
        let mut indexes: Vec<usize> = (0..plants.len()).collect();
        shuffle(&mut rng, &mut indexes);
        let stale_probes: Vec<usize> = indexes.into_iter().take(stale_count).collect();
        let live_count = percent(shape.queries, params::LIVE_PROBE_PERCENT);
        let mut live: Vec<usize> = (0..shape.capsules)
            .filter(|cue| !fates[*cue].revoked())
            .collect();
        shuffle(&mut rng, &mut live);
        let live_probes: Vec<usize> = live.into_iter().take(live_count).collect();
        let stride = (shape.queries / params::REPEAT_QUERIES).max(1);
        let repeats: Vec<usize> = (0..shape.queries)
            .step_by(stride)
            .take(params::REPEAT_QUERIES)
            .collect();
        Ok(Self {
            shape,
            fates,
            queries,
            plants,
            late_writes,
            stale_probes,
            live_probes,
            repeats,
            strong,
        })
    }

    /// The events of one phase, in order of capsule.
    pub fn events(&self, phase: Phase) -> Vec<Event> {
        let mut out = Vec::new();
        for (cue, fate) in self.fates.iter().enumerate() {
            let history = fate.history();
            let kind = match phase {
                Phase::Commits => Some(EventKind::Commit),
                Phase::FirstSupersessions if history >= 2 => {
                    Some(EventKind::Supersede { old: 1, new: 2 })
                }
                Phase::SecondSupersessions if history >= 3 => {
                    Some(EventKind::Supersede { old: 2, new: 3 })
                }
                Phase::Revocations if fate.revoked() => Some(EventKind::Revoke {
                    generation: history,
                }),
                _ => None,
            };
            if let Some(kind) = kind {
                out.push(Event { kind, cue });
            }
        }
        out
    }

    pub fn queries_at(&self, checkpoint: u8) -> impl Iterator<Item = (usize, &Query)> {
        self.queries
            .iter()
            .enumerate()
            .filter(move |(_, query)| query.checkpoint == checkpoint)
    }

    pub fn plants_at(&self, checkpoint: u8) -> impl Iterator<Item = (usize, &Plant)> {
        self.plants
            .iter()
            .enumerate()
            .filter(move |(_, plant)| plant.checkpoint == checkpoint)
    }

    /// The canonical rendering the digest is taken over.
    pub fn canonical(&self, seed: u64) -> String {
        let mut out = format!(
            "q003-workload seed={seed} capsules={} queries={}\n",
            self.shape.capsules, self.shape.queries
        );
        out.push_str("fates:");
        for fate in &self.fates {
            out.push_str(&format!(" {}", fate.name()));
        }
        out.push('\n');
        for query in &self.queries {
            out.push_str(&format!(
                "query {} {} {} {} {}\n",
                query.checkpoint,
                query.cue,
                query.paraphrase,
                u8::from(query.revoked_target),
                query.generation
            ));
        }
        for plant in &self.plants {
            out.push_str(&format!(
                "plant {} {} {} {}\n",
                plant.checkpoint,
                plant.cue,
                plant.generation,
                u8::from(plant.revoked)
            ));
        }
        for late in &self.late_writes {
            out.push_str(&format!(
                "late {} {} {}\n",
                late.kind.name(),
                late.cue,
                late.generation
            ));
        }
        out.push_str(&format!("stale_probes {:?}\n", self.stale_probes));
        out.push_str(&format!("live_probes {:?}\n", self.live_probes));
        out.push_str(&format!("repeats {:?}\n", self.repeats));
        let strong: Vec<usize> = (0..self.strong.len())
            .filter(|index| self.strong[*index])
            .collect();
        out.push_str(&format!("strong {strong:?}\n"));
        out
    }

    pub fn digest(&self, seed: u64) -> String {
        hex(&sha256(self.canonical(seed).as_bytes()))
    }
}

fn assign_fates(rng: &mut Rng, capsules: usize) -> Vec<Fate> {
    let stay = capsules * params::STAY_PERMILLE / 1000;
    let once = capsules * params::SUPERSEDED_ONCE_PERMILLE / 1000;
    let twice = capsules * params::SUPERSEDED_TWICE_PERMILLE / 1000;
    let revoked1 = capsules * params::REVOKED_AT_1_PERMILLE / 1000;
    let revoked2 = capsules * params::REVOKED_AT_2_PERMILLE / 1000;
    let revoked3 = capsules - stay - once - twice - revoked1 - revoked2;
    let mut fates = Vec::with_capacity(capsules);
    for (fate, count) in [
        (Fate::Stay, stay),
        (Fate::Once, once),
        (Fate::Twice, twice),
        (Fate::Revoked(1), revoked1),
        (Fate::Revoked(2), revoked2),
        (Fate::Revoked(3), revoked3),
    ] {
        fates.extend(std::iter::repeat_n(fate, count));
    }
    shuffle(rng, &mut fates);
    fates
}

/// Draw `count` unused cues whose fate satisfies `accept`.
fn draw(
    rng: &mut Rng,
    fates: &[Fate],
    used: &mut HashSet<usize>,
    count: usize,
    accept: impl Fn(Fate) -> bool,
) -> Result<Vec<usize>, String> {
    let mut candidates: Vec<usize> = (0..fates.len())
        .filter(|cue| !used.contains(cue) && accept(fates[*cue]))
        .collect();
    if candidates.len() < count {
        return Err(format!(
            "only {} unused capsules for {count} queries",
            candidates.len()
        ));
    }
    shuffle(rng, &mut candidates);
    candidates.truncate(count);
    used.extend(candidates.iter().copied());
    Ok(candidates)
}

fn pick_queries(rng: &mut Rng, fates: &[Fate], shape: Shape) -> Result<Vec<Query>, String> {
    let first = percent(shape.queries, params::FIRST_CHECKPOINT_PERCENT);
    let revoked = percent(shape.queries, params::REVOKED_TARGET_PERCENT);
    let second_live = shape.queries - first - revoked;
    let mut used = HashSet::new();
    let mut queries = Vec::new();
    let mut add = |rng: &mut Rng, cues: Vec<usize>, checkpoint: u8, revoked_target: bool| {
        for cue in cues {
            queries.push(Query {
                cue,
                paraphrase: rng.below(3) as u8,
                checkpoint,
                revoked_target,
                generation: fates[cue].history(),
            });
        }
    };
    // First checkpoint: every capsule is still answerable; the queries ask for capsules that stay
    // (their live generation is the history), so a revoked capsule is asked only after its
    // revocation.
    for (generation, count) in split_evenly(first, 3).into_iter().enumerate() {
        let generation = generation as u64 + 1;
        let cues = draw(rng, fates, &mut used, count, |fate| {
            !fate.revoked() && fate.history() == generation
        })?;
        add(rng, cues, 1, false);
    }
    for (generation, count) in split_evenly(revoked, 3).into_iter().enumerate() {
        let generation = generation as u64 + 1;
        let cues = draw(rng, fates, &mut used, count, |fate| {
            fate == Fate::Revoked(generation)
        })?;
        add(rng, cues, 2, true);
    }
    for (generation, count) in split_evenly(second_live, 3).into_iter().enumerate() {
        let generation = generation as u64 + 1;
        let cues = draw(rng, fates, &mut used, count, |fate| {
            !fate.revoked() && fate.history() == generation
        })?;
        add(rng, cues, 2, false);
    }
    // Interleave the order the queries are asked in within each checkpoint, so latency does not
    // follow the generation.
    let mut order: Vec<usize> = (0..queries.len()).collect();
    shuffle(rng, &mut order);
    let mut shuffled: Vec<Query> = order.iter().map(|index| queries[*index]).collect();
    shuffled.sort_by_key(|query| query.checkpoint);
    Ok(shuffled)
}

/// The queries the strong arm answers: `want` of them, drawn from the answerable queries only (a
/// revoked target has no relevant document, so a strong answer to it could never enter a recall),
/// from every stratum (checkpoint, live generation) in proportion to its size (largest remainder),
/// the first ones of the stratum in query order, which is already shuffled by the seed. The rule
/// reads nothing but the strata and that order, so it is independent of every outcome.
fn pick_strong(queries: &[Query], want: usize) -> Vec<bool> {
    let answerable = queries.iter().filter(|query| !query.revoked_target).count();
    let mut chosen = vec![false; queries.len()];
    if want >= answerable {
        for (index, query) in queries.iter().enumerate() {
            chosen[index] = !query.revoked_target;
        }
        return chosen;
    }
    let mut strata: std::collections::BTreeMap<(u8, u64), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (index, query) in queries.iter().enumerate() {
        if !query.revoked_target {
            strata
                .entry((query.checkpoint, query.generation))
                .or_default()
                .push(index);
        }
    }
    let mut quotas: Vec<((u8, u64), usize, usize)> = strata
        .iter()
        .map(|(key, members)| {
            (
                *key,
                members.len() * want / answerable,
                members.len() * want % answerable,
            )
        })
        .collect();
    let mut leftover = want - quotas.iter().map(|(_, floor, _)| floor).sum::<usize>();
    let mut by_remainder: Vec<usize> = (0..quotas.len()).collect();
    by_remainder.sort_by(|a, b| quotas[*b].2.cmp(&quotas[*a].2).then(a.cmp(b)));
    for position in by_remainder {
        if leftover == 0 {
            break;
        }
        quotas[position].1 += 1;
        leftover -= 1;
    }
    for (key, quota, _) in quotas {
        for index in strata[&key].iter().take(quota) {
            chosen[*index] = true;
        }
    }
    chosen
}

fn pick_late_writes(
    rng: &mut Rng,
    fates: &[Fate],
    shape: Shape,
    used: &mut HashSet<(usize, u64)>,
) -> Result<Vec<LateWrite>, String> {
    let mut out = Vec::new();
    for (kind, per) in [
        (LateKind::Superseded, params::LATE_SUPERSEDED_PERCENT),
        (LateKind::Revoked, params::LATE_REVOKED_PERCENT),
        (LateKind::Future, params::LATE_FUTURE_PERCENT),
        (LateKind::Ghost, params::LATE_GHOST_PERCENT),
    ] {
        let count = percent(shape.queries, per).max(1);
        let mut candidates: Vec<(usize, u64)> = Vec::new();
        for (cue, fate) in fates.iter().enumerate() {
            match kind {
                LateKind::Superseded => {
                    candidates.extend((1..fate.history()).map(|generation| (cue, generation)));
                }
                LateKind::Revoked if fate.revoked() => candidates.push((cue, fate.history())),
                LateKind::Future if !fate.revoked() && fate.history() < 3 => {
                    candidates.push((cue, fate.history() + 1));
                }
                LateKind::Ghost => candidates.push((cue, 1)),
                _ => {}
            }
        }
        if kind != LateKind::Ghost {
            candidates.retain(|pair| !used.contains(pair));
        }
        shuffle(rng, &mut candidates);
        if candidates.len() < count {
            return Err(format!(
                "only {} candidates for {count} {} late writes",
                candidates.len(),
                kind.name()
            ));
        }
        for (cue, generation) in candidates.into_iter().take(count) {
            if kind != LateKind::Ghost {
                used.insert((cue, generation));
            }
            out.push(LateWrite {
                kind,
                cue,
                generation,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned(seed: u64) -> Workload {
        Workload::generate(seed, Shape::PINNED).expect("the pinned workload generates")
    }

    #[test]
    fn the_pinned_shape_has_the_mix_and_the_floors_the_criteria_need() {
        for seed in params::SEEDS {
            let workload = pinned(seed);
            let count = |wanted: Fate| {
                workload
                    .fates
                    .iter()
                    .filter(|fate| **fate == wanted)
                    .count()
            };
            assert_eq!(count(Fate::Stay), 700);
            assert_eq!(count(Fate::Once), 500);
            assert_eq!(count(Fate::Twice), 400);
            assert_eq!(count(Fate::Revoked(1)), 150);
            assert_eq!(count(Fate::Revoked(2)), 124);
            assert_eq!(count(Fate::Revoked(3)), 126);
            let queries = &workload.queries;
            assert_eq!(queries.len(), params::QUERIES);
            let revoked = queries.iter().filter(|q| q.revoked_target).count() as i64;
            let answerable = queries.len() as i64 - revoked;
            let superseded = queries
                .iter()
                .filter(|q| !q.revoked_target && q.generation >= 2)
                .count() as i64;
            assert!(
                revoked >= params::MIN_REVOKED_TARGETS_QUERIED,
                "seed {seed}: {revoked}"
            );
            assert!(
                answerable >= params::MIN_ANSWERABLE_QUERIES,
                "seed {seed}: {answerable}"
            );
            assert!(
                superseded >= params::MIN_SUPERSEDED_TARGETS_QUERIED,
                "seed {seed}: {superseded}"
            );
            assert!(workload.plants.len() as i64 >= params::MIN_PLANTED_STALE_ROWS);
            assert!(workload.late_writes.len() as i64 >= params::MIN_LATE_WRITE_ATTEMPTS);
            assert!(
                (workload.stale_probes.len() + workload.live_probes.len()) as i64
                    >= params::MIN_PROBE_QUERIES
            );
            assert_eq!(workload.repeats.len(), params::REPEAT_QUERIES);
        }
    }

    #[test]
    fn every_target_is_distinct_and_every_query_is_asked_in_a_state_that_matches_its_label() {
        let workload = pinned(17);
        let mut seen = HashSet::new();
        for query in &workload.queries {
            assert!(seen.insert(query.cue), "cue {} asked twice", query.cue);
            let fate = workload.fates[query.cue];
            assert_eq!(query.generation, fate.history());
            assert_eq!(
                query.revoked_target,
                fate.revoked() && query.checkpoint == 2
            );
            if fate.revoked() {
                assert_eq!(
                    query.checkpoint, 2,
                    "a revoked capsule is asked only after revocation"
                );
                assert!(query.revoked_target);
            }
            assert!(query.paraphrase < 3);
        }
        // Checkpoint 1 comes first.
        let order: Vec<u8> = workload.queries.iter().map(|q| q.checkpoint).collect();
        assert!(order.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn plants_and_late_writes_are_never_live_and_never_the_same_pair() {
        let workload = pinned(29);
        let mut pairs = HashSet::new();
        for plant in &workload.plants {
            let fate = workload.fates[plant.cue];
            assert!(plant.generation >= 1 && plant.generation <= fate.history());
            let is_live_state = plant.generation == fate.history() && !fate.revoked();
            assert!(
                !is_live_state,
                "a plant must not be a live generation: {plant:?}"
            );
            if plant.revoked {
                assert!(fate.revoked() && plant.generation == fate.history());
                assert_eq!(plant.checkpoint, 2);
            } else {
                assert!(plant.generation < fate.history());
            }
            assert!(
                pairs.insert((plant.cue, plant.generation)),
                "{plant:?} planted twice"
            );
        }
        for late in &workload.late_writes {
            let fate = workload.fates[late.cue];
            match late.kind {
                LateKind::Superseded => assert!(late.generation < fate.history()),
                LateKind::Revoked => {
                    assert!(fate.revoked() && late.generation == fate.history());
                }
                LateKind::Future => {
                    assert!(late.generation > fate.history() && late.generation <= 3)
                }
                LateKind::Ghost => assert_eq!(late.generation, 1),
            }
            if late.kind != LateKind::Ghost {
                assert!(
                    pairs.insert((late.cue, late.generation)),
                    "{late:?} is also planted or written twice"
                );
            }
        }
    }

    #[test]
    fn events_replay_to_the_fates_and_a_seed_reproduces_its_workload() {
        let workload = pinned(43);
        let mut live: Vec<(u64, bool)> = vec![(0, false); workload.fates.len()];
        for phase in PHASES {
            for event in workload.events(phase) {
                match event.kind {
                    EventKind::Commit => live[event.cue] = (1, false),
                    EventKind::Supersede { old, new } => {
                        assert_eq!(live[event.cue].0, old);
                        live[event.cue].0 = new;
                    }
                    EventKind::Revoke { generation } => {
                        assert_eq!(live[event.cue], (generation, false));
                        live[event.cue].1 = true;
                    }
                }
            }
        }
        for (cue, fate) in workload.fates.iter().enumerate() {
            assert_eq!(live[cue], (fate.history(), fate.revoked()));
        }
        assert_eq!(pinned(43).digest(43), workload.digest(43));
        assert_ne!(pinned(43).digest(43), pinned(71).digest(71));
        assert_ne!(
            workload.digest(43),
            workload.digest(44),
            "the seed is part of the digest"
        );
    }

    #[test]
    fn the_strong_subset_is_answerable_only_the_size_asked_for_and_stratified_like_the_whole() {
        for seed in params::SEEDS {
            let workload = pinned(seed);
            let chosen: Vec<&Query> = workload
                .queries
                .iter()
                .zip(&workload.strong)
                .filter(|(_, picked)| **picked)
                .map(|(query, _)| query)
                .collect();
            assert_eq!(chosen.len(), params::STRONG_QUERIES, "seed {seed}");
            assert!(
                chosen.iter().all(|query| !query.revoked_target),
                "seed {seed}"
            );
            let answerable: Vec<&Query> = workload
                .queries
                .iter()
                .filter(|query| !query.revoked_target)
                .collect();
            let share = |queries: &[&Query], wanted: fn(&Query) -> bool| {
                queries.iter().filter(|query| wanted(query)).count()
            };
            for (name, wanted) in [
                (
                    "first checkpoint",
                    (|q: &Query| q.checkpoint == 1) as fn(&Query) -> bool,
                ),
                ("generation 2", |q: &Query| q.generation == 2),
                ("generation 3", |q: &Query| q.generation == 3),
            ] {
                let whole = share(&answerable, wanted) * params::STRONG_QUERIES;
                let part = share(&chosen, wanted) * answerable.len();
                assert!(
                    part.abs_diff(whole) <= answerable.len() * 2,
                    "seed {seed}: {name} is {part} against {whole} (scaled)"
                );
            }
        }
        // A shape with at most the strong count of answerable queries asks the strong arm all of them.
        let small = Workload::generate(
            3,
            Shape {
                capsules: 2000,
                queries: 150,
            },
        )
        .unwrap();
        for (query, picked) in small.queries.iter().zip(&small.strong) {
            assert_eq!(*picked, !query.revoked_target);
        }
    }

    #[test]
    fn every_pinned_seed_generates_and_keeps_its_plants_and_late_writes_disjoint() {
        for seed in params::SEEDS {
            let workload = pinned(seed);
            let mut pairs = HashSet::new();
            for plant in &workload.plants {
                assert!(
                    pairs.insert((plant.cue, plant.generation)),
                    "seed {seed}: {plant:?}"
                );
            }
            for late in workload
                .late_writes
                .iter()
                .filter(|l| l.kind != LateKind::Ghost)
            {
                assert!(
                    pairs.insert((late.cue, late.generation)),
                    "seed {seed}: {late:?}"
                );
            }
            assert_eq!(workload.digest(seed).len(), 64);
        }
    }

    #[test]
    fn doc_ids_name_the_capsule_and_the_generation() {
        assert_eq!(doc_id("cap-0007", 3), "cap-0007@3");
    }

    #[test]
    fn shapes_the_generator_cannot_serve_are_refused() {
        assert!(Workload::generate(
            1,
            Shape {
                capsules: 10,
                queries: 12
            }
        )
        .is_err());
        assert!(Workload::generate(
            1,
            Shape {
                capsules: 100,
                queries: 101
            }
        )
        .is_err());
        assert!(Workload::generate(1, Shape::REDUCED).is_ok());
        for seed in params::SEEDS {
            assert!(
                Workload::generate(seed, Shape::REDUCED).is_ok(),
                "seed {seed}"
            );
        }
    }
}
