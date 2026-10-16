//! Multi-objective selection: Pareto fronts, crowding distance and seeded tournaments.
//!
//! Every point carries a verification [`Gate`] that is compared lexicographically before any
//! other objective. A point in a better gate dominates every point in a worse gate, so a
//! candidate whose verification failed or carries a hard finding is never ranked above one that
//! passed, whatever its learned score, cost or length (Invariant 11). Within one gate the usual
//! Pareto dominance over the declared objectives applies. Ranking is the non-dominated sorting
//! of NSGA-II (O(n^2 m)); ties are broken by index so the result is deterministic.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

use ptr_types::VerificationLevel;
use ptr_verifier::VerificationStatus;

use crate::candidate::{Candidate, Population};
use crate::cluster::mix;
use crate::text::CandidateText;

/// Most points one selection call accepts.
pub const MAX_SELECTION_POINTS: usize = 4096;
/// Most objectives a space declares.
pub const MAX_OBJECTIVES: usize = 16;
/// Most winners one [`Tournament::select`] call draws (one per point of a maximal population).
pub const MAX_TOURNAMENT_WINNERS: usize = MAX_SELECTION_POINTS;

/// Why selection could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionError {
    /// The space declares no objective.
    NoObjectives,
    /// The space declares more than [`MAX_OBJECTIVES`].
    TooManyObjectives,
    /// More than [`MAX_SELECTION_POINTS`] points.
    TooManyPoints,
    /// The point at this index has a different number of values than the space has objectives.
    DimensionMismatch { point: usize },
    /// The point at this index has a NaN or infinite value.
    NonFinite { point: usize },
    /// A front index is outside the points.
    FrontOutOfRange,
    /// The tournament size is zero or larger than the number of ranked points.
    InvalidTournamentSize,
    /// There are no ranked points to select from.
    NothingToSelect,
    /// More than [`MAX_TOURNAMENT_WINNERS`] winners were requested.
    TooManyWinners,
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObjectives => f.write_str("the objective space is empty"),
            Self::TooManyObjectives => f.write_str("too many objectives"),
            Self::TooManyPoints => f.write_str("too many points to rank"),
            Self::DimensionMismatch { point } => {
                write!(f, "point {point} does not match the objective count")
            }
            Self::NonFinite { point } => write!(f, "point {point} has a non-finite value"),
            Self::FrontOutOfRange => f.write_str("a front index is out of range"),
            Self::InvalidTournamentSize => f.write_str("invalid tournament size"),
            Self::NothingToSelect => f.write_str("nothing to select from"),
            Self::TooManyWinners => f.write_str("too many tournament winners requested"),
        }
    }
}

impl Error for SelectionError {}

/// Which way an objective improves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    Maximize,
    Minimize,
}

/// The verification gate of a candidate, ordered worst to best.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Gate {
    /// The verifier reported `Fail` or a hard finding.
    Rejected,
    /// No report, `Unknown`, `Disputed`, or a clean `Pass` below the required level.
    Undecided,
    /// [`Candidate::passes`] holds at the required level.
    Passed,
}

/// The gate of `candidate` at `required` depth. Only [`Candidate::passes`] reaches
/// [`Gate::Passed`]; a score is never consulted.
pub fn gate_of<T>(candidate: &Candidate<T>, required: VerificationLevel) -> Gate {
    if candidate.passes(required) {
        return Gate::Passed;
    }
    match &candidate.verification {
        Some(report)
            if report.status == VerificationStatus::Fail
                || report.findings.iter().any(|finding| finding.hard) =>
        {
            Gate::Rejected
        }
        _ => Gate::Undecided,
    }
}

/// The declared objectives (after the gate) and their directions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectiveSpace {
    directions: Vec<Direction>,
}

impl ObjectiveSpace {
    /// # Errors
    /// [`SelectionError::NoObjectives`] or [`SelectionError::TooManyObjectives`].
    pub fn new(directions: Vec<Direction>) -> Result<Self, SelectionError> {
        if directions.is_empty() {
            return Err(SelectionError::NoObjectives);
        }
        if directions.len() > MAX_OBJECTIVES {
            return Err(SelectionError::TooManyObjectives);
        }
        Ok(Self { directions })
    }

    pub fn directions(&self) -> &[Direction] {
        &self.directions
    }
}

/// A candidate's place in the objective space.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub gate: Gate,
    pub values: Vec<f64>,
}

/// The rank (0 is the Pareto front) and crowding distance of one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranked {
    pub index: usize,
    pub rank: usize,
    /// Infinite on a front boundary.
    pub crowding: f64,
}

fn check_points(space: &ObjectiveSpace, points: &[Point]) -> Result<(), SelectionError> {
    if points.len() > MAX_SELECTION_POINTS {
        return Err(SelectionError::TooManyPoints);
    }
    for (index, point) in points.iter().enumerate() {
        if point.values.len() != space.directions.len() {
            return Err(SelectionError::DimensionMismatch { point: index });
        }
        if point.values.iter().any(|value| !value.is_finite()) {
            return Err(SelectionError::NonFinite { point: index });
        }
    }
    Ok(())
}

/// Whether `a` dominates `b`: a better gate always does; within a gate, `a` is no worse on every
/// objective and strictly better on one.
pub fn dominates(space: &ObjectiveSpace, a: &Point, b: &Point) -> bool {
    if a.gate != b.gate {
        return a.gate > b.gate;
    }
    let mut strictly_better = false;
    for ((x, y), direction) in a.values.iter().zip(&b.values).zip(&space.directions) {
        let order = match direction {
            Direction::Maximize => x.total_cmp(y),
            Direction::Minimize => y.total_cmp(x),
        };
        match order {
            Ordering::Less => return false,
            Ordering::Greater => strictly_better = true,
            Ordering::Equal => {}
        }
    }
    strictly_better
}

/// Non-dominated sorting: the fronts in rank order, each front's indices ascending.
///
/// # Errors
/// Returns [`SelectionError`] for too many, mis-sized or non-finite points.
pub fn pareto_fronts(
    space: &ObjectiveSpace,
    points: &[Point],
) -> Result<Vec<Vec<usize>>, SelectionError> {
    check_points(space, points)?;
    let n = points.len();
    let mut dominated_by_count = vec![0usize; n];
    let mut dominating: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(space, &points[i], &points[j]) {
                dominating[i].push(j);
                dominated_by_count[j] += 1;
            } else if dominates(space, &points[j], &points[i]) {
                dominating[j].push(i);
                dominated_by_count[i] += 1;
            }
        }
    }
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|i| dominated_by_count[*i] == 0).collect();
    while !current.is_empty() {
        let mut next: Vec<usize> = Vec::new();
        for &i in &current {
            for &j in &dominating[i] {
                dominated_by_count[j] -= 1;
                if dominated_by_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        next.sort_unstable();
        fronts.push(std::mem::replace(&mut current, next));
    }
    Ok(fronts)
}

/// The first (non-dominated) front, indices ascending. Empty for no points.
///
/// # Errors
/// As [`pareto_fronts`].
pub fn pareto_front(
    space: &ObjectiveSpace,
    points: &[Point],
) -> Result<Vec<usize>, SelectionError> {
    Ok(pareto_fronts(space, points)?
        .into_iter()
        .next()
        .unwrap_or_default())
}

/// Crowding distance of each member of `front`, aligned with `front`. A member that is a
/// boundary on some objective is infinite; an interior member sums, over objectives, the gap
/// between its neighbours divided by the front's range on that objective (objectives with no
/// range add nothing). A front of at most two members is all infinite.
///
/// # Errors
/// Returns [`SelectionError`] for bad points or a front index outside `points`.
pub fn crowding_distance(
    space: &ObjectiveSpace,
    points: &[Point],
    front: &[usize],
) -> Result<Vec<f64>, SelectionError> {
    check_points(space, points)?;
    if front.iter().any(|index| *index >= points.len()) {
        return Err(SelectionError::FrontOutOfRange);
    }
    let size = front.len();
    let mut distance = vec![0.0f64; size];
    if size <= 2 {
        distance.iter_mut().for_each(|d| *d = f64::INFINITY);
        return Ok(distance);
    }
    for objective in 0..space.directions.len() {
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_by(|a, b| {
            points[front[*a]].values[objective]
                .total_cmp(&points[front[*b]].values[objective])
                .then_with(|| front[*a].cmp(&front[*b]))
        });
        let value = |slot: usize| points[front[order[slot]]].values[objective];
        let range = value(size - 1) - value(0);
        distance[order[0]] = f64::INFINITY;
        distance[order[size - 1]] = f64::INFINITY;
        if range > 0.0 {
            for slot in 1..size - 1 {
                distance[order[slot]] += (value(slot + 1) - value(slot - 1)) / range;
            }
        }
    }
    Ok(distance)
}

/// Rank and crowding distance of every point, in point order.
///
/// # Errors
/// As [`pareto_fronts`].
pub fn rank_points(
    space: &ObjectiveSpace,
    points: &[Point],
) -> Result<Vec<Ranked>, SelectionError> {
    let fronts = pareto_fronts(space, points)?;
    let mut ranked = vec![
        Ranked {
            index: 0,
            rank: 0,
            crowding: 0.0
        };
        points.len()
    ];
    for (rank, front) in fronts.iter().enumerate() {
        let crowding = crowding_distance(space, points, front)?;
        for (member, distance) in front.iter().zip(crowding) {
            ranked[*member] = Ranked {
                index: *member,
                rank,
                crowding: distance,
            };
        }
    }
    Ok(ranked)
}

/// Seeded k-way tournament selection. The same seed over the same ranking always yields the
/// same winners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tournament {
    pub size: usize,
    pub seed: u64,
}

impl Tournament {
    /// `winners` point indices (a point may win more than once). Each round draws `size`
    /// distinct contestants; the lowest rank wins, then the larger crowding distance, then the
    /// lower index. A contestant in a worse Pareto rank never beats a better one, so a rejected
    /// candidate loses to any passing candidate it meets.
    ///
    /// # Errors
    /// [`SelectionError::NothingToSelect`] without ranked points and
    /// [`SelectionError::InvalidTournamentSize`] when `size` is zero or above their number,
    /// [`SelectionError::TooManyWinners`] when `winners` exceeds [`MAX_TOURNAMENT_WINNERS`]
    /// (winners are drawn with replacement, so the input size does not bound them).
    pub fn select(&self, ranked: &[Ranked], winners: usize) -> Result<Vec<usize>, SelectionError> {
        if ranked.is_empty() {
            return Err(SelectionError::NothingToSelect);
        }
        if self.size == 0 || self.size > ranked.len() {
            return Err(SelectionError::InvalidTournamentSize);
        }
        if winners > MAX_TOURNAMENT_WINNERS {
            return Err(SelectionError::TooManyWinners);
        }
        let mut state = self.seed;
        let mut next_u64 = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            mix(state)
        };
        let mut chosen = Vec::with_capacity(winners);
        for _ in 0..winners {
            let mut pool: Vec<usize> = (0..ranked.len()).collect();
            let mut best: Option<usize> = None;
            for draw in 0..self.size {
                let remaining = (pool.len() - draw) as u64;
                let pick = draw + (next_u64() % remaining) as usize;
                pool.swap(draw, pick);
                let contender = pool[draw];
                best = Some(match best {
                    Some(held) if !beats(&ranked[contender], &ranked[held]) => held,
                    _ => contender,
                });
            }
            if let Some(winner) = best {
                chosen.push(ranked[winner].index);
            }
        }
        Ok(chosen)
    }
}

fn beats(a: &Ranked, b: &Ranked) -> bool {
    a.rank
        .cmp(&b.rank)
        .then_with(|| b.crowding.total_cmp(&a.crowding))
        .then_with(|| a.index.cmp(&b.index))
        == Ordering::Less
}

/// The standard objective space of [`standard_points`]: verifier score (maximize), cost
/// (minimize), text length in characters (minimize).
pub fn standard_space() -> ObjectiveSpace {
    ObjectiveSpace {
        directions: vec![
            Direction::Maximize,
            Direction::Minimize,
            Direction::Minimize,
        ],
    }
}

/// Points for every candidate of `population` in [`standard_space`]. The gate comes from
/// [`gate_of`]; the verifier score is 0 for an unverified candidate; `cost` is supplied by the
/// caller (for example tokens or latency) and must be finite.
pub fn standard_points<T: CandidateText>(
    population: &Population<T>,
    required: VerificationLevel,
    cost: impl Fn(&Candidate<T>) -> f64,
) -> Vec<Point> {
    population
        .candidates
        .iter()
        .map(|candidate| Point {
            gate: gate_of(candidate, required),
            values: vec![
                candidate
                    .verification
                    .as_ref()
                    .map_or(0.0, |report| f64::from(report.score.get())),
                cost(candidate),
                candidate.value.candidate_text().chars().count() as f64,
            ],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{CandidateId, Probability};
    use ptr_verifier::{Finding, VerificationReport};

    fn min2() -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Direction::Minimize, Direction::Minimize]).unwrap()
    }

    fn pt(a: f64, b: f64) -> Point {
        Point {
            gate: Gate::Passed,
            values: vec![a, b],
        }
    }

    /// A(1,5) B(2,3) C(3,1) are mutually non-dominated; D(2,4) is dominated by B; E(4,4) by
    /// B, C and D.
    fn example() -> Vec<Point> {
        vec![
            pt(1.0, 5.0),
            pt(2.0, 3.0),
            pt(3.0, 1.0),
            pt(2.0, 4.0),
            pt(4.0, 4.0),
        ]
    }

    #[test]
    fn fronts_match_the_hand_checked_example() {
        let fronts = pareto_fronts(&min2(), &example()).unwrap();
        assert_eq!(fronts, [vec![0, 1, 2], vec![3], vec![4]]);
        assert_eq!(pareto_front(&min2(), &example()).unwrap(), [0, 1, 2]);
        let ranks: Vec<usize> = rank_points(&min2(), &example())
            .unwrap()
            .iter()
            .map(|r| r.rank)
            .collect();
        assert_eq!(ranks, [0, 0, 0, 1, 2]);
    }

    #[test]
    fn crowding_boundaries_are_infinite_and_the_interior_is_hand_checked() {
        let distance = crowding_distance(&min2(), &example(), &[0, 1, 2]).unwrap();
        assert!(distance[0].is_infinite() && distance[2].is_infinite());
        // objective 0: (3 - 1) / (3 - 1) = 1; objective 1: (5 - 1) / (5 - 1) = 1.
        assert_eq!(distance[1], 2.0);
        let pair = crowding_distance(&min2(), &example(), &[3, 4]).unwrap();
        assert!(pair.iter().all(|d| d.is_infinite()));
        assert_eq!(crowding_distance(&min2(), &example(), &[]).unwrap(), []);
    }

    #[test]
    fn equal_points_do_not_dominate_each_other() {
        let points = vec![pt(1.0, 1.0), pt(1.0, 1.0)];
        assert_eq!(pareto_fronts(&min2(), &points).unwrap(), [vec![0, 1]]);
        let distance = crowding_distance(&min2(), &points, &[0, 1]).unwrap();
        assert!(distance.iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn maximize_direction_flips_dominance() {
        let space = ObjectiveSpace::new(vec![Direction::Maximize]).unwrap();
        let points = vec![
            Point {
                gate: Gate::Passed,
                values: vec![1.0],
            },
            Point {
                gate: Gate::Passed,
                values: vec![2.0],
            },
        ];
        assert_eq!(pareto_fronts(&space, &points).unwrap(), [vec![1], vec![0]]);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        assert_eq!(
            ObjectiveSpace::new(vec![]),
            Err(SelectionError::NoObjectives)
        );
        assert_eq!(
            ObjectiveSpace::new(vec![Direction::Minimize; MAX_OBJECTIVES + 1]),
            Err(SelectionError::TooManyObjectives)
        );
        let bad_dim = vec![Point {
            gate: Gate::Passed,
            values: vec![1.0],
        }];
        assert_eq!(
            pareto_fronts(&min2(), &bad_dim),
            Err(SelectionError::DimensionMismatch { point: 0 })
        );
        let nan = vec![pt(1.0, f64::NAN)];
        assert_eq!(
            pareto_fronts(&min2(), &nan),
            Err(SelectionError::NonFinite { point: 0 })
        );
        let many = vec![pt(0.0, 0.0); MAX_SELECTION_POINTS + 1];
        assert_eq!(
            pareto_fronts(&min2(), &many),
            Err(SelectionError::TooManyPoints)
        );
        assert_eq!(
            crowding_distance(&min2(), &example(), &[9]),
            Err(SelectionError::FrontOutOfRange)
        );
        let tournament = Tournament { size: 1, seed: 0 };
        assert_eq!(
            tournament.select(&[], 1),
            Err(SelectionError::NothingToSelect)
        );
        let ranked = rank_points(&min2(), &example()).unwrap();
        for size in [0, 6] {
            assert_eq!(
                Tournament { size, seed: 0 }.select(&ranked, 1),
                Err(SelectionError::InvalidTournamentSize)
            );
        }
    }

    #[test]
    fn a_better_gate_dominates_whatever_the_objectives_say() {
        let space = min2();
        let rejected = Point {
            gate: Gate::Rejected,
            values: vec![0.0, 0.0],
        };
        let passed = Point {
            gate: Gate::Passed,
            values: vec![9.0, 9.0],
        };
        assert!(dominates(&space, &passed, &rejected));
        assert!(!dominates(&space, &rejected, &passed));
    }

    #[test]
    fn tournament_is_deterministic_by_seed_and_varies_across_seeds() {
        let ranked = rank_points(&min2(), &example()).unwrap();
        let a = Tournament { size: 2, seed: 42 }
            .select(&ranked, 12)
            .unwrap();
        let b = Tournament { size: 2, seed: 42 }
            .select(&ranked, 12)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        let others: Vec<Vec<usize>> = (0..8)
            .map(|seed| Tournament { size: 2, seed }.select(&ranked, 12).unwrap())
            .collect();
        assert!(others.iter().any(|o| *o != a));
        // The whole-population tournament always returns the best-ranked, best-crowded point.
        let all = Tournament { size: 5, seed: 3 }.select(&ranked, 4).unwrap();
        assert!(all.iter().all(|i| ranked[*i].rank == 0));
    }

    fn report(
        status: VerificationStatus,
        level: VerificationLevel,
        score: f32,
    ) -> VerificationReport {
        VerificationReport {
            status,
            level,
            score: Probability::new(score).unwrap(),
            findings: vec![],
        }
    }

    fn cand(id: &str, text: &str, report: Option<VerificationReport>) -> Candidate<String> {
        Candidate {
            id: CandidateId::from(id),
            value: text.to_owned(),
            verification: report,
        }
    }

    #[test]
    fn gates_follow_the_acceptance_rule_and_never_the_score() {
        use VerificationLevel::Deterministic as D;
        let pass = cand("p", "t", Some(report(VerificationStatus::Pass, D, 0.1)));
        let fail = cand("f", "t", Some(report(VerificationStatus::Fail, D, 1.0)));
        let unknown = cand("u", "t", Some(report(VerificationStatus::Unknown, D, 1.0)));
        let disputed = cand("d", "t", Some(report(VerificationStatus::Disputed, D, 1.0)));
        let shallow = cand(
            "s",
            "t",
            Some(report(
                VerificationStatus::Pass,
                VerificationLevel::LatentAgreement,
                1.0,
            )),
        );
        let mut forged_report = report(VerificationStatus::Pass, D, 1.0);
        forged_report.findings.push(Finding {
            code: "target".into(),
            message: String::new(),
            hard: true,
        });
        let forged = cand("x", "t", Some(forged_report));
        assert_eq!(gate_of(&pass, D), Gate::Passed);
        assert_eq!(gate_of(&fail, D), Gate::Rejected);
        assert_eq!(gate_of(&forged, D), Gate::Rejected);
        for undecided in [&unknown, &disputed, &shallow, &cand("n", "t", None)] {
            assert_eq!(gate_of(undecided, D), Gate::Undecided);
        }
    }

    /// Invariant 11: a candidate with a hard finding or a Fail is never ranked above a passing
    /// one, however high its learned score and however cheap and short it is.
    #[test]
    fn invariant_11_a_failed_high_score_never_outranks_a_passing_candidate() {
        use VerificationLevel::Deterministic as D;
        let mut forged_report = report(VerificationStatus::Pass, D, 1.0);
        forged_report.findings.push(Finding {
            code: "effect".into(),
            message: String::new(),
            hard: true,
        });
        let population = Population {
            candidates: vec![
                cand("fail", "x", Some(report(VerificationStatus::Fail, D, 1.0))),
                cand("forged", "x", Some(forged_report)),
                cand(
                    "pass",
                    "a much longer and costlier passing candidate text",
                    Some(report(VerificationStatus::Pass, D, 0.01)),
                ),
            ],
        };
        let cost = |c: &Candidate<String>| {
            if c.id.to_string() == "pass" {
                100.0
            } else {
                0.0
            }
        };
        let points = standard_points(&population, D, cost);
        let ranked = rank_points(&standard_space(), &points).unwrap();
        assert_eq!(ranked[2].rank, 0);
        assert!(ranked[0].rank > ranked[2].rank);
        assert!(ranked[1].rank > ranked[2].rank);
        // Every tournament, of any seed and size, that includes the passing candidate picks it
        // over the rejected ones.
        for seed in 0..32 {
            let winners = Tournament { size: 3, seed }.select(&ranked, 3).unwrap();
            assert_eq!(winners, [2, 2, 2]);
            let pair = Tournament { size: 2, seed }.select(&ranked, 5).unwrap();
            assert!(pair.iter().all(|w| ranked[*w].rank <= 1));
        }
    }

    #[test]
    fn standard_points_use_score_cost_and_character_length() {
        use VerificationLevel::Deterministic as D;
        let population = Population {
            candidates: vec![cand(
                "a",
                "héllo",
                Some(report(VerificationStatus::Pass, D, 0.5)),
            )],
        };
        let points = standard_points(&population, D, |_| 7.0);
        assert_eq!(points[0].values, [0.5, 7.0, 5.0]);
        assert_eq!(points[0].gate, Gate::Passed);
    }

    #[test]
    fn an_unbounded_winner_count_is_rejected_before_allocating() {
        let ranked = [Ranked {
            index: 0,
            rank: 0,
            crowding: f64::INFINITY,
        }];
        let tournament = Tournament { size: 1, seed: 7 };
        assert_eq!(
            tournament.select(&ranked, usize::MAX),
            Err(SelectionError::TooManyWinners)
        );
        assert_eq!(
            tournament.select(&ranked, MAX_TOURNAMENT_WINNERS + 1),
            Err(SelectionError::TooManyWinners)
        );
        assert_eq!(
            tournament
                .select(&ranked, MAX_TOURNAMENT_WINNERS)
                .map(|w| w.len()),
            Ok(MAX_TOURNAMENT_WINNERS)
        );
        assert!(!SelectionError::TooManyWinners.to_string().is_empty());
    }
}
