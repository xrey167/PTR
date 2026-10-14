use ptr_feedback::{
    crowding_distance, pareto_fronts, rank_points, standard_points, standard_space, Candidate,
    Direction, Gate, ObjectiveSpace, Point, Population, Tournament,
};
use ptr_types::{CandidateId, Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

fn point(gate: Gate, values: [f64; 2]) -> Point {
    Point {
        gate,
        values: values.to_vec(),
    }
}

#[test]
fn fronts_and_crowding_on_a_hand_checked_example() {
    let space = ObjectiveSpace::new(vec![Direction::Minimize, Direction::Minimize]).unwrap();
    let points: Vec<Point> = [[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [2.0, 4.0], [4.0, 4.0]]
        .into_iter()
        .map(|v| point(Gate::Passed, v))
        .collect();
    assert_eq!(
        pareto_fronts(&space, &points).unwrap(),
        [vec![0, 1, 2], vec![3], vec![4]]
    );
    let crowding = crowding_distance(&space, &points, &[0, 1, 2]).unwrap();
    assert!(crowding[0].is_infinite() && crowding[2].is_infinite());
    assert_eq!(crowding[1], 2.0);
}

#[test]
fn gates_come_before_every_other_objective() {
    let space = ObjectiveSpace::new(vec![Direction::Minimize, Direction::Minimize]).unwrap();
    let points = vec![
        point(Gate::Rejected, [0.0, 0.0]),
        point(Gate::Undecided, [5.0, 5.0]),
        point(Gate::Passed, [9.0, 9.0]),
    ];
    let ranks: Vec<usize> = rank_points(&space, &points)
        .unwrap()
        .iter()
        .map(|r| r.rank)
        .collect();
    assert_eq!(ranks, [2, 1, 0]);
}

fn report(status: VerificationStatus, score: f32, hard: bool) -> VerificationReport {
    VerificationReport {
        status,
        level: VerificationLevel::Deterministic,
        score: Probability::new(score).unwrap(),
        findings: if hard {
            vec![Finding {
                code: "effect".into(),
                message: String::new(),
                hard: true,
            }]
        } else {
            vec![]
        },
    }
}

/// Invariant 11 through the public API: whatever the learned score and the seed, a tournament
/// that meets a passing candidate never prefers a failed one.
#[test]
fn invariant_11_failed_or_hard_finding_candidates_never_win_against_a_pass() {
    let make = |id: &str, text: &str, r: VerificationReport| Candidate {
        id: CandidateId::from(id),
        value: text.to_owned(),
        verification: Some(r),
    };
    let population = Population {
        candidates: vec![
            make(
                "fail-high",
                "a",
                report(VerificationStatus::Fail, 0.999, true),
            ),
            make(
                "forged-pass",
                "b",
                report(VerificationStatus::Pass, 0.999, true),
            ),
            make(
                "pass-low",
                "long long text",
                report(VerificationStatus::Pass, 0.01, false),
            ),
            make(
                "fail-high-2",
                "c",
                report(VerificationStatus::Fail, 0.99, false),
            ),
        ],
    };
    let points = standard_points(&population, VerificationLevel::Deterministic, |_| 1.0);
    let ranked = rank_points(&standard_space(), &points).unwrap();
    assert_eq!(ranked[2].rank, 0);
    assert!(ranked.iter().filter(|r| r.index != 2).all(|r| r.rank > 0));
    for seed in 0..64 {
        let winners = Tournament { size: 4, seed }.select(&ranked, 4).unwrap();
        assert!(winners.iter().all(|w| *w == 2), "seed {seed}: {winners:?}");
    }
}

#[test]
fn tournament_selection_is_reproducible_by_seed() {
    let space = ObjectiveSpace::new(vec![Direction::Maximize, Direction::Minimize]).unwrap();
    let points: Vec<Point> = (0..20)
        .map(|i| point(Gate::Passed, [f64::from(i % 7), f64::from(i % 5)]))
        .collect();
    let ranked = rank_points(&space, &points).unwrap();
    let run = |seed| Tournament { size: 3, seed }.select(&ranked, 30).unwrap();
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}
