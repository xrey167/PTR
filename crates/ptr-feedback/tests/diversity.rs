use ptr_feedback::{
    cluster_candidates, dedup_population, population_diversity, Candidate, ClusterOptions,
    Population, SimilarityThreshold,
};
use ptr_types::{CandidateId, Probability, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

fn options(threshold: f64) -> ClusterOptions {
    ClusterOptions::new(
        SimilarityThreshold::new(threshold).unwrap(),
        VerificationLevel::Deterministic,
    )
}

fn candidate(id: &str, text: &str, status: Option<(VerificationStatus, f32)>) -> Candidate<String> {
    Candidate {
        id: CandidateId::from(id),
        value: text.to_owned(),
        verification: status.map(|(status, score)| VerificationReport {
            status,
            level: VerificationLevel::Deterministic,
            score: Probability::new(score).unwrap(),
            findings: vec![],
        }),
    }
}

fn population() -> Population<String> {
    let mut population = Population::default();
    population.push(candidate(
        "c0",
        "Rename the file to report.txt and then upload it to the shared folder.",
        Some((VerificationStatus::Fail, 0.9)),
    ));
    population.push(candidate(
        "c1",
        "rename the FILE to report.txt, and then upload it to the shared folder",
        Some((VerificationStatus::Pass, 0.7)),
    ));
    population.push(candidate(
        "c2",
        "Rename the file to report.txt and then upload it to the shared folder right away.",
        Some((VerificationStatus::Pass, 0.8)),
    ));
    population.push(candidate(
        "c3",
        "Delete every cached thumbnail older than a week from the image directory.",
        None,
    ));
    population
}

#[test]
fn near_duplicates_form_one_cluster_with_the_best_verified_representative() {
    let clusters = cluster_candidates(&population(), &options(0.6)).unwrap();
    assert_eq!(clusters.len(), 2);
    assert_eq!(clusters[0].members, [0, 1, 2]);
    // The failed 0.9 candidate does not represent the cluster; the higher passing score does.
    assert_eq!(clusters[0].representative_id.to_string(), "c2");
    assert_eq!(clusters[1].members, [3]);
}

#[test]
fn dedup_keeps_one_candidate_per_cluster() {
    let kept = dedup_population(&population(), &options(0.6)).unwrap();
    let ids: Vec<String> = kept.candidates.iter().map(|c| c.id.to_string()).collect();
    assert_eq!(ids, ["c2", "c3"]);
}

#[test]
fn a_strict_threshold_keeps_variants_apart_and_dedup_raises_diversity() {
    let loose = options(0.6);
    let strict = options(1.0);
    assert_eq!(cluster_candidates(&population(), &strict).unwrap().len(), 3);
    let diversity = population_diversity(&population(), &loose).unwrap();
    assert!(diversity > 0.0 && diversity < 1.0, "{diversity}");
    let deduped = dedup_population(&population(), &loose).unwrap();
    assert!(population_diversity(&deduped, &loose).unwrap() > diversity);
}
