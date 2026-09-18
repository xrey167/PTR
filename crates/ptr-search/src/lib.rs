use ptr_types::{CapsuleId, Generation};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceStage {
    SearchCandidate,
    PossibleEvidence,
    Observed,
    VerifiedKnown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub capsule: CapsuleId,
    pub generation: Generation,
    pub score: f32,
    pub backend: String,
    pub stage: EvidenceStage,
}

pub trait SearchIndex {
    fn search(&self, query: &str, limit: usize) -> Vec<SearchHit>;
}

pub fn reciprocal_rank_fusion(lists: &[Vec<SearchHit>], k: f32) -> Vec<(CapsuleId, f32)> {
    let mut scores: BTreeMap<CapsuleId, f32> = BTreeMap::new();
    for list in lists {
        for (rank, hit) in list.iter().enumerate() {
            *scores.entry(hit.capsule.clone()).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<_> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}
