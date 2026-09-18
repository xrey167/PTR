use ptr_types::Probability;

#[derive(Clone, Debug, PartialEq)]
pub struct BranchState {
    pub id: u64,
    pub parent: Option<u64>,
    pub probability: Probability,
    pub summary: String,
}

#[derive(Clone, Debug, Default)]
pub struct BranchFrontier {
    pub branches: Vec<BranchState>,
}

impl BranchFrontier {
    pub fn prune_below(&mut self, threshold: f32) {
        self.branches.retain(|b| b.probability.get() >= threshold);
    }
}
