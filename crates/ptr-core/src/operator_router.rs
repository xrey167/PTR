use ptr_types::Probability;

pub use ptr_types::ReasoningOperator;

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedOperator {
    pub operator: ReasoningOperator,
    pub weight: Probability,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterDecision {
    pub operators: Vec<WeightedOperator>,
}

impl RouterDecision {
    pub fn top(&self) -> Option<ReasoningOperator> {
        self.operators
            .iter()
            .max_by(|a, b| a.weight.get().total_cmp(&b.weight.get()))
            .map(|x| x.operator)
    }
}
