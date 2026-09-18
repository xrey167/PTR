use ptr_types::Probability;

#[derive(Clone, Debug, PartialEq)]
pub struct RouteScore { pub target: String, pub score: Probability }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteDecision { pub candidates: Vec<RouteScore> }
impl RouteDecision {
    pub fn best(&self) -> Option<&RouteScore> { self.candidates.iter().max_by(|a,b| a.score.get().total_cmp(&b.score.get())) }
}

#[derive(Clone, Debug)]
pub struct RoutingPolicy { pub max_parallel: usize, pub uncertainty_trigger: f32 }
impl Default for RoutingPolicy { fn default() -> Self { Self { max_parallel: 4, uncertainty_trigger: 0.35 } } }
