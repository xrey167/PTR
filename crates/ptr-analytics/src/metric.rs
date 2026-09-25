/// A platform metric, defined in PTR terms rather than in any storage dialect.
///
/// Each metric is a proportion `numerator / denominator` over non-authoritative
/// operational records (triage logs, branch outcomes). A backend compiles a
/// [`MetricSpec`] to its own query language — the Postgres substrate compiles
/// it to SQL — and returns [`MetricRow`]s; this crate only defines what the
/// numbers mean and how their intervals are computed. None of these numbers is
/// evidence about semantic state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Metric {
    /// Eligible triaged branches that were auto-proposed, over eligible
    /// triaged branches.
    AutoProposeShare,
    /// Triaged branches that were escalated to a person, over all triaged
    /// branches.
    EscalationShare,
    /// Certified branches whose certification ended in conflict, over branches
    /// with any recorded outcome.
    ConflictRate,
    /// Calibration-slice branches adjudicated harmful, over adjudicated
    /// calibration-slice branches. Because the slice is a uniform random sample
    /// of eligible branches, this is an unbiased estimate of the harm rate the
    /// arbiter's threshold is calibrated against.
    AdjudicatedHarmRate,
}

impl Metric {
    pub const ALL: [Metric; 4] = [
        Metric::AutoProposeShare,
        Metric::EscalationShare,
        Metric::ConflictRate,
        Metric::AdjudicatedHarmRate,
    ];

    /// Stable machine name.
    pub fn name(self) -> &'static str {
        match self {
            Self::AutoProposeShare => "auto_propose_share",
            Self::EscalationShare => "escalation_share",
            Self::ConflictRate => "conflict_rate",
            Self::AdjudicatedHarmRate => "adjudicated_harm_rate",
        }
    }
}

/// How rows are grouped.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Grouping {
    Overall,
    /// By the principal that authored the branch.
    ByPrincipal,
    /// By the triage policy version in force.
    ByPolicyVersion,
}

/// A metric request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricSpec {
    pub metric: Metric,
    pub grouping: Grouping,
}

/// One result row: a group label (empty for [`Grouping::Overall`]) and the two
/// counts of the proportion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricRow {
    pub group: String,
    pub numerator: u64,
    pub denominator: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_metric_has_a_distinct_name() {
        let names: BTreeSet<&str> = Metric::ALL.iter().map(|m| m.name()).collect();
        assert_eq!(names.len(), Metric::ALL.len());
    }
}
