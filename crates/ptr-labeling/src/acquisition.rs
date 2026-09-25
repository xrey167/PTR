use crate::model::LabelOutcome;

/// How items are ranked for human annotation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Acquisition {
    /// Highest posterior entropy first.
    Entropy,
    /// Smallest gap between the two most probable classes first.
    Margin,
}

/// Rank items for annotation, most informative first, returning at most
/// `budget` item indices.
///
/// Items verifiers already determined are never proposed: a person would be
/// asked to confirm a fact. Disputed items are always proposed first, because
/// only a person can settle checks that ruled out every class.
pub fn rank_for_annotation(
    posteriors: &[Vec<f64>],
    outcomes: &[LabelOutcome],
    strategy: Acquisition,
    budget: usize,
) -> Vec<usize> {
    let mut disputed = Vec::new();
    let mut scored = Vec::new();
    for (item, (posterior, outcome)) in posteriors.iter().zip(outcomes).enumerate() {
        match outcome {
            LabelOutcome::Determined { .. } => {}
            LabelOutcome::Disputed { .. } => disputed.push(item),
            LabelOutcome::Estimated { .. } | LabelOutcome::Unknown => {
                scored.push((informativeness(posterior, strategy), item));
            }
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    disputed
        .into_iter()
        .chain(scored.into_iter().map(|(_, item)| item))
        .take(budget)
        .collect()
}

fn informativeness(posterior: &[f64], strategy: Acquisition) -> f64 {
    match strategy {
        Acquisition::Entropy => -posterior
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>(),
        Acquisition::Margin => {
            let mut sorted: Vec<f64> = posterior.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let gap = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            -gap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn disputed_items_come_first_and_determined_items_never() {
        let posteriors = vec![
            vec![0.5, 0.5],
            vec![0.99, 0.01],
            vec![0.6, 0.4],
            vec![0.5, 0.5],
        ];
        let outcomes = vec![
            LabelOutcome::Determined { class: 0 },
            LabelOutcome::Estimated {
                class: 0,
                probability: 0.99,
            },
            LabelOutcome::Unknown,
            LabelOutcome::Disputed {
                vetoed: BTreeSet::from([0, 1]),
            },
        ];
        assert_eq!(
            rank_for_annotation(&posteriors, &outcomes, Acquisition::Entropy, 10),
            vec![3, 2, 1]
        );
        assert_eq!(
            rank_for_annotation(&posteriors, &outcomes, Acquisition::Margin, 2),
            vec![3, 2]
        );
    }
}
