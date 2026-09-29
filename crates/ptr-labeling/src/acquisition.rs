use crate::model::{LabelOutcome, Resolution};

/// How items are ranked for human annotation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Acquisition {
    /// Highest posterior entropy first.
    Entropy,
    /// Smallest gap between the two most probable classes first.
    Margin,
}

/// Rank the items of one resolution for annotation, most informative first,
/// returning at most `budget` item indices.
///
/// Items verifiers already determined are never proposed: a person would be
/// asked to confirm a fact. Disputed items are always proposed first, in item
/// order, because only a person can settle checks that ruled out every class.
/// Every other item is scored by `strategy` on the posterior its outcome was
/// resolved from ([`Resolution::posterior`]), not on the model's raw
/// posterior: vetoed classes have probability zero, so the model's mass on a
/// class a verifier ruled out neither makes an item look settled nor makes a
/// settled one look uncertain, and an item the model gives no share (no
/// probabilistic vote, or mass left below `f64::MIN_POSITIVE`) is scored as
/// uniform over the classes left. Equal scores keep item order.
///
/// Each item's outcome and posterior come from the same [`Resolution`], which
/// only [`crate::resolve`] makes, so they cannot be mispaired, reordered,
/// truncated or replaced, and every posterior ranked is one `resolve` checked
/// and renormalized; nothing is left to refuse here.
pub fn rank_for_annotation(
    resolution: &Resolution,
    strategy: Acquisition,
    budget: usize,
) -> Vec<usize> {
    let mut disputed = Vec::new();
    let mut scored = Vec::new();
    for (item, outcome) in resolution.outcomes().iter().enumerate() {
        match outcome {
            LabelOutcome::Determined { .. } => {}
            LabelOutcome::Disputed { .. } => disputed.push(item),
            LabelOutcome::Estimated { .. } | LabelOutcome::Unknown => {
                let posterior = resolution
                    .posterior(item)
                    .expect("resolve pairs every estimated or unknown item with a posterior");
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
    use crate::model::{fit_label_model, resolve, DawidSkeneParams};
    use crate::votes::{FunctionKind, LabelSchema, LabelingFunction, Vote, VoteMatrix};

    #[test]
    fn disputed_items_come_first_and_determined_items_never() {
        let (c0, c1, a) = (Vote::Class(0), Vote::Class(1), Vote::Abstain);
        let mut votes = vec![
            vec![c0, a, a, Vote::Veto(1), a],
            vec![c0, c0, c0, a, a],
            vec![c0, c1, a, a, a],
            vec![c0, a, a, Vote::Veto(0), Vote::Veto(1)],
        ];
        // Agreeing items of both classes, from which the model learns that
        // the heuristics are accurate.
        for item in 0..10 {
            let agreed = Vote::Class(item % 2);
            votes.push(vec![agreed, agreed, agreed, a, a]);
        }
        let matrix = VoteMatrix::new(
            LabelSchema::new(["a", "b"]).unwrap(),
            vec![
                LabelingFunction::new("h1", FunctionKind::Heuristic),
                LabelingFunction::new("h2", FunctionKind::Heuristic),
                LabelingFunction::new("h3", FunctionKind::Heuristic),
                LabelingFunction::new("v1", FunctionKind::Verifier),
                LabelingFunction::new("v2", FunctionKind::Verifier),
            ],
            votes,
        )
        .unwrap();
        let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
        let resolution = resolve(&matrix, &model, 0.9).unwrap();
        assert!(matches!(
            resolution.outcomes()[..4],
            [
                LabelOutcome::Determined { class: 0 },
                LabelOutcome::Estimated { class: 0, .. },
                LabelOutcome::Unknown,
                LabelOutcome::Disputed { .. },
            ]
        ));
        // The dispute, then the unknown item, then every estimated one; the
        // determined item never.
        let ranked = rank_for_annotation(&resolution, Acquisition::Entropy, 20);
        assert_eq!(ranked[..2], [3, 2]);
        assert_eq!(ranked.len(), matrix.items() - 1);
        assert!(ranked.contains(&1) && !ranked.contains(&0));
        assert_eq!(
            rank_for_annotation(&resolution, Acquisition::Margin, 2),
            vec![3, 2]
        );
    }
}
