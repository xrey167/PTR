use ptr_labeling::{
    evaluate, fit_label_model, function_accuracy, rank_for_annotation, resolve, Acquisition,
    DawidSkeneParams, EvaluationSet, FunctionKind, GoldLabel, GoldSampling, GoldSource,
    LabelOutcome, LabelSchema, LabelingError, LabelingFunction, ModelWarning, Vote, VoteMatrix,
};

fn function(name: &str, kind: FunctionKind) -> LabelingFunction {
    LabelingFunction::new(name, kind)
}

/// A deterministic pseudo-random stream for synthetic votes.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 400 items, balanced truth; three heuristic functions with accuracies 0.9,
/// 0.75 and 0.55 that abstain 20% of the time.
fn synthetic() -> (VoteMatrix, Vec<usize>) {
    let schema = LabelSchema::new(["refund", "no_refund"]).unwrap();
    let accuracies = [0.9, 0.75, 0.55];
    let mut stream = Stream(0x5eed);
    let mut truth = Vec::new();
    let mut votes = Vec::new();
    for item in 0..400 {
        let label = item % 2;
        truth.push(label);
        votes.push(
            accuracies
                .iter()
                .map(|&accuracy| {
                    if stream.next() < 0.2 {
                        Vote::Abstain
                    } else if stream.next() < accuracy {
                        Vote::Class(label)
                    } else {
                        Vote::Class(1 - label)
                    }
                })
                .collect(),
        );
    }
    let functions = vec![
        function("strong", FunctionKind::Heuristic),
        function("medium", FunctionKind::Model),
        function("weak", FunctionKind::Agent),
    ];
    (VoteMatrix::new(schema, functions, votes).unwrap(), truth)
}

fn gold(truth: &[usize]) -> EvaluationSet {
    let mut set = EvaluationSet::new(GoldSampling::Uniform);
    for (item, &class) in truth.iter().enumerate() {
        set.push(GoldLabel {
            item,
            class,
            source: GoldSource::Human {
                annotator: "annotator-1".into(),
            },
        })
        .unwrap();
    }
    set
}

#[test]
fn the_label_model_recovers_function_accuracies_and_beats_majority_vote() {
    let (matrix, truth) = synthetic();
    let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();

    // Estimated diagonal accuracy per function, averaged over classes. With
    // symmetric noise the likelihood trades class prior against confusion
    // asymmetry, so individual cells are only weakly identified at this sample
    // size; the ranking of the functions is what must come out right.
    let accuracy = |j: usize| {
        let table = model.confusion[j].as_ref().unwrap();
        (table[0][0] + table[1][1]) / 2.0
    };
    assert!(accuracy(0) > accuracy(1) && accuracy(1) > accuracy(2));
    assert!(accuracy(0) > 0.75, "{}", accuracy(0));
    assert!((accuracy(2) - 0.55).abs() < 0.1, "{}", accuracy(2));

    let report = evaluate(&model.posteriors, &gold(&truth), 10).unwrap();
    assert!(report.calibration.is_some());
    let majority_correct = (0..matrix.items())
        .filter(|&item| {
            let mut counts = [0usize; 2];
            for vote in matrix.row(item) {
                if let Vote::Class(class) = vote {
                    counts[*class] += 1;
                }
            }
            counts[truth[item]] > counts[1 - truth[item]]
        })
        .count() as f64
        / matrix.items() as f64;
    assert!(
        report.accuracy > majority_correct,
        "label model {} majority {}",
        report.accuracy,
        majority_correct
    );
}

#[test]
fn a_verifier_veto_overrides_a_confident_model_and_vetoing_everything_is_a_dispute() {
    let schema = LabelSchema::new(["refund", "no_refund", "partial"]).unwrap();
    let functions = vec![
        function("h1", FunctionKind::Heuristic),
        function("h2", FunctionKind::Heuristic),
        function("h3", FunctionKind::Model),
        function("policy-check", FunctionKind::Verifier),
        function("amount-check", FunctionKind::Verifier),
        function("stock-check", FunctionKind::Verifier),
    ];
    let a = Vote::Abstain;
    let class = Vote::Class;
    let veto = Vote::Veto;
    let votes = vec![
        // Heuristics say refund; checks rule refund and partial out:
        // no_refund by elimination, however confident the model is.
        vec![class(0), class(0), class(0), veto(0), veto(2), a],
        // Checks rule out every class between them.
        vec![class(0), a, a, veto(0), veto(1), veto(2)],
        // A heuristic may not veto; this row is refused below and replaced.
        vec![class(2), veto(2), a, a, a, a],
        // Nothing probabilistic voted: the class prior is not a label.
        vec![a, a, a, a, a, a],
        // Heuristics disagree; not confident enough.
        vec![class(0), class(1), class(2), a, a, a],
    ];
    assert!(matches!(
        VoteMatrix::new(schema.clone(), functions.clone(), votes.clone()),
        Err(LabelingError::VoteKind { item: 2, .. })
    ));
    let mut votes = votes;
    votes[2] = vec![class(0), a, a, veto(0), veto(1), a];
    let matrix = VoteMatrix::new(schema, functions, votes).unwrap();
    let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    let outcomes = resolve(&matrix, &model, 0.95).unwrap();
    assert_eq!(outcomes[0], LabelOutcome::Determined { class: 1 });
    assert!(matches!(outcomes[1], LabelOutcome::Disputed { .. }));
    // Determined by elimination against the only heuristic vote.
    assert_eq!(outcomes[2], LabelOutcome::Determined { class: 2 });
    assert_eq!(outcomes[3], LabelOutcome::Unknown);
    assert_eq!(outcomes[4], LabelOutcome::Unknown);
    // The dispute is the first thing a person is asked about.
    let ranked =
        rank_for_annotation(&model.posteriors, &outcomes, Acquisition::Entropy, 1).unwrap();
    assert_eq!(ranked, vec![1]);
}

#[test]
fn a_verifier_casting_a_class_vote_is_refused() {
    let schema = LabelSchema::new(["a", "b"]).unwrap();
    let result = VoteMatrix::new(
        schema,
        vec![function("check", FunctionKind::Verifier)],
        vec![vec![Vote::Class(0)]],
    );
    assert!(matches!(result, Err(LabelingError::VoteKind { .. })));
}

#[test]
fn fewer_than_three_modelled_functions_are_reported_as_unidentifiable() {
    let schema = LabelSchema::new(["a", "b"]).unwrap();
    let matrix = VoteMatrix::new(
        schema,
        vec![
            function("h1", FunctionKind::Heuristic),
            function("h2", FunctionKind::Heuristic),
        ],
        vec![vec![Vote::Class(0), Vote::Class(0)]; 4],
    )
    .unwrap();
    let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    assert!(model
        .warnings
        .contains(&ModelWarning::FewerThanThreeFunctions { count: 2 }));
}

#[test]
fn a_vote_for_a_class_outside_the_schema_is_refused() {
    let schema = LabelSchema::new(["a", "b"]).unwrap();
    let result = VoteMatrix::new(
        schema,
        vec![function("f", FunctionKind::Heuristic)],
        vec![vec![Vote::Class(2)]],
    );
    assert!(result.is_err());
}

#[test]
fn abstentions_do_not_become_labels_even_with_a_low_resolution_threshold() {
    let matrix = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![
            function("model", FunctionKind::Model),
            function("check", FunctionKind::Verifier),
        ],
        vec![vec![Vote::Abstain, Vote::Abstain]; 3],
    )
    .unwrap();
    assert_eq!(matrix.coverage(), 0.0);
    let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    assert_eq!(model.priors, vec![0.5, 0.5]);
    assert!(model.confusion[1].is_none());
    assert!(model.warnings.contains(&ModelWarning::NoOverlap {
        function: "model".into()
    }));
    assert_eq!(
        resolve(&matrix, &model, 0.1).unwrap(),
        vec![LabelOutcome::Unknown; 3]
    );
}

#[test]
fn verifiers_alone_can_determine_a_class_only_by_eliminating_every_alternative() {
    let matrix = VoteMatrix::new(
        LabelSchema::new(["a", "b", "c"]).unwrap(),
        vec![
            function("first", FunctionKind::Verifier),
            function("second", FunctionKind::Verifier),
        ],
        vec![
            vec![Vote::Veto(0), Vote::Abstain],
            vec![Vote::Veto(0), Vote::Veto(1)],
        ],
    )
    .unwrap();
    let model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    assert_eq!(model.confusion, vec![None, None]);
    assert!(model
        .warnings
        .contains(&ModelWarning::FewerThanThreeFunctions { count: 0 }));
    assert_eq!(
        resolve(&matrix, &model, 1.0).unwrap(),
        vec![LabelOutcome::Unknown, LabelOutcome::Determined { class: 2 }]
    );
}

#[test]
fn ragged_vote_rows_report_the_item_and_expected_function_count() {
    let result = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![
            function("first", FunctionKind::Agent),
            function("second", FunctionKind::Model),
        ],
        vec![vec![Vote::Abstain, Vote::Class(1)], vec![Vote::Class(0)]],
    );
    assert_eq!(
        result.unwrap_err(),
        LabelingError::RaggedVotes {
            item: 1,
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn annotation_ties_are_stable_and_budget_zero_never_requests_work() {
    let posteriors = vec![vec![0.5, 0.5], vec![0.9, 0.1], vec![0.5, 0.5]];
    let outcomes = vec![LabelOutcome::Unknown; 3];
    for strategy in [Acquisition::Entropy, Acquisition::Margin] {
        assert_eq!(
            rank_for_annotation(&posteriors, &outcomes, strategy, 2),
            Ok(vec![0, 2])
        );
        assert_eq!(
            rank_for_annotation(&posteriors, &outcomes, strategy, 9),
            Ok(vec![0, 2, 1])
        );
        assert_eq!(
            rank_for_annotation(&posteriors, &outcomes, strategy, 0),
            Ok(vec![])
        );
    }
}

#[test]
fn evaluation_scores_only_gold_items_and_reports_missing_predictions() {
    let mut set = EvaluationSet::new(GoldSampling::Uniform);
    set.push(GoldLabel {
        item: 2,
        class: 1,
        source: GoldSource::Oracle,
    })
    .unwrap();
    let predictions = vec![vec![], vec![], vec![0.0, 1.0]];
    let report = evaluate(&predictions, &set, 5).unwrap();
    assert_eq!(report.accuracy, 1.0);
    let calibration = report.calibration.unwrap();
    assert_eq!(calibration.brier, 0.0);
    assert_eq!(calibration.expected_calibration_error, 0.0);
    assert_eq!(
        evaluate(&predictions[..2], &set, 5).unwrap_err(),
        LabelingError::LengthMismatch {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn only_a_model_function_may_be_attributed_to_an_adapter() {
    let schema = LabelSchema::new(["a", "b"]).unwrap();
    let votes = vec![vec![Vote::Class(0)]];
    let attributed = VoteMatrix::new(
        schema.clone(),
        vec![LabelingFunction::model("ranker", "adapter-7")],
        votes.clone(),
    )
    .unwrap();
    assert_eq!(
        attributed.functions()[0].adapter.as_deref(),
        Some("adapter-7")
    );
    // Every kind but a model is refused. The function abstains, which every
    // kind may do, so only the attribution can be the reason.
    for kind in [
        FunctionKind::Verifier,
        FunctionKind::Heuristic,
        FunctionKind::Agent,
    ] {
        let mut function = LabelingFunction::new("rule", kind);
        function.adapter = Some("adapter-7".into());
        assert_eq!(
            VoteMatrix::new(schema.clone(), vec![function], vec![vec![Vote::Abstain]]).unwrap_err(),
            LabelingError::AdapterAttribution {
                function: "rule".into()
            },
            "{kind:?}"
        );
    }
    assert_eq!(
        VoteMatrix::new(schema, vec![LabelingFunction::model("ranker", "")], votes).unwrap_err(),
        LabelingError::AdapterAttribution {
            function: "ranker".into()
        }
    );
}

#[test]
fn per_function_accuracy_is_attributed_to_the_adapter_and_brackets_the_truth() {
    let schema = LabelSchema::new(["refund", "no_refund"]).unwrap();
    let accuracies = [0.9, 0.6];
    let mut stream = Stream(0xadab7e4);
    let mut truth = Vec::new();
    let mut votes = Vec::new();
    for item in 0..600 {
        let label = item % 2;
        truth.push(label);
        let mut row: Vec<Vote> = accuracies
            .iter()
            .map(|&accuracy| {
                if stream.next() < accuracy {
                    Vote::Class(label)
                } else {
                    Vote::Class(1 - label)
                }
            })
            .collect();
        // A verifier rules out the wrong class on every item.
        row.push(Vote::Veto(1 - label));
        votes.push(row);
    }
    let functions = vec![
        LabelingFunction::model("ranker-v2", "adapter-v2"),
        LabelingFunction::model("ranker-v1", "adapter-v1"),
        function("schema-check", FunctionKind::Verifier),
    ];
    let matrix = VoteMatrix::new(schema, functions, votes).unwrap();
    let scores = function_accuracy(&matrix, &gold(&truth), 1.96).unwrap();
    assert_eq!(scores.len(), 3);
    for (score, (adapter, accuracy)) in scores
        .iter()
        .zip([("adapter-v2", 0.9), ("adapter-v1", 0.6)])
    {
        assert_eq!(score.adapter.as_deref(), Some(adapter));
        let estimate = score.estimate.unwrap();
        assert_eq!(estimate.trials, 600);
        assert!(
            estimate.low <= accuracy && accuracy <= estimate.high,
            "{adapter}: {estimate:?}"
        );
    }
    assert!(scores[0].estimate.unwrap().low > scores[1].estimate.unwrap().high);
    // A verifier casts no class vote, so it has no accuracy to report.
    assert_eq!(scores[2].adapter, None);
    assert_eq!(scores[2].estimate, None);
}

#[test]
fn per_function_accuracy_refuses_an_actively_sampled_or_misaligned_gold_set() {
    let (matrix, truth) = synthetic();
    let mut active = EvaluationSet::new(GoldSampling::Active);
    active
        .push(GoldLabel {
            item: 0,
            class: truth[0],
            source: GoldSource::Oracle,
        })
        .unwrap();
    assert!(matches!(
        function_accuracy(&matrix, &active, 1.96),
        Err(LabelingError::InvalidParameter { .. })
    ));
    assert_eq!(
        function_accuracy(&matrix, &EvaluationSet::new(GoldSampling::Uniform), 1.96).unwrap_err(),
        LabelingError::Empty {
            field: "evaluation set"
        }
    );
    let mut beyond = EvaluationSet::new(GoldSampling::Uniform);
    beyond
        .push(GoldLabel {
            item: 400,
            class: 0,
            source: GoldSource::Oracle,
        })
        .unwrap();
    assert_eq!(
        function_accuracy(&matrix, &beyond, 1.96).unwrap_err(),
        LabelingError::LengthMismatch {
            expected: 401,
            actual: 400
        }
    );
    assert_eq!(
        function_accuracy(&matrix, &gold(&truth), 0.0).unwrap_err(),
        LabelingError::InvalidParameter {
            field: "z",
            message: "must be finite and positive"
        }
    );
}

#[test]
fn per_function_accuracy_scores_only_class_votes_on_the_gold_items_it_names() {
    let a = Vote::Abstain;
    let class = Vote::Class;
    let veto = Vote::Veto;
    let matrix = VoteMatrix::new(
        LabelSchema::new(["refund", "no_refund"]).unwrap(),
        vec![
            LabelingFunction::model("ranker", "adapter-3"),
            function("rule", FunctionKind::Heuristic),
            function("sparse", FunctionKind::Agent),
            function("check", FunctionKind::Verifier),
        ],
        vec![
            vec![class(1), a, class(0), a],
            vec![class(1), a, a, veto(0)],
            vec![class(0), class(1), class(1), a],
            vec![a, class(0), a, a],
            vec![class(1), a, a, veto(1)],
            vec![class(0), class(1), class(0), a],
        ],
    )
    .unwrap();
    // A subset of the items, not in item order.
    let mut gold = EvaluationSet::new(GoldSampling::Uniform);
    for (item, class) in [(4, 0), (1, 1), (3, 0)] {
        gold.push(GoldLabel {
            item,
            class,
            source: GoldSource::Oracle,
        })
        .unwrap();
    }
    let scores = function_accuracy(&matrix, &gold, 1.96).unwrap();
    let scored: Vec<_> = scores
        .iter()
        .map(|score| {
            (
                score.function.as_str(),
                score.adapter.as_deref(),
                score
                    .estimate
                    .map(|estimate| (estimate.successes, estimate.trials)),
            )
        })
        .collect();
    // The ranker is wrong on item 4, right on item 1 and abstains on item 3;
    // the rule abstains on items 4 and 1 and is right on item 3. Scoring an
    // abstention would add trials; reading the votes of the gold set's
    // positions (items 0, 1 and 2) instead of its items would give 2 of 3 and
    // 0 of 1. The agent votes only off the gold items, so it has no estimate,
    // and a verifier's vetoes are never scored.
    assert_eq!(
        scored,
        vec![
            ("ranker", Some("adapter-3"), Some((1, 2))),
            ("rule", None, Some((1, 1))),
            ("sparse", None, None),
            ("check", None, None),
        ]
    );
}

#[test]
fn per_function_accuracy_refuses_an_invalid_z_whatever_the_votes() {
    // No function casts a class vote on the gold item, so no interval is
    // computed; z is refused anyway.
    let silent = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![
            LabelingFunction::model("ranker", "adapter-1"),
            function("check", FunctionKind::Verifier),
        ],
        vec![vec![Vote::Abstain, Vote::Veto(1)]],
    )
    .unwrap();
    let mut gold_item = EvaluationSet::new(GoldSampling::Uniform);
    gold_item
        .push(GoldLabel {
            item: 0,
            class: 0,
            source: GoldSource::Oracle,
        })
        .unwrap();
    assert_eq!(
        function_accuracy(&silent, &gold_item, 1.96).unwrap()[0].estimate,
        None
    );
    let (voting, truth) = synthetic();
    for (matrix, gold) in [(&silent, gold_item), (&voting, gold(&truth))] {
        for z in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(
                function_accuracy(matrix, &gold, z).unwrap_err(),
                LabelingError::InvalidParameter {
                    field: "z",
                    message: "must be finite and positive"
                },
                "{z}"
            );
        }
        // Its square overflows, so no interval at it is finite.
        assert_eq!(
            function_accuracy(matrix, &gold, 1e200).unwrap_err(),
            LabelingError::InvalidParameter {
                field: "z",
                message: "is too large for a finite interval"
            }
        );
    }
}

#[test]
fn an_evaluation_set_holds_one_gold_label_per_item() {
    let mut set = EvaluationSet::new(GoldSampling::Uniform);
    let label = |item, class| GoldLabel {
        item,
        class,
        source: GoldSource::Human {
            annotator: format!("annotator-{class}"),
        },
    };
    set.push(label(3, 0)).unwrap();
    // A conflicting label, and a repeat of the same class, for item 3.
    for class in [1, 0] {
        assert_eq!(
            set.push(label(3, class)).unwrap_err(),
            LabelingError::DuplicateGoldItem { item: 3 }
        );
    }
    set.push(label(0, 1)).unwrap();
    assert_eq!(set.len(), 2);

    // The refused labels leave no trace, so each item is scored once: had the
    // conflicting label been kept, the posterior on item 3 would count as
    // right and wrong, and the rule's vote on it as two trials.
    let posteriors = vec![vec![0.2, 0.8], vec![], vec![], vec![0.9, 0.1]];
    let report = evaluate(&posteriors, &set, 1).unwrap();
    assert_eq!(report.accuracy, 1.0);
    let matrix = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![function("rule", FunctionKind::Heuristic)],
        vec![
            vec![Vote::Class(0)],
            vec![Vote::Abstain],
            vec![Vote::Abstain],
            vec![Vote::Class(0)],
        ],
    )
    .unwrap();
    let estimate = function_accuracy(&matrix, &set, 1.96).unwrap()[0]
        .estimate
        .unwrap();
    assert_eq!((estimate.successes, estimate.trials), (1, 2));
}

#[test]
fn a_gold_class_outside_the_schema_is_refused_before_any_vote_is_scored() {
    let matrix = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![LabelingFunction::model("ranker", "adapter-1")],
        vec![vec![Vote::Class(0)], vec![Vote::Class(1)]],
    )
    .unwrap();
    let posteriors = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
    for sampling in [GoldSampling::Uniform, GoldSampling::Active] {
        let mut set = EvaluationSet::new(sampling);
        for (item, class) in [(0, 0), (1, 2)] {
            set.push(GoldLabel {
                item,
                class,
                source: GoldSource::Oracle,
            })
            .unwrap();
        }
        let unknown = LabelingError::UnknownClass {
            class: 2,
            classes: 2,
        };
        assert_eq!(
            evaluate(&posteriors, &set, 1).unwrap_err(),
            unknown,
            "{sampling:?}"
        );
        if sampling == GoldSampling::Uniform {
            assert_eq!(function_accuracy(&matrix, &set, 1.96).unwrap_err(), unknown);
        }
    }
}

#[test]
fn an_em_tolerance_outside_zero_to_one_is_refused_and_zero_waits_for_a_fixed_point() {
    let (matrix, _) = synthetic();
    for tolerance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-12, 1.0, 5.0] {
        let params = DawidSkeneParams {
            tolerance,
            ..DawidSkeneParams::default()
        };
        assert_eq!(
            fit_label_model(&matrix, params).unwrap_err(),
            LabelingError::InvalidParameter {
                field: "tolerance",
                message: "must lie in [0, 1)"
            },
            "{tolerance}"
        );
    }
    // No probabilistic vote: the posteriors never move, so a zero tolerance
    // is met on the first iteration.
    let silent = VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![function("model", FunctionKind::Model)],
        vec![vec![Vote::Abstain]; 3],
    )
    .unwrap();
    let exact = DawidSkeneParams {
        tolerance: 0.0,
        ..DawidSkeneParams::default()
    };
    let model = fit_label_model(&silent, exact).unwrap();
    assert_eq!(model.iterations, 1);
    assert!(!model
        .warnings
        .iter()
        .any(|warning| matches!(warning, ModelWarning::NotConverged { .. })));
}

#[test]
fn annotation_ranking_refuses_posteriors_and_outcomes_of_different_lengths() {
    let disputed = LabelOutcome::Disputed {
        vetoed: std::collections::BTreeSet::from([0, 1]),
    };
    // Pairing them up to the shorter list would drop item 1, the dispute a
    // person must always be asked about first.
    let outcomes = vec![LabelOutcome::Unknown, disputed, LabelOutcome::Unknown];
    for strategy in [Acquisition::Entropy, Acquisition::Margin] {
        assert_eq!(
            rank_for_annotation(&[vec![0.5, 0.5]], &outcomes, strategy, 10),
            Err(LabelingError::LengthMismatch {
                expected: 3,
                actual: 1
            })
        );
        assert_eq!(
            rank_for_annotation(
                &[vec![0.9, 0.1], vec![0.5, 0.5], vec![0.5, 0.5]],
                &[LabelOutcome::Unknown],
                strategy,
                10
            ),
            Err(LabelingError::LengthMismatch {
                expected: 1,
                actual: 3
            })
        );
    }
}

#[test]
fn annotation_ranking_refuses_a_scored_posterior_that_is_not_a_distribution() {
    let outcomes = vec![LabelOutcome::Unknown; 3];
    let confident = vec![0.99, 0.01];
    let even = vec![0.5, 0.5];
    // An empty posterior has no margin, a NaN one ranked below a confident
    // item, and one outside [0, 1] has a negative entropy.
    for invalid in [
        vec![],
        vec![f64::NAN, f64::NAN],
        vec![2.0, -1.0],
        vec![0.5, 0.6],
    ] {
        for strategy in [Acquisition::Entropy, Acquisition::Margin] {
            assert_eq!(
                rank_for_annotation(
                    &[confident.clone(), invalid.clone(), even.clone()],
                    &outcomes,
                    strategy,
                    10
                ),
                Err(LabelingError::InvalidPosterior { item: 1 }),
                "{invalid:?} {strategy:?}"
            );
        }
    }
    // Posteriors of items that are not ranked by them are not read.
    let outcomes = vec![
        LabelOutcome::Determined { class: 0 },
        LabelOutcome::Disputed {
            vetoed: std::collections::BTreeSet::from([0, 1]),
        },
        LabelOutcome::Unknown,
    ];
    assert_eq!(
        rank_for_annotation(&[vec![], vec![], even], &outcomes, Acquisition::Margin, 10),
        Ok(vec![1, 2])
    );
}

fn two_heuristics_and_a_third(votes: Vec<Vec<Vote>>) -> VoteMatrix {
    VoteMatrix::new(
        LabelSchema::new(["a", "b"]).unwrap(),
        vec![
            function("f0", FunctionKind::Heuristic),
            function("f1", FunctionKind::Heuristic),
            function("f2", FunctionKind::Heuristic),
        ],
        votes,
    )
    .unwrap()
}

#[test]
fn a_smoothing_near_the_largest_finite_value_fits_finite_uniform_probabilities() {
    let (c0, c1) = (Vote::Class(0), Vote::Class(1));
    let matrix = two_heuristics_and_a_third(vec![vec![c0, c0, c1], vec![c1, c1, c1]]);
    // Two smoothed counts of 1e308 overflow their total; normalizing by it
    // gave zero priors and NaN posteriors, reported as a converged fit.
    for smoothing in [f64::MAX, 1e308] {
        let model = fit_label_model(
            &matrix,
            DawidSkeneParams {
                smoothing,
                ..DawidSkeneParams::default()
            },
        )
        .unwrap();
        // The votes are negligible against the smoothing: every probability
        // is uniform.
        let uniform = vec![0.5, 0.5];
        assert_eq!(model.priors, uniform, "{smoothing}");
        for table in model.confusion.iter().flatten() {
            assert_eq!(
                table,
                &vec![uniform.clone(), uniform.clone()],
                "{smoothing}"
            );
        }
        assert_eq!(
            model.posteriors,
            vec![uniform.clone(), uniform.clone()],
            "{smoothing}"
        );
        assert_eq!(
            resolve(&matrix, &model, 0.9).unwrap(),
            vec![LabelOutcome::Unknown, LabelOutcome::Unknown]
        );
    }
}

#[test]
fn a_smoothing_too_small_to_keep_every_probability_positive_is_refused() {
    let (c0, c1, a) = (Vote::Class(0), Vote::Class(1), Vote::Abstain);
    let mut votes = vec![vec![c0, c0, c0]; 20];
    votes.extend(vec![vec![c1, c1, c1]; 20]);
    votes.push(vec![c0, a, a]);
    let matrix = two_heuristics_and_a_third(votes);
    let params = |smoothing| DawidSkeneParams {
        smoothing,
        ..DawidSkeneParams::default()
    };
    // With 41 items, 5e-324 / 41 is below the smallest positive f64: the
    // confusion matrices held exact zeros and one vote zeroed a class.
    for smoothing in [5e-324, f64::MIN_POSITIVE, 1e-307] {
        assert_eq!(
            fit_label_model(&matrix, params(smoothing)).unwrap_err(),
            LabelingError::InvalidParameter {
                field: "smoothing",
                message:
                    "is too small for this many items to keep every smoothed probability positive",
            },
            "{smoothing}"
        );
    }
    // The smallest smoothed probability of 1e-300 is about 2.4e-302, a
    // normal number: every prior and confusion probability stays positive,
    // and so does the single-vote item's posterior of the other class.
    let model = fit_label_model(&matrix, params(1e-300)).unwrap();
    assert!(model.priors.iter().all(|p| *p > 0.0));
    for table in model.confusion.iter().flatten() {
        assert!(
            table.iter().flatten().all(|p| p.is_finite() && *p > 0.0),
            "{table:?}"
        );
    }
    assert!(model.posteriors[40][1] > 0.0, "{:?}", model.posteriors[40]);
}

#[test]
fn resolution_refuses_a_posterior_that_is_not_a_distribution_over_the_schema() {
    let matrix = two_heuristics_and_a_third(vec![vec![Vote::Class(0); 3]]);
    let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    // Resolved as they stood, these gave a probability of 1.5, a class from
    // negative mass, and a class of a two-class schema while 0.8 of the mass
    // sat on a third entry.
    for posterior in [
        vec![1.5, -0.5],
        vec![-1.0, -3.0],
        vec![0.1, 0.1, 0.8],
        vec![1.0],
        vec![f64::NAN, 0.5],
        vec![0.6, 0.6],
    ] {
        model.posteriors = vec![posterior.clone()];
        assert_eq!(
            resolve(&matrix, &model, 0.5),
            Err(LabelingError::InvalidPosterior { item: 0 }),
            "{posterior:?}"
        );
    }
    model.posteriors = vec![vec![0.25, 0.75]];
    assert_eq!(
        resolve(&matrix, &model, 0.5).unwrap(),
        vec![LabelOutcome::Estimated {
            class: 1,
            probability: 0.75
        }]
    );
}

#[test]
fn a_model_with_no_mass_on_the_classes_left_resolves_to_unknown() {
    let schema = LabelSchema::new(["a", "b", "c"]).unwrap();
    let matrix = VoteMatrix::new(
        schema,
        vec![
            function("rule", FunctionKind::Heuristic),
            function("check", FunctionKind::Verifier),
        ],
        vec![vec![Vote::Class(0), Vote::Veto(0)]],
    )
    .unwrap();
    let mut model = fit_label_model(&matrix, DawidSkeneParams::default()).unwrap();
    // Every share of the classes left is 0 / 0, which reaches no required
    // probability however small, whatever the sign of the zeros.
    for posterior in [vec![1.0, 0.0, 0.0], vec![1.0, -0.0, 0.0]] {
        model.posteriors = vec![posterior];
        for min_probability in [f64::MIN_POSITIVE, 0.5, 1.0] {
            assert_eq!(
                resolve(&matrix, &model, min_probability).unwrap(),
                vec![LabelOutcome::Unknown]
            );
        }
    }
}

#[test]
fn a_gold_item_at_the_largest_index_is_reported_missing_rather_than_overflowing() {
    let mut set = EvaluationSet::new(GoldSampling::Uniform);
    set.push(GoldLabel {
        item: usize::MAX,
        class: 0,
        source: GoldSource::Oracle,
    })
    .unwrap();
    // `item + 1` panicked in debug builds and wrapped to 0 in release ones.
    assert_eq!(
        evaluate(&[vec![1.0, 0.0]], &set, 5).unwrap_err(),
        LabelingError::LengthMismatch {
            expected: usize::MAX,
            actual: 1
        }
    );
    let matrix = two_heuristics_and_a_third(vec![vec![Vote::Class(0); 3]]);
    assert_eq!(
        function_accuracy(&matrix, &set, 1.96).unwrap_err(),
        LabelingError::LengthMismatch {
            expected: usize::MAX,
            actual: 1
        }
    );
}

#[test]
fn evaluation_refuses_a_posterior_that_is_not_a_distribution_whatever_the_sampling() {
    for sampling in [GoldSampling::Active, GoldSampling::Uniform] {
        let mut set = EvaluationSet::new(sampling);
        set.push(GoldLabel {
            item: 0,
            class: 0,
            source: GoldSource::Oracle,
        })
        .unwrap();
        // total_cmp orders NaN above every number, so a NaN posterior was
        // the argmax of class 0 and counted as a correct prediction.
        for posterior in [vec![f64::NAN, 0.1], vec![1.5, -0.5], vec![0.6, 0.6]] {
            assert_eq!(
                evaluate(std::slice::from_ref(&posterior), &set, 5),
                Err(LabelingError::InvalidPosterior { item: 0 }),
                "{sampling:?} {posterior:?}"
            );
        }
    }
}

#[test]
fn labeling_function_names_must_be_non_empty_and_distinct() {
    let schema = LabelSchema::new(["yes", "no"]).unwrap();
    // Accepted, the second "rule" counted the same function's votes twice as
    // independent evidence in the label model.
    assert_eq!(
        VoteMatrix::new(
            schema.clone(),
            vec![
                function("rule", FunctionKind::Heuristic),
                function("model", FunctionKind::Model),
                function("rule", FunctionKind::Agent),
            ],
            vec![vec![Vote::Class(0), Vote::Class(1), Vote::Class(0)]],
        )
        .unwrap_err(),
        LabelingError::DuplicateFunction {
            function: "rule".into()
        }
    );
    assert_eq!(
        VoteMatrix::new(
            schema.clone(),
            vec![
                function("rule", FunctionKind::Heuristic),
                function("", FunctionKind::Heuristic),
            ],
            vec![],
        )
        .unwrap_err(),
        LabelingError::Empty {
            field: "labeling function name"
        }
    );
    assert_eq!(
        LabelingError::DuplicateFunction {
            function: "rule".into()
        }
        .code(),
        "PTR_LABELING_DUPLICATE_FUNCTION"
    );
    assert!(VoteMatrix::new(
        schema,
        vec![
            function("rule", FunctionKind::Heuristic),
            function("Rule", FunctionKind::Heuristic),
        ],
        vec![],
    )
    .is_ok());
}

#[test]
fn a_model_is_refused_with_any_matrix_but_the_one_it_was_fitted_on() {
    let functions = || {
        vec![
            function("rule", FunctionKind::Heuristic),
            function("model", FunctionKind::Model),
            function("agent", FunctionKind::Agent),
            function("check", FunctionKind::Verifier),
        ]
    };
    let schema = || LabelSchema::new(["yes", "no"]).unwrap();
    let fitted = VoteMatrix::new(
        schema(),
        functions(),
        vec![
            vec![
                Vote::Class(0),
                Vote::Class(0),
                Vote::Class(0),
                Vote::Abstain,
            ],
            vec![
                Vote::Class(1),
                Vote::Class(1),
                Vote::Class(1),
                Vote::Abstain,
            ],
        ],
    )
    .unwrap();
    let model = fit_label_model(&fitted, DawidSkeneParams::default()).unwrap();
    assert_eq!(model.matrix_digest(), fitted.digest());
    let rows = || -> Vec<Vec<Vote>> {
        (0..fitted.items())
            .map(|item| fitted.row(item).to_vec())
            .collect()
    };
    // An equal matrix built again is the same matrix.
    let rebuilt = VoteMatrix::new(schema(), functions(), rows()).unwrap();
    assert_eq!(rebuilt.digest(), fitted.digest());
    assert_eq!(
        resolve(&rebuilt, &model, 0.5).unwrap(),
        resolve(&fitted, &model, 0.5).unwrap()
    );

    // Same shape, different votes: this resolved, combining the other
    // matrix's veto and votes with this model's posteriors.
    let other_votes = VoteMatrix::new(
        schema(),
        functions(),
        vec![
            vec![Vote::Abstain, Vote::Abstain, Vote::Abstain, Vote::Veto(1)],
            vec![
                Vote::Class(0),
                Vote::Class(0),
                Vote::Class(0),
                Vote::Abstain,
            ],
        ],
    )
    .unwrap();
    let other_functions = VoteMatrix::new(
        schema(),
        vec![
            function("rule", FunctionKind::Heuristic),
            function("model", FunctionKind::Model),
            function("other", FunctionKind::Agent),
            function("check", FunctionKind::Verifier),
        ],
        rows(),
    )
    .unwrap();
    let other_schema = VoteMatrix::new(
        LabelSchema::new(["spam", "ham"]).unwrap(),
        functions(),
        rows(),
    )
    .unwrap();
    for other in [&other_votes, &other_functions, &other_schema] {
        assert_ne!(other.digest(), fitted.digest());
        assert_eq!(
            resolve(other, &model, 0.5),
            Err(LabelingError::MatrixMismatch)
        );
    }
    assert_eq!(
        LabelingError::MatrixMismatch.code(),
        "PTR_LABELING_MATRIX_MISMATCH"
    );
}
