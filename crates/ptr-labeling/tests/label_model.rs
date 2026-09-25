use ptr_labeling::{
    evaluate, fit_label_model, rank_for_annotation, resolve, Acquisition, DawidSkeneParams,
    EvaluationSet, FunctionKind, GoldLabel, GoldSampling, GoldSource, LabelOutcome, LabelSchema,
    LabelingError, LabelingFunction, ModelWarning, Vote, VoteMatrix,
};

fn function(name: &str, kind: FunctionKind) -> LabelingFunction {
    LabelingFunction {
        name: name.into(),
        kind,
    }
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
    let ranked = rank_for_annotation(&model.posteriors, &outcomes, Acquisition::Entropy, 1);
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
