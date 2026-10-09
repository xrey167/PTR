//! The self-model and the attention schema must not even name the authority types. The
//! `compile_fail` doctests only prove that some conversions are missing; this scan pins that the
//! two source files never reach for a registry, a registry update, the router itself or the
//! `Accepted` constructor (stable rustdoc does not check the error codes of compile_fail blocks).

fn code_lines(source: &str) -> Vec<(usize, &str)> {
    source
        .lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line.trim_start()))
        .filter(|(_, line)| !line.starts_with("//"))
        .collect()
}

#[test]
fn model_sources_never_name_authority_types() {
    let sources = [
        ("self_model.rs", include_str!("../src/self_model.rs")),
        (
            "attention_schema.rs",
            include_str!("../src/attention_schema.rs"),
        ),
    ];
    let forbidden = [
        "FindingRegistry",
        "RegistryUpdate",
        "RegistryChange",
        "update_registry",
        "Accepted(",
        "FindingRouter",
        "RouterConfig",
        "FallbackPolicy",
        "RouterLimits::default",
    ];
    for (name, source) in sources {
        for (line_number, line) in code_lines(source) {
            for token in forbidden {
                // `SelfOutcome::from_report` takes `&RouterLimits` to call `classify`; that is the
                // one sanctioned use and does not match the tokens above.
                assert!(
                    !line.contains(token),
                    "{name}:{line_number} names `{token}`: {line}"
                );
            }
        }
    }
}

#[test]
fn only_the_controlled_scorer_implements_the_scorer_trait() {
    let model = include_str!("../src/self_model.rs");
    assert!(code_lines(model)
        .iter()
        .all(|(_, line)| !line.contains("AdmissibleScorer for")));
    let schema = include_str!("../src/attention_schema.rs");
    let implementors: Vec<&str> = code_lines(schema)
        .into_iter()
        .map(|(_, line)| line)
        .filter(|line| line.contains("AdmissibleScorer for"))
        .collect();
    assert_eq!(implementors.len(), 1, "{implementors:?}");
    assert!(implementors[0].contains("ControlledScorer<S>"));
}
