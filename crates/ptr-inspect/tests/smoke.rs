use ptr_inspect::{InspectNode, Inspectable, Secret};
#[test]
fn secret_is_redacted() {
    assert_eq!(Secret::new("token").inspect(), InspectNode::Redacted);
}

// `Secret<T>` used to derive `Debug` over a public field, so `{:?}` printed the
// plaintext and so did every derived `Debug` that contained one.
#[test]
fn secret_debug_never_prints_the_value() {
    let secret = Secret::new("hunter2");
    assert_eq!(format!("{secret:?}"), "Secret([redacted])");
    assert_eq!(format!("{secret:#?}"), "Secret([redacted])");

    #[derive(Debug)]
    #[allow(dead_code)]
    struct Credentials {
        user: &'static str,
        password: Secret<&'static str>,
    }
    let credentials = Credentials {
        user: "operator",
        password: Secret::new("hunter2"),
    };
    for rendered in [format!("{credentials:?}"), format!("{credentials:#?}")] {
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(rendered.contains("Secret([redacted])"), "{rendered}");
    }
}

#[test]
fn secret_value_is_reachable_only_by_asking_for_it() {
    let secret = Secret::new(String::from("hunter2"));
    assert_eq!(secret.expose(), "hunter2");
    assert_eq!(secret.clone().into_inner(), "hunter2");
    assert_eq!(secret, Secret::new(String::from("hunter2")));
}

#[test]
fn redaction_does_not_require_debug_or_inspectable_on_the_inner_value() {
    struct Opaque(u8);

    let secret = Secret::new(Opaque(42));
    assert_eq!(format!("{secret:?}"), "Secret([redacted])");
    assert_eq!(format!("{secret:#?}"), "Secret([redacted])");
    assert_eq!(secret.inspect(), InspectNode::Redacted);
    assert_eq!(secret.expose().0, 42);
    assert_eq!(secret.into_inner().0, 42);
}

#[test]
fn redaction_never_calls_the_inner_debug_implementation() {
    struct PanickingDebug;

    impl std::fmt::Debug for PanickingDebug {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("secret debug must not inspect the wrapped value");
        }
    }

    let secret = Secret::new(PanickingDebug);
    assert_eq!(format!("{secret:?}"), "Secret([redacted])");
    assert_eq!(format!("{secret:#?}"), "Secret([redacted])");
    assert_eq!(secret.inspect(), InspectNode::Redacted);
}
