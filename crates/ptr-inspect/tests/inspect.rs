use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use ptr_inspect::{
    diff, inspect_struct, redact_keys, redact_sensitive, render_compact, render_json, render_tree,
    walk, ChangeKind, InspectBytes, InspectNode, InspectPath, Inspectable, PathSegment,
    PreviewPolicy, Private, RawEvidence, RenderOptions, Secret, Visitor, WalkError, WalkLimits,
};
use ptr_types::{
    ConfidenceEstimate, ConfidenceTarget, Epistemic, EpistemicState, Generation, PodId,
    Probability, Revision, SemanticRole, TypedValue, Validity,
};

struct Account {
    name: String,
    password: Secret<String>,
    email: Private<String>,
    payload: RawEvidence<Vec<u8>>,
    tags: Vec<&'static str>,
}
inspect_struct!(Account {
    name,
    password,
    email,
    payload,
    tags
});

fn account() -> Account {
    Account {
        name: "ann".into(),
        password: Secret::new("hunter2-SECRET".into()),
        email: Private::with_length_class("ann@example.org".into()),
        payload: RawEvidence::new(b"RAW-EVIDENCE-BYTES".to_vec()),
        tags: vec!["a", "b"],
    }
}

#[derive(Default)]
struct Events(Vec<String>);

impl Visitor for Events {
    fn enter_struct(&mut self, path: &InspectPath, name: &str) {
        self.0.push(format!("+struct {name} {path}"));
    }
    fn leave_struct(&mut self, path: &InspectPath, name: &str) {
        self.0.push(format!("-struct {name} {path}"));
    }
    fn enter_list(&mut self, path: &InspectPath, len: usize) {
        self.0.push(format!("+list{len} {path}"));
    }
    fn leave_list(&mut self, path: &InspectPath, len: usize) {
        self.0.push(format!("-list{len} {path}"));
    }
    fn enter_map(&mut self, path: &InspectPath, len: usize) {
        self.0.push(format!("+map{len} {path}"));
    }
    fn leave_map(&mut self, path: &InspectPath, len: usize) {
        self.0.push(format!("-map{len} {path}"));
    }
    fn enter_enum(&mut self, path: &InspectPath, name: &str, variant: &str) {
        self.0.push(format!("+enum {name}::{variant} {path}"));
    }
    fn leave_enum(&mut self, path: &InspectPath, name: &str, variant: &str) {
        self.0.push(format!("-enum {name}::{variant} {path}"));
    }
    fn leaf(&mut self, path: &InspectPath, node: &InspectNode) {
        self.0.push(format!("leaf {} {path}", node.kind()));
    }
}

#[test]
fn visitor_order_covers_map_enum_struct_and_list() {
    let mut map = BTreeMap::new();
    map.insert("k".to_owned(), Some(1u8));
    let node = InspectNode::new_struct(
        "Root",
        vec![
            ("m", map.inspect()),
            (
                "e",
                InspectNode::variant("E", "V", vec![("0", true.inspect())]),
            ),
            ("l", vec![1u8, 2].inspect()),
        ],
    );
    let mut events = Events::default();
    let visited = walk(&node, &mut events, &WalkLimits::default()).unwrap();
    assert_eq!(visited, 8);
    assert_eq!(
        events.0,
        vec![
            "+struct Root $",
            "+map1 $.m",
            "leaf uint $.m{\"k\"}",
            "-map1 $.m",
            "+enum E::V $.e",
            "leaf bool $.e.0",
            "-enum E::V $.e",
            "+list2 $.l",
            "leaf uint $.l[0]",
            "leaf uint $.l[1]",
            "-list2 $.l",
            "-struct Root $",
        ]
    );
}

#[test]
fn walk_limits_fail_with_typed_errors() {
    let mut deep = InspectNode::Null;
    for _ in 0..10 {
        deep = InspectNode::List(vec![deep]);
    }
    let shallow = WalkLimits {
        max_depth: 3,
        max_nodes: 1000,
    };
    let err = walk(&deep, &mut Events::default(), &shallow).unwrap_err();
    assert!(matches!(err, WalkError::DepthExceeded { limit: 3, .. }));

    let wide = InspectNode::List((0..50u8).map(|n| n.inspect()).collect());
    let few = WalkLimits {
        max_depth: 8,
        max_nodes: 10,
    };
    let err = walk(&wide, &mut Events::default(), &few).unwrap_err();
    assert_eq!(
        err,
        WalkError::NodeLimitExceeded {
            limit: 10,
            path: InspectPath::root().child(PathSegment::Index(9)),
        }
    );
    let huge = WalkLimits {
        max_depth: usize::MAX,
        max_nodes: usize::MAX,
    };
    assert!(walk(&deep, &mut Events::default(), &huge).is_ok());
}

#[test]
fn every_diff_kind_is_reported() {
    let old = InspectNode::new_struct(
        "S",
        vec![
            ("same", 1u8.inspect()),
            ("changed", "a".inspect()),
            ("removed", 2u8.inspect()),
            ("retyped", 3u8.inspect()),
            ("list", vec![1u8, 2, 3].inspect()),
            ("variant", InspectNode::unit_variant("E", "A")),
            ("enum_name", InspectNode::unit_variant("E", "A")),
            ("map", BTreeMap::from([("x", 1u8), ("y", 2)]).inspect()),
        ],
    );
    let new = InspectNode::new_struct(
        "S",
        vec![
            ("same", 1u8.inspect()),
            ("changed", "b".inspect()),
            ("retyped", "3".inspect()),
            ("added", true.inspect()),
            ("list", vec![1u8, 9].inspect()),
            ("variant", InspectNode::unit_variant("E", "B")),
            ("enum_name", InspectNode::unit_variant("F", "A")),
            ("map", BTreeMap::from([("y", 5u8), ("z", 6)]).inspect()),
        ],
    );
    let changes: Vec<(String, &'static str)> = diff(&old, &new)
        .iter()
        .map(|c| {
            let kind = match c.kind {
                ChangeKind::Added(_) => "added",
                ChangeKind::Removed(_) => "removed",
                ChangeKind::Changed { .. } => "changed",
                ChangeKind::TypeChanged { .. } => "typechanged",
            };
            (c.path.to_string(), kind)
        })
        .collect();
    let expected = [
        ("$.changed", "changed"),
        ("$.removed", "removed"),
        ("$.retyped", "typechanged"),
        ("$.list[1]", "changed"),
        ("$.list[2]", "removed"),
        ("$.variant", "changed"),
        ("$.enum_name", "typechanged"),
        ("$.map{\"x\"}", "removed"),
        ("$.map{\"y\"}", "changed"),
        ("$.map{\"z\"}", "added"),
        ("$.added", "added"),
    ];
    let expected: Vec<(String, &str)> = expected
        .iter()
        .map(|(p, k)| ((*p).to_owned(), *k))
        .collect();
    assert_eq!(changes, expected);
}

#[test]
fn diff_of_growing_list_reports_added_positions() {
    let changes = diff(&vec![1u8].inspect(), &vec![1u8, 2].inspect());
    assert_eq!(changes.len(), 1);
    assert!(matches!(changes[0].kind, ChangeKind::Added(_)));
    assert_eq!(changes[0].path.to_string(), "$[1]");
}

#[test]
fn diff_never_exposes_a_redacted_side() {
    let before = account().inspect();
    let mut after_account = account();
    after_account.password = Secret::new("a-different-secret".into());
    after_account.email = Private::with_length_class("zed@example.org".into());
    assert!(diff(&before, &after_account.inspect()).is_empty());

    let plaintext_side = InspectNode::new_struct("S", vec![("k", InspectNode::text("LEAK-ME"))]);
    let redacted_side = InspectNode::new_struct("S", vec![("k", InspectNode::Redacted)]);
    for changes in [
        diff(&redacted_side, &plaintext_side),
        diff(&plaintext_side, &redacted_side),
    ] {
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ChangeKind::Changed { .. }));
        assert!(!format!("{changes:?}").contains("LEAK-ME"));
    }
}

#[test]
fn secrets_private_and_raw_evidence_never_leak_anywhere() {
    let value = account();
    let node = value.inspect();
    let outputs = [
        format!("{:?}", value.password),
        format!("{:#?}", value.password),
        format!("{:?}", value.email),
        format!("{:?}", value.payload),
        format!("{:#?}", value.payload),
        format!("{node:?}"),
        format!("{node:#?}"),
        render_tree(&node, &RenderOptions::default()),
        render_compact(&node, &RenderOptions::default()),
        render_json(&node),
        format!("{:?}", diff(&node, &InspectNode::Null)),
        format!("{:?}", diff(&InspectNode::Null, &node)),
    ];
    for output in &outputs {
        for needle in ["hunter2", "SECRET", "ann@example", "RAW-EVIDENCE"] {
            assert!(!output.contains(needle), "{needle} leaked in {output}");
        }
    }
    assert!(outputs[7].contains("password: <redacted>"));
    assert!(outputs[7].contains("email: <redacted type=alloc::string::String len=1-15>"));
    assert!(outputs[7].contains("len: 18"));
    assert!(outputs[7].contains("fnv1a64:"));
}

#[test]
fn raw_evidence_previews_only_under_explicit_policy() {
    let evidence = RawEvidence::with_policy(
        b"hello evidence".to_vec(),
        PreviewPolicy::Allow { max_bytes: 5 },
    );
    let text = render_compact(&evidence.inspect(), &RenderOptions::default());
    assert!(text.contains("preview:\"hello\""), "{text}");
    assert!(text.contains("truncated:true"), "{text}");
    assert!(!text.contains("evidence"), "{text}");
    assert_eq!(evidence.policy(), PreviewPolicy::Allow { max_bytes: 5 });
}

#[test]
fn key_redaction_pass_hides_credentials_by_name() {
    let mut headers = HashMap::new();
    headers.insert("Authorization".to_owned(), "Bearer abc123".to_owned());
    headers.insert("Accept".to_owned(), "text/plain".to_owned());
    let node = InspectNode::new_struct(
        "Request",
        vec![
            ("headers", headers.inspect()),
            ("API_KEY", "k-999".inspect()),
            ("url", "/x".inspect()),
        ],
    );
    let exact = redact_keys(&node, &["api_key", "authorization"]);
    let json = render_json(&exact);
    assert!(
        !json.contains("abc123") && !json.contains("k-999"),
        "{json}"
    );
    assert!(
        json.contains("text/plain") && json.contains("\"/x\""),
        "{json}"
    );
    assert_eq!(redact_sensitive(&node), exact);
}

#[test]
fn tree_compact_and_json_renderings_are_stable() {
    let node = InspectNode::new_struct(
        "Task",
        vec![
            ("id", 7u64.inspect()),
            ("note", "line1\nline2 \"q\"".inspect()),
            ("ratio", f64::NAN.inspect()),
            ("blob", InspectBytes(&[0xde, 0xad, 0xbe, 0xef]).inspect()),
        ],
    );
    let tree = render_tree(&node, &RenderOptions::default());
    assert_eq!(
        tree,
        "Task\n  id: 7\n  note: \"line1\\nline2 \\\"q\\\"\"\n  ratio: NaN\n  blob: bytes(len=4, fnv1a64:fa5aa4a7ae5f4fa1, 0xdeadbeef)"
            .replace("fnv1a64:fa5aa4a7ae5f4fa1", &fingerprint_of(&[0xde, 0xad, 0xbe, 0xef]))
    );
    let json = render_json(&node);
    assert!(
        json.starts_with("{\"struct\":\"Task\",\"fields\":{\"id\":7,"),
        "{json}"
    );
    assert!(
        json.contains("\"note\":\"line1\\nline2 \\\"q\\\"\""),
        "{json}"
    );
    assert!(json.contains("\"ratio\":\"NaN\""), "{json}");
    assert!(json.contains("\"bytes\":\"deadbeef\",\"len\":4"), "{json}");
}

fn fingerprint_of(bytes: &[u8]) -> String {
    match &InspectNode::bytes(bytes) {
        InspectNode::Bytes(preview) => format!("fnv1a64:{:016x}", preview.fingerprint()),
        other => panic!("not bytes: {other:?}"),
    }
}

#[test]
fn json_escapes_control_characters_and_preserves_unicode() {
    let text: String = (0u8..0x20).map(char::from).collect::<String>() + "\"\\ é 日本 😀 \u{7f}";
    let json = render_json(&InspectNode::text(text));
    for code in 0u32..0x20 {
        let ch = char::from_u32(code).unwrap();
        assert!(!json.contains(ch), "raw control {code:#x} in output");
    }
    assert!(json.contains("\\u0000") && json.contains("\\u001f"));
    assert!(json.contains("\\b") && json.contains("\\f") && json.contains("\\n"));
    assert!(json.contains("\\\"\\\\ é 日本 😀"));
    assert!(json.starts_with('"') && json.ends_with('"'));
}

#[test]
fn std_impls_cover_the_requested_types() {
    assert_eq!(Inspectable::inspect(&Some(3u8)), InspectNode::UInt(3));
    assert_eq!(Inspectable::inspect(&None::<u8>), InspectNode::Null);
    assert_eq!(
        [1i8, -2].inspect(),
        InspectNode::List(vec![InspectNode::Int(1), InspectNode::Int(-2)])
    );
    assert_eq!(
        (1u8, "x", true, 2.5f32).inspect(),
        InspectNode::List(vec![
            InspectNode::UInt(1),
            InspectNode::text("x"),
            InspectNode::Bool(true),
            InspectNode::Number(2.5)
        ])
    );
    assert_eq!(Box::new(1u8).inspect(), Rc::new(1u8).inspect());
    assert_eq!(Arc::new("s").inspect(), InspectNode::text("s"));
    assert_eq!(
        Inspectable::inspect(&Ok::<u8, u8>(1)),
        InspectNode::variant("Result", "Ok", vec![("0", 1u8.inspect())])
    );
    let first: HashMap<&str, u8> = HashMap::from([("b", 2), ("a", 1), ("c", 3)]);
    let second: HashMap<&str, u8> = HashMap::from([("c", 3), ("a", 1), ("b", 2)]);
    assert_eq!(first.inspect(), second.inspect());
    assert_eq!(
        render_compact(&first.inspect(), &RenderOptions::default()),
        "{\"a\"=>1,\"b\"=>2,\"c\"=>3}"
    );
}

#[test]
fn ptr_types_domain_values_inspect() {
    assert_eq!(
        Revision(4).inspect(),
        InspectNode::new_struct("Revision", vec![("value", InspectNode::UInt(4))])
    );
    assert_eq!(
        Generation(2).inspect(),
        InspectNode::new_struct("Generation", vec![("value", InspectNode::UInt(2))])
    );
    assert_eq!(
        SemanticRole::Claim.inspect(),
        InspectNode::unit_variant("SemanticRole", "Claim")
    );
    assert_eq!(
        EpistemicState::Verified.inspect(),
        InspectNode::unit_variant("EpistemicState", "Verified")
    );
    let probability = Probability::new(0.5).unwrap();
    let estimate = ConfidenceEstimate::new(
        ConfidenceTarget::SemanticRole(SemanticRole::Goal),
        probability,
    );
    let text = render_compact(&estimate.inspect(), &RenderOptions::default());
    assert_eq!(
        text,
        "ConfidenceEstimate{target:ConfidenceTarget::SemanticRole{0:SemanticRole::Goal},probability:Probability{value:0.5}}"
    );
    let typed = TypedValue {
        value: Epistemic::Known(PodId::from("pod-1")),
        generation: Generation(1),
        validity: Validity::Live,
        provenance: Vec::new(),
    };
    let text = render_compact(&typed.inspect(), &RenderOptions::default());
    assert!(
        text.contains("Epistemic::Known{0:PodId{value:\"pod-1\"}}"),
        "{text}"
    );
    assert!(text.contains("validity:Validity::Live"), "{text}");
}
