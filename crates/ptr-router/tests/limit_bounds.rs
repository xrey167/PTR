//! Hard maxima of caller-set limits: an unbounded value is refused or clamped.

use ptr_router::{
    AttentionSchema, FindingRegistry, JsonLinesSink, RouterLimits, SchemaError, SchemaLimits,
    SelfModel, SelfModelLimits, TelemetryError, MAX_JSON_LINES_BYTES, MAX_REGISTRY_CODES,
    MAX_REGISTRY_PODS_PER_ROLE,
};

fn limits_with(set: impl FnOnce(&mut SchemaLimits)) -> SchemaLimits {
    let mut limits = SchemaLimits::default();
    set(&mut limits);
    limits
}

#[test]
fn attention_schema_refuses_unbounded_limits() {
    let cases = [
        ("max_contexts", limits_with(|l| l.max_contexts = usize::MAX)),
        (
            "max_pods_per_context",
            limits_with(|l| l.max_pods_per_context = usize::MAX),
        ),
        ("window", limits_with(|l| l.window = usize::MAX)),
        (
            "max_preferences",
            limits_with(|l| l.max_preferences = usize::MAX),
        ),
        ("dedup_window", limits_with(|l| l.dedup_window = usize::MAX)),
        ("max_id_len", limits_with(|l| l.max_id_len = usize::MAX)),
    ];
    for (field, limits) in cases {
        assert_eq!(
            AttentionSchema::new(limits).err(),
            Some(SchemaError::InvalidLimits { field })
        );
    }
    let at_max = limits_with(|l| {
        l.max_contexts = SchemaLimits::HARD_MAX_CONTEXTS;
        l.window = SchemaLimits::HARD_MAX_WINDOW;
    });
    assert!(AttentionSchema::new(at_max).is_ok());
}

#[test]
fn self_model_clamps_unbounded_limits() {
    let model = SelfModel::new(SelfModelLimits {
        max_contexts: usize::MAX,
        max_codes_per_context: usize::MAX,
        max_pods_per_context: usize::MAX,
        max_repair_pairs: usize::MAX,
        max_repair_codes: usize::MAX,
        max_context_codes: usize::MAX,
        max_code_len: usize::MAX,
        max_log: usize::MAX,
    });
    let limits = model.limits();
    assert_eq!(limits.max_contexts, SelfModelLimits::HARD_MAX_CONTEXTS);
    assert_eq!(limits.max_log, SelfModelLimits::HARD_MAX_LOG);
    assert_eq!(limits.max_code_len, SelfModelLimits::HARD_MAX_CODE_LEN);
    assert_eq!(
        limits.max_context_codes,
        SelfModelLimits::HARD_MAX_CONTEXT_CODES
    );
    assert_eq!(
        limits.max_repair_pairs,
        SelfModelLimits::HARD_MAX_REPAIR_PAIRS
    );
    assert_eq!(
        limits.max_repair_codes,
        SelfModelLimits::HARD_MAX_REPAIR_CODES
    );
    assert_eq!(
        limits.max_codes_per_context,
        SelfModelLimits::HARD_MAX_PER_CONTEXT
    );
    assert_eq!(
        limits.max_pods_per_context,
        SelfModelLimits::HARD_MAX_PER_CONTEXT
    );
    let defaults = SelfModel::new(SelfModelLimits::default());
    assert_eq!(*defaults.limits(), SelfModelLimits::default());
}

#[test]
fn router_limits_clamp_to_hard_maxima() {
    let clamped = RouterLimits {
        max_codes_per_report: usize::MAX,
        max_code_len: usize::MAX,
        max_tracked_unroutable_codes: usize::MAX,
    }
    .clamped();
    assert_eq!(
        clamped.max_codes_per_report,
        RouterLimits::HARD_MAX_CODES_PER_REPORT
    );
    assert_eq!(clamped.max_code_len, RouterLimits::HARD_MAX_CODE_LEN);
    assert_eq!(
        clamped.max_tracked_unroutable_codes,
        RouterLimits::HARD_MAX_TRACKED_UNROUTABLE_CODES
    );
    assert_eq!(RouterLimits::default().clamped().max_codes_per_report, 32);
}

#[test]
fn finding_registry_accepts_an_unbounded_request_and_documents_its_ceiling() {
    // The registry's bounds are private; the clamp itself is unit-tested beside the code.
    let _registry = FindingRegistry::with_capacity(usize::MAX, usize::MAX);
    assert_eq!(MAX_REGISTRY_CODES, 1 << 16);
    assert_eq!(MAX_REGISTRY_PODS_PER_ROLE, 1 << 12);
}

#[test]
fn json_lines_sink_refuses_an_unbounded_byte_bound() {
    let path = std::env::temp_dir().join(format!("ptr-router-limit-{}.jsonl", std::process::id()));
    for bound in [u64::MAX, MAX_JSON_LINES_BYTES + 1] {
        assert_eq!(
            JsonLinesSink::open(&path, bound).err(),
            Some(TelemetryError::LimitTooLarge)
        );
    }
    assert!(!path.exists(), "a refused bound must not create the file");
    assert!(JsonLinesSink::open(&path, MAX_JSON_LINES_BYTES).is_ok());
    let _ = std::fs::remove_file(path);
}
