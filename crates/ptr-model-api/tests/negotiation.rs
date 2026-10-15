use ptr_model_api::{
    negotiate, BackendCapabilities, ByteTokenizer, CapabilityRequirements, DType, KvReuseMode,
    KvReuseSupport, ModelRequest, NegotiationError, RejectionReason, Tokenizer,
};
use ptr_types::{RequestId, Revision, SemanticContext};

fn caps(id: &str, ctx: usize) -> BackendCapabilities {
    let mut c = BackendCapabilities::new(id, ctx);
    c.dtypes = vec![DType::F16];
    c
}

fn reasons(err: NegotiationError, id: &str) -> Vec<RejectionReason> {
    match err {
        NegotiationError::NoBackendSatisfies { rejected } => rejected
            .into_iter()
            .find(|r| r.backend_id == id)
            .map(|r| r.reasons)
            .unwrap_or_default(),
        other => panic!("unexpected {other:?}"),
    }
}

fn byte_id() -> ptr_model_api::TokenizerIdentity {
    ByteTokenizer::new().identity().clone()
}

#[test]
fn every_rejection_reason_is_reported_with_all_reasons_in_order() {
    let req = CapabilityRequirements {
        context_tokens: 100,
        streaming: true,
        resumable: true,
        kv_reuse: Some(KvReuseMode::Approximate),
        tokenizer: Some(byte_id()),
        batch: 4,
        acceptable_dtypes: vec![DType::Q4],
    };
    let mut weak = caps("weak", 10);
    weak.kv_reuse = KvReuseSupport::ExactOnly;
    let err = negotiate(&req, &[weak]).unwrap_err();
    assert_eq!(
        reasons(err, "weak"),
        vec![
            RejectionReason::ContextTooSmall {
                required: 100,
                offered: 10
            },
            RejectionReason::StreamingUnsupported,
            RejectionReason::ResumeUnsupported,
            RejectionReason::KvReuseUnsupported {
                required: KvReuseMode::Approximate,
                offered: KvReuseSupport::ExactOnly
            },
            RejectionReason::TokenizerMismatch {
                required: byte_id(),
                offered: None
            },
            RejectionReason::BatchTooLarge {
                required: 4,
                offered: 1
            },
            RejectionReason::DTypeUnsupported {
                acceptable: vec![DType::Q4],
                offered: vec![DType::F16]
            },
        ]
    );
}

#[test]
fn each_reason_in_isolation() {
    let base = CapabilityRequirements {
        batch: 1,
        ..CapabilityRequirements::default()
    };
    let cases: Vec<(CapabilityRequirements, RejectionReason)> = vec![
        (
            CapabilityRequirements {
                context_tokens: 11,
                ..base.clone()
            },
            RejectionReason::ContextTooSmall {
                required: 11,
                offered: 10,
            },
        ),
        (
            base.clone().with_streaming(),
            RejectionReason::StreamingUnsupported,
        ),
        (
            base.clone().with_resumable(),
            RejectionReason::ResumeUnsupported,
        ),
        (
            base.clone().with_kv_reuse(KvReuseMode::Exact),
            RejectionReason::KvReuseUnsupported {
                required: KvReuseMode::Exact,
                offered: KvReuseSupport::None,
            },
        ),
        (
            CapabilityRequirements {
                tokenizer: Some(byte_id()),
                ..base.clone()
            },
            RejectionReason::TokenizerMismatch {
                required: byte_id(),
                offered: None,
            },
        ),
        (
            base.clone().with_batch(2),
            RejectionReason::BatchTooLarge {
                required: 2,
                offered: 1,
            },
        ),
        (
            base.clone().with_dtypes(vec![DType::Bf16]),
            RejectionReason::DTypeUnsupported {
                acceptable: vec![DType::Bf16],
                offered: vec![DType::F16],
            },
        ),
    ];
    for (req, expected) in cases {
        let err = negotiate(&req, &[caps("only", 10)]).unwrap_err();
        assert_eq!(reasons(err, "only"), vec![expected]);
    }
}

#[test]
fn selects_satisfying_backend_and_lists_rejections() {
    let mut good = caps("good", 1000);
    good.streaming = true;
    let small = caps("small", 5);
    let req = CapabilityRequirements {
        context_tokens: 50,
        batch: 1,
        ..CapabilityRequirements::default()
    }
    .with_streaming();
    let plan = negotiate(&req, &[small, good]).unwrap();
    assert_eq!(plan.backend_id, "good");
    assert_eq!(plan.selected_index, 1);
    assert_eq!(plan.dtype, Some(DType::F16));
    assert_eq!(plan.rejected.len(), 1);
    assert_eq!(plan.rejected[0].backend_id, "small");
}

#[test]
fn tie_break_order_and_offer_order_independence() {
    let req = CapabilityRequirements {
        batch: 1,
        ..CapabilityRequirements::default()
    };
    let mut hi_fidelity = caps("z-hi", 100);
    hi_fidelity.dtypes = vec![DType::F32];
    let mut big_ctx = caps("b-ctx", 500);
    big_ctx.dtypes = vec![DType::F16];
    let mut big_batch = caps("a-batch", 500);
    big_batch.dtypes = vec![DType::F16];
    big_batch.max_batch = 8;
    let mut twin = caps("a-twin", 500);
    twin.dtypes = vec![DType::F16];
    twin.max_batch = 8;

    // 1: fidelity beats context, batch and id
    let all = vec![
        big_ctx.clone(),
        big_batch.clone(),
        twin.clone(),
        hi_fidelity.clone(),
    ];
    assert_eq!(negotiate(&req, &all).unwrap().backend_id, "z-hi");
    // 3 then 4: equal dtype/context -> larger batch; equal batch -> smaller id
    let rest = vec![twin.clone(), big_ctx.clone(), big_batch.clone()];
    assert_eq!(negotiate(&req, &rest).unwrap().backend_id, "a-batch");
    // 2: larger context wins before batch
    let mut small_ctx_big_batch = caps("c", 100);
    small_ctx_big_batch.max_batch = 64;
    let pair = vec![small_ctx_big_batch, big_ctx.clone()];
    assert_eq!(negotiate(&req, &pair).unwrap().backend_id, "b-ctx");

    // order independence over all rotations/reversals
    let expected = negotiate(&req, &all).unwrap().backend_id;
    let mut reversed = all.clone();
    reversed.reverse();
    assert_eq!(negotiate(&req, &reversed).unwrap().backend_id, expected);
    let mut rotated = all.clone();
    rotated.rotate_left(2);
    assert_eq!(negotiate(&req, &rotated).unwrap().backend_id, expected);
}

#[test]
fn dtype_preference_order_is_honoured() {
    let mut a = caps("a", 10);
    a.dtypes = vec![DType::F32, DType::Q8];
    let req = CapabilityRequirements {
        batch: 1,
        ..CapabilityRequirements::default()
    }
    .with_dtypes(vec![DType::Q8, DType::F32]);
    assert_eq!(negotiate(&req, &[a]).unwrap().dtype, Some(DType::Q8));
}

#[test]
fn empty_and_duplicate_offers_are_errors() {
    let req = CapabilityRequirements::default();
    assert_eq!(negotiate(&req, &[]), Err(NegotiationError::NoCandidates));
    assert_eq!(
        negotiate(&req, &[caps("x", 1), caps("x", 2)]),
        Err(NegotiationError::DuplicateBackendId("x".into()))
    );
}

#[test]
fn requirements_from_request_use_text_length() {
    let request = ModelRequest {
        request_id: RequestId::from("r"),
        semantic_context: SemanticContext {
            revision: Revision(1),
            entries: vec![],
        },
        raw_text: "hello".into(),
    };
    let req = CapabilityRequirements::from_request(&request);
    assert_eq!(req.context_tokens, 5);
    assert_eq!(req.batch, 1);
    assert!(negotiate(&req, &[caps("fits", 5)]).is_ok());
    assert!(negotiate(&req, &[caps("tiny", 4)]).is_err());
}
