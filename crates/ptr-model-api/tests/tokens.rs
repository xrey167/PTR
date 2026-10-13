use ptr_model_api::{
    ByteTokenizer, KvReuseHint, KvReuseMode, KvReusePolicy, KvReuseReport, KvReusingBackend,
    LineageContribution, ModelError, ModelEvent, ModelRun, PromptSegment, SegmentKind,
    SegmentSource, SegmentedModelRequest, TokenContractError, TokenId, TokenizeError, Tokenizer,
    TokenizerIdentity,
};
use ptr_types::{
    Digest, EvidenceId, Generation, KnowledgeObjectId, RequestId, Revision, SemanticContext,
};

fn hex(digest: &Digest) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn context() -> SemanticContext {
    SemanticContext {
        revision: Revision(1),
        entries: vec![],
    }
}

fn raw(text: &str) -> PromptSegment {
    PromptSegment::raw_from_text(&ByteTokenizer::new(), text).unwrap()
}

fn semantic(text: &str, key: &str, generation: u64) -> PromptSegment {
    PromptSegment::from_text(
        &ByteTokenizer::new(),
        SegmentSource::Semantic {
            key: key.into(),
            generation: Generation(generation),
        },
        text,
    )
    .unwrap()
}

fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        context(),
        ByteTokenizer::new().identity().clone(),
        segments,
    )
    .unwrap()
}

#[test]
fn byte_tokenizer_round_trips_multibyte_utf8() {
    let tokenizer = ByteTokenizer::new();
    assert_eq!(tokenizer.identity().name(), "ptr.byte");
    assert_eq!(tokenizer.identity().version(), "1");
    let text = "aé€😀";
    let tokens = tokenizer.encode(text);
    assert_eq!(tokens.len(), text.len());
    assert_eq!(tokens[0], TokenId(97));
    assert_eq!(tokenizer.decode(&tokens).unwrap(), text);
    assert_eq!(tokenizer.decode(&[]).unwrap(), "");
}

#[test]
fn byte_tokenizer_decode_rejects_bad_input() {
    let tokenizer = ByteTokenizer::new();
    assert_eq!(
        tokenizer.decode(&[TokenId(65), TokenId(256)]),
        Err(TokenizeError::UnknownToken(TokenId(256)))
    );
    assert_eq!(
        tokenizer.decode(&[TokenId(0xFF)]),
        Err(TokenizeError::InvalidUtf8)
    );
    // A truncated multi-byte sequence is invalid UTF-8.
    assert_eq!(
        tokenizer.decode(&[TokenId(0xC3)]),
        Err(TokenizeError::InvalidUtf8)
    );
}

#[test]
fn tokenizer_identity_rejects_empty_parts() {
    assert_eq!(
        TokenizerIdentity::new(" ", "1"),
        Err(TokenContractError::EmptyTokenizerName)
    );
    assert_eq!(
        TokenizerIdentity::new("x", ""),
        Err(TokenContractError::EmptyTokenizerVersion)
    );
}

#[test]
fn segment_rejects_empty_tokens_and_empty_semantic_key() {
    let tokenizer = ByteTokenizer::new();
    assert_eq!(
        PromptSegment::raw_from_text(&tokenizer, ""),
        Err(TokenContractError::EmptySegment)
    );
    assert_eq!(
        PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Semantic {
                key: String::new(),
                generation: Generation(1)
            },
            "x"
        ),
        Err(TokenContractError::EmptySemanticKey)
    );
}

#[test]
fn content_digest_depends_on_tokens_and_tokenizer_identity() {
    assert_eq!(raw("abc").content_digest(), raw("abc").content_digest());
    assert_ne!(raw("abc").content_digest(), raw("abd").content_digest());

    let tokens = ByteTokenizer::new().encode("abc");
    let other = TokenizerIdentity::new("ptr.byte", "2").unwrap();
    let other_version = PromptSegment::raw_from_tokens(other, tokens.clone()).unwrap();
    assert_ne!(raw("abc").content_digest(), other_version.content_digest());
    assert_eq!(other_version.tokens(), tokens.as_slice());
}

#[test]
fn source_and_generation_change_the_lineage_digest_not_the_content_digest() {
    let a = semantic("abc", "k", 1);
    let b = semantic("abc", "k", 2);
    let c = semantic("abc", "other", 1);
    assert_eq!(a.content_digest(), b.content_digest());
    assert_eq!(a.content_digest(), raw("abc").content_digest());

    let digest = |s: PromptSegment| request(vec![s]).lineage_digest();
    assert_ne!(digest(a.clone()), digest(b));
    assert_ne!(digest(a.clone()), digest(c));
    assert_ne!(digest(a), digest(raw("abc")));
}

#[test]
fn memory_and_derived_sources_carry_identity_and_generation() {
    let tokenizer = ByteTokenizer::new();
    let memory = PromptSegment::from_text(
        &tokenizer,
        SegmentSource::Memory {
            id: KnowledgeObjectId::from("m1"),
            generation: Generation(3),
        },
        "x",
    )
    .unwrap();
    let derived = PromptSegment::from_text(
        &tokenizer,
        SegmentSource::Derived {
            id: EvidenceId::from("e1"),
            generation: Generation(3),
        },
        "x",
    )
    .unwrap();
    match memory.lineage_contribution() {
        LineageContribution::Bound {
            kind,
            identity,
            generation,
            ..
        } => {
            assert_eq!(kind, SegmentKind::Memory);
            assert_eq!(identity, "m1");
            assert_eq!(generation, Generation(3));
        }
        other @ LineageContribution::Raw { .. } => panic!("unexpected {other:?}"),
    }
    assert_eq!(derived.lineage_contribution().kind(), SegmentKind::Derived);
    assert_ne!(
        request(vec![memory]).lineage_digest(),
        request(vec![derived]).lineage_digest()
    );
}

#[test]
fn raw_contribution_has_no_identity_or_generation() {
    let segment = raw("abc");
    assert_eq!(
        segment.lineage_contribution(),
        LineageContribution::Raw {
            content_digest: *segment.content_digest()
        }
    );
    assert_eq!(segment.lineage_contribution().kind(), SegmentKind::Raw);
    // Identical raw prefixes share a lineage regardless of the request.
    let first = SegmentedModelRequest::new(
        RequestId::from("one"),
        context(),
        ByteTokenizer::new().identity().clone(),
        vec![raw("abc")],
    )
    .unwrap();
    assert_eq!(
        first.lineage_digest(),
        request(vec![raw("abc")]).lineage_digest()
    );
}

#[test]
fn lineage_digest_is_order_dependent_and_deterministic() {
    let ab = request(vec![raw("a"), raw("b")]);
    let ba = request(vec![raw("b"), raw("a")]);
    assert_ne!(ab.lineage_digest(), ba.lineage_digest());
    assert_eq!(
        ab.lineage_digest(),
        request(vec![raw("a"), raw("b")]).lineage_digest()
    );
    assert_eq!(ab.tokens(), vec![TokenId(97), TokenId(98)]);
    assert_eq!(ab.total_tokens(), 2);
}

#[test]
fn lineage_digest_format_is_pinned() {
    let req = request(vec![raw("a"), semantic("b", "k", 7)]);
    assert_eq!(
        hex(&req.lineage_digest()),
        "cc61a596b42a3efd0282f098fc5f19eb76e96a851f628e5ec43f028f3c3e57d7"
    );
}

#[test]
fn segmented_request_rejects_empty_list_and_tokenizer_mismatch() {
    let identity = ByteTokenizer::new().identity().clone();
    assert_eq!(
        SegmentedModelRequest::new(RequestId::from("r"), context(), identity.clone(), vec![]),
        Err(TokenContractError::EmptySegmentList)
    );
    let foreign = TokenizerIdentity::new("hf.other", "1").unwrap();
    let segment = PromptSegment::raw_from_tokens(foreign.clone(), vec![TokenId(1)]).unwrap();
    assert_eq!(
        SegmentedModelRequest::new(
            RequestId::from("r"),
            context(),
            identity.clone(),
            vec![raw("a"), segment]
        ),
        Err(TokenContractError::TokenizerMismatch {
            expected: identity,
            found: foreign
        })
    );
}

#[test]
fn kv_reuse_report_rejects_hit_beyond_total() {
    let report = KvReuseReport::new(3, KvReuseMode::Exact, 3, KvReusePolicy::ExactOnly).unwrap();
    assert_eq!(report.hit_tokens(), 3);
    assert_eq!(report.mode(), KvReuseMode::Exact);
    assert_eq!(
        KvReuseReport::new(
            4,
            KvReuseMode::Approximate,
            3,
            KvReusePolicy::AllowApproximate
        ),
        Err(TokenContractError::HitExceedsTotal {
            hit_tokens: 4,
            total_tokens: 3
        })
    );
}

struct PrefixBackend;

impl KvReusingBackend for PrefixBackend {
    fn name(&self) -> &str {
        "prefix-test"
    }

    fn infer_segmented(
        &self,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
    ) -> Result<ModelRun, ModelError> {
        let total = request.total_tokens();
        let kv_report = match hint {
            Some(hint) => Some(
                KvReuseReport::new(
                    hint.requested_prefix_tokens.min(total),
                    KvReuseMode::Exact,
                    total,
                    hint.policy,
                )
                .map_err(|e| ModelError(e.to_string()))?,
            ),
            None => None,
        };
        Ok(ModelRun {
            events: vec![ModelEvent::Finished],
            kv_report,
        })
    }
}

#[test]
fn trivial_backend_implements_kv_reusing_backend() {
    let backend: &dyn KvReusingBackend = &PrefixBackend;
    let req = request(vec![raw("abcd")]);
    assert_eq!(backend.name(), "prefix-test");

    let cold = backend.infer_segmented(&req, None).unwrap();
    assert_eq!(cold.kv_report, None);

    let hint = KvReuseHint {
        requested_prefix_tokens: 10,
        policy: KvReusePolicy::ExactOnly,
    };
    let warm = backend.infer_segmented(&req, Some(&hint)).unwrap();
    assert_eq!(warm.events, vec![ModelEvent::Finished]);
    let report = warm.kv_report.unwrap();
    assert_eq!(report.hit_tokens(), 4);
    assert_eq!(report.mode(), KvReuseMode::Exact);
}

#[test]
fn approximate_report_requires_a_policy_that_allows_it() {
    assert_eq!(
        KvReuseReport::new(1, KvReuseMode::Approximate, 3, KvReusePolicy::ExactOnly),
        Err(TokenContractError::ApproximateNotAllowed)
    );
    let ok = KvReuseReport::new(
        1,
        KvReuseMode::Approximate,
        3,
        KvReusePolicy::AllowApproximate,
    )
    .unwrap();
    assert_eq!(ok.mode(), KvReuseMode::Approximate);
    assert!(KvReuseReport::new(1, KvReuseMode::Exact, 3, KvReusePolicy::AllowApproximate).is_ok());
}

#[test]
fn raw_source_is_rejected_by_the_general_constructors() {
    let tokenizer = ByteTokenizer::new();
    assert_eq!(
        PromptSegment::from_text(&tokenizer, SegmentSource::Raw, "x"),
        Err(TokenContractError::RawSourceNeedsRawConstructor)
    );
    assert_eq!(
        PromptSegment::from_tokens(
            tokenizer.identity().clone(),
            SegmentSource::Raw,
            vec![TokenId(1)]
        ),
        Err(TokenContractError::RawSourceNeedsRawConstructor)
    );
}

#[test]
fn empty_memory_and_derived_ids_are_rejected() {
    let tokenizer = ByteTokenizer::new();
    assert_eq!(
        PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Memory {
                id: KnowledgeObjectId::from(""),
                generation: Generation(1)
            },
            "x"
        ),
        Err(TokenContractError::EmptySourceId)
    );
    assert_eq!(
        PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Derived {
                id: EvidenceId::from(""),
                generation: Generation(1)
            },
            "x"
        ),
        Err(TokenContractError::EmptySourceId)
    );
}

#[test]
fn revision_and_tokenizer_change_the_lineage_digest() {
    let identity = ByteTokenizer::new().identity().clone();
    let at = |revision: u64| {
        SegmentedModelRequest::new(
            RequestId::from("r"),
            SemanticContext {
                revision: Revision(revision),
                entries: vec![],
            },
            identity.clone(),
            vec![raw("abc")],
        )
        .unwrap()
        .lineage_digest()
    };
    assert_eq!(at(1), at(1));
    assert_ne!(at(1), at(2));

    let v2 = TokenizerIdentity::new("ptr.byte", "2").unwrap();
    let seg =
        PromptSegment::raw_from_tokens(v2.clone(), ByteTokenizer::new().encode("abc")).unwrap();
    let other = SegmentedModelRequest::new(RequestId::from("r"), context(), v2, vec![seg]).unwrap();
    assert_ne!(other.lineage_digest(), at(1));
}
