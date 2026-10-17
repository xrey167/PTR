use ptr_model_api::{
    collect_stream, BoundedStream, CancellationToken, Deadline, EventStream, FinishReason,
    GenerationError, InferenceBackend, InferenceStreamAdapter, ManualClock, ModelError, ModelEvent,
    ModelRequest, ReferenceByteStreamingBackend, ReferenceEchoBackend, ScenarioBackend,
    StreamEvent, StreamItem, StreamLimits, StreamOptions, StreamSource, StreamingBackend, TokenId,
    Tokenizer, MAX_STREAM_CHUNK_CHARS, MAX_STREAM_MODEL_EVENTS,
};
use ptr_types::{RequestId, Revision, SemanticContext};
use std::sync::Arc;
use std::time::Duration;

fn request(text: &str) -> ModelRequest {
    ModelRequest {
        request_id: RequestId::from("r"),
        semantic_context: SemanticContext {
            revision: Revision(1),
            entries: vec![],
        },
        raw_text: text.to_owned(),
    }
}

fn options(max_tokens: usize, buffer: usize) -> StreamOptions {
    StreamOptions {
        limits: StreamLimits::new(max_tokens, buffer).unwrap(),
        ..StreamOptions::default()
    }
}

#[test]
fn byte_backend_streams_every_byte_then_completes() {
    let backend = ReferenceByteStreamingBackend::new();
    let mut stream = backend
        .open_stream(&request("héy"), options(100, 4))
        .unwrap();
    let collected = collect_stream(stream.as_mut()).unwrap();
    assert_eq!(collected.finish_reason, Some(FinishReason::Completed));
    assert_eq!(collected.tokens.len(), "héy".len());
    assert_eq!(
        backend.tokenizer().decode(&collected.tokens).unwrap(),
        "héy"
    );
    assert_eq!(
        stream.next_event(),
        StreamEvent::Error(GenerationError::StreamClosed)
    );
}

#[test]
fn max_tokens_cuts_off_with_finish_reason() {
    let backend = ReferenceByteStreamingBackend::new();
    let mut stream = backend
        .open_stream(&request("abcdef"), options(3, 2))
        .unwrap();
    let collected = collect_stream(stream.as_mut()).unwrap();
    assert_eq!(
        collected.tokens,
        vec![TokenId(97), TokenId(98), TokenId(99)]
    );
    assert_eq!(collected.finish_reason, Some(FinishReason::MaxTokens));
}

#[test]
fn exactly_max_tokens_of_output_is_completed_not_cut() {
    let backend = ReferenceByteStreamingBackend::new();
    let mut stream = backend.open_stream(&request("abc"), options(3, 8)).unwrap();
    let collected = collect_stream(stream.as_mut()).unwrap();
    assert_eq!(collected.tokens.len(), 3);
    assert_eq!(collected.finish_reason, Some(FinishReason::Completed));
}

#[test]
fn cancellation_mid_stream_reports_delivered_count() {
    let backend = ReferenceByteStreamingBackend::new();
    let cancel = CancellationToken::new();
    let opts = StreamOptions {
        limits: StreamLimits::new(100, 4).unwrap(),
        cancel: cancel.child(),
        deadline: None,
    };
    let mut stream = backend.open_stream(&request("abcdefgh"), opts).unwrap();
    assert!(matches!(stream.next_event(), StreamEvent::Token(_)));
    assert!(matches!(stream.next_event(), StreamEvent::Token(_)));
    cancel.cancel(); // parent cancels the child the stream holds
    assert_eq!(
        stream.next_event(),
        StreamEvent::Error(GenerationError::Cancelled { at_token: 2 })
    );
    assert_eq!(
        stream.next_event(),
        StreamEvent::Error(GenerationError::StreamClosed)
    );
}

#[test]
fn already_cancelled_token_refuses_to_open() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let opts = StreamOptions {
        cancel,
        ..StreamOptions::default()
    };
    let result = ReferenceByteStreamingBackend::new().open_stream(&request("x"), opts);
    assert!(matches!(
        result,
        Err(GenerationError::Cancelled { at_token: 0 })
    ));
    let adapter = InferenceStreamAdapter::new(ReferenceEchoBackend, 2).unwrap();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let opts = StreamOptions {
        cancel: cancelled,
        ..StreamOptions::default()
    };
    assert!(matches!(
        adapter.open_stream(&request("x"), opts),
        Err(GenerationError::Cancelled { at_token: 0 })
    ));
}

#[test]
fn deadline_expiry_with_manual_clock() {
    let clock = ManualClock::new();
    let deadline = Deadline::start(Arc::new(clock.clone()), Duration::from_millis(10));
    let opts = StreamOptions {
        limits: StreamLimits::new(100, 1).unwrap(),
        cancel: CancellationToken::new(),
        deadline: Some(deadline),
    };
    let backend = ReferenceByteStreamingBackend::new();
    let mut stream = backend.open_stream(&request("abcdef"), opts).unwrap();
    assert!(matches!(stream.next_event(), StreamEvent::Token(_)));
    clock.advance(Duration::from_millis(10));
    assert!(matches!(stream.next_event(), StreamEvent::Token(_)));
    clock.advance(Duration::from_millis(1));
    match stream.next_event() {
        StreamEvent::Error(GenerationError::DeadlineExceeded { elapsed, budget }) => {
            assert_eq!(budget, Duration::from_millis(10));
            assert_eq!(elapsed, Duration::from_millis(11));
        }
        other => panic!("expected deadline error, got {other:?}"),
    }
}

#[test]
fn expired_deadline_refuses_to_open_adapter() {
    let clock = ManualClock::new();
    let deadline = Deadline::start(Arc::new(clock.clone()), Duration::ZERO);
    clock.advance(Duration::from_nanos(1));
    let opts = StreamOptions {
        deadline: Some(deadline),
        ..StreamOptions::default()
    };
    let adapter = InferenceStreamAdapter::new(ReferenceEchoBackend, 2).unwrap();
    assert!(matches!(
        adapter.open_stream(&request("x"), opts),
        Err(GenerationError::DeadlineExceeded { .. })
    ));
}

#[test]
fn adapter_concatenation_equals_non_streamed_output() {
    let req = request("the quick brown fox ünïcode");
    let direct: String = ReferenceEchoBackend
        .infer(&req)
        .unwrap()
        .into_iter()
        .filter_map(|e| match e {
            ModelEvent::Token(t) => Some(t),
            _ => None,
        })
        .collect();
    for chunk in [1usize, 3, 7, 1000] {
        let adapter = InferenceStreamAdapter::new(ReferenceEchoBackend, chunk).unwrap();
        let mut stream = adapter.open_stream(&req, StreamOptions::default()).unwrap();
        let collected = collect_stream(stream.as_mut()).unwrap();
        assert_eq!(collected.text, direct, "chunk size {chunk}");
        assert_eq!(collected.finish_reason, Some(FinishReason::Completed));
        assert!(collected.model_events.is_empty());
    }
}

#[test]
fn adapter_passes_non_text_events_and_drops_finished() {
    let req = request("note");
    let adapter = InferenceStreamAdapter::new(ScenarioBackend, 2).unwrap();
    let mut stream = adapter.open_stream(&req, StreamOptions::default()).unwrap();
    let collected = collect_stream(stream.as_mut()).unwrap();
    assert_eq!(collected.text, "note");
    assert_eq!(collected.model_events.len(), 1);
    assert!(matches!(
        collected.model_events[0],
        ModelEvent::ActionReady(_)
    ));
}

#[test]
fn adapter_max_tokens_counts_text_chunks() {
    let adapter = InferenceStreamAdapter::new(ReferenceEchoBackend, 2).unwrap();
    let mut stream = adapter
        .open_stream(&request("abcdefgh"), options(2, 4))
        .unwrap();
    let collected = collect_stream(stream.as_mut()).unwrap();
    assert_eq!(collected.text, "abcd");
    assert_eq!(collected.finish_reason, Some(FinishReason::MaxTokens));
}

#[test]
fn adapter_surfaces_backend_error_and_zero_chunk_is_rejected() {
    struct Failing;
    impl InferenceBackend for Failing {
        fn name(&self) -> &'static str {
            "failing"
        }
        fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
            Err(ModelError("down".into()))
        }
    }
    let adapter = InferenceStreamAdapter::new(Failing, 4).unwrap();
    let error = adapter
        .open_stream(&request("x"), StreamOptions::default())
        .err()
        .unwrap();
    assert_eq!(error, GenerationError::Backend(ModelError("down".into())));
    assert_eq!(ModelError::from(error), ModelError("down".into()));
    assert!(matches!(
        InferenceStreamAdapter::new(ReferenceEchoBackend, 0),
        Err(GenerationError::InvalidLimits(_))
    ));
}

#[test]
fn collect_stream_returns_first_error() {
    let cancel = CancellationToken::new();
    let opts = StreamOptions {
        cancel: cancel.clone(),
        ..StreamOptions::default()
    };
    let mut stream = ReferenceByteStreamingBackend::new()
        .open_stream(&request("abc"), opts)
        .unwrap();
    cancel.cancel();
    assert_eq!(
        collect_stream(stream.as_mut()),
        Err(GenerationError::Cancelled { at_token: 0 })
    );
}

/// A source that never ends and never emits text.
struct EndlessModelEvents;

impl StreamSource for EndlessModelEvents {
    fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
        Ok(Some(StreamItem::Model(ModelEvent::Token("x".to_owned()))))
    }
}

#[test]
fn endless_non_text_events_hit_the_model_event_limit_even_with_max_tokens_one() {
    let mut stream = BoundedStream::new(EndlessModelEvents, options(1, 2));
    let error = collect_stream(&mut stream).unwrap_err();
    assert_eq!(
        error,
        GenerationError::ModelEventLimit {
            limit: MAX_STREAM_MODEL_EVENTS
        }
    );
    assert_eq!(
        stream.next_event(),
        StreamEvent::Error(GenerationError::StreamClosed)
    );
}

#[test]
fn oversize_chunk_chars_is_refused() {
    assert!(matches!(
        InferenceStreamAdapter::new(ReferenceEchoBackend, MAX_STREAM_CHUNK_CHARS + 1),
        Err(GenerationError::InvalidLimits(_))
    ));
    assert!(InferenceStreamAdapter::new(ReferenceEchoBackend, MAX_STREAM_CHUNK_CHARS).is_ok());
}
