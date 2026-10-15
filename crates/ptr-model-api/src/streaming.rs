//! Streaming backend contract: a synchronous pull iterator over generation events.
//!
//! The contract is deliberately `async`-free; an async adapter (tokio/futures) is future
//! work. [`BoundedStream`] enforces, around any [`StreamSource`], that cancellation and the
//! deadline are checked before every delivered event and before every source step, that no
//! more than `max_tokens` token/text events are delivered (then `Done { MaxTokens }`) and
//! that the internal look-ahead buffer never exceeds `buffer_capacity` items (Invariant 8).

use crate::{
    ByteTokenizer, CancellationToken, Deadline, GenerationError, InferenceBackend, ModelEvent,
    ModelRequest, TokenId, Tokenizer,
};
use std::collections::VecDeque;

/// Upper bound accepted for [`StreamLimits::buffer_capacity`].
pub const MAX_STREAM_BUFFER: usize = 4096;
/// Upper bound accepted for [`StreamLimits::max_tokens`].
pub const MAX_STREAM_TOKENS: usize = 1 << 20;

/// Why a stream ended normally.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// The source ran out of output.
    Completed,
    /// `max_tokens` token/text events were delivered and more output was available.
    MaxTokens,
}

/// One pulled event. `Done` and `Error` are terminal; afterwards `next_event` returns
/// `Error(GenerationError::StreamClosed)`.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    Token(TokenId),
    Text(String),
    /// Any non-text [`ModelEvent`] (slot update, action, ...). `ModelEvent::Finished`
    /// is never forwarded; it is expressed by `Done`.
    Model(ModelEvent),
    Done {
        finish_reason: FinishReason,
    },
    Error(GenerationError),
}

/// Bounds of one stream. `max_tokens` counts delivered `Token` and `Text` events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamLimits {
    pub max_tokens: usize,
    pub buffer_capacity: usize,
}

impl StreamLimits {
    pub fn new(max_tokens: usize, buffer_capacity: usize) -> Result<Self, GenerationError> {
        if max_tokens == 0 || max_tokens > MAX_STREAM_TOKENS {
            return Err(GenerationError::InvalidLimits("max_tokens out of range"));
        }
        if buffer_capacity == 0 || buffer_capacity > MAX_STREAM_BUFFER {
            return Err(GenerationError::InvalidLimits(
                "buffer_capacity out of range",
            ));
        }
        Ok(Self {
            max_tokens,
            buffer_capacity,
        })
    }
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            buffer_capacity: 64,
        }
    }
}

/// Limits plus the cooperative stop conditions of one stream.
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    pub limits: StreamLimits,
    pub cancel: CancellationToken,
    pub deadline: Option<Deadline>,
}

/// An item produced by a [`StreamSource`].
#[derive(Clone, Debug, PartialEq)]
pub enum StreamItem {
    Token(TokenId),
    Text(String),
    Model(ModelEvent),
}

impl StreamItem {
    fn counts_toward_max(&self) -> bool {
        matches!(self, Self::Token(_) | Self::Text(_))
    }

    fn into_event(self) -> StreamEvent {
        match self {
            Self::Token(token) => StreamEvent::Token(token),
            Self::Text(text) => StreamEvent::Text(text),
            Self::Model(event) => StreamEvent::Model(event),
        }
    }
}

/// Step producer behind a stream. One call is one generation step.
pub trait StreamSource: Send {
    /// `Ok(None)` means the output is complete.
    fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError>;
}

/// Pull iterator over generation events.
pub trait EventStream: Send {
    fn next_event(&mut self) -> StreamEvent;
}

/// Backend that opens bounded, cancellable event streams.
pub trait StreamingBackend: Send + Sync {
    fn name(&self) -> &str;
    fn open_stream(
        &self,
        request: &ModelRequest,
        options: StreamOptions,
    ) -> Result<Box<dyn EventStream>, GenerationError>;
}

/// Enforcing wrapper that turns a [`StreamSource`] into an [`EventStream`].
pub struct BoundedStream<S: StreamSource> {
    source: S,
    options: StreamOptions,
    buffer: VecDeque<StreamItem>,
    /// Effective look-ahead bound: `buffer_capacity` clamped into `1..=MAX_STREAM_BUFFER`.
    buffer_limit: usize,
    /// Effective token bound: `max_tokens` clamped into `1..=MAX_STREAM_TOKENS`.
    token_limit: usize,
    source_done: bool,
    delivered: usize,
    ended: bool,
}

impl<S: StreamSource> BoundedStream<S> {
    pub fn new(source: S, options: StreamOptions) -> Self {
        let capacity = options.limits.buffer_capacity.clamp(1, MAX_STREAM_BUFFER);
        let token_limit = options.limits.max_tokens.clamp(1, MAX_STREAM_TOKENS);
        Self {
            source,
            options,
            buffer: VecDeque::with_capacity(capacity),
            buffer_limit: capacity,
            token_limit,
            source_done: false,
            delivered: 0,
            ended: false,
        }
    }

    /// Number of items currently buffered; never exceeds `buffer_capacity`.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    fn check_stop(&self) -> Result<(), GenerationError> {
        if self.options.cancel.is_cancelled() {
            return Err(GenerationError::Cancelled {
                at_token: self.delivered,
            });
        }
        match &self.options.deadline {
            Some(deadline) => deadline.check(),
            None => Ok(()),
        }
    }

    fn refill(&mut self) -> Result<(), GenerationError> {
        while !self.source_done && self.buffer.len() < self.buffer_limit {
            // Checked between every source step, not once per refill.
            self.check_stop()?;
            match self.source.next_item()? {
                Some(item) => self.buffer.push_back(item),
                None => self.source_done = true,
            }
        }
        Ok(())
    }

    fn step(&mut self) -> Result<StreamEvent, GenerationError> {
        self.check_stop()?;
        if self.buffer.is_empty() {
            self.refill()?;
        }
        let Some(item) = self.buffer.pop_front() else {
            return Ok(StreamEvent::Done {
                finish_reason: FinishReason::Completed,
            });
        };
        if item.counts_toward_max() {
            if self.delivered >= self.token_limit {
                self.buffer.clear();
                return Ok(StreamEvent::Done {
                    finish_reason: FinishReason::MaxTokens,
                });
            }
            self.delivered += 1;
        }
        Ok(item.into_event())
    }
}

impl<S: StreamSource> EventStream for BoundedStream<S> {
    fn next_event(&mut self) -> StreamEvent {
        if self.ended {
            return StreamEvent::Error(GenerationError::StreamClosed);
        }
        let event = match self.step() {
            Ok(event) => event,
            Err(error) => StreamEvent::Error(error),
        };
        if matches!(event, StreamEvent::Done { .. } | StreamEvent::Error(_)) {
            self.ended = true;
            self.buffer.clear();
        }
        event
    }
}

/// Everything a stream delivered, as collected by [`collect_stream`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollectedStream {
    pub tokens: Vec<TokenId>,
    pub text: String,
    pub model_events: Vec<ModelEvent>,
    pub finish_reason: Option<FinishReason>,
}

/// Drain `stream` to its end. The first `Error` event is returned as `Err`.
pub fn collect_stream(stream: &mut dyn EventStream) -> Result<CollectedStream, GenerationError> {
    let mut out = CollectedStream::default();
    loop {
        match stream.next_event() {
            StreamEvent::Token(token) => out.tokens.push(token),
            StreamEvent::Text(text) => out.text.push_str(&text),
            StreamEvent::Model(event) => out.model_events.push(event),
            StreamEvent::Done { finish_reason } => {
                out.finish_reason = Some(finish_reason);
                return Ok(out);
            }
            StreamEvent::Error(error) => return Err(error),
        }
    }
}

/// Adaptor: any [`InferenceBackend`] as a [`StreamingBackend`].
///
/// `infer` still returns its whole `Vec` at open time (that is the existing contract), so
/// the wrapped output is held in memory; only the delivery is chunked and bounded.
/// `ModelEvent::Token(s)` becomes `Text` chunks of at most `chunk_chars` characters whose
/// concatenation equals `s`; `Finished` becomes `Done`; other events pass as `Model`.
#[derive(Clone, Debug)]
pub struct InferenceStreamAdapter<B> {
    backend: B,
    chunk_chars: usize,
}

impl<B: InferenceBackend> InferenceStreamAdapter<B> {
    pub fn new(backend: B, chunk_chars: usize) -> Result<Self, GenerationError> {
        if chunk_chars == 0 {
            return Err(GenerationError::InvalidLimits(
                "chunk_chars must be positive",
            ));
        }
        Ok(Self {
            backend,
            chunk_chars,
        })
    }
}

/// Lazy source behind [`InferenceStreamAdapter`]. The backend's events are held as returned
/// (the `infer` contract), but a `Token` text is cut into chunks only when pulled: memory is
/// the token text plus one chunk, and the stream's stop checks run between chunks.
struct ChunkingSource {
    events: VecDeque<ModelEvent>,
    /// Token text being chunked and the byte offset of its next unread character.
    pending: Option<(String, usize)>,
    chunk_chars: usize,
    /// Chunks cut so far (observability for tests).
    #[cfg(test)]
    chunks_cut: usize,
}

impl ChunkingSource {
    fn new(events: Vec<ModelEvent>, chunk_chars: usize) -> Self {
        Self {
            events: events.into(),
            pending: None,
            chunk_chars,
            #[cfg(test)]
            chunks_cut: 0,
        }
    }

    fn cut_chunk(&mut self) -> Option<StreamItem> {
        let (text, offset) = self.pending.as_mut()?;
        let rest = &text[*offset..];
        let end = rest
            .char_indices()
            .nth(self.chunk_chars)
            .map_or(rest.len(), |(index, _)| index);
        let chunk = rest[..end].to_owned();
        *offset += end;
        if *offset >= text.len() {
            self.pending = None;
        }
        #[cfg(test)]
        {
            self.chunks_cut += 1;
        }
        Some(StreamItem::Text(chunk))
    }
}

impl StreamSource for ChunkingSource {
    fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
        loop {
            if let Some(item) = self.cut_chunk() {
                return Ok(Some(item));
            }
            match self.events.pop_front() {
                None => return Ok(None),
                Some(ModelEvent::Token(text)) => {
                    if !text.is_empty() {
                        self.pending = Some((text, 0));
                    }
                }
                Some(ModelEvent::Finished) => {}
                Some(other) => return Ok(Some(StreamItem::Model(other))),
            }
        }
    }
}

impl<B: InferenceBackend> StreamingBackend for InferenceStreamAdapter<B> {
    fn name(&self) -> &str {
        self.backend.name()
    }

    fn open_stream(
        &self,
        request: &ModelRequest,
        options: StreamOptions,
    ) -> Result<Box<dyn EventStream>, GenerationError> {
        if options.cancel.is_cancelled() {
            return Err(GenerationError::Cancelled { at_token: 0 });
        }
        if let Some(deadline) = &options.deadline {
            deadline.check()?;
        }
        let events = self.backend.infer(request)?;
        let source = ChunkingSource::new(events, self.chunk_chars);
        Ok(Box::new(BoundedStream::new(source, options)))
    }
}

struct ByteSource {
    tokens: std::vec::IntoIter<TokenId>,
}

impl StreamSource for ByteSource {
    fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
        Ok(self.tokens.next().map(StreamItem::Token))
    }
}

/// Reference streaming backend: emits one `Token` per UTF-8 byte of the request's raw
/// text, tokenized by [`ByteTokenizer`]. A deterministic plumbing backend, not a model.
#[derive(Clone, Debug, Default)]
pub struct ReferenceByteStreamingBackend {
    tokenizer: ByteTokenizer,
}

impl ReferenceByteStreamingBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tokenizer(&self) -> &ByteTokenizer {
        &self.tokenizer
    }
}

impl StreamingBackend for ReferenceByteStreamingBackend {
    fn name(&self) -> &str {
        "reference-byte-stream"
    }

    fn open_stream(
        &self,
        request: &ModelRequest,
        options: StreamOptions,
    ) -> Result<Box<dyn EventStream>, GenerationError> {
        if options.cancel.is_cancelled() {
            return Err(GenerationError::Cancelled { at_token: 0 });
        }
        let tokens = self.tokenizer.encode(&request.raw_text).into_iter();
        Ok(Box::new(BoundedStream::new(ByteSource { tokens }, options)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Endless source that fails after `stop_after` pulls so a defect cannot exhaust memory.
    struct EndlessSource {
        pulled: usize,
        stop_after: usize,
    }

    impl StreamSource for EndlessSource {
        fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
            self.pulled += 1;
            if self.pulled > self.stop_after {
                return Err(GenerationError::StreamClosed);
            }
            Ok(Some(StreamItem::Model(ModelEvent::Finished)))
        }
    }

    /// Endless token source that fails after `stop_after` pulls (test-side stop).
    struct EndlessTokens {
        pulled: usize,
        stop_after: usize,
    }

    impl StreamSource for EndlessTokens {
        fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
            self.pulled += 1;
            if self.pulled > self.stop_after {
                return Err(GenerationError::StreamClosed);
            }
            Ok(Some(StreamItem::Token(TokenId(1))))
        }
    }

    #[test]
    fn oversized_public_max_tokens_is_clamped_by_new() {
        for max_tokens in [usize::MAX, MAX_STREAM_TOKENS + 1] {
            let options = StreamOptions {
                limits: StreamLimits {
                    max_tokens,
                    buffer_capacity: 4,
                },
                ..StreamOptions::default()
            };
            let source = EndlessTokens {
                pulled: 0,
                stop_after: MAX_STREAM_TOKENS + 1000,
            };
            let mut stream = BoundedStream::new(source, options);
            let mut delivered = 0usize;
            let last = loop {
                match stream.next_event() {
                    StreamEvent::Token(_) => delivered += 1,
                    other => break other,
                }
            };
            assert_eq!(
                last,
                StreamEvent::Done {
                    finish_reason: FinishReason::MaxTokens
                },
                "max_tokens {max_tokens}: ended after {delivered} tokens"
            );
            assert_eq!(delivered, MAX_STREAM_TOKENS);
        }
    }

    #[test]
    fn oversized_public_buffer_capacity_is_clamped_by_new() {
        for capacity in [usize::MAX, MAX_STREAM_BUFFER + 1, 0] {
            let options = StreamOptions {
                limits: StreamLimits {
                    max_tokens: 4,
                    buffer_capacity: capacity,
                },
                ..StreamOptions::default()
            };
            let source = EndlessSource {
                pulled: 0,
                stop_after: 3 * MAX_STREAM_BUFFER,
            };
            let mut stream = BoundedStream::new(source, options);
            let _ = stream.next_event();
            assert!(stream.buffered() <= MAX_STREAM_BUFFER);
            assert!(
                stream.source.pulled <= MAX_STREAM_BUFFER,
                "pulled {} items with capacity {capacity}",
                stream.source.pulled
            );
        }
    }

    #[test]
    fn limits_reject_zero_and_oversize() {
        assert!(StreamLimits::new(0, 4).is_err());
        assert!(StreamLimits::new(4, 0).is_err());
        assert!(StreamLimits::new(4, MAX_STREAM_BUFFER + 1).is_err());
        assert!(StreamLimits::new(MAX_STREAM_TOKENS + 1, 4).is_err());
        assert!(StreamLimits::new(4, 4).is_ok());
    }

    fn drain_text(source: &mut ChunkingSource) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(item) = source.next_item().unwrap() {
            if let StreamItem::Text(t) = item {
                out.push(t);
            }
        }
        out
    }

    #[test]
    fn chunking_preserves_text() {
        let mut source = ChunkingSource::new(vec![ModelEvent::Token("héllo wörld".into())], 3);
        let chunks = drain_text(&mut source);
        assert_eq!(chunks.concat(), "héllo wörld");
        assert!(chunks.iter().all(|t| t.chars().count() <= 3));
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn large_token_is_chunked_lazily_within_the_stream_bound() {
        let big = "x".repeat(2_000_000);
        let source = ChunkingSource::new(vec![ModelEvent::Token(big)], 1);
        let options = StreamOptions {
            limits: StreamLimits::new(1, 1).unwrap(),
            ..StreamOptions::default()
        };
        let mut stream = BoundedStream::new(source, options);
        assert!(matches!(stream.next_event(), StreamEvent::Text(_)));
        assert_eq!(
            stream.next_event(),
            StreamEvent::Done {
                finish_reason: FinishReason::MaxTokens
            }
        );
        assert!(
            stream.source.chunks_cut <= 3,
            "cut {}",
            stream.source.chunks_cut
        );
    }

    #[test]
    fn cancellation_stops_chunk_production() {
        let cancel = CancellationToken::default();
        let options = StreamOptions {
            limits: StreamLimits::new(1000, 4).unwrap(),
            cancel: cancel.clone(),
            ..StreamOptions::default()
        };
        let source = ChunkingSource::new(vec![ModelEvent::Token("y".repeat(100_000))], 1);
        let mut stream = BoundedStream::new(source, options);
        assert!(matches!(stream.next_event(), StreamEvent::Text(_)));
        cancel.cancel();
        assert!(matches!(
            stream.next_event(),
            StreamEvent::Error(GenerationError::Cancelled { .. })
        ));
        assert!(
            stream.source.chunks_cut <= 6,
            "cut {}",
            stream.source.chunks_cut
        );
    }

    struct Endless;
    impl StreamSource for Endless {
        fn next_item(&mut self) -> Result<Option<StreamItem>, GenerationError> {
            Ok(Some(StreamItem::Token(TokenId(1))))
        }
    }

    #[test]
    fn buffer_never_exceeds_capacity_on_endless_source() {
        let options = StreamOptions {
            limits: StreamLimits::new(100, 4).unwrap(),
            ..StreamOptions::default()
        };
        let mut stream = BoundedStream::new(Endless, options);
        for _ in 0..100 {
            assert!(matches!(stream.next_event(), StreamEvent::Token(_)));
            assert!(stream.buffered() <= 4);
        }
        assert_eq!(
            stream.next_event(),
            StreamEvent::Done {
                finish_reason: FinishReason::MaxTokens
            }
        );
        assert_eq!(
            stream.next_event(),
            StreamEvent::Error(GenerationError::StreamClosed)
        );
    }
}
