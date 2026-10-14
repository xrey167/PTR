//! What a Pod may send out of the host: which destinations and topics its link allows, and how
//! often.
//!
//! Three rules live here and nowhere else, so an executor cannot apply a weaker copy:
//!
//! - **Topics.** An [`EgressPolicy`] lists topic *filters* in MQTT syntax: levels separated by
//!   `/`, `+` for exactly one level and `#` for the remaining levels, last. A publication names a
//!   concrete topic, never a filter, so a topic carrying a wildcard is refused before it is
//!   matched. As in MQTT, a filter that starts with a wildcard does not match a `$`-prefixed
//!   topic.
//! - **No executor here binds a topic to the wire.** `TcpProtocolExecutor` and
//!   `UdpProtocolExecutor` send a request's payload unchanged and read no topic from it, so the
//!   `topic` a request declares cannot be checked against what is published. Under a topic
//!   policy, or on an MQTT link, they refuse with `EgressRefusalReason::TopicUnverifiable`.
//!   [`EgressPolicy::check_publication`] is the check for a future MQTT executor that takes the
//!   topic from the frame it encodes.
//! - **Destinations.** A host and a port are each listed or refused; the reason says which.
//! - **Rate.** A [`FrameRateLimiter`] is a token bucket: it admits a burst of `max_per_window`
//!   frames and refills at `max_per_window` per `window`. It is bounded by construction (one
//!   counter, no queue), so a Pod that floods its link is refused rather than buffered. One
//!   limiter belongs to one Pod lease: whoever holds the lease creates it and hands the same
//!   `Arc` to every executor that sends on the lease's behalf.
//!
//! A refusal names only the redacted target (see [`redact_endpoint`] and [`redact_topic`]): a hashed host with its port, or a hashed topic,
//! never anything a payload could ride in.
use crate::{EgressPolicy, PodLinkError};
use ptr_types::EgressRefusalReason;
use sha2::{Digest, Sha256};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The longest topic a publication may name, in bytes. The same bound the ledger puts on a
/// recorded refusal target.
pub const MAX_TOPIC_BYTES: usize = 256;

/// The longest window a rate limit may use. The bucket's counter is scaled by the window in
/// nanoseconds, and a bound keeps that product far from overflow.
pub const MAX_RATE_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

impl EgressPolicy {
    /// Whether this policy lists `host` and `port`. The host is checked first, so the reason
    /// names the first thing wrong.
    pub fn check_endpoint(&self, host: &str, port: u16) -> Result<(), EgressRefusalReason> {
        if !self.hosts.iter().any(|allowed| allowed == host) {
            return Err(EgressRefusalReason::HostNotAllowed);
        }
        if !self.ports.contains(&port) {
            return Err(EgressRefusalReason::PortNotAllowed);
        }
        Ok(())
    }

    /// Whether `topic` is a well-formed concrete topic that at least one listed filter matches.
    pub fn check_topic(&self, topic: &str) -> Result<(), EgressRefusalReason> {
        if !is_concrete_topic(topic) {
            return Err(EgressRefusalReason::InvalidTopic);
        }
        if self
            .topics
            .iter()
            .any(|filter| topic_filter_matches(filter, topic))
        {
            Ok(())
        } else {
            Err(EgressRefusalReason::TopicNotAllowed)
        }
    }

    /// Check one publication against the policy. A policy that lists topics governs every
    /// publication on its link, so a request that names no topic is refused rather than let
    /// through unchecked ([`EgressRefusalReason::TopicRequired`]); a policy with no topics
    /// leaves a request without a topic to the host and port checks alone.
    pub fn check_publication(&self, topic: Option<&str>) -> Result<(), EgressRefusalReason> {
        match topic {
            Some(topic) => self.check_topic(topic),
            None if self.topics.is_empty() => Ok(()),
            None => Err(EgressRefusalReason::TopicRequired),
        }
    }

    pub(crate) fn validate_topic_filters(&self) -> Result<(), PodLinkError> {
        if self
            .topics
            .iter()
            .all(|filter| is_valid_topic_filter(filter))
        {
            Ok(())
        } else {
            Err(PodLinkError::InvalidEgressPolicy)
        }
    }
}

fn has_forbidden_character(value: &str) -> bool {
    value
        .chars()
        .any(|character| character.is_control() || character.is_whitespace())
}

/// A concrete topic: non-empty, bounded, without control characters, whitespace or wildcards.
pub fn is_concrete_topic(topic: &str) -> bool {
    !topic.is_empty()
        && topic.len() <= MAX_TOPIC_BYTES
        && !has_forbidden_character(topic)
        && !topic.contains(['+', '#'])
}

/// A topic filter: bounded, without control characters or whitespace, `+` only as a whole
/// level and `#` only as the whole last level.
pub fn is_valid_topic_filter(filter: &str) -> bool {
    if filter.is_empty() || filter.len() > MAX_TOPIC_BYTES || has_forbidden_character(filter) {
        return false;
    }
    let mut levels = filter.split('/').peekable();
    while let Some(level) = levels.next() {
        match level {
            "#" if levels.peek().is_some() => return false,
            "#" | "+" => {}
            other if other.contains(['+', '#']) => return false,
            _ => {}
        }
    }
    true
}

/// Whether `filter` matches the concrete `topic`, with MQTT's rules: `+` matches exactly one
/// level (an empty one too), `#` matches every remaining level including none, and a filter
/// whose first level is a wildcard does not match a topic whose first level starts with `$`.
/// An invalid filter or topic matches nothing.
pub fn topic_filter_matches(filter: &str, topic: &str) -> bool {
    if !is_valid_topic_filter(filter) || !is_concrete_topic(topic) {
        return false;
    }
    let first_filter_level = filter.split('/').next().unwrap_or_default();
    if topic.starts_with('$') && matches!(first_filter_level, "+" | "#") {
        return false;
    }
    let mut topic_levels = topic.split('/');
    for filter_level in filter.split('/') {
        match filter_level {
            "#" => return true,
            "+" => {
                if topic_levels.next().is_none() {
                    return false;
                }
            }
            literal => {
                if topic_levels.next() != Some(literal) {
                    return false;
                }
            }
        }
    }
    topic_levels.next().is_none()
}

/// A topic as an audit trail may hold it: a digest prefix and the length, never the topic. A
/// topic can carry an identifier or a credential a path should not (`tokens/<secret>`), and
/// the record needs only to tell two refused topics apart and to be matched to a log line by
/// whoever knows the topic.
pub fn redact_topic(topic: &str) -> String {
    let digest = Sha256::digest(topic.as_bytes());
    let prefix: String = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("topic:{prefix}:{}", topic.len())
}

/// A destination fit to be recorded: `host:<16 hex of the SHA-256 prefix>:<host length>:<port>`.
/// The host is a name the request chose (and may carry a secret, e.g. a tenant or token label),
/// so it is hashed like a topic; the port is a number and stays readable. Same host, same
/// digest, so refusals still group.
pub fn redact_endpoint(host: &str, port: u16) -> String {
    let digest = Sha256::digest(host.as_bytes());
    let prefix: String = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("host:{prefix}:{}:{port}", host.len())
}

/// The budget of one Pod lease: `max_per_window` frames per `window`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRateLimit {
    pub max_per_window: u32,
    pub window: Duration,
}

/// Why a [`FrameRateLimit`] cannot be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RateLimitError {
    /// A limit of zero frames refuses everything; say so by not opening the link.
    NoFrames,
    /// A zero window has no rate.
    EmptyWindow,
    /// A window over [`MAX_RATE_WINDOW`].
    WindowTooLong { limit: Duration, actual: Duration },
}

impl FrameRateLimit {
    pub fn validate(&self) -> Result<(), RateLimitError> {
        if self.max_per_window == 0 {
            return Err(RateLimitError::NoFrames);
        }
        if self.window.is_zero() {
            return Err(RateLimitError::EmptyWindow);
        }
        if self.window > MAX_RATE_WINDOW {
            return Err(RateLimitError::WindowTooLong {
                limit: MAX_RATE_WINDOW,
                actual: self.window,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Bucket {
    /// Tokens scaled by the window in nanoseconds, so one frame costs `window_nanos` and the
    /// refill is whole-number arithmetic: no floating point, no rounding drift.
    tokens: u128,
    last: Instant,
}

/// A token-bucket limiter for the frames one Pod lease sends. Safe to share: the bucket is one
/// counter behind a lock.
#[derive(Debug)]
pub struct FrameRateLimiter {
    limit: FrameRateLimit,
    window_nanos: u128,
    capacity: u128,
    bucket: Mutex<Bucket>,
}

impl FrameRateLimiter {
    /// A limiter that starts full: a lease may burst `max_per_window` frames at once.
    pub fn new(limit: FrameRateLimit, now: Instant) -> Result<Self, RateLimitError> {
        limit.validate()?;
        let window_nanos = limit.window.as_nanos();
        let capacity = window_nanos * u128::from(limit.max_per_window);
        Ok(Self {
            limit,
            window_nanos,
            capacity,
            bucket: Mutex::new(Bucket {
                tokens: capacity,
                last: now,
            }),
        })
    }

    pub fn limit(&self) -> FrameRateLimit {
        self.limit
    }

    /// Take one frame's token at `now`, or refuse with [`EgressRefusalReason::RateLimited`]. A
    /// refusal takes nothing: a Pod that keeps pushing does not push its own recovery further
    /// away.
    pub fn try_acquire_at(&self, now: Instant) -> Result<(), EgressRefusalReason> {
        let mut bucket = self.bucket.lock().unwrap_or_else(PoisonError::into_inner);
        let elapsed = now.saturating_duration_since(bucket.last).as_nanos();
        bucket.last = bucket.last.max(now);
        bucket.tokens = bucket
            .tokens
            .saturating_add(elapsed.saturating_mul(u128::from(self.limit.max_per_window)))
            .min(self.capacity);
        if bucket.tokens < self.window_nanos {
            return Err(EgressRefusalReason::RateLimited);
        }
        bucket.tokens -= self.window_nanos;
        Ok(())
    }

    /// [`Self::try_acquire_at`] at the current instant.
    pub fn try_acquire(&self) -> Result<(), EgressRefusalReason> {
        self.try_acquire_at(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plus_matches_one_level_and_hash_matches_the_rest() {
        assert!(topic_filter_matches("sensors/+/temp", "sensors/a/temp"));
        assert!(!topic_filter_matches("sensors/+/temp", "sensors/a/b/temp"));
        assert!(!topic_filter_matches("sensors/+/temp", "sensors/temp"));
        assert!(topic_filter_matches("sensors/#", "sensors/a/b/c"));
        assert!(topic_filter_matches("sensors/#", "sensors"));
        assert!(topic_filter_matches("#", "anything/at/all"));
        assert!(!topic_filter_matches("sensors/#", "other/a"));
        assert!(topic_filter_matches("a/b", "a/b"));
        assert!(!topic_filter_matches("a/b", "a/b/c"));
        assert!(!topic_filter_matches("a/b/c", "a/b"));
    }

    #[test]
    fn a_leading_wildcard_does_not_reach_a_system_topic() {
        assert!(!topic_filter_matches("#", "$SYS/broker"));
        assert!(!topic_filter_matches("+/broker", "$SYS/broker"));
        assert!(topic_filter_matches("$SYS/#", "$SYS/broker"));
    }

    #[test]
    fn malformed_filters_and_topics_match_nothing() {
        for filter in ["a/#/b", "a+", "a/b#", "", "a b", "a/+b"] {
            assert!(!is_valid_topic_filter(filter), "{filter:?}");
            assert!(!topic_filter_matches(filter, "a/x"), "{filter:?}");
        }
        for topic in ["", "a/+", "a/#", "a b", "a\nb"] {
            assert!(!is_concrete_topic(topic), "{topic:?}");
            assert!(!topic_filter_matches("#", topic), "{topic:?}");
        }
        assert!(!is_concrete_topic(&"t".repeat(MAX_TOPIC_BYTES + 1)));
    }

    #[test]
    fn a_redacted_endpoint_hides_the_host_and_keeps_the_port() {
        let redacted = redact_endpoint("secret-tenant.example", 1883);
        assert!(redacted.starts_with("host:"), "{redacted}");
        assert!(redacted.ends_with(":21:1883"), "{redacted}");
        assert!(!redacted.contains("secret"), "{redacted}");
        assert_eq!(redacted, redact_endpoint("secret-tenant.example", 1883));
        assert_ne!(redacted, redact_endpoint("other.example", 1883));
        assert_eq!(redact_endpoint("", 1).matches(':').count(), 3);
        assert!(redact_endpoint(&"x".repeat(10_000), 9).len() < 64);
    }

    #[test]
    fn the_bucket_bursts_then_refills_at_the_configured_rate() {
        let start = Instant::now();
        let limiter = FrameRateLimiter::new(
            FrameRateLimit {
                max_per_window: 4,
                window: Duration::from_secs(1),
            },
            start,
        )
        .unwrap();
        for _ in 0..4 {
            assert_eq!(limiter.try_acquire_at(start), Ok(()));
        }
        assert_eq!(
            limiter.try_acquire_at(start),
            Err(EgressRefusalReason::RateLimited)
        );
        // 250 ms refills exactly one token at 4 per second.
        let later = start + Duration::from_millis(250);
        assert_eq!(limiter.try_acquire_at(later), Ok(()));
        assert_eq!(
            limiter.try_acquire_at(later),
            Err(EgressRefusalReason::RateLimited)
        );
    }

    #[test]
    fn the_bucket_never_holds_more_than_one_window_of_frames() {
        let start = Instant::now();
        let limiter = FrameRateLimiter::new(
            FrameRateLimit {
                max_per_window: 2,
                window: Duration::from_secs(1),
            },
            start,
        )
        .unwrap();
        let much_later = start + Duration::from_secs(3600);
        assert_eq!(limiter.try_acquire_at(much_later), Ok(()));
        assert_eq!(limiter.try_acquire_at(much_later), Ok(()));
        assert_eq!(
            limiter.try_acquire_at(much_later),
            Err(EgressRefusalReason::RateLimited),
            "an idle hour buys a burst of one window, not an hour of frames"
        );
    }

    #[test]
    fn a_refusal_takes_nothing_and_a_clock_that_runs_backwards_adds_nothing() {
        let start = Instant::now() + Duration::from_secs(10);
        let limiter = FrameRateLimiter::new(
            FrameRateLimit {
                max_per_window: 1,
                window: Duration::from_secs(1),
            },
            start,
        )
        .unwrap();
        assert_eq!(limiter.try_acquire_at(start), Ok(()));
        let earlier = start - Duration::from_secs(5);
        assert_eq!(
            limiter.try_acquire_at(earlier),
            Err(EgressRefusalReason::RateLimited)
        );
        assert_eq!(
            limiter.try_acquire_at(start + Duration::from_secs(1)),
            Ok(())
        );
    }

    #[test]
    fn unusable_limits_are_refused_at_construction() {
        let now = Instant::now();
        assert_eq!(
            FrameRateLimiter::new(
                FrameRateLimit {
                    max_per_window: 0,
                    window: Duration::from_secs(1)
                },
                now
            )
            .unwrap_err(),
            RateLimitError::NoFrames
        );
        assert_eq!(
            FrameRateLimiter::new(
                FrameRateLimit {
                    max_per_window: 1,
                    window: Duration::ZERO
                },
                now
            )
            .unwrap_err(),
            RateLimitError::EmptyWindow
        );
        let too_long = MAX_RATE_WINDOW + Duration::from_secs(1);
        assert_eq!(
            FrameRateLimiter::new(
                FrameRateLimit {
                    max_per_window: 1,
                    window: too_long
                },
                now
            )
            .unwrap_err(),
            RateLimitError::WindowTooLong {
                limit: MAX_RATE_WINDOW,
                actual: too_long
            }
        );
    }
}
