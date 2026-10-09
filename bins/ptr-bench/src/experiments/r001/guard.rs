//! Containment of the product: every call into `ptr_exec` goes through [`contained`].
//!
//! A mailbox that panics under load is a finding about the mailbox ("predictable overload
//! behavior" is false), not a fault of the harness. So a panic inside a product call is caught
//! here, counted by the caller into the hard counter `product_panics`, and answered with a
//! stand-in so that the run goes on to its end and prints its judgement. A panic anywhere
//! else (harness code) is not caught: it stays a harness fault (exit status 101, an INVALID
//! seed).

use std::panic::{catch_unwind, AssertUnwindSafe};

/// The result of `call`, or `None` when it panicked. The caller counts the `None`.
pub fn contained<R>(call: impl FnOnce() -> R) -> Option<R> {
    catch_unwind(AssertUnwindSafe(call)).ok()
}

/// Silence the default panic message for the rest of the process. The binary calls it once,
/// before the run: a mailbox that panics on every send would otherwise print one message per
/// send to stderr, and every contained panic is counted anyway.
pub fn silence_contained_panics() {
    std::panic::set_hook(Box::new(|_| {}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_caught_and_a_return_value_passes_through() {
        assert_eq!(contained(|| 7), Some(7));
        assert_eq!(contained(|| -> u8 { panic!("boom") }), None);
    }
}
