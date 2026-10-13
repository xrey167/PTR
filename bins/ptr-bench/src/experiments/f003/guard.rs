//! Containment of the product: every call into `ptr_branch` and `ptr_analytics` goes through
//! [`contained`] (directly, or through `stream::product`, or inside a closure that is).
//!
//! A triage or a certification that panics is a finding about the arbiter, not a fault of the
//! harness. A panic inside a product call is caught here, counted by the caller into the hard
//! counter `product_panics` and answered with a stand-in so that the run goes on to its end and
//! prints its judgement. A panic anywhere else (harness code) is not caught: it stays a harness
//! fault (exit status 101, an INVALID seed).

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

thread_local! {
    /// Whether this thread is inside a contained product call.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// The result of `call`, or `None` when it panicked. The caller counts the `None`.
pub fn contained<R>(call: impl FnOnce() -> R) -> Option<R> {
    let was = INSIDE.with(|flag| flag.replace(true));
    let result = catch_unwind(AssertUnwindSafe(call)).ok();
    INSIDE.with(|flag| flag.set(was));
    result
}

/// Silence the default panic message for panics inside a contained product call only. A panic
/// anywhere else is a harness fault (exit status 101) and keeps its message.
pub fn silence_contained_panics() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !INSIDE.with(Cell::get) {
            default(info);
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_caught_and_a_return_value_passes_through() {
        assert_eq!(contained(|| 7), Some(7));
        assert_eq!(contained(|| -> u8 { panic!("boom") }), None);
        // The flag is restored, so a later panic outside is not silenced.
        assert!(!INSIDE.with(Cell::get));
    }
}
