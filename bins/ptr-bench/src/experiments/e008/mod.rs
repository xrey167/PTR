//! Experiment E008, split-pod extensions: the harnesses of its parts, one subcommand
//! (`ptr-bench split-pod-extensions ...`).
//!
//! * Part A (CPU, no network): [`part_a`], the input lineage and raw ancestry, the full
//!   upstream chain revalidation, the mode and taint namespaces, the never-persisted
//!   downstream cache and the 45-cell invalidation coverage matrix. `split-pod-extensions
//!   <seed>` and `split-pod-extensions <cases> <seed>` (the mutation checker's form).
//! * Part B (CPU and loopback iroh, the `podwire-experiments` feature): `part_b`, the
//!   chunked frames between a real `PodHost` and `PodClient`, its own subcommand
//!   (`ptr-bench split-pod-loopback <seed>`).
//! * Part C (a GPU or a real model) has no harness here.
//!
//! E008 is `planned`; nothing in these harnesses is frozen.

pub mod part_a;
#[cfg(feature = "podwire-experiments")]
pub mod part_b;

/// Run the harness the arguments after `split-pod-extensions` select. Today that is Part A
/// for every form.
pub fn run(args: &[String]) {
    part_a::run(args);
}
