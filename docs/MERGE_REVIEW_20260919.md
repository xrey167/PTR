# Main integration review — 2026-09-19

## Scope and order

Review PR #9 (open architecture, shared cognitive axes and Rust conventions)
first, then PR #11 (target-aware confidence). PR #11 must be retargeted to main
only after #9 is integrated. No codebook, SemDB payload migration, A0 architecture
change or model-training claim is part of this integration.

Reviewed starting points: main `617c6655`, PR #9 `eb660eb2`, PR #11 `81c2711`.
The read-only preflight branch is tooling for this review only and is not a merge
candidate. It exports exact source snapshots and rustfmt output, with checksums;
it never pushes code or modifies main.

## Repairs before integration

- Rustfmt-normalize the affected ptr-observe files and ptr-types whitespace.
- Keep API inventory paths repository-relative for repository files, and use a
  stable crate-relative path for external/temporary crates. Previously the latter
  raised ValueError before the inventory test could run.
- Add regression checks for external paths, nested modules, temporary-root
  independence, repository-relative paths and normalized parent segments.
- Restore RUSTDOCFLAGS to the documentation command: inserting commands in PR #9
  had attached the existing environment block to the tracing-adapter test.
- Explicitly lint the optional tracing adapter and test the reference crate on
  the stated Rust 1.85 MSRV, in addition to the existing stable checks.
- Replace the boolean match in the trace filter with guarded matches! without
  suppressing the Clippy warning; preserve lazy FnMut predicate semantics.
- Implement the empty template BackendRegistry Default without requiring
  T: Default. The previous derive prevented Arc<dyn Backend> registries from
  compiling. Add a non-Default entry regression and remove an unnecessary
  explicit dereference in the template's lazy name filter; elide a redundant
  single-input lifetime in its validation helper.
- Separate the reference service's iterator/predicate lifetime from the lifetime
  of borrowed backend names. The original FnMut test exposed E0502 by reading its
  counter before using the collected names. Keep that test and add a regression
  where captured locals leave scope before the names are used. Simplify the
  explicit best-effort telemetry pattern without changing its behavior.
- Regenerate component README/status from component.toml after source changes.

The existing Rust, documentation, repository, research and security checks are
not disabled or downgraded. CI execution and review decisions are recorded in the
PR conversations against exact head/integration SHAs; this document is not a
substitute for a successful test run.

## Dependency audit scope — not a clean security result

A parsed comparison of Cargo.lock at main `617c6655` and PR #9 `eb660eb2` shows
611 identical package identities, versions, sources and checksums. The only
changed package record adds the already-present tracing crate to ptr-observe.
That dependency is optional (`tracing-adapter`); default features stay empty.
PR #11 changes no dependency manifest or lockfile.

The security audit of PR #9 in run `35411950976` reported eight vulnerability
findings affecting hickory-proto 0.25.2, protobuf 2.28.0, rustls-webpki 0.102.8 and
time 0.3.41, plus maintenance/unsoundness warnings. These exact package records
are already present in main. The deny job also fails for advisories and disallowed licenses, including
Unicode-3.0 expressions. Its policy is unchanged.
Neither PR fixes those findings. Their existence must not be represented as an
all-green repository or a production/security release approval.

Known dependency debt and a reviewed, non-regressive source increment are
separate decisions. No vulnerable package version is introduced/upgraded by
these two PRs; no advisory ignore entry, license waiver, required-check bypass,
backend enablement or branch-protection change is part of this work. A security
remediation of the affected backend dependency trees needs its own compatibility
and feature tests, rather than an unreviewed lockfile upgrade in this type step.

Remediation is tracked in [issue #12](https://github.com/xrey167/PTR/issues/12).

## Component evidence boundary

Confidence tests check representation and API behavior, not learned accuracy,
conflict resolution, source revocation closure or effect authorization. The
existing full project Definition of Done is unchanged. Complete the current
component review before starting the next architectural component.
