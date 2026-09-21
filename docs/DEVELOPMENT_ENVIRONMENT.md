# PTR Development Environment

PTR's development contract is editor-independent.

## Authoritative developer commands

The repository/CI commands remain authoritative:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
python scripts/check_repo.py
python scripts/check_architecture_catalog.py
python scripts/check_rust_conventions.py
python scripts/run_experiment.py validate
python scripts/run_component_eval.py validate
```

Editor integrations may wrap these commands but may not redefine them.

## Every Cargo package belongs to a workspace

`cargo fmt --all`, `cargo test --workspace` and `cargo clippy --workspace` are
authoritative only over the packages the workspace knows about, and cargo reports
nothing at all for a directory it was never told about. So a package that matches
no `members` glob is not lightly linted — it is built, tested, linted and formatted
by *nothing*, silently, for as long as it exists.

That is what `members = ["crates/ptr-*", "bins/ptr-*"]` did to `bins/ptrctl` and
`bins/ptrd`: the second glob needs the hyphen, neither name has one, and neither
appeared in `exclude` either. `bins/ptrctl/tests/doctor.rs` had never run.

Two rules follow, and `scripts/check_repo.py` enforces the second:

- **A package a glob does not reach is named outright**, not accommodated by
  widening the glob. Widening it is how the next package inherits a decision
  nobody made.
- **Every directory under `crates/` or `bins/` holding a `Cargo.toml` matches a
  `members` entry or an `exclude` entry.** Being outside the workspace is allowed;
  being outside it by accident is not. `scripts/tests/test_check_repo.py` drives
  the rule against the original defect's exact shape.

One consequence of the first fix, recorded here rather than left to be found:
`bins/ptrctl/tests/doctor.rs` now runs, and it asserts on repository *layout* —
`Cargo.toml`, `Cargo.lock`, `config/default.toml`, `training/uv.lock`, the three
registries and `docs/components/STATUS.md`. Moving or renaming any of those now
fails a test, which was not true while the binary was outside the workspace.

## Every package is linted at the MSRV it declares

`clippy.toml` wins over `rust-version`, and clippy finds it by walking upward from
a crate. So a nested workspace under a root that carries a `clippy.toml` is linted
at the *root's* MSRV however much newer its own is, and every lint whose
suggestion needs a newer compiler is quietly suppressed.

`model/burn-a0` requires 1.95 and was linted at the root's 1.85.0. Clippy is not
silent about this — it prints *"the MSRV in `clippy.toml` and `Cargo.toml` differ;
using `1.85.0` from `clippy.toml`"* on every run of that workflow — but the notice
is not a named lint, so `-D warnings` cannot promote it and the job exits 0. A
printed warning that nothing enforces is indistinguishable from no warning.

`scripts/check_msrv_alignment.py` requires that the `clippy.toml` governing a
package declares that package's `rust-version`. Two details are load-bearing and
both are tested in `scripts/tests/test_check_msrv_alignment.py`:

- it compares **semantically**, because the root declares `rust-version = "1.85"`
  against `msrv = "1.85.0"`, and a string comparison would fail on the root
  workspace the day the check is added;
- it resolves `rust-version.workspace = true`, which every crate and both binaries
  use, so it compares the version that actually governs each package.

A package that declares no `rust-version` — `fuzz`, `templates/rust-crate` — makes
no claim, so there is nothing to contradict and nothing is reported.

## Existing environments

- Dev Container: `.devcontainer/`
- VS Code extension hints: rust-analyzer + TOML support
- Emacs optional profile: `tooling/emacs/`

## Emacs Rust candidates

PTR tracks rust-mode, rustic and cargo-mode under `integrations/editor/` and compares their profiles under `evaluations/components/rust-editor-emacs/`.

No editor package is a runtime, build or release dependency.
