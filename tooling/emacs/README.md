# PTR Emacs Rust Tooling

PTR does not require Emacs. This directory provides an optional project-local layer whose commands stay identical to the repository CLI/CI workflow.

## Candidate profiles

- **minimal** — rust-mode (or rust-ts-mode) + rust-analyzer via the developer's chosen LSP client.
- **cargo-focused** — rust-mode + cargo-mode.
- **integrated** — rust-mode + rustic.
- **integrated-plus** — rustic + cargo-mode only when the overlapping Cargo UI is intentionally desired.

See the tracked evaluation in `evaluations/components/rust-editor-emacs/`.

## PTR commands

Load `ptr-rust.el` from your Emacs config to get editor-independent project commands:

- `M-x ptr-rust-check`
- `M-x ptr-rust-test`
- `M-x ptr-rust-clippy`
- `M-x ptr-rust-fmt-check`
- `M-x ptr-rust-repository-invariants`

These commands shell out to the same Cargo/Python entry points used by the repository. They do not install or require rust-mode, rustic or cargo-mode.

## Package composition

The external package layer is intentionally not auto-installed here. Use your normal Emacs package manager and choose a profile after evaluating overlap.

Do not make generated documentation, tests, formatting or release behavior depend on editor state.


## Example package profiles

These snippets are examples only; PTR does not install packages.

### Minimal rust-mode

```elisp
(use-package rust-mode
  :mode "\\.rs\\'"
  :custom
  (rust-format-on-save nil))
```

Use Eglot or lsp-mode with rust-analyzer according to your Emacs setup. Keep formatting explicit so repository-wide checks remain visible.

### rust-mode + cargo-mode

```elisp
(use-package rust-mode
  :mode "\\.rs\\'")

(use-package cargo-mode
  :after rust-mode
  :hook (rust-mode . cargo-minor-mode))
```

This profile keeps editing relatively light while adding dynamic Cargo task/test commands.

### rust-mode + rustic

```elisp
(use-package rust-mode
  :init
  (setq rust-mode-treesitter-derive t))

(use-package rustic
  :after rust-mode)
```

Use this profile when the integrated Cargo/test/Clippy/LSP/macro workflow is preferred.

## Overlap rule

Do not automatically enable both rustic's Cargo command surface and cargo-mode in the repository profile. Test that combination explicitly first, because both provide Cargo/test ergonomics and can duplicate commands/keybindings.
