;;; ptr-rust.el --- Optional PTR Rust project commands -*- lexical-binding: t; -*-

(defgroup ptr-rust nil
  "Optional Emacs commands for the PTR Rust repository."
  :group 'tools)

(defcustom ptr-rust-cargo-program "cargo"
  "Cargo executable used by PTR project commands."
  :type 'string
  :group 'ptr-rust)

(defun ptr-rust--compile (command)
  "Run COMMAND using Emacs compilation mode."
  (compile command))

(defun ptr-rust-check ()
  "Run the PTR workspace Cargo check contract."
  (interactive)
  (ptr-rust--compile
   (format "%s check --workspace --all-targets --locked"
           ptr-rust-cargo-program)))

(defun ptr-rust-test ()
  "Run the PTR workspace test contract."
  (interactive)
  (ptr-rust--compile
   (format "%s test --workspace --locked"
           ptr-rust-cargo-program)))

(defun ptr-rust-clippy ()
  "Run the PTR workspace Clippy contract."
  (interactive)
  (ptr-rust--compile
   (format "%s clippy --workspace --all-targets -- -D warnings"
           ptr-rust-cargo-program)))

(defun ptr-rust-fmt-check ()
  "Run the PTR rustfmt check contract."
  (interactive)
  (ptr-rust--compile
   (format "%s fmt --all -- --check"
           ptr-rust-cargo-program)))

(defun ptr-rust-repository-invariants ()
  "Run PTR repository and architecture invariant checks."
  (interactive)
  (ptr-rust--compile
   (mapconcat
    #'identity
    '("python scripts/check_repo.py"
      "python scripts/check_architecture_catalog.py"
      "python scripts/check_rust_conventions.py"
      "python scripts/run_experiment.py validate"
      "python scripts/run_component_eval.py validate")
    " && ")))

(provide 'ptr-rust)
;;; ptr-rust.el ends here
