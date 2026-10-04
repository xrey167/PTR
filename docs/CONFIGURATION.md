# Configuration Layout

Every first-class workspace area has a local `config.toml` describing its role and test location. Runtime behavior is configured through `config/default.toml` and parsed by `ptr-config`.

Precedence: defaults → file → environment → CLI. All three layers are implemented in `ptr-config`: `PtrConfig::from_path` loads and validates the file (unknown keys are refused), `apply_env` applies the `PTR_*` variables, and `CliOverrides` applies command-line values. Several keys are parsed and validated but not yet read by any runtime path; `crates/ptr-config/component.toml` is the record of what is wired. Secrets must be referenced rather than committed as plaintext.
