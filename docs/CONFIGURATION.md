# Configuration Layout

Every first-class workspace area has a local `config.toml` describing its role and test location. Runtime behavior is configured through `config/default.toml` and parsed by `ptr-config`.

Precedence target: defaults → file → environment → CLI. Only the file layer is implemented now. Secrets must be referenced rather than committed as plaintext.
