#!/usr/bin/env bash
set -euo pipefail
command -v git >/dev/null || { echo "git missing"; exit 1; }
command -v rustup >/dev/null || { echo "rustup missing"; exit 1; }
rustup toolchain install 1.85.0 --profile minimal --component rustfmt --component clippy
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
echo "PTR base developer prerequisites are available."
echo "Optional GPU/protoc/backend dependencies are checked by ptrctl doctor as adapters are enabled."
