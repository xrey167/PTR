$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git missing" }
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) { throw "rustup missing" }
rustup toolchain install 1.85.0 --profile minimal --component rustfmt --component clippy
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python missing" }
Write-Host "PTR base developer prerequisites are available."
