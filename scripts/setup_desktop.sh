#!/usr/bin/env bash
# Set up a fresh Ubuntu 24.04 desktop with NVIDIA GPUs (reference hosts: two RTX 3090 or one
# RTX 4090; the setup is the same, the driver sees every card) so PTR GPU
# experiments can run. Idempotent: every step checks first and skips what is done.
#
#   scripts/setup_desktop.sh [--dry-run] [--yes]
#
#   --dry-run  print every step and the commands it would run; change nothing
#   --yes      do not ask before installing rustup and uv
#
# Never runs sudo. The NVIDIA driver is the one thing it does not install: if
# nvidia-smi is missing it prints the exact commands and stops (see docs/DESKTOP_SETUP.md
# for Secure Boot / MOK enrolment, which needs a reboot and a console).
set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TOTAL=7
VENV="training/.venv"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

step() { printf '\nSTEP %s/%s: %s\n' "$1" "$TOTAL" "$2"; }
note() { printf '  %s\n' "$*"; }
# Run a command, or only print it under --dry-run.
run() {
  printf '  + %s\n' "$*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}
confirm() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ "$DRY_RUN" -eq 1 ]; then note "(would ask: $1 [y/N], --yes skips)"; return 0; fi
  if [ ! -t 0 ]; then echo "  $1 needs confirmation and stdin is not a terminal; rerun with --yes" >&2; return 1; fi
  read -r -p "  $1 [y/N] " reply
  case "$reply" in y|Y|yes|YES) return 0 ;; *) echo "  declined" >&2; return 1 ;; esac
}
dry_note() { [ "$DRY_RUN" -eq 1 ] && note "(dry run: nothing changed)"; return 0; }

[ "$DRY_RUN" -eq 1 ] && echo "DRY RUN: printing steps only, nothing is installed or changed."

# --- 1. OS ---------------------------------------------------------------------
step 1 "check the OS is Ubuntu and the base tools exist"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
else
  ID=unknown; VERSION_ID=unknown
fi
note "detected: ${ID:-unknown} ${VERSION_ID:-unknown}"
if [ "${ID:-}" != "ubuntu" ]; then
  if [ "$DRY_RUN" -eq 1 ] || [ "${PTR_ALLOW_NON_UBUNTU:-0}" = "1" ]; then
    note "WARNING: not Ubuntu (this runbook targets Ubuntu 24.04 LTS)"
  else
    echo "  not Ubuntu; this script targets Ubuntu 24.04 LTS. Set PTR_ALLOW_NON_UBUNTU=1 to override." >&2
    exit 1
  fi
elif [ "${VERSION_ID:-}" != "24.04" ]; then
  note "WARNING: Ubuntu ${VERSION_ID}, tested on 24.04"
fi
missing=()
for tool in git curl cc pkg-config; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ "${#missing[@]}" -gt 0 ]; then
  note "missing base tools: ${missing[*]}"
  note "install them yourself (this script never runs sudo):"
  note "  sudo apt update && sudo apt install -y git curl build-essential pkg-config"
  [ "$DRY_RUN" -eq 1 ] || exit 1
fi

# --- 2. NVIDIA driver ------------------------------------------------------------
step 2 "check the NVIDIA driver (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  note "nvidia-smi works: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
else
  note "nvidia-smi is missing or not working. Install the driver (580, open kernel module) yourself:"
  note "  sudo apt update"
  note "  sudo apt install -y nvidia-driver-580-open"
  note "  sudo reboot"
  note "With Secure Boot on, enrol the MOK key when prompted at reboot (docs/DESKTOP_SETUP.md)."
  note "Then verify with: nvidia-smi   and rerun: make desktop-setup"
  if [ "$DRY_RUN" -eq 1 ]; then
    note "(dry run: continuing to list the remaining steps)"
  else
    exit 1
  fi
fi

# --- 3. Rust ---------------------------------------------------------------------
step 3 "install rustup and the toolchain pinned in rust-toolchain.toml"
CHANNEL="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml | head -n1)"
[ -n "$CHANNEL" ] || { echo "  cannot read channel from rust-toolchain.toml" >&2; exit 1; }
note "pinned channel: $CHANNEL"
if command -v rustup >/dev/null 2>&1; then
  note "rustup present"
else
  note "rustup missing"
  confirm "Install rustup from https://sh.rustup.rs?" || exit 1
  run bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal"
fi
# rustup creates ~/.rustup on first use, so a dry run does not ask it anything.
if [ "$DRY_RUN" -eq 0 ] && rustup toolchain list 2>/dev/null | grep -q "^$CHANNEL"; then
  note "toolchain $CHANNEL already installed"
else
  run rustup toolchain install "$CHANNEL" --profile minimal --component rustfmt --component clippy
fi

# --- 4. uv -----------------------------------------------------------------------
step 4 "install uv (official installer)"
if command -v uv >/dev/null 2>&1; then
  note "uv present: $(uv --version)"
else
  note "uv missing"
  confirm "Install uv from https://astral.sh/uv/install.sh?" || exit 1
  run bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# --- 5. training venv --------------------------------------------------------------
step 5 "create or sync $VENV (Python 3.11, extra moe, locked CUDA wheels)"
note "needs the network for the first sync (torch and CUDA wheels are several GB)"
run uv sync --project training --python 3.11 --extra moe --locked

# --- 6. ptr_py -------------------------------------------------------------------
step 6 "build ptr_py into $VENV with maturin"
MATURIN_VERSION="$(sed -n 's/^MATURIN ?= *//p' Makefile | head -n1)"
run uv pip install --python "$VENV/bin/python" "maturin==${MATURIN_VERSION:-1.15.0}"
note "maturin develop runs with $VENV active so the module lands in that venv"
if [ "$DRY_RUN" -eq 1 ]; then
  # The dry run prints the command; $PATH stays literal on purpose.
  # shellcheck disable=SC2016
  printf '  + VIRTUAL_ENV=%s/%s PATH=%s/%s/bin:$PATH maturin develop --uv --locked -m bindings/ptr-py/Cargo.toml\n' \
    "$ROOT" "$VENV" "$ROOT" "$VENV"
else
  VIRTUAL_ENV="$ROOT/$VENV" PATH="$ROOT/$VENV/bin:$PATH" \
    maturin develop --uv --locked -m bindings/ptr-py/Cargo.toml
fi

# --- 7. doctor -------------------------------------------------------------------
step 7 "run the GPU doctor"
if [ "$DRY_RUN" -eq 1 ]; then
  printf '  + python3 scripts/gpu_doctor.py --require-gpu\n'
  dry_note
else
  python3 scripts/gpu_doctor.py --require-gpu
  echo
  echo "Setup finished. Next: make smoke-gpu, then make olmoe-sft-smoke (docs/DESKTOP_SETUP.md)."
fi
