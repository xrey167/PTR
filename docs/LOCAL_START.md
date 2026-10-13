# Local start: from "the machine is on" to "the head session is running"

One page, in order. Every step ends with a check; do not continue past a failed one. The target
is the owner's desktop: Ubuntu 24.04 with **two RTX 3090** (24 GiB each, 48 GiB in total,
PCIe only, no NVLink bridge assumed): profiles [`hardware/dual-rtx3090.toml`](../hardware/dual-rtx3090.toml)
(the machine) and [`hardware/rtx3090.toml`](../hardware/rtx3090.toml) (one card, 24 GiB, fp32, fp16,
bf16, int8; no fp8). `DESKTOP_SETUP.md` was written for one RTX 4090; its section 10 lists what differs.
This page only sequences the other runbooks and does not repeat them:
[`DESKTOP_SETUP.md`](DESKTOP_SETUP.md) (OS, driver, `make desktop-setup`, Hugging Face),
[`GPU_HANDOVER.md`](GPU_HANDOVER.md) (the GPU run plan and owner decisions),
[`HOSTING.md`](HOSTING.md) (GitHub primary, GitLab alternative), [`CLAUDE_TOOLING.md`](CLAUDE_TOOLING.md) (plugins),
[`agents/KICKOFF.md`](agents/KICKOFF.md) (the prompts for step f). Nothing here has run on this
GPU hardware; the first real run is the first real test. Commands marked "(Windows)" are for the
case that the repository first arrives on a Windows machine (step g).

Secrets rule for the whole page: a token, password or connection string is typed into the
terminal session or a credential manager and never written into a repository file, a log, a
decision note or a prompt. This page names environment variables by name only.

## a. Get the repository

The repository comes from GitHub, whose `main` carries waves 12 to 14 since PR #132 (`git clone https://github.com/xrey167/PTR.git ~/PTR`), or as git bundles (wave 12, then wave 13) or from GitLab.

**Bundles.** Apply wave 12 first, then wave 13. Each wave comes with an apply script
(`apply-ptr-wave12.sh` / `apply-ptr-wave13.sh` on Ubuntu, `.ps1` on Windows). Read the script
before running it, and verify the checksums that were delivered with the bundles:

```bash
mkdir -p ~/ptr-import && cd ~/ptr-import          # the bundles and scripts are copied here
sha256sum -c *.sha256 2>&1 | tail                 # every line must say OK; use the checksum file(s) delivered with the bundles
git bundle verify <wave-12 bundle>                # "is okay"
bash apply-ptr-wave12.sh                          # then:
bash apply-ptr-wave13.sh
```

If a wave arrives as split parts with a `SHA256SUMS` file, the generic helper does the same
job: `python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py join <dir>`
then `apply <dir>/repo.bundle` (it writes only `import/*` branches; merge explicitly). The file
names of the wave bundles and what the apply scripts do internally are not described in the
repository; the scripts' own header is authoritative.

**GitLab.** Setup, token handling and the publish script are in `HOSTING.md` (sections 3 and 6).

```bash
git clone https://gitlab.com/<user>/ptr.git ~/PTR      # token goes to the credential helper, never into the URL
```

**Check.** In `~/PTR` (use the branch the apply script created or the cloned `main`):

```bash
git merge-base --is-ancestor 941f5ddb HEAD && echo wave12-ok   # wave 12 tip is in your history
git merge-base --is-ancestor a9b4a453 HEAD && echo wave13-ok   # wave 13 tip (final evidence pass) is in your history
test -f AGENTS.md -a -d scripts/freeze && echo wave14-ok       # wave 14 applied (AGENTS.md, scripts/freeze/)
git tag -l '*-prereg' | wc -l   # 32 at wave 13; the list includes q003v3-prereg, c001-prereg, r007-prereg
git status --short          # empty
```

If any of the three checks prints nothing (exit code 1), stop: that wave was not applied. Later waves on top are fine.

## b. Toolchains

`rust-toolchain.toml` pins 1.99.0 (also the workspace MSRV). `model/burn-a0` is a separate
workspace that also needs 1.95.0 (`make a0`).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # skip if `make desktop-setup` will do it (step c)
rustup toolchain install 1.99.0 --profile minimal -c rustfmt -c clippy
rustup toolchain install 1.95.0 --profile minimal
rustup update stable
rustc +stable --version       # must be 1.99 or newer
cd ~/PTR && rustc --version   # 1.99.0 (the pin)
```

**Why `stable` matters.** The `make ci-*` recipes call `cargo +stable` literally. A `stable`
older than 1.99 (the earlier shared host had 1.97) fails them before compiling; that is an
environment finding, not a product bug (`STATUS.md`, "Findings of the final CPU wave").
Fix it with `rustup update stable`. If you must keep an old `stable`, run the recipe lines with
`+1.99.0` by hand (the targets `ci-rust-msrv`, `ci-state-postgres` and `make a0 A0_STABLE=1.99.0`
already take or use a version). `make release-package RELEASE_TOOLCHAIN=1.99.0` is the same idea.

## c. Machine setup, GPU, Hugging Face, PostgreSQL

1. OS, Secure Boot and driver 580 (`nvidia-driver-580-open`): `DESKTOP_SETUP.md` sections 1 and 2.
   Check: `nvidia-smi -L` lists **two** "NVIDIA GeForce RTX 3090" lines (indices 0 and 1) and
   `nvidia-smi` shows driver 580.x. Record the exact
   driver version; M016 and E006 pin `580.65.06` (GPU_HANDOVER OD-6).
2. `make desktop-setup` (idempotent, never runs `sudo`; `scripts/setup_desktop.sh --dry-run`
   previews). It builds `training/.venv` with the locked CUDA torch wheel and installs `ptr_py`
   into it. Open a new shell afterwards.
3. `make gpu-doctor`, stricter: `python3 scripts/gpu_doctor.py --require-gpu`. Must exit 0
   (it warns below 60 GB free). On this machine expect a `gpu 0` and a `gpu 1` line (name, MiB,
   `sm_86`), a summary line `gpu` reading "2x NVIDIA GeForce RTX 3090, N MiB each", a torch probe
   (allocation and matmul, bf16) on every device, and a `hardware profile` line that is OK and
   names `hardware/dual-rtx3090.toml` (single-card runs: `hardware/rtx3090.toml`). A card the doctor
   does not know (not a 3090 or 4090) is reported as an untested profile with a WARN.
4. Hugging Face: `training/.venv/bin/hf auth login` (read token typed at the prompt) and the
   pinned downloads of `GPU_HANDOVER.md` section 1.3 (about 14 GB for OLMoE). For frozen runs keep
   the default cache location; if the disk is small, make `~/.cache/huggingface` a symlink to the
   big disk instead of exporting `HF_HOME` (`GPU_HANDOVER.md` section 1.1 item 8).
5. Local PostgreSQL 16 with pgvector, port **5433**, for the `ptr-pg` tests (`make ci-state-postgres`)
   and the Q003 follow-up. Install through the distribution or PGDG packages (this installs
   system packages, so it is an owner action), create a disposable database and role for tests
   only, and give the Makefile the connection through two environment variables, set in the
   shell and never written to a file:

   ```bash
   read -rs PGPW; export PGPASSWORD="$PGPW"       # typed, not echoed
   export PTR_PG_TEST_DSN="host=127.0.0.1 port=5433 user=<test-role> dbname=<test-db> password=$PGPW"
   export PTR_PG_EXPERIMENT_DSN="$PTR_PG_TEST_DSN"
   ```

   `PTR_PG_TEST_DSN` serves the `ptr-pg` tests, `PTR_PG_EXPERIMENT_DSN` the `ptr-bench`
   experiment steps. The Makefile defaults point at port 5432 (CI's disposable container), so the
   exports are required. Check: `psql "$PTR_PG_TEST_DSN" -c 'select 1'` prints one row. Do not
   `echo` the variables, do not paste them into a prompt or a decision note.

## d. Baseline verification before any new work

Run on the clean checkout, in this order, and record the wall time of each in
`~/ptr-head/baseline.md` (outside the repository; step e). Any red item is reported first and is
not fixed silently; a failure on a commit that was green elsewhere points at the environment.

```bash
export PYTHONDONTWRITEBYTECODE=1
cd ~/PTR
time make check                                                         # cargo check --workspace --all-targets --locked
time make test                                                          # cargo test --workspace --locked
time make fmt                                                           # root workspace, model/burn-a0, evaluations/bench/kit
time cargo clippy --workspace --all-targets --locked -- -D warnings     # not `cargo xclippy`: it has no --locked
time make docs-check                                                    # generated component docs fresh
time make repo-check                                                    # check_repo, MSRV alignment, contract citations
time python3 scripts/run_experiment.py validate
time python3 scripts/check_research_gates.py
time PYTHONPATH=training/src make python-test
time make py-bindings                                                   # needs Python >= 3.11; builds bindings/ptr-py/.venv
time make vcw-demo                                                      # builds ptrd, runs scripts/vcw_demo.py
time BASE=<commit> make ci-local                                        # all CI jobs incl. feature backends; needs step b and PostgreSQL
```

Notes for the last line: `ci-repository-invariants` needs `origin/main` or `BASE=<commit>`;
use the wave-12 tip `941f5ddb` if there is no remote. `ci-state-postgres` needs the exports of
step c. A single job: `make ci-quality`, `make ci-ledger-raft-rs`, and so on (names in the
`Makefile`). `make ci-local` does not cover `make a0` (run it separately), the Windows leg, the
Python 3.11 leg, `security.yml` or `sdk.yml`. Candle CUDA is compiled by no workflow, and
`wintun-backend` only on CI's Windows leg (`CLAUDE.md`).

**Check.** Everything exits 0, or the exceptions are written down with the exact error. Known
local-only causes: `stable` older than 1.99 (step b), missing PostgreSQL variables (step c), no
`origin/main` for the metadata gate, `Cargo.lock` drift in an excluded workspace (fix is a
reviewed lock update, not `--locked` removal).

## e. Claude Code, plugins, working layout

```bash
curl -fsSL https://claude.ai/install.sh | bash       # official installer, then a new shell
cd ~/PTR && claude                                    # first start: /login with your Claude account
```

In the repository root (the `./` matters; a bare `.` is rejected, `CLAUDE_TOOLING.md`):

```bash
claude plugin marketplace add ./
claude plugin install ptr-dev@ptr
claude plugin install ptr-mods@ptr      # optional: the /runs pane and status line
claude plugin list                      # both show "Read from: <repo>/.claude/plugins/..."
```

`ptr-dev` also brings the head's helper subagents `ptr-channel` (coding), `ptr-rereviewer`
and `ptr-fact-checker` (reviews), besides the three read-only reviewers; check with `/agents`
inside a session. The repository root `AGENTS.md` holds the binding working method (roles,
rules); the freeze tooling is described in `docs/FREEZE_PROCEDURE.md` and lives in
`scripts/freeze/`. If either is missing, the checkout is older than the kickoff material:
stop and re-check step a.

**Permission mode.** Start in the default mode. `.claude/settings.json` pre-approves only commands
that run no repository code (`git status/diff/log/show`, `nvidia-smi`, read-only `glab`). On your own
machine, with a checkout you trust, copy `.claude/settings.local.example.json` to
`.claude/settings.local.json` (git-ignored, personal) to also pre-approve the checks, the validators,
`make gpu-doctor`, `make smoke-gpu` and `make olmoe-sft-smoke`; never do that for a checkout of
someone else's branch, because those commands run the checkout's Makefile, scripts and build code.
Everything else asks. For a long head session you may allow edits
without prompts (`claude --permission-mode acceptEdits`); keep shell commands, pushes and
package installs on ask. Do not run with permission checks disabled on a machine that holds
tokens or credentials. The head never pushes, force-pushes, installs system packages or changes
hosting settings without a question to you (`GPU_HANDOVER.md` section 7, rules 5 and 6).

**Layout (all outside the repository, nothing from here is committed).**

| Path | Purpose |
| --- | --- |
| `~/PTR` | the integration checkout; the head works and merges here |
| `~/ptr-wt/<package>` | one `git worktree` per channel: `git worktree add ~/ptr-wt/<package> -b claude/<package> <base-sha>`; exclusive file sets per channel |
| `~/ptr-head/decisions.md` | the head's decision log: date, decision, alternatives, why, how to reverse |
| `~/ptr-head/baseline.md` | step d timings and results |
| `~/ptr-head/handback/` | channel and reviewer reports |

**Resources for the two-3090 desktop.** Cargo uses all cores by default; with up to four channels
building at once cap each with `CARGO_BUILD_JOBS=<cores/4>` rather than serialising. Every
worktree has its own `target/` (several tens of GB each, `DESKTOP_SETUP.md` step 1): remove a
channel's worktree and `target/` when it is merged, and keep at least **6 GB** free at all times
(the hard floor for one heavy build, `evaluations/BATCH_DECISIONS.md`) and preferably the 60 GB
the doctor asks for. Heavy stacks (rocksdb, datafusion, torch) one at a time per disk. Never
run a CPU build wave while a GPU run is in progress: it distorts timings. Two GPU runs may run
at the same time, one per card (`CUDA_VISIBLE_DEVICES`), but not two frozen seed runs: see `DESKTOP_SETUP.md` section 10. Note that
`scripts/mutation_check.py` runs `<root>/target/release/<package>` and ignores `CARGO_TARGET_DIR`
(`STATUS.md`), so a freeze run needs the default target directory.

## f. Start the head session

Open `docs/agents/KICKOFF.md`. In `~/PTR`:

1. `claude`, then paste the block **Head prompt** as the first message.
2. Set the goal with `/goal` and paste the block **Goal**.
3. The head first verifies and reports the baseline of step d (it may reuse your
   `~/ptr-head/baseline.md`), then works Stage 0 to Stage 4. It replies in German; files stay English.

The goal is staged so that you can stop after any stage. Smoke runs are pipeline evidence only;
research claims exist only for experiments frozen and run per `docs/FREEZE_PROCEDURE.md`.

## g. Windows notes

- Apply the bundles with the `.ps1` scripts (`apply-ptr-wave12.ps1`, then `apply-ptr-wave13.ps1`;
  PowerShell, `py` instead of `python3`). Set `git config --global core.autocrlf false`
  first: `.gitattributes` forces LF and the freeze digests hash file bytes.
- Verify as in step a (`Get-FileHash -Algorithm SHA256`, `git log -1`, `git tag -l "*-prereg"`).
- Work in **WSL2 Ubuntu** with the repository in the Linux filesystem (`~/PTR`, not `/mnt/c`),
  or move it to the Ubuntu desktop (a new bundle, `HOSTING.md` section 8). Steps b, c(5), d, e, f
  then run unchanged inside WSL2. The Windows `rust-stable` CI leg and the wintun backend are
  not reproduced by any local target.
- **GPU work only on the Ubuntu desktop.** `DESKTOP_SETUP.md` and `GPU_HANDOVER.md` define the
  target as native Ubuntu 24.04, not WSL2; `make desktop-setup`, `make gpu-doctor`,
  `make smoke-gpu`, `make olmoe-sft-smoke` and every GPU experiment run there.
