# Hosting: GitLab as the primary remote

**Kurzfassung (Deutsch).** Der GitHub-Account `xrey167` ist gesperrt (vermutlich wegen vieler
automatisierter Aktionen: viele PRs, Kommentare, CI-Wiederholungen). Bis das geklärt ist, ist
GitLab.com (privates Projekt `ptr`) der primäre Remote; `.github/workflows` bleiben für eine
spätere Rückkehr unverändert. Lokal arbeiten und Git-Bundles (`ptr-dev:ptr-offline-sync`)
funktionieren immer. Ablauf: GitLab-Konto, privates leeres Projekt, Personal Access Token
(`write_repository` und `api`, mit Ablaufdatum) nur lokal im Credential Manager speichern,
dann `scripts/publish_remote.py` zuerst mit `--dry-run`, danach echt. CI läuft über
`.gitlab-ci.yml` (gleiche Job-Namen wie bei GitHub, jeweils `make ci-<job>`); für GPU-Jobs wird
der Desktop als Projekt-Runner mit dem Tag `ptr-gpu` registriert. Damit keine neue Sperre
droht, gelten Regeln für wenig Automatisierung (Abschnitt 6): Pushes bündeln, keine
Schleifen aus Kommentaren oder Re-Runs, wenige Pipelines pro Stunde. Der Rest ist Englisch,
wie die übrigen Docs.

## 1. Status

- The GitHub account `xrey167` is suspended. The probable trigger is a high volume of
  automated actions (many PRs, comments, CI re-runs). The cause is not confirmed.
- `.github/workflows/` stays untouched so a return to GitHub (or a Forgejo/Gitea runner, see
  section 7) needs no rework.
- GitLab is the primary remote for now. The repository is private research code, dual-licensed
  Apache-2.0 OR MIT (`LICENSE`, `LICENSE-APACHE`, `LICENSE-MIT`).
- Working locally and moving the repository with git bundles always works and needs no
  host: see the skill `ptr-dev:ptr-offline-sync` and `docs/CLAUDE_TOOLING.md`.

## 2. Options compared

For cost and limits, check the provider's current pricing page; this document states no
numbers because they change.

| Option | CI | Cost / limits | Control | Risk |
|---|---|---|---|---|
| GitLab.com (recommended) | GitLab CI (`.gitlab-ci.yml`), shared runners plus your own project runners | check the provider's current pricing page (shared-runner minutes and storage are limited) | hosted, but private projects and own runners are available | account moderation can also happen here, so keep automation low (section 6) and keep bundles as backup |
| Self-hosted Forgejo/Gitea on the desktop | Forgejo Actions, largely compatible with `.github/workflows` (verify per workflow) | own hardware and electricity, no provider limits | full | desktop is a single point of failure; you do the updates, backups and network exposure. Use as second remote, not the only one |
| Codeberg (Forgejo) | Forgejo Actions or Woodpecker where offered | check the provider's current pricing page | hosted | intended for free and open-source projects. This repo is licensed Apache-2.0 OR MIT, but it is private research code; confirm Codeberg's current terms permit private or not-yet-published work before using it |
| Bitbucket / Azure DevOps | Bitbucket Pipelines / Azure Pipelines (both need new pipeline files) | check the provider's current pricing page | hosted | all CI would have to be rewritten; no benefit over GitLab here |
| Local only (bundles) | none, run `make ci-local` | free | full | no off-machine copy unless you back up bundles yourself (section 8) |

## 3. GitLab setup, step by step

1. Create an account on gitlab.com.
2. Create a **private blank project** named `ptr`. Do not add a README, `.gitignore` or
   license, so the first push is not rejected as non-fast-forward.
3. Create a Personal Access Token (User settings, Access tokens) with scopes
   `write_repository` and `api`, and **an expiry date**.
4. Store the token **only locally**. Never paste it into a chat, a file, an issue or a commit,
   and never put it into a remote URL.
   - Windows: Git Credential Manager prompts on the first push. Username is your GitLab
     username, password is the token.
   - Linux: `git config --global credential.helper store` (plain text in `~/.git-credentials`,
     mode 600) or a libsecret helper if installed.
   - CLI: `glab auth login` (install `glab` first; it keeps the token in its own config).
5. Dry run, inspect the output, then publish. On Windows use `py` instead of `python3`:

   ```bash
   python3 scripts/publish_remote.py --remote-url https://gitlab.com/<user>/ptr.git --dry-run
   python3 scripts/publish_remote.py --remote-url https://gitlab.com/<user>/ptr.git
   ```

   The script performs one `git push`, never forces, refuses URLs with embedded credentials
   and exits with code 3 when refs are rejected. Other options: `--name gitlab`,
   `--branches main,claude/*`, `--tags`, `--yes`, `--set-url`, `--set-upstream-main`. See
   `python3 scripts/publish_remote.py --help` once the script is in your checkout.
6. In GitLab (Settings, Repository): set the default branch to `main` and protect `main`
   (merge only through MRs, force push not allowed). Under Protected tags, protect `v*`
   (Maintainers only): the release jobs run only for protected `v*` tags on a commit of `main`.

## 4. CI on GitLab

`.gitlab-ci.yml` mirrors `.github/workflows/ci.yml`: the jobs have the same names and each
runs `make ci-<job>` (`make ci-local` covers the fast ones locally).

| Jobs | When |
|---|---|
| `quality`, `rust-stable`, `rust-msrv`, `python-training`, `repository-invariants` | every push and every MR |
| backend jobs (`lifecycle-failpoints`, `ledger-raft-engine`, `ledger-raft-rs`, `state-turso`, `state-postgres`, `network-iroh`, `cluster-wire`, `execution-wire`, `pod-wire`, `storage-tiers`) | automatic on the default branch, manual in MRs |
| `burn-a0` | `make a0`; automatic on the default branch, manual elsewhere |
| `security` | scheduled or manual (cargo-audit, cargo-deny) |
| `py-bindings` | manual |
| `gpu-smoke` | manual, needs a runner tagged `ptr-gpu` |
| `release-package`, `release` | on `v*` tags: `make release-package`, then a GitLab release with the assets |

`state-postgres` uses a PostgreSQL service container. The Windows leg of `rust-stable` is
not mirrored. Read `.gitlab-ci.yml` for the authoritative rules; this table can lag.

Shared-runner minutes are limited (see the provider's pricing page), so register the desktop
as a **project runner**:

```bash
# Ubuntu 24.04 desktop. Add GitLab's official apt repository first, following
# https://docs.gitlab.com/runner/install/linux-repository.html
sudo apt install -y gitlab-runner
# Docker engine: https://docs.docker.com/engine/install/ubuntu/
# NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
sudo gitlab-runner register        # interactive; or pass flags
```

Create the runner in GitLab (Settings, CI/CD, Runners, new project runner) and use the
token it shows once; do not paste it anywhere else. Registration values:

Register two runners (or one runner with two `[[runners]]` entries):

- **Docker runner** for the Rust and Python jobs: executor `docker`, default image `rust:1-bookworm`,
  optional tag `ptr-linux`. The jobs set their own images.
- **Shell runner** for `gpu-smoke`: executor `shell`, tag `ptr-gpu`. The job has no image on
  purpose: `make smoke-gpu` uses the host's `training/.venv` with the CUDA torch wheel that
  `make desktop-setup` installed, and the host driver (docs/DESKTOP_SETUP.md). The
  `gitlab-runner` user needs read access to the Hugging Face cache for the OLMoE configs.
  It runs the checked-out Makefile on the host, so register it with "Protected" set (it then
  takes jobs only from protected refs) and as a project runner, not a shared one; `gpu-smoke`
  itself starts only on protected refs.

Check the flags against `gitlab-runner register --help` and the current runner docs; they
change between versions. Before relying on it, test that a container sees the GPU:
`sudo -u gitlab-runner nvidia-smi` for the shell runner (driver setup is in `docs/DESKTOP_SETUP.md`).

To run `gpu-smoke`: open the pipeline in GitLab and press play on the manual job, or use
`glab ci view` and trigger it from there. It is pipeline evidence only, like `make smoke-gpu`.

### Releases without GitHub

`make release-package` builds what `.github/workflows/release.yml` published: `dist/ptr-linux-x86_64.tar.gz`
(ptrd, ptrctl, ptr-worker, ptr-bench), `dist/ptr-cyclonedx.json` (CycloneDX SBOM),
`dist/cargo-metadata.json` and `.sha256` files. Locally, if your `stable` is older than the MSRV:
`make release-package RELEASE_TOOLCHAIN=1.99.0`. On GitLab, pushing a tag `v*` runs the
`release-package` job and creates a release that links those artifacts:

```bash
git tag -a v0.1.0 -m "PTR 0.1.0"
git push gitlab v0.1.0
```

GitHub's build attestations have no GitLab equivalent here; the SHA-256 files and the SBOM ship instead.

## 5. Working with merge requests

The old PR rules carry over: a green pipeline before merge.

1. One branch per change (`claude/<topic>` or similar), small and reviewable.
2. Push once the change is ready (batch commits), then `glab mr create --draft --fill`.
3. `glab ci status` for the pipeline; `glab ci view` for jobs. Manual backend jobs are started
   on purpose, not in a loop.
4. Review, mark ready, then merge with a **merge commit** (project setting Merge method:
   merge commit) once the pipeline is green.
5. Pull `main`, delete the branch.

## 6. Low-automation rules (to avoid another suspension)

- Batch pushes: several commits, one push.
- No loops of comments, MR updates or pipeline re-runs. Never retry in a loop.
- At most a few pipelines per hour. Use `make ci-local` before pushing.
- Claude Code agents never mass-create issues or MRs and never script the GitLab API in bulk.
- A failing pipeline is read (`glab ci trace`) and fixed locally, not re-run to see if it passes.

## 7. Optional: self-hosted Forgejo as second remote

Run Forgejo with docker compose on the desktop (see the Forgejo docs for the compose file and
current image tags; keep it on the LAN or behind a VPN, not on the open internet). Create a
private repository, then:

```bash
python3 scripts/publish_remote.py --name forgejo --remote-url https://<host>/<user>/ptr.git --dry-run
```

Forgejo Actions is largely compatible with `.github/workflows`, but not identical (some
actions, contexts and caching differ). Register a Forgejo runner and verify each workflow
rather than assuming it passes. It doubles as a backup of the GitLab remote.

## 8. Backup

Periodically create bundles on an external disk (details and verification in the
`ptr-offline-sync` skill):

```bash
python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py create --out /path/to/external/ptr-bundle
```

Bundles contain the full history, so keep the disk private and put no secrets in the repo.
