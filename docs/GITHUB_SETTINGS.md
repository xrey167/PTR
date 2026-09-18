# GitHub Repository Setup

The source tree is configured for CI, security scanning, SDK tests, release attestations and CODEOWNERS. Repository-administration settings require an authenticated repository administrator and cannot be changed through the current ChatGPT GitHub connection.

## Apply metadata and security settings

From a local clone with GitHub CLI authenticated as an administrator:

    ./scripts/setup_github_repo.sh

PowerShell:

    ./scripts/setup_github_repo.ps1

This configures the description, homepage, topics, automatic branch cleanup, auto-merge, vulnerability alerts and automated security fixes.

## Protect main

Protection is intentionally opt-in so setup cannot accidentally lock out the owner before all required checks exist.

    PTR_APPLY_BRANCH_PROTECTION=1 ./scripts/setup_github_repo.sh

The template at `.github/branch-protection-main.json` requires quality, stable Linux/Windows, Rust 1.85 MSRV, repository-invariants and lifecycle-failpoint checks. It blocks force pushes/deletion, requires linear history and requires PR conversation resolution. Administrator enforcement remains disabled so the owner retains an emergency bypass.

## Current limitation

The repository metadata currently exposed by GitHub still has no description/topics and branch administration is not writable through the connected app. The setup scripts are the reproducible handoff for those settings.
