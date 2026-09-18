#!/usr/bin/env bash
set -euo pipefail

REPO="${PTR_GITHUB_REPO:-xrey167/PTR}"
APPLY_PROTECTION="${PTR_APPLY_BRANCH_PROTECTION:-0}"

command -v gh >/dev/null || { echo "gh CLI is required"; exit 1; }
gh auth status >/dev/null

gh repo edit "$REPO" \
  --description "Probabilistically Typed Reasoning — typed cognitive runtime and model architecture research" \
  --homepage "https://github.com/$REPO" \
  --add-topic rust \
  --add-topic llm \
  --add-topic agents \
  --add-topic reasoning \
  --add-topic ai-systems \
  --add-topic machine-learning \
  --add-topic distributed-systems \
  --delete-branch-on-merge \
  --enable-auto-merge \
  --enable-vulnerability-alerts

gh api -X PUT "repos/$REPO/automated-security-fixes" >/dev/null || true

if [[ "$APPLY_PROTECTION" == "1" ]]; then
  gh api -X PUT -H "Accept: application/vnd.github+json" "repos/$REPO/branches/main/protection" --input .github/branch-protection-main.json
  echo "Applied main branch protection."
else
  echo "Skipped branch protection. Set PTR_APPLY_BRANCH_PROTECTION=1 to apply it."
fi

echo "Repository metadata/security settings applied to $REPO."
