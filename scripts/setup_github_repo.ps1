$ErrorActionPreference = "Stop"
$Repo = if ($env:PTR_GITHUB_REPO) { $env:PTR_GITHUB_REPO } else { "xrey167/PTR" }
$ApplyProtection = $env:PTR_APPLY_BRANCH_PROTECTION -eq "1"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "gh CLI is required" }
gh auth status | Out-Null

gh repo edit $Repo --description "Probabilistically Typed Reasoning — typed cognitive runtime and model architecture research" --homepage "https://github.com/$Repo" --add-topic rust --add-topic llm --add-topic agents --add-topic reasoning --add-topic ai-systems --add-topic machine-learning --add-topic distributed-systems --delete-branch-on-merge --enable-auto-merge --enable-vulnerability-alerts

try { gh api -X PUT "repos/$Repo/automated-security-fixes" | Out-Null } catch {}

if ($ApplyProtection) {
  Get-Content ".github/branch-protection-main.json" -Raw | gh api -X PUT -H "Accept: application/vnd.github+json" "repos/$Repo/branches/main/protection" --input -
  Write-Host "Applied main branch protection."
} else {
  Write-Host "Skipped branch protection. Set PTR_APPLY_BRANCH_PROTECTION=1 to apply it."
}

Write-Host "Repository metadata/security settings applied to $Repo."
