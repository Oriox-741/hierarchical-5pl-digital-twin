# Push Instructions

GitHub CLI (`gh`) was not available in the source environment, so this sanitized repository was prepared and committed locally only.

Before pushing, choose a concrete private GitHub target:

```powershell
$OWNER = "YOUR_GITHUB_OWNER"
$REPO = "codex-proje-sanitized"
cd reports\github_publish\codex-proje-sanitized
gh auth login
gh repo create "$OWNER/$REPO" --private --source . --remote origin --push
```

If you do not use GitHub CLI:

```powershell
cd reports\github_publish\codex-proje-sanitized
git remote add origin https://github.com/YOUR_GITHUB_OWNER/codex-proje-sanitized.git
git push -u origin main
```

Do not push to a public repository unless the owner explicitly approves public visibility after reviewing the scan reports.
