# Reproducibility

This repository is designed for safe, lightweight verification first.

## Safe Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Safe Checks

```powershell
python -m py_compile (Get-ChildItem -Recurse src,scripts,tests -Filter *.py | ForEach-Object FullName)
```

`python -m unittest discover -s tests` is not enabled by default in CI because some tests depend on optional research packages or local artifacts. Run focused tests manually only after reviewing their behavior.

## Do Not Run By Default

Do not run training, offline scenario evaluation, long-run gates, model registration, activation, production promotion, dataset downloads, tunnel setup, or dashboard servers from this sanitized repository without explicit approval.

## Exact-Resume Evidence

Exact-resume evidence is documented in:

- `reports/thesis_handoff/chapter3_training_and_resume_truth_table.md`
- `docs/runs/20260610_level2_continuation_exact_resume_patch_report.md` if included

Checkpoint bodies are intentionally excluded.

## Ablation Scaffolding

The repository includes public-safe ablation configuration scaffolding under `configs/ablation/` and a dry-run CLI:

```powershell
python scripts\run_ablation_matrix.py --dry-run --all
```

The dry run validates planned ablation definitions without requiring private checkpoints or running evaluation. Actual ablation execution may require checkpoint artifacts and episode outputs that are intentionally excluded from this sanitized public repository. No ablation results are claimed by default; results should be claimed only when generated reports are present and intentionally reviewed.

## Known Missing or Ambiguous Fields

Human-dependent thesis and publication fields remain in:

- `reports/thesis_handoff/thesis_manual_fields_checklist.md`
- `reports/thesis_handoff/chapter3_missing_or_ambiguous_fields.md`
