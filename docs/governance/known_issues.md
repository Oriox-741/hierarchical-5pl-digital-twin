# Known Issues

Documented defects found during repository packaging review on 2026-08-08.

These entries are **documentation only**. No file under `src/` was modified while
recording them. Each entry states the evidence and the available remediation
paths without selecting one; the choice is left to the repository owner.

---

## Issue 1 — `fine_tune.py` cannot read its own default config

**Severity:** functional — the documented default invocation path raises an
unhandled exception before any work begins.

| Field | Value |
| --- | --- |
| File | `src/learn/fine_tune.py` |
| Line | 28 |
| Statement | `with path.open("r", encoding="utf-8") as handle:` |
| Default config | `src/learn/fine_tune.py:23` → `DEFAULT_JOINT_CONFIG = Path("configs/training_joint.json")` |
| Resolution site | `src/learn/fine_tune.py:203` → `config_path = args.config or DEFAULT_JOINT_CONFIG` |

`configs/training_joint.json` begins with a UTF-8 byte order mark. Reading it as
`utf-8` leaves the BOM as `U+FEFF` at the head of the string, and `json.loads`
rejects it.

**Evidence** — isolated parse of the real file, no project code executed:

```
$ python -c "import json, io; json.loads(io.open('configs/training_joint.json', encoding='utf-8').read())"
utf-8      : JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)
utf-8-sig  : OK

first 4 bytes: b'\xef\xbb\xbf{'
```

The same pattern appears at `src/archive_aborted_sb3/evaluate_aborted_models.py:403`
(`json.loads(CONFIG_PATH.read_text(encoding="utf-8"))`, where `CONFIG_PATH` is
`configs/training_joint.json` at line 22). That module sits in an archived
directory, so its practical impact is lower.

### Remediation path A — change the reader to `utf-8-sig`

Edit the two call sites to decode with `utf-8-sig`, which strips a BOM when
present and is a no-op when absent.

*Impact:* touches `src/`, which the packaging scope forbids. Two files change,
no data files change. Matches the encoding already used by
`src/learn/train_joint_torch.py:1911` (see Issue 3), so it moves the codebase
toward one convention. Any config file, present or future, is then readable
regardless of BOM.

### Remediation path B — strip the BOMs from the config files

Rewrite the 35 BOM-carrying files under `configs/` without the marker.

*Impact:* leaves `src/` untouched but rewrites tracked data files, changing their
SHA-256 digests. Those digests are recorded in `SANITIZED_FILE_MANIFEST.json`,
so the manifest would no longer match the tree unless it is also regenerated —
which would edit a point-in-time audit record. Readers hardcoded to `utf-8`
remain fragile against any BOM reintroduced later by an editor.

---

## Issue 2 — UTF-8 BOM is applied inconsistently across the repository

**Severity:** latent — harmless on its own, but it is the precondition for
Issue 1 and makes the failure depend on which config a caller happens to pick.

51 tracked text files begin with a UTF-8 BOM (`EF BB BF`). Within `configs/` the
marker is present on some files and absent on others:

| Scope | With BOM | Without BOM | Total |
| --- | --- | --- | --- |
| `configs/**/*.json` | 35 | 41 | 76 |

Two configs that sit side by side therefore behave differently through a
`utf-8` reader. `configs/training_joint.json` carries a BOM;
`configs/simulation.json` and `configs/reward_weights.json` do not.

### Affected paths

`configs/` (35):

```
configs/training_joint.json
configs/training_joint_curriculum.json
configs/training_joint_curriculum_1m_targeted_demand_spike_10k.json
configs/training_joint_curriculum_1m_targeted_demand_spike_10k_post_repair.json
configs/training_joint_curriculum_1m_targeted_premium_sla_10k.json
configs/training_joint_curriculum_1m_targeted_premium_sla_10k_post_repair.json
configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json
configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json
configs/training_joint_curriculum_clean_cold_start_1m.json
configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_hold_neutralization_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_shortest_dispatch_context_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json
configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json
configs/training_joint_curriculum_smoke.json
configs/training_joint_curriculum_smoke_demand_spike.json
configs/training_joint_curriculum_smoke_demand_spike_strong.json
configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json
configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json
configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4.json
configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4_50k.json
configs/training_joint_curriculum_smoke_high_holding_cost.json
configs/training_joint_curriculum_smoke_lead_time_delay.json
configs/training_joint_curriculum_smoke_mixed_stress.json
configs/training_joint_curriculum_smoke_premium_sla.json
configs/training_joint_curriculum_smoke_premium_sla_strong.json
configs/training_joint_curriculum_smoke_route_disruption.json
configs/training_joint_curriculum_smoke_vehicle_scarcity.json
configs/training_joint_smoke.json
```

Outside `configs/` (16):

```
docs/goals/bounded_stability_autopilot_goal.txt
docs/goals/level2_research_autopilot_goal.txt
docs/runs/20260608_clean_rewardfix_perfclean_1m_autopilot_status.md
docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_gate_result.json
reports/benchmarks/sota_pathway_20260613/preflight_protected_state.json
reports/benchmarks/sota_pathway_20260613/codex_5pl_benchmark_suite/codex_5pl_benchmark_suite_spec.json
reports/benchmarks/sota_pathway_20260613/congestion_analogs/congestion_analog_preflight_report.json
reports/benchmarks/sota_pathway_20260613/final_synthesis/protected_state_post_audit.json
reports/benchmarks/sota_pathway_20260613/final_synthesis/sota_pathway_execution_summary.json
reports/benchmarks/sota_pathway_20260613/fleet_dispatch/fleet_dispatch_benchmark_report.json
reports/benchmarks/sota_pathway_20260613/fleet_dispatch/fleet_dispatch_summary.csv
reports/benchmarks/sota_pathway_20260613/stochastic_routing_svrpbench/svrpbench_report.json
reports/thesis_handoff/chapter3_method_evidence/docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_gate_result.json
reports/thesis_handoff/chapter3_method_evidence/models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json
reports/thesis_handoff/chapter3_method_evidence/reports/benchmarks/sota_pathway_20260613/codex_5pl_benchmark_suite/codex_5pl_benchmark_suite_spec.json
tests/learn/test_curriculum_config.py
```

### Remediation path A — accept the BOM and normalise every reader

Leave the files as they are and require all JSON/text readers to use
`utf-8-sig`.

*Impact:* no tracked data file changes, so `SANITIZED_FILE_MANIFEST.json` stays
valid. Correctness then depends on every current and future reader choosing the
right encoding, and there is no mechanical check enforcing it.

### Remediation path B — normalise the files and forbid the BOM

Strip the marker from all 51 files and add an enforcement rule (for example a
`.gitattributes` entry or a CI check) so it cannot return.

*Impact:* readers become encoding-agnostic and the class of defect disappears at
the source. In exchange, 51 tracked files change content and digest. Files under
`reports/benchmarks/` and `reports/thesis_handoff/` are evidence artifacts whose
hashes are referenced by the manifest and by protected-state audit records, so
rewriting them alters material that the governance model treats as
point-in-time evidence.

---

## Issue 3 — the two config readers disagree on encoding

**Severity:** consistency — not a defect by itself, but it is why the failure in
Issue 1 is intermittent rather than total.

| Reader | File | Line | Encoding | BOM handling |
| --- | --- | --- | --- | --- |
| `load_config` | `src/learn/train_joint_torch.py` | 1911 | `utf-8-sig` | correct |
| `load_config` | `src/learn/fine_tune.py` | 28 | `utf-8` | fails |

```python
# src/learn/train_joint_torch.py:1911
return json.loads(path.read_text(encoding="utf-8-sig"))
```

Two functions with the same name, loading configs from the same directory,
disagree about the encoding. The `utf-8-sig` spelling in `train_joint_torch.py`
is deliberate — it is the exception rather than the default across the codebase,
which suggests this failure was encountered and fixed at that call site without
the fix being propagated.

The practical consequence is that whether a BOM breaks a run depends on which
entry point is used, not on the config being loaded. A config that works under
one command fails under another.

### Remediation path A — converge on `utf-8-sig`

Adopt the `train_joint_torch.py` spelling everywhere config files are read.

*Impact:* one convention, tolerant of both BOM and BOM-less files. Requires
edits under `src/`. Does not remove the underlying inconsistency in the data
files, only its effect.

### Remediation path B — converge on `utf-8` and guarantee BOM-free inputs

Keep the plain `utf-8` readers, remove every BOM, and enforce their absence.

*Impact:* the encoding contract becomes explicit and testable, and a stray BOM
fails loudly instead of silently depending on the entry point. Requires the data
rewrite described in Issue 2 path B, with the same effect on manifest digests
and evidence artifacts.

---

## Scan methodology

Audit record of how the findings above, and the related character-corruption
findings, were produced. Recorded so the result can be reproduced or challenged.

### Patterns searched

| Target | Pattern | Result |
| --- | --- | --- |
| Latin-1 mojibake | `[ÃÄÅ]` followed by a continuation byte — covers `Ä±` (ı), `ÅŸ` (ş), `Äž` (ğ), `Ã§` (ç), `Ã¶` (ö), `Ã¼` (ü) | 0 findings |
| Replacement character | `U+FFFD` | 0 findings |
| Undecodable bytes | every text file decoded as UTF-8 | 0 findings |
| Byte order mark | leading `EF BB BF` | 51 files |
| Destroyed characters | ASCII `?` (`0x3F`) where a letter belongs | 3 files, 233 characters, plus `CITATION.cff` |

Scope: the whole repository, excluding `.git/`, `__pycache__/`, `.obsidian/` and
binary extensions. `SANITIZED_FILE_MANIFEST.json` was excluded from `?` matching
because it records filenames and digests rather than prose.

### Discriminator for destroyed characters

A `?` is ambiguous: it is both valid punctuation and the byte an ASCII-lossy
encoder leaves behind. The zero-false-positive signal used was **`?` wedged
between two letters** (`letter?letter`), which cannot be punctuation in any
language present in this repository.

This was cross-checked against a second, independent signal: whether the file
still contains any real Turkish letter (`çşğıöüÇŞĞİÖÜ`). The three flagged files
contain Turkish prose and **zero** surviving Turkish letters, while 54 other
files in the repository retain theirs intact — up to 47,766 in
`reports/demo_control_room_v8/traces/route_disruption_congestion_trace.json`.
Both signals agree on the same three files.

### Excluded as false positives

An initial broader heuristic — any `?` at the end of a word — produced 354
additional matches. All were rejected on inspection:

- **Genuine English questions**, mostly markdown table cells:
  `| Was there current work? |`, `### Should production remain active?`
- **Genuine Turkish questions in files whose Turkish is intact**:
  `"Şimdi ne yapmalıyım?"`, `"eğer model şu kararı verirse ne olur?"` — these
  carry correct `ş`, `ı`, `ğ`, so the `?` is punctuation.
- **JavaScript ternary operators**:
  `${r.selected?'selected':''}` in `reports/demo_control_room_v6/app.js`
- **URL query strings**: `openreview.net/forum?id=...`, `app.js?v=20260616v5`

### Not treated as corruption

Most `*_tr.md` files are written in deliberate ASCII transliteration — `gore`
for *göre*, `calistirildi` for *çalıştırıldı* — with no Turkish letters and no
`?` substitution. This is a writing convention, not data loss, and was excluded
from the findings. The distinguishing test is the presence of `?` in letter
positions: transliterated files have none.

### Root cause boundary

`git log --follow` shows all three corrupted files arriving already broken in
`ceb2e84`, the initial sanitized commit, so the loss predates this repository.
Every encoding declaration in this repository's own Python sources is explicit
UTF-8, and no ASCII-forcing conversion exists here. The corruption signature —
every non-ASCII character replaced, including `ç`, `ö` and `ü`, which Windows
cp1252 can represent — is consistent with an `encode("ascii", errors="replace")`
step or an ASCII-only output stream in the upstream build. That build lives
outside this repository and was not inspected.

Twenty-one sibling `*_tr.md` files carrying the same date are unaffected, so the
loss came from a path taken by those three files specifically, not from a
repository-wide transformation.
