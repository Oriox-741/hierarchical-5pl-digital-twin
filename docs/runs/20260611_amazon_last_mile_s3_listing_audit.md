# Amazon Last Mile S3 Listing Audit

Date: 2026-06-11

## Scope

This audit records the manual S3 listing summary supplied by the user and the
metadata-only per-key listing used before selective download. It does not
download or approve the full Amazon Last Mile dataset.

## Manual Listing Evidence

User-supplied S3 listing summary:

- Total objects: `42`
- Total size: `3.1 GiB`
- Relevant small files are under:
  - `almrrc2021-data-training/model_apply_inputs`
  - `almrrc2021-data-training/model_score_inputs`

## AWS CLI

AWS CLI was not on PATH, but the full-path CLI was available:

```text
C:\Program Files\Amazon\AWSCLIV2\aws.exe
aws-cli/2.35.2 Python/3.14.5 Windows/11 exe/AMD64
```

## Approved Key Size Listing

Only the seven approved keys were listed before download.

| Size bytes | S3 key |
| ---: | --- |
| `19342` | `s3://amazon-last-mile-challenges/almrrc2021/License.txt` |
| `422` | `s3://amazon-last-mile-challenges/almrrc2021/Readme.txt` |
| `178703` | `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_route_data.json` |
| `717877` | `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_package_data.json` |
| `4310705` | `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_travel_times.json` |
| `21937` | `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_score_inputs/new_actual_sequences.json` |
| `884` | `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_score_inputs/new_invalid_sequence_scores.json` |

Total approved size:

- `5,249,870` bytes
- `5.007` MiB

Size gate:

- PASS: below the `10 MiB` stop threshold.

## Download Scope

Downloaded only:

- `License.txt`
- `Readme.txt`
- `new_route_data.json`
- `new_package_data.json`
- `new_travel_times.json`
- `new_actual_sequences.json`
- `new_invalid_sequence_scores.json`

Target:

- `data/public/almrrc2021_small/`

Explicitly not downloaded:

- full `3.1 GiB` dataset
- training `model_build_inputs` bulk route/package/travel-time files
- evaluation travel-time/package-data files
- any unapproved S3 key

## Local File Hashes

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `License.txt` | `19342` | `48F99D700291C586C53CEED67E424E4513EF0A2BA1B61D1472D4113086D820C9` |
| `Readme.txt` | `422` | `1325A6079AD41DA79C150DD34BF20B866CE26330E3DAC2D28F8753E0F812A54D` |
| `new_actual_sequences.json` | `21937` | `174625CCF3E0AED722F1129FEEE558EE5672372A3757D475072F7E04C300C2DB` |
| `new_invalid_sequence_scores.json` | `884` | `D401C4A1C8D441962D2ADD760ADF14325FE9D08AF5573C0AFC287EEAB18D199B` |
| `new_package_data.json` | `717877` | `27FC132C7F9DE5498301844E27E0D57EBA7898BD2F2A247C51742D5B87400C95` |
| `new_route_data.json` | `178703` | `5B6292CF03ED6ABC1A8E37A61405C05E14751DC2820A0856F04FD14563F362A3` |
| `new_travel_times.json` | `4310705` | `3A96C295D70F1EB372C89126E77E69F60D7BB1B9D38B2A913777D9341D560AB6` |

## Protected No-Mutation Position

Allowed mutation for this data-access step:

- create `data/public/almrrc2021_small/`
- write the seven approved small files
- write this audit report

No training, project offline eval, registry mutation, production mutation,
baseline mutation, DB mutation, checkpoint mutation, or existing eval-output edit
was performed.
