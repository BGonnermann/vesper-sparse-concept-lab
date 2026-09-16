# Verification, provenance and continuation

Campaign start: **2026-09-15T21:57:06.690291-04:00**. Original experiment cutoff: **2026-09-16T04:57:06.690291-04:00**; eight-hour ceiling: **2026-09-16T05:57:06.690291-04:00**. Closeout audit: **2026-09-16T00:53:11.448593-04:00** (2.93hours elapsed). All experiment training stopped before the final-test freeze at **2026-09-15T23:53:08.791418-04:00**.

The predeclared experiments and final test stage completed early. Do not describe this as eight hours of training or eight elapsed hours. No extra training was launched merely to consume the remaining budget after opening test results.

Final tests: **130 CPU and 130 CUDA tests passed**. All15validation replays passed; all15final-test evaluations verified unchanged weights. Every8,388,608-token training tape was reconstructed on CPU with identical source counts and tensor hash. These audits perform no optimizer updates.

Maximum sampled training temperature: **63°C**, below the85°C sustained stop threshold. Maximum sampled whole-board memory: **4094MiB** (includes desktop/driver allocations; sampling is not an exact peak). Free disk at closeout: **211.43GiB**, above the30GiB floor. No CUDA-device failures or non-finite pilot losses were observed. The first smoke's tensor-layout error was a software validation failure, not a GPU health failure.

## Data and license checks

- Original output reconstruction was byte-identical. Expanded output reconstruction, when present, is recorded separately (wall-time metadata is not expected to match). Independent literal-shingle pair audits checked549,676original-corpus pairs and2,973,141expanded-corpus pairs, with zero retained exact or >=.8Jaccard near duplicates and zero shared long paragraphs across retained documents. These are lexical, not semantic/benchmark-wide guarantees.
- New training sources: pinned WikiText raw (CC-BY-SA3.0/GFDL) and CPython documentation (PSF terms, examples additionally0BSD). Source URLs/revisions/byte counts and hashes are in the manifests; licenses/cards remain in local raw directories.
- Existing TinyStories cache provenance was resolved to `karpathy/tinystories-gpt4-clean` revision `0397e27157956705a0260709da3095bb9c43d6a7`: the pinned LFS hash exactly matches the sealed parquet. License: CDLA-Sharing1.0. The original launcher used a main URL; this revision match was established afterward, not invented retroactively. No parquet redownload occurred. Full-corpus token count is unknown and recorded as null.

## Code and artifact identities

Baseline reproduction recorded revision `e1c55b36f14d5213f43d50d6455a74738f8e6682`. Pilot training revisions: `1b999a1d3479d3345a49843ca310d306368654b0, 687907c3b9926d97d05bc418fb35d7550711ec81, 81e1be2c7e447825718232d5ef3bf6f05a3e97ee, 8b505a45e568a1e5be1d64c5dd1fca156d96f4f8, c816e9627791e095f766953f8fea9dc9f0dd9d01`; differences include reporting/orchestration commits. Shared executed trainer SHA256: `0887ac9b5789a4c4dfd9ce862b2815e6d9344abadd7434689b778b7050ac5b91`. Each run retains its own complete source hash inventory and copies.

- [Changed files](CHANGED-FILES.txt) and [implementation commits through this audit](IMPLEMENTATION-COMMITS.txt). The final publication commit is necessarily identified by Git history/final response rather than embedding its own hash into itself.
- All checkpoints, tokenizer files, raw/processed datasets, logs and evaluation text are local under `runs/foundation_campaign/`, `data/foundation-v1/`, and `data/foundation-expanded-v1/`; none are tracked as model/corpus artifacts.
- [Runtime identity](runtime-identity.json) records installed versions, pinned upstream and retained resolved-lock location/hash. Existing dependencies were not upgraded. Exact runtime inputs are retained under `runs/foundation_campaign/runtime-inputs/`.
- Original and newly appearing unrelated user files were left untouched and uncommitted. No push: the working tree began dirty. No history rewrite or artifact deletion.

## Exact native PowerShell commands

Existing sealed runtime/cache required; do not run setup over it or replace its dependencies. A fresh-machine reproduction must use the recorded upstream revision, resolved lock and pinned data identities rather than resolve latest dependencies.

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py scripts/foundation_data.py verify
& $py scripts/foundation_expand.py
& $py -m unittest discover -s tests -v
$env:MOE_TEST_DEVICE = 'cuda'; & $py -m unittest discover -s tests -v
& $py scripts/evaluate_foundation.py runs/foundation_campaign/data-expanded-s211 --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-replay-01.json
& $py scripts/audit_foundation.py --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-pair-audit-01.json
git log --oneline e1c55b3..HEAD
```

Future study only: the monitored profile launcher refuses to place new training into the closed campaign. It is unit-tested; its underlying trainer completed the15real GPU pilots. No new training was executed to test this wrapper after final-test opening.

```powershell
& $py scripts/run_foundation_profile.py --profile experiments/mainline/foundation-general-v2.json --seed 301 --output runs/foundation-next/trial-s301 --timeout 1200
```

Freeze a new study and fresh held-out data before making new improvement claims. This campaign's opened test set is now a regression set. For tokenizer reconstruction use `foundation_tokenizer.py` with a new output directory and verify the recorded candidate SHA256. The old frozen queue commands are historical controllers, not a request to restart training after test opening.

Live campaign log: `runs/foundation_campaign/progress.log`. Retained failure and health logs remain alongside every attempt. Unknown measurements have not been filled with estimates.
