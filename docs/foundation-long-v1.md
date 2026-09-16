# Long-run dense reference: predeclared protocol

Campaign starts 2026-09-16 06:16:33 EDT; eight-hour ceiling 14:16:33. Training cutoff 12:46:33 leaves 90 minutes. Seed **301**, chosen before calibration, is not selected for favorable performance. Foundation commit **4c5b255** was pushed normally before new work; unrelated user files remain untouched.

## Locked foundation and permitted changes

Source of truth: `experiments/mainline/foundation-general-v2.json`. D12/width768 dense model, full attention pattern L, 135,267,480 total/structurally active parameters, no memory table, vocabulary8192, BF16, sequence512, microbatch2,16,384targets/update, no activation checkpointing. The accepted tokenizer, data fingerprint,80/20token-mass mixture and all evaluation texts are unchanged.

Derived config: `experiments/mainline/foundation-long-v1.json`.

Only the total update/token budget and existing budget-relative schedule horizon, diagnostics, checkpoint/resume and logging machinery differ. The exact original `fixed_schedule(N)` formula is reused: extending N extends its coupled LR/weight-decay progress; coefficients and optimizer family are unchanged. Momentum still ramps over300updates. No new schedule family, optimizer tuning, gradient clipping or architecture change.

The old full in-memory tape is replaced by on-demand materialization of **byte-identical microbatches**, tested directly against `foundation_train.make_tape`. Same seeded per-domain document permutation, cyclic streams, BOS placement, Bernoulli microbatch source and offsets. No epoch reshuffling or changed batch policy.

## Calibration and fixed admission

96updates measured25,678steady tok/s; mean update0.63805s, p95 0.69867s; validation2.058s/evaluation, checkpoint1.673s/save, six-prompt generation5.000s. Full checkpoint1,019,770,587bytes. Measured allocated/reserved peak2,414,750,208/2,552,233,984bytes.

Select the largest1024-update multiple fitting the remaining training window using p95 update time times1.05,6seconds per checkpoint/evaluation,300seconds fixed overhead and180seconds admission allowance. This yields **28,672updates /469,762,048tokens /3.47284tokens per parameter**. Nominal total5.13hours; conservative estimate5.98hours. Estimates are not measurements. No budget adjustment for attractive or disappointing validation results.

The accepted corpus contains11,378,155one-pass token positions (including2,263BOS):9,995,910general and1,382,245technical. The fixed run deliberately implies approximately37.60general and67.97technical passes. This is a **heavily repeated closed-pool baseline**, not hundreds of millions of unique tokens. Keeping the pool and mixture locked makes this unavoidable at the calibrated budget; no repetition is hidden. Measure overfitting and plateau rather than stop early for quality. Unique-corpus exposure is only about0.084tokens/parameter.

## Preflight and numerical reproducibility

Profile compared with Git4c5b255; all dataset/tokenizer and evidence hashes checked. Held-out files byte-identical to foundation-v1; prior independent lexical pair audits and all three accepted joint pilot checkpoints verified. Original published test scores are already open; the test is a fixed regression set, not pristine external evidence.

Eight focused tests run on CPU and CUDA, not the full260tests. Full native-CUDA model/optimizer save+resume tested at a two-update interruption of a four-update run. Restoration is asserted **bit-exact** for model, optimizer, sampler, scheduler position and all RNG states before continuing.

The initial demand for bit-identical *future trajectories* failed. A separate uninterrupted repeat also diverged (maximum parameter difference1.08789; final loss difference0.00002277), so native BF16 training is not bitwise deterministic. Do not misreport this as exact trajectory reproduction. Failed attempts remain. Immediate restored states matched exactly; revised continuation checks require loss/BPB differences below0.0001 and preserve identical sampling/schedule/RNG. No mathematical training policy or attention backend was changed to obtain this result. Final saved-weight evaluation must reproduce within0.000001.

## Measurements and retention

Validation at0,32,128,512 and every1024updates through the fixed final endpoint:32points. Training loss, gradient norm (observed, not clipped), LR, throughput, wall time, allocator peaks and both domain pass counts logged every update. GPU temperature, utilization and whole-board memory sampled by the supervisor every approximately5seconds; sustained85°C stops owned work. A temporary Windows system-awake request is released when supervision exits.

Samples at512,14336,28672updates. Same five foundation prompts plus `Write one sentence explaining what a dictionary is.`; seed20260915,top-k40,temperature0.8,64new tokens. These are diagnostic base-model continuations, not capability tests.

Atomic full-state checkpoints include model, optimizer, original schedule definition/N/next step, sampler offsets/source hashes/RNG, Python/NumPy/CPU/CUDA RNG states, cumulative metrics and provenance. Each has a SHA256 receipt. Retain latest+predecessor, three meaningful milestones and best validation checkpoint (at most six, approximately6.2GB plus an atomic temporary file). Only newly generated superseded checkpoints in this run may be pruned; hash receipts and logs remain. Budget15GB for reference artifacts, plus preserved preflights. Disk reserve30GiB; checkpoint admission requires extra headroom.

On interruption use the last validated checkpoint, same config and source bytes. Attempts and any replayed work remain identifiable. No automatic retry after numerical or provenance failure. No test evaluation during training. Final test once, on the final fixed-budget checkpoint, never the best-validation checkpoint.

## Native PowerShell

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py scripts/long_baseline_campaign.py train --config experiments/mainline/foundation-long-v1.json
# If explicitly resuming after diagnosis (same original configuration):
$latest = Get-Content runs/foundation_long_20260916/reference/latest.json | ConvertFrom-Json
& $py scripts/long_baseline.py --config experiments/mainline/foundation-long-v1.json --output runs/foundation_long_20260916/reference --resume $latest.path
Get-Content -Path "C:\Users\bgonn\Desktop\vesper-sparse-concept-lab\runs\foundation_long_20260916\progress.log" -Tail 30 -Wait
```

The direct resume command retains model/data safeguards; use supervised execution for thermal/deadline monitoring. Final evaluation, chart and inference commands will be recorded with their delivered tools. Never restart a completed run or overwrite earlier attempts.
