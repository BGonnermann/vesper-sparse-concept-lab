# Longer-training depth campaign

Started 2026-09-15 19:50:45 UTC. Deadline 2026-09-16 03:50:45 UTC.
Training admission ends by 02:35:45 UTC, reserving 75 minutes for final evaluation, verification and reporting.

## Question and frozen matrix

On this sealed TinyStories corpus, compare dense D6-width768 and D12-width768 at 512, 1024 and 2048 optimizer updates: 8,388,608, 16,777,216 and 33,554,432 tokens. No architectural or optimizer search. Matrix LR0.04, BF16, context512, microbatch2, 16 accumulated microbatches, 16,384 tokens/update, full attention, activation checkpointing off. Reuse pinned autoresearch runtime, dataset and tokenizer. Preserve all existing checkpoints and the production baseline.

Initial paired seeds101,102,103. A scan of211 prior result files found only42–46. Additional seeds follow104,105,... in numerical order, never chosen from scores. Complete all six cells for a seed before admitting another seed. The minimum first three seeds are mandatory. Subsequent whole-seed blocks require a conservative measured runtime estimate plus the reporting reserve. No trial-count cap or result-based early stopping. Alternate depth execution order by seed and budget to reduce order bias. Each cell starts from fresh initialization using the same existing initializer policy and seed; different-depth parameter tensors need not all be equal. Same-architecture initial tensors must match across budgets. Data order must match exactly across depths and be nested across budgets.

## Schedule and training stream

Each budget is independently trained from initialization, not resumed from a shorter-budget endpoint. For update i=0,...,N-1, use the existing policy at progress i/N: LR multiplier=min(1,2*(1-i/N)); Muon weight decay=.2*(1-i/N); momentum=.85+.1*min(i/300,1). No LR warmup; momentum warmup remains300 absolute updates. The final applied LR is2/N of base, with zero only at the unexecuted endpoint N. Both depths have identical schedules for each budget. These are independently annealed budget endpoints, not checkpoints from one continuous learning trajectory. Exclude11 initial updates only from steady-state throughput, not total training time.

Build one immutable32,768-microbatch tape from the pinned training loader and sealed training split. Record source/data hashes, epochs, per-batch hashes, exact duplicates and overlap with the previous tape. For each seed, shuffle the entire tape once, then consume its first8192/16384/32768 rows. No repeated index within a run; longer budgets add stream rows. Natural duplicate content and repeated exposure across independent runs must be reported. Verify the complete tape against a fresh captured-loader execution before training. No test data is used to build this tape.

## Comparison and decision rules

Validation uses the existing fixed65,536-token protocol and order. Report all seeds and D12-minus-D6 differences at each equal-token budget, paired mean and standard deviation, plus exact parameter, allocator-memory and timing costs. Never call equal-token comparisons equal compute. Plot each seed and mean BPB against tokens and measured all-update time; connect budget endpoints descriptively, without interpolation, extrapolation or a convergence claim. If4x beats2x, describe the curves as unfinished.

Predeclare two approximately time-matched contrasts: D6@2x versus D12@1x, and D6@4x versus D12@2x. A contrast qualifies as approximately matched only if each paired all-update-time ratio lies in[0.85,1.15]. Recommend D6 for the overlapping measured time range only if both contrasts qualify, every seed favors D6, and both mean improvements are at least0.001 BPB. Recommend D12 under the symmetric quality rule. Otherwise recommend inconclusive, describing any budget-dependent tradeoff. D12@4x may achieve the lowest observed BPB at greater cost; no D6@8x extrapolation is allowed. All decisions use validation only.

After all training stops, freeze exact checkpoint identities for the highest common-time comparison D6@4x versus D12@2x on every completed seed. Candidate is the validation-favored mean condition, reference is the other; ties favor the cheaper measured mean condition. This designation does not override an inconclusive recommendation. Include frozen D12@4x only as a separately labeled maximum-budget context condition. Perform one final test stage on these predeclared conditions, same65,536-token accounting and batch size, from test rows0–9999. Do not tune, train or reselect after opening test results. Report all test outcomes, including regressions. Validate test split dispatch independently before opening scores.

## Gates, failures and publication

Before training: protocol/stream/schedule tests, existing CPU/CUDA checks, captured-source validation, runtime/data seal verification, free-disk check, and public plan/source freeze. Use the existing snapshot bootstrap, checkpoint retention and report index. Add only required budget/tape support; old protocol remains unchanged. Run GPU jobs sequentially under the shared OS lock. Preserve every failed attempt; at most two retries per cell after diagnosis. Shared provenance/correctness failure blocks further training until repaired; resource insufficiency or deadline stops admission. No cloud, paid services, dependency upgrades, deletion or production promotion.

Estimate18 mandatory checkpoints at roughly0.2–0.5GiB each plus0.5GiB tape and logs, about8GiB initially; allow25GiB for additional seeds and20GiB free-disk floor. Refresh the measured estimate after the first seed. Publish compact trial evidence and progress after each cell. Source archives and source/data/config hashes are mandatory. Preserve local evidence if publication fails.

Final verification replays saved validation checkpoints, checks all token/order/schedule/pairing receipts, optimizer coverage, evaluation immutability, source bytes, final test freeze and plot values. Tests: unittest discovery on CPU/CUDA plus campaign-specific stream and decision tests. Routine repair loops are bounded to three iterations per issue before diagnosing a shared blocker. Final report covers every attempt, curves, paired results, costs, repeated data, test limitations and reproducibility. No claims about V20 trading or general coding ability. Stop at this deadline; no subsequent campaign.
