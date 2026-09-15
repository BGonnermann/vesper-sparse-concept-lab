# Prospective shape extension after the first screen

The 29-condition screen and the first depth8 repeat are complete. Both depth8
runs scored .617737 BPB, versus dense6 .634848. The original depth8 all-update
GPU cost was 1.2253 times dense6; its allocated peak was 988,913,664 bytes.
These observations motivate two additional standalone shape probes rather than
only repeating tiny learning-rate or memory gains for the remaining budget.

This is an adaptive post-screen amendment, not an originally preregistered study.
No seeds43/44 have run. The original search, preflight and controller versions
remain preserved. No previous result or executed source is rewritten.

- Add dense depths10 and12, matrix LR.04, no memory/MoE/NCP. Compare against
  depth8 under the same 512 updates / 8,388,608 tokens, data tape/order, step
  schedule, checkpointing-off, full-attention and validation protocol.
- Existing shape construction yields widths640/768 and exact parameter counts
  85,852,980 / 135,267,480. Changing shape retains the optimizer's existing
  width-dependent scaling policy; record the resolved optimizer groups.
- Change only the candidate depth validation ceiling from8 to12. Transformer,
  attention, tokenizer, training loop and evaluation implementation are unchanged.
- Before either trial, run the complete CPU/CUDA suites plus full-context GPU
  causal-prefix, finite-gradient, optimizer-coverage, CE-only-evaluation and
  save/load checks. Randomize zero-initialized output projections in synthetic
  fixtures so causal checks exercise attention rather than an inactive path.
- Fit probes use synthetic data, three warmup and three measured optimizer
  updates at the real accumulation/context settings. They are correctness/fit
  evidence, not TinyStories scores or campaign training-throughput claims.
- Require at least2GiB free VRAM, reserved peak below75% of device memory, and
  a conservative 512-update forecast plus45seconds below900seconds. Preserve
  diagnostics and stop on a failed gate; do not increase the timeout.
- Keep the original 03:30-11:30UTC eight-hour envelope, no trial-count cap, and
  the existing confirmation/reporting reserves. No upgrades, cloud, deletion,
  new data, NCP or combined mechanisms.

After these screens, prospectively update the recorded replication target list
to include the best-BPB measured frontier point and its matching control, plus
the lower-cost alternatives. Freeze final candidate/control before seeds43/44.
Equal tokens do not mean equal parameters, compute, wall time or efficiency.
