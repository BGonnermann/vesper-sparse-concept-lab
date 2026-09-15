# Prospective depth-12 optimizer follow-up

Recorded after trial 42, before any combined depth-12/LR outcome.

Depth 12 is the best observed BPB, with larger parameter and compute budgets.
Depth and matrix LR were screened separately first. The next four configurations
combine depth 12 with matrix LR 0.03, 0.05, 0.02, and 0.06. Their matching control
is depth 12 at 0.04, not depth 6. All other settings stay unchanged.

Hypothesis: the larger model's optimization may benefit from a different LR;
neither the small depth-6 LR differences nor the shape result establish this.
Order is prospective; completed results remain available to the controller.
These are explicitly combined, already-gated configuration axes, not new model
mechanisms. No NCP or memory combination is added.

The existing passed CPU/CUDA and depth-fit gates cover the unchanged executed
model, runner, training and evaluation sources. A versioned gate receipt inherits
those exact source/test/log hashes and records the amended search hash and plan.
Each proposed configuration is validated and differs from the depth-12 control
only in matrix_lr. No correctness evidence is fabricated or relabeled as rerun.

512 updates, 8,388,608 tokens, seed 42, checkpointing off, original batches,
schedule and evaluation remain fixed. Keep the original 900-second timeout,
11:30 UTC deadline, failure safeguards, and reserved confirmations/reporting.
Prior search/gate/controller receipts remain immutable.
