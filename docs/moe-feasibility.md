# First MoE feasibility contract

Implement a selectable dense / four-expert top-1 feedforward adapter at depth 6.
Keep the pinned upstream checkout, attention, tokenizer, data, context 512,
effective batch 16384 tokens, microbatch 2, evaluation and 300-second timed
training budget unchanged. Preserve every existing run artifact.

The MoE replaces every dense MLP with four MLPs of the same dimensions. Each
token goes to exactly one expert, without a capacity limit or token dropping.
Multiply its expert output by the selected full-softmax probability, preserving
a task-loss gradient to the router. The explicit auxiliary objective is
`0.01 * mean_layers(4 * sum(mean_assignment * mean_router_probability))`.
Assignment fractions are detached; probabilities are differentiable. Router
logits/softmax use float32. Router AdamW LR is 0.001 with the existing schedule;
expert matrices use the existing Muon policy and matrix LR 0.04.

Checkpointed blocks return hidden states, auxiliary loss and routing counts.
Accumulate detached statistics only outside checkpoint recomputation. Empty
expert batches still execute their empty differentiable path, providing zero
gradients for the upstream optimizer. Validate optimizer membership by parameter
identity, including every expert and router exactly once.

Validation BPB contains token cross-entropy only. Record training cross-entropy,
unweighted and weighted routing auxiliary loss, train/eval utilization per layer,
zero dropped tokens, and router gradient magnitudes separately. Active parameter
count means all shared parameter tensors plus one expert per layer and all router
weights. This convention counts entire embedding tables, not only lookup rows;
it is a structural per-token count, not a measured FLOP or speed estimate.

Acceptance: shape, future-token causality, differentiable task routing and load
balance, finite expert/router gradients and updates including empty experts,
exact routing counts, checkpoint on/off loss and gradient equivalence, and pure
validation loss. Verify runner configuration rejection, variant/source smoke
gate isolation, and retained result artifacts. Run CPU correctness tests and a
native CUDA BF16 counterpart before fresh dense/MoE smoke tests, followed by
one dense and one MoE baseline. Send each baseline's live-log command immediately.

Use up to three implementation repair/review iterations. Expected test-first
failures are recorded; stop and diagnose any unexpected correctness or training
failure. No NCP, n-gram memory, autonomous search, additional training trials,
cloud jobs, promotion, or resumable-checkpoint claims. Existing checkpoints are
weights-only; this task does not implement resumption.

Compare fresh runs under identical protocol/data/runtime/source hashes, except
for the explicit feedforward architecture and router-specific settings. Report
BPB, total/active parameters, tokens, elapsed time, memory, expert utilization,
and unequal parameter budgets. All quality conclusions are provisional.

## Captured execution

Version-3 runs copy all four project modules, pinned upstream train/prepare
sources and runtime identity files, plus resolved candidate/protocol JSON into
the run directory. Python isolated mode executes the captured bootstrap;
project imports resolve only from captured source directories. The bootstrap
verifies hashes before and after execution. `execution.json` identifies the
Python source files actually executed, and the parent checks this receipt
against its captured manifest. Installed dependencies stay in the existing
recorded venv; the dataset stays in the verified shared cache. This is source
and configuration isolation, not a copied dependency environment or sandbox
against deliberately malicious code. Bootstrap hashes also enter smoke gates.

The CUDA routing reference uses FP32 logits/softmax and vector probability
weights before casting to input dtype, matching the production operation order.
Its tolerance is 1e-6 absolute / 1e-5 relative, with explicit finite and nonzero
router task-gradient assertions. No production precision change is required.
