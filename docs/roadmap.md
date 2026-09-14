# Roadmap

## Runnable baseline added

The [autoresearch integration](autoresearch.md) now provides Windows setup, a GPU probe, a bounded dense smoke/baseline runner, and instructions for up to three candidate experiments. Runner tests are separate from GPU validation. No local GPU result or custom MoE/NCP/n-gram implementation is claimed.

## Completed in this research pass

- Identified primary NCP, MoE, and Engram sources and reviewed the available implementation descriptions.
- Verified a public NCP-ArchPreview model card; separated inference/evaluation availability from training-code availability.
- Defined a local-first experiment path, memory accounting, controlled comparisons, and cloud decision criteria.
- Created the initial research repository. No architecture has been implemented or benchmarked.

## Next: establish the desktop training environment

1. Capture the [hardware preflight](hardware.md#desktop-preflight) on the actual Windows PC.
2. Verify native Windows GPU training with a compatible runtime; pin the working environment.
3. Audit exact upstream code revisions and licenses. Finalize a minimal architecture and parameter count.
4. Select a permitted small corpus and freeze document-level splits.
5. Implement and validate the dense baseline, then add MoE, NCP, and memory behind independent switches.

## Then: learn whether the design is worthwhile

Execute the [experiment stages](experiments.md). Rank designs by quality, memory, and measured elapsed time. A failed combination is a useful result; preserve the evidence and simplify.

If a promising mechanism survives, compare training a larger custom model with adapting a capable pretrained model. The latter may yield workflow value sooner. Choose a larger token and parameter budget only after the small runs show the actual resource cost.

## Later: cloud training and local deployment

Use cloud GPUs for validated runs whose memory or time requirements exceed the local budget. Make each run resumable and portable before moving it. Cloud training size is separate from the deployed model's memory envelope.

Investigate distillation, quantization, and runtime integration for the selected architecture. Validate any transformation against the same quality suite. Demonstrate actual 8GB and 16GB deployment at explicit context lengths and concurrency. Include the memory table, concept module, cache, and any host-memory offload in reporting.

## Open questions

- Which actual user tasks and latency targets define success?
- What is the current free GPU memory, installed runtime, and usable host RAM?
- Which NCP training implementation can be reused under compatible terms?
- How should concept targets and feedback be aligned without future leakage?
- Does the small n-gram table help coding after hash collisions and its optimizer cost?
- Does a sparse implementation run efficiently enough on this GPU to beat dense alternatives?
- Which custom modules can the eventual quantization/inference runtime support?
