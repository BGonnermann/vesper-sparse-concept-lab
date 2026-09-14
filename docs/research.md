# Initial research review

Reviewed 2026-09-14. This is a targeted first review, not an exhaustive literature survey. No reviewed implementation has been executed locally.

## What the three features do

| Feature | Evidence | Implication for this project |
| --- | --- | --- |
| Mixture of experts | Switch Transformer routes examples through selected parameters and discusses training stability and communication costs. | Sparse activation can reduce computation, but desktop speed must be measured. All experts still need storage. |
| Next Concept Prediction (NCP) | ConceptLM learns discrete latent representations and predicts concepts alongside tokens. | NCP is a modeling/training mechanism, not a replacement tokenizer or guaranteed full-answer planning. |
| Conditional n-gram memory | Engram uses hashed lookup and contextual gating to complement neural computation. | Memory lookup can complement experts; its table and transfer costs must be included. |

Sources: [Switch Transformer](https://arxiv.org/abs/2101.03961), [ConceptLM](https://arxiv.org/abs/2602.08984), [Engram](https://arxiv.org/abs/2601.07372).

## NCP: original paper and available implementation

ConceptLM studies models from 70M to 1.5B parameters trained from scratch and continued pretraining of an 8B model. Predicted concepts feed back into token generation. Its small-scale studies make it relevant to a local mechanics experiment; they do not establish feasibility for our combined architecture. The paper's HTML was inspected. [Paper](https://arxiv.org/html/2602.08984v1).

The official repository exposes model/evaluation material and references GPT2, Pythia, and Llama. Its historical installation recipe includes FlashAttention. A complete reusable training pipeline and native Windows/Blackwell compatibility were not established by this review. Verify licensing and pin a revision before copying code. [ConceptLM repository](https://github.com/LUMIA-Group/ConceptLM).

## Newer NCP checkpoint: useful reference, larger than the first prototype

The September 2026 NCP-ArchPreview report describes an 8.9B model trained on 5.73T tokens. The authors report improved convergence and downstream results. Those results are not measurements of 5070 Ti training speed or inference latency. The abstract was reviewed; the full report could not be retrieved in this session. [Technical report](https://arxiv.org/abs/2609.10715).

A public Stage 1 checkpoint exists. Its model card describes approximately 8.94B BF16 parameters, a base model, custom Transformers code, and an Apache-2.0 weight license. It retains autoregressive token generation; the card labels training code as coming soon. This is a reference for NCP, not evidence of a combined MoE+NCP+Engram checkpoint. [Stage 1 model card](https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage1).

The official evaluation repository lists public Stage 1 and Stage 2 checkpoints, exposes a vLLM workflow, and explicitly excludes the private training stack. Its presence does not establish a native Windows training path. [Evaluation repository](https://github.com/LUMIA-Group/ncp_olmo_eval).

## Engram: integration work remains

Engram adds conditional memory through deterministic n-gram lookup and gating. It preserves the ordinary token path. Its large-scale results and host-memory prefetch claims do not establish a benefit for small consumer GPUs. The original HTML's architecture sections were inspected; the later revision was identified but not fully compared. [Paper, v1](https://arxiv.org/html/2601.07372v1).

The official repository's demo mocks the attention, MoE, and mHC components. It demonstrates the memory mechanism rather than providing a complete trainable language model. The repository displays Apache-2.0 licensing. [Engram repository](https://github.com/deepseek-ai/Engram).

## What remains unproven

We have not verified a ready-made checkpoint combining all three features. We have not established that their gains add together, that expert identities map cleanly to human skills, or that removing supposedly irrelevant experts preserves quality. Broad training can support the user's tasks indirectly.

Next research tasks: audit a pinned NCP implementation's causal alignment and training availability; compare the Engram revisions; verify Windows runtime support; select licensed training data; investigate conversion/quantization support for custom modules. Keep tokenizer changes out of the first ablation so they do not obscure architectural effects.
