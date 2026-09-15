"""Local dense/MoE extension of the sealed upstream GPT; no data or runtime edits."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from autoresearch_memory import CausalNgramMemory
from autoresearch_ncp import NextConcept


class Top1FeedForward(nn.Module):
    """Dropless sparse dispatch. Empty expert paths retain zero parameter gradients."""

    def __init__(self, first_expert, expert_factory, width, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([first_expert] + [expert_factory() for _ in range(num_experts - 1)])
        self.router = nn.Linear(width, num_experts, bias=False)
        self.num_experts = num_experts

    @property
    def c_fc(self):
        # Let upstream initialize expert 0 in the same RNG order as dense.
        return self.experts[0].c_fc

    @property
    def c_proj(self):
        return self.experts[0].c_proj

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            probabilities = F.softmax(self.router(flat.float()), dim=-1)
        selected = probabilities.argmax(dim=-1)
        # A fixed-size histogram avoids bincount's dynamic-size host checks.
        counts = torch.zeros(self.num_experts, device=selected.device, dtype=selected.dtype)
        counts.scatter_add_(0, selected, torch.ones_like(selected))
        fractions = counts.to(probabilities.dtype) / flat.shape[0]
        auxiliary = self.num_experts * (fractions.detach() * probabilities.mean(dim=0)).sum()
        # Stable packing preserves the original row order inside every expert.
        order = torch.argsort(selected, stable=True)
        packed = flat.index_select(0, order)
        # One boundary transfer replaces four dynamic nonzero synchronizations.
        chunks = packed.split(counts.tolist(), dim=0)
        # Keep empty expert paths: Muon requires tensor grads for every matrix.
        values = torch.cat([expert(chunk) for expert, chunk in zip(self.experts, chunks)], dim=0)
        weights = probabilities.gather(1, selected.unsqueeze(-1)).index_select(0, order)
        values = (values * weights).to(flat.dtype)
        # order is a complete permutation, so every output row is written once.
        output = torch.empty_like(flat).index_copy_(0, order, values)
        return output.reshape_as(x), auxiliary, counts


class ExpertBlock(nn.Module):
    def __init__(self, original, upstream, config, num_experts):
        super().__init__()
        self.attn = original.attn
        self.mlp = Top1FeedForward(original.mlp, lambda: upstream.MLP(config), config.n_embd, num_experts)
        self.norm = upstream.norm

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(self.norm(x), ve, cos_sin, window_size)
        feedforward, auxiliary, counts = self.mlp(self.norm(x))
        return x + feedforward, auxiliary, counts


def optimizer_coverage(model, optimizer):
    named = dict(model.named_parameters())
    identities = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in named.values() if p.requires_grad}
    missing = [name for name, p in named.items() if p.requires_grad and id(p) not in identities]
    duplicate_count = len(identities) - len(set(identities))
    unexpected_count = len(set(identities) - expected)
    if missing or duplicate_count or unexpected_count:
        raise RuntimeError(f"Optimizer coverage failed: missing={missing}, duplicates={duplicate_count}, unexpected={unexpected_count}")
    names_by_id = {id(p): name for name, p in named.items()}
    return {
        "verified": True, "missing": missing, "duplicate_count": duplicate_count,
        "unexpected_count": unexpected_count,
        "parameter_tensors": len(identities),
        "groups": [{"kind": group["kind"], "initial_lr": group["initial_lr"],
                    "parameters": [names_by_id[id(p)] for p in group["params"]]}
                   for group in optimizer.param_groups],
    }


def model_class(upstream, candidate):
    """Create a configured class without modifying upstream source or global Block."""
    variant = candidate["feedforward"]
    num_experts = candidate.get("num_experts", 0)

    class ExperimentGPT(upstream.GPT):
        def __init__(self, config):
            super().__init__(config)
            self.feedforward = variant
            self.aux_loss_weight = candidate.get("aux_loss_weight", 0.0)
            if variant == "moe":
                self.transformer.h = nn.ModuleList([
                    ExpertBlock(block, upstream, config, num_experts) for block in self.transformer.h
                ])
            memory = candidate.get("memory")
            self.ngram_memory = None if memory is None else CausalNgramMemory(
                config.n_embd, memory["bos_token_id"], config.vocab_size, upstream.norm)
            self.ncp = None if candidate.get('ncp') is None else NextConcept(config.n_embd,config.n_head,candidate['ncp'])
            for phase in ("train", "eval"):
                self.register_buffer(f"_stats_{phase}_counts", torch.zeros(config.n_layer, num_experts, dtype=torch.int64), persistent=False)
                self.register_buffer(f"_stats_{phase}_tokens", torch.zeros((), dtype=torch.int64), persistent=False)
            self.register_buffer("_stats_ce_sum", torch.zeros((), dtype=torch.float64), persistent=False)
            self.register_buffer("_stats_aux_sum", torch.zeros((), dtype=torch.float64), persistent=False)
            self.register_buffer("_stats_router_grad_max", torch.zeros(config.n_layer if num_experts else 0), persistent=False)
            self.register_buffer("_stats_router_grad_finite", torch.ones(config.n_layer if num_experts else 0, dtype=torch.bool), persistent=False)
            self.optimizer_report = None

        @torch.no_grad()
        def init_weights(self, embed_dtype=torch.bfloat16):
            super().init_weights(embed_dtype=embed_dtype)
            for name, buffer in self.named_buffers():
                if name.startswith("_stats_"):
                    buffer.fill_(True if name.endswith("_finite") else 0)
            if variant == "moe":
                device = self.transformer.wte.weight.device
                devices = [device.index] if device.type == "cuda" else []
                # Preserve the shared initialization and subsequent data RNG stream.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(4200 + torch.initial_seed())
                    scale = math.sqrt(3 / self.config.n_embd)
                    for block in self.transformer.h:
                        nn.init.normal_(block.mlp.router.weight, std=.02)
                        for expert in block.mlp.experts[1:]:
                            nn.init.uniform_(expert.c_fc.weight, -scale, scale)
                            nn.init.zeros_(expert.c_proj.weight)
            if self.ngram_memory is not None:
                self.ngram_memory.initialize(8400 + torch.initial_seed())
            if self.ncp is not None:
                self.ncp.initialize(12600 + torch.initial_seed())

        def setup_optimizer(self, **kwargs):
            # Upstream asserts its fixed backbone parameter partition. Temporarily
            # unregister the independent memory module, restoring even on failure.
            memory = self.ngram_memory
            ncp = self.ncp
            self.ngram_memory = None
            self.ncp = None
            try:
                optimizer = super().setup_optimizer(**kwargs)
            finally:
                self.ngram_memory = memory
                self.ncp = ncp
            if variant == "moe":
                routers = [block.mlp.router.weight for block in self.transformer.h]
                router_ids = {id(p) for p in routers}
                for group in optimizer.param_groups:
                    group["params"] = [p for p in group["params"] if id(p) not in router_ids]
                optimizer.param_groups = [group for group in optimizer.param_groups if group["params"]]
                optimizer.add_param_group(dict(kind="adamw", params=routers, lr=candidate["router_lr"],
                                               initial_lr=candidate["router_lr"], betas=(.9, .999),
                                               eps=1e-8, weight_decay=0.0))
                for index, parameter in enumerate(routers):
                    def record_gradient(grad, layer=index):
                        with torch.no_grad():
                            self._stats_router_grad_max[layer].copy_(torch.maximum(self._stats_router_grad_max[layer], grad.abs().max()))
                            self._stats_router_grad_finite[layer].logical_and_(torch.isfinite(grad).all())
                        return grad
                    parameter.register_hook(record_gradient)
            if memory is not None:
                memory_lr = candidate["memory"].get("lr", .001)
                optimizer.add_param_group(dict(kind="adamw", params=list(memory.parameters()),
                                               lr=memory_lr, initial_lr=memory_lr, betas=(.9, .999),
                                               eps=1e-8, weight_decay=0.0))
            if ncp is not None:
                lr = candidate['ncp']['lr']
                optimizer.add_param_group(dict(kind='adamw', params=list(ncp.parameters()),
                    lr=lr, initial_lr=lr, betas=(.9,.999), eps=1e-8, weight_decay=0.0))
            self.optimizer_report = optimizer_coverage(self, optimizer)
            return optimizer

        def parameter_report(self):
            total = sum(p.numel() for p in self.parameters())
            expert_parameters = 0
            router_parameters = 0
            inactive = 0
            if variant == "moe":
                for block in self.transformer.h:
                    per_expert = sum(p.numel() for p in block.mlp.experts[0].parameters())
                    expert_parameters += num_experts * per_expert
                    inactive += (num_experts - 1) * per_expert
                    router_parameters += sum(p.numel() for p in block.mlp.router.parameters())
            memory_parameters = 0 if self.ngram_memory is None else sum(
                p.numel() for p in self.ngram_memory.parameters())
            memory_table_bytes = 0
            if self.ngram_memory is not None:
                inactive += 2 * (8192 - 1) * 64
                memory_table_bytes = 2 * 8192 * 64 * 4
            return {"feedforward": variant, "depth": self.config.n_layer, "width": self.config.n_embd,
                    "expert_dispatch": "packed_stable" if variant == "moe" else None,
                    "activation_checkpointing": bool(self.config.use_activation_checkpointing),
                    "total_parameters": total, "active_parameters": total - inactive,
                    "expert_parameters": expert_parameters, "router_parameters": router_parameters,
                    "memory": None if self.ngram_memory is None else "ngram_v1",
                    "memory_parameters": memory_parameters,
                    "memory_after_layer": None if self.ngram_memory is None else candidate["memory"].get("after_layer", 1),
                    "memory_lr": None if self.ngram_memory is None else candidate["memory"].get("lr", .001),
                    "memory_initialization_seed": None if self.ngram_memory is None else self.ngram_memory.initialization_seed,
                    "active_parameter_convention": "All shared tensors including full embedding tables, all router weights, and one expert per layer; structural per-token count, not measured FLOPs",
                    "memory_active_parameter_convention": "At most two selected 64-value rows plus all projection/gate parameters; excludes unselected table rows, not optimizer work",
                    "ncp_parameters": 0 if self.ncp is None else sum(p.numel() for p in self.ncp.parameters()),
                    "ncp_codebook_parameters": 0 if self.ncp is None else sum(p.numel() for p in self.ncp.codebook.parameters()),
                    "ncp_codebook_buffer_bytes": 0 if self.ncp is None else self.ncp.codebook.basis.numel()*4,
                    "ncp_effective_codebook_bytes": 0 if self.ncp is None else self.ncp.entries*self.config.n_embd*4,
                    "memory_table_bytes": memory_table_bytes, "num_experts": num_experts, "top_k": 1 if num_experts else 0}

        def estimate_flops(self):
            report = self.parameter_report()
            # Retain upstream's rough matmul estimate, excluding inactive experts.
            # This still excludes routing/dispatch overhead; it is not a measurement.
            return super().estimate_flops() - 6 * (report["total_parameters"] - report["active_parameters"])

        def num_scaling_params(self):
            report = super().num_scaling_params()
            added = 0 if self.ngram_memory is None else sum(p.numel() for p in self.ngram_memory.parameters())
            report["ngram_memory"] = added
            report["total"] += added
            ncp_added = 0 if self.ncp is None else sum(p.numel() for p in self.ncp.parameters())
            report['ncp'] = ncp_added
            report['total'] += ncp_added
            return report

        def forward(self, idx, targets=None, reduction="mean"):
            counts = None
            auxiliary = None
            concept_losses = {}
            if variant == "dense" and self.ngram_memory is None and self.ncp is None:
                value = super().forward(idx, targets, reduction)
            else:
                _, length = idx.shape
                assert length <= self.cos.size(1)
                cos_sin = self.cos[:, :length], self.sin[:, :length]
                x = upstream.norm(self.transformer.wte(idx))
                x0 = x
                losses, layer_counts = [], []
                for index, block in enumerate(self.transformer.h):
                    x = self.resid_lambdas[index] * x + self.x0_lambdas[index] * x0
                    ve = self.value_embeds[str(index)](idx) if str(index) in self.value_embeds else None
                    args = (x, ve, cos_sin, self.window_sizes[index])
                    if self.config.use_activation_checkpointing:
                        result = checkpoint(block, *args, use_reentrant=False)
                    else:
                        result = block(*args)
                    if variant == "moe":
                        x, aux, count = result
                        losses.append(aux)
                        layer_counts.append(count)
                    else:
                        x = result
                    if self.ngram_memory is not None and index == candidate["memory"].get("after_layer", 1):
                        if self.config.use_activation_checkpointing:
                            x = checkpoint(self.ngram_memory, x, idx, use_reentrant=False)
                        else:
                            x = self.ngram_memory(x, idx)
                    if self.ncp is not None and index == candidate['ncp']['after_layer']:
                        x, concept_losses = self.ncp(x, compute_loss=targets is not None and self.training and reduction=='mean')
                if losses:
                    auxiliary = torch.stack(losses).mean()
                    counts = torch.stack(layer_counts)
                logits = self.lm_head(upstream.norm(x)).float()
                logits = 15 * torch.tanh(logits / 15)
                value = logits if targets is None else F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1, reduction=reduction)
            if targets is not None:
                phase = "train" if self.training else "eval"
                tokens = idx.numel()
                with torch.no_grad():
                    getattr(self, f"_stats_{phase}_tokens").add_(tokens)
                    if counts is not None:
                        getattr(self, f"_stats_{phase}_counts").add_(counts)
                    if self.training:
                        self._stats_ce_sum.add_(value.detach().double().mean() * tokens)
                        if auxiliary is not None:
                            self._stats_aux_sum.add_(auxiliary.detach().double() * tokens)
                # The validation API uses reduction='none'; never mix routing loss into it.
                if self.training and reduction == "mean" and auxiliary is not None:
                    return value + self.aux_loss_weight * auxiliary + concept_losses.get("total", 0)
                if self.training and reduction == 'mean' and concept_losses:
                    return value + concept_losses['total']
            return value

        def routing_report(self):
            result = {"feedforward": variant, "num_experts": num_experts,
                      "aux_loss_weight": self.aux_loss_weight,
                      "router_gradient_max_abs": self._stats_router_grad_max.tolist(),
                      "router_gradients_finite": bool(self._stats_router_grad_finite.all().item())}
            for phase in ("train", "eval"):
                tokens = int(getattr(self, f"_stats_{phase}_tokens").item())
                counts = getattr(self, f"_stats_{phase}_counts").tolist() if num_experts else []
                result[phase] = {"tokens": tokens, "counts": counts,
                                 "fractions": [[count / tokens if tokens else 0.0 for count in layer] for layer in counts],
                                 "dropped_tokens": tokens * self.config.n_layer - sum(map(sum, counts)) if num_experts else 0}
            denominator = max(result["train"]["tokens"], 1)
            result["training_cross_entropy"] = self._stats_ce_sum.item() / denominator
            result["training_auxiliary_loss"] = self._stats_aux_sum.item() / denominator
            result["weighted_training_auxiliary_loss"] = result["training_auxiliary_loss"] * self.aux_loss_weight
            return result

    return ExperimentGPT
