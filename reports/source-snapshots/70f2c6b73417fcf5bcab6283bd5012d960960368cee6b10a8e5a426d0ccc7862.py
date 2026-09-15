"""Frozen ngram_v1: causal, BOS-local lookup with no mutable forward history."""
import math

import torch
from torch import nn
from torch.nn import functional as F


HASH_VERSION = "polynomial-mod-v1"
TABLE_ROWS = 8192
TABLE_WIDTH = 64
HASH_MODULUS = 2147483647


def suffix_keys(ids, bos_token_id, vocab_size, boundaries=None):
    """Return exact int64 addresses and validity [batch, time, (2, 3)].

    A boundary marks an excluded boundary token, like BOS. No preceding batch,
    row or call is consulted. Input IDs, never target IDs, are the only keys.
    """
    if ids.ndim != 2 or ids.dtype != torch.int64:
        raise ValueError("lookup IDs must be a rank-two int64 tensor")
    if not 0 <= bos_token_id < vocab_size <= HASH_MODULUS:
        raise ValueError("invalid tokenizer bounds")
    bounded = ((ids >= 0) & (ids < vocab_size)).all()
    if ids.device.type == "cuda":
        torch._assert_async(bounded, "lookup input token outside tokenizer vocabulary")
    elif not bool(bounded):
        raise ValueError("lookup input token outside tokenizer vocabulary")
    excluded = ids == bos_token_id
    if boundaries is not None:
        if boundaries.shape != ids.shape or boundaries.dtype != torch.bool:
            raise ValueError("boundaries must be a matching boolean tensor")
        excluded = excluded | boundaries
    addresses, masks = [], []
    for order in (2, 3):
        hashed = torch.full_like(ids, order)
        valid = torch.ones_like(ids, dtype=torch.bool)
        for offset in range(order - 1, -1, -1):
            shifted = torch.full_like(ids, bos_token_id)
            allowed = torch.zeros_like(ids, dtype=torch.bool)
            if offset == 0:
                shifted = ids
                allowed = ~excluded
            elif offset < ids.shape[1]:
                shifted[:, offset:] = ids[:, :-offset]
                allowed[:, offset:] = ~excluded[:, :-offset]
            # h < 2**31 and u < 2**31: product stays far below int64 max.
            hashed = (hashed * 1000003 + shifted + 1).remainder(HASH_MODULUS)
            valid = valid & allowed
        # Invalid addresses are a sentinel, not a trainable padding row.
        addresses.append(torch.where(valid, hashed.remainder(TABLE_ROWS), -1))
        masks.append(valid)
    return torch.stack(addresses, -1), torch.stack(masks, -1)


class CausalNgramMemory(nn.Module):
    def __init__(self, width, bos_token_id, vocab_size, norm):
        super().__init__()
        self.E2 = nn.Parameter(torch.empty(TABLE_ROWS, TABLE_WIDTH, dtype=torch.float32))
        self.E3 = nn.Parameter(torch.empty(TABLE_ROWS, TABLE_WIDTH, dtype=torch.float32))
        self.Wv = nn.Parameter(torch.empty(width, 2 * TABLE_WIDTH, dtype=torch.float32))
        self.wg = nn.Parameter(torch.empty(width, dtype=torch.float32))
        self.bg = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.width, self.bos_token_id, self.vocab_size = width, bos_token_id, vocab_size
        self.norm = norm
        self.diagnostics_enabled = False
        self.last_diagnostics = None
        self.initialization_seed = None

    @torch.no_grad()
    def initialize(self, seed):
        device = self.E2.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            nn.init.normal_(self.E2, std=.02)
            nn.init.normal_(self.E3, std=.02)
            nn.init.normal_(self.Wv, std=1 / math.sqrt(128))
            self.wg.zero_()
            self.bg.fill_(-2)
        self.initialization_seed = seed

    def keys(self, ids, boundaries=None):
        return suffix_keys(ids, self.bos_token_id, self.vocab_size, boundaries)

    def lookup(self, ids, boundaries=None):
        addresses, valid = self.keys(ids, boundaries)
        return torch.cat([
            F.embedding(addresses[..., i].clamp_min(0), table) * valid[..., i, None]
            for i, table in enumerate((self.E2, self.E3))
        ], -1)

    def components(self, hidden, ids):
        memory = self.lookup(ids).to(hidden.dtype)
        projected = F.linear(memory, self.Wv)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            gate = torch.sigmoid(F.linear(self.norm(hidden.float()), self.wg[None], self.bg[None]))
        residual = (gate * projected).to(hidden.dtype)
        return gate, residual

    def forward(self, hidden, ids):
        gate, residual = self.components(hidden, ids)
        if self.diagnostics_enabled:
            self.last_diagnostics = self.diagnostics(hidden, ids, gate=gate)
        return hidden + residual

    @torch.no_grad()
    def diagnostics(self, hidden, ids, gate=None):
        """Explicit expensive observation only; never enabled in timed training."""
        if gate is None:
            gate, _ = self.components(hidden, ids)
        addresses, valid = self.keys(ids)
        flat_gate = gate.detach().float().flatten()
        result = {
            "hash_version": HASH_VERSION, "initialization_seed": self.initialization_seed,
            "gate_mean": flat_gate.mean().item(), "gate_min": flat_gate.min().item(),
            "gate_max": flat_gate.max().item(),
            "gate_quantiles_0_25_50_75_100": torch.quantile(flat_gate, torch.tensor(
                [0., .25, .5, .75, 1.], device=flat_gate.device)).tolist(),
            "valid_rates": valid.float().mean((0, 1)).tolist(),
            "accessed_rows": [addresses[..., i][valid[..., i]].unique().numel() for i in range(2)],
            "gradient_norms": {name: None if p.grad is None else p.grad.float().norm().item()
                               for name, p in self.named_parameters()},
            "scope": "explicit diagnostic batch, not all training tokens",
        }
        return result


class IncrementalLookupState:
    """Lookup-only generation state. This is not a model KV-cache implementation."""
    def __init__(self, batch_size, bos_token_id, vocab_size):
        self.bos_token_id, self.vocab_size = bos_token_id, vocab_size
        self.histories = [[] for _ in range(batch_size)]
        self.identities = [None] * batch_size

    def reset(self, rows=None):
        for row in range(len(self.histories)) if rows is None else rows:
            self.histories[row] = []
            self.identities[row] = None

    def reorder(self, permutation):
        if sorted(permutation) != list(range(len(self.histories))):
            raise ValueError("history reorder must be a complete batch permutation")
        self.histories = [self.histories[i][:] for i in permutation]
        self.identities = [self.identities[i] for i in permutation]

    def step(self, ids, identities=None):
        if ids.ndim != 1 or ids.shape[0] != len(self.histories):
            raise ValueError("one token per sequence is required")
        if identities is not None and len(identities) != len(self.histories):
            raise ValueError("one identity per sequence is required")
        rows = []
        for i, token in enumerate(ids.tolist()):
            if not 0 <= token < self.vocab_size:
                raise ValueError("lookup input token outside tokenizer vocabulary")
            if identities is not None and identities[i] != self.identities[i]:
                self.histories[i] = []
                self.identities[i] = identities[i]
            previous = self.histories[i]
            rows.append([self.bos_token_id] * (2 - len(previous)) + previous + [token])
            self.histories[i] = [] if token == self.bos_token_id else (previous + [token])[-2:]
        keys, masks = suffix_keys(torch.tensor(rows, device=ids.device, dtype=torch.int64),
                                 self.bos_token_id, self.vocab_size)
        return keys[:, -1], masks[:, -1]

    def state_dict(self):
        return {"histories": [row[:] for row in self.histories], "identities": self.identities[:],
                "bos_token_id": self.bos_token_id, "vocab_size": self.vocab_size}

    def load_state_dict(self, state):
        if state["bos_token_id"] != self.bos_token_id or state["vocab_size"] != self.vocab_size:
            raise ValueError("lookup-state tokenizer mismatch")
        histories = state["histories"]
        if len(histories) != len(self.histories) or len(state["identities"]) != len(histories):
            raise ValueError("lookup-state batch mismatch")
        if any(len(row) > 2 or any(type(t) is not int or not 0 <= t < self.vocab_size
                                  or t == self.bos_token_id for t in row) for row in histories):
            raise ValueError("invalid lookup history")
        self.histories = [row[:] for row in histories]
        self.identities = state["identities"][:]
