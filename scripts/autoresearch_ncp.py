"""Small ConceptLM-inspired chunk predictor. No future target enters feedback."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConceptBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        b,t,d = x.shape
        normalized = F.rms_norm(x, (d,))
        q,k,v = self.qkv(normalized).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        attended = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2),
            v.transpose(1,2), is_causal=True).transpose(1,2).reshape(b,t,d)
        x = x + self.proj(attended)
        return x + self.down(F.relu(self.fc(F.rms_norm(x,(d,)))).square())


class SegmentCodebook(nn.Module):
    """Frozen random basis with learned per-segment two-layer transforms."""
    def __init__(self, segments, entries, dimension):
        super().__init__()
        self.register_buffer('basis',torch.empty(segments,entries,dimension))
        self.first = nn.Parameter(torch.empty(segments,dimension,dimension))
        self.second = nn.Parameter(torch.empty(segments,dimension,dimension))

    def forward(self):
        # Quantization distances and dictionary transform stay float32.
        with torch.autocast(self.basis.device.type, enabled=False):
            return torch.bmm(F.relu(torch.bmm(self.basis,self.first)),self.second)


class NextConcept(nn.Module):
    def __init__(self, width, heads, settings):
        super().__init__()
        self.settings = dict(settings)
        self.width, self.segments = width, heads
        self.chunk_size, self.entries = settings['chunk_size'], settings['entries']
        self.blocks = nn.ModuleList([ConceptBlock(width, heads) for _ in range(settings['layers'])])
        self.head = nn.Linear(width, heads * self.entries, bias=False)
        self.codebook = SegmentCodebook(heads, self.entries, width // heads)
        self.register_buffer('_loss_sums',torch.zeros(4,dtype=torch.float64),persistent=False)
        self.register_buffer('_loss_count',torch.zeros((),dtype=torch.int64),persistent=False)
        self.register_buffer('_last_token_ce',torch.zeros(()),persistent=False)

    @torch.no_grad()
    def initialize(self, seed):
        device = self.head.weight.device
        with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            for block in self.blocks:
                for parameter in block.parameters():
                    nn.init.normal_(parameter,std=.02)
            nn.init.normal_(self.head.weight,std=.02)
            nn.init.normal_(self.codebook.basis,std=1.)
            d = self.width // self.segments
            nn.init.normal_(self.codebook.first,std=1/math.sqrt(d))
            nn.init.normal_(self.codebook.second,std=.5/math.sqrt(d))
        self._loss_sums.zero_(); self._loss_count.zero_()
        self._last_token_ce.zero_()

    def pool(self, hidden):
        b,t,d = hidden.shape
        n = t // self.chunk_size
        pooled=hidden[:,:n*self.chunk_size].reshape(b,n,self.chunk_size,d).mean(2)
        if self.settings.get('pool_normalization','none')=='rms':
            with torch.autocast(hidden.device.type,enabled=False):
                pooled=F.rms_norm(pooled.float(),(d,),eps=1e-6).to(hidden.dtype)
        return pooled

    def broadcast(self, prediction, length):
        repeated = prediction.repeat_interleave(self.chunk_size,dim=1)
        zeros = prediction.new_zeros(prediction.shape[0],self.chunk_size-1,self.width)
        return torch.cat((zeros,repeated),dim=1)[:,:length]

    def predict(self, pooled, codes):
        x = pooled
        # Absolute sinusoidal positions identify chunk starts; computed from prefix only.
        position = torch.arange(x.shape[1],device=x.device,dtype=torch.float32) * self.chunk_size
        frequency = torch.exp(torch.arange(0,self.width,2,device=x.device,dtype=torch.float32)*(-math.log(10000)/self.width))
        angle = position[:,None]*frequency[None]
        positional = torch.stack((angle.sin(),angle.cos()),dim=-1).flatten(-2)
        x = x + positional.to(x.dtype)[None] * .1
        for block in self.blocks: x = block(x)
        logits = self.head(F.rms_norm(x,(self.width,))).float().reshape(*x.shape[:2],self.segments,self.entries)
        with torch.autocast(x.device.type, enabled=False):
            weights = logits if self.settings.get('mixing','softmax')=='raw_logits' else logits.softmax(-1)
            prediction = torch.einsum('btsn,snd->btsd', weights, codes).flatten(-2)
        return prediction, logits

    def quantize(self, pooled, codes):
        with torch.autocast(pooled.device.type, enabled=False):
            target = pooled.detach().float().reshape(*pooled.shape[:2],self.segments,-1)
            distances = target.square().sum(-1,keepdim=True) + codes.square().sum(-1)[None,None] - 2*torch.einsum('btsd,snd->btsn',target,codes)
            indices = distances.detach().argmin(-1)
            selected = codes[torch.arange(self.segments,device=codes.device)[None,None,:],indices]
        return selected, indices

    def forward(self, hidden, compute_loss=False, target_hidden=None):
        pooled = self.pool(hidden)
        losses = {}
        if pooled.shape[1] == 0:
            return hidden, losses
        codes = self.codebook()
        prediction, logits = self.predict(pooled,codes)
        fused = hidden
        if self.settings['mode'] == 'feedback':
            fused = hidden + self.settings['feedback_scale'] * self.broadcast(prediction,hidden.shape[1]).to(hidden.dtype)
        if compute_loss:
            # This branch is downstream of the complete predictive computation.
            target = self.pool(hidden if target_hidden is None else target_hidden).detach().float()
            selected, indices = self.quantize(target,codes)
            losses['vq'] = F.mse_loss(selected.flatten(-2),target)
            if pooled.shape[1] > 1:
                losses['prediction'] = F.mse_loss(prediction[:,:-1],target[:,1:])
                losses['concept_ce'] = F.cross_entropy(logits[:,:-1].reshape(-1,self.entries),indices[:,1:].reshape(-1))
            else:
                losses['prediction'] = prediction.sum()*0
                losses['concept_ce'] = logits.sum()*0
            losses['total'] = (self.settings['prediction_weight']*losses['prediction'] +
                self.settings['vq_weight']*losses['vq'] + self.settings['ce_weight']*losses['concept_ce'])
            if self.training:
                with torch.no_grad():
                    self._loss_sums.add_(torch.stack([losses[k].detach().double() for k in ('prediction','vq','concept_ce','total')]))
                    self._loss_count.add_(1)
        return fused, losses

    @torch.no_grad()
    def diagnostics(self, hidden):
        pooled = self.pool(hidden)
        codes = self.codebook()
        prediction, logits = self.predict(pooled,codes)
        _, indices = self.quantize(pooled,codes)
        counts = torch.stack([torch.bincount(indices[:,:,s].flatten(),minlength=self.entries) for s in range(self.segments)])
        predictions = torch.stack([torch.bincount(logits[:,:,s].argmax(-1).flatten(),minlength=self.entries) for s in range(self.segments)])
        probabilities = logits.softmax(-1)
        return dict(target_counts=counts.tolist(), prediction_counts=predictions.tolist(),
            mean_prediction_entropy=float(-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean()),
            next_concept_mse=float(F.mse_loss(prediction[:,:-1],pooled[:,1:].float())),
            next_concept_accuracy=float((logits[:,:-1].argmax(-1)==indices[:,1:]).float().mean()),
            feedback_rms=float(self.broadcast(prediction,hidden.shape[1]).square().mean().sqrt()),
            hidden_rms=float(hidden.float().square().mean().sqrt()),
            pooled_rms=float(pooled.float().square().mean().sqrt()),
            codebook_rms=float(codes.square().mean().sqrt()))
