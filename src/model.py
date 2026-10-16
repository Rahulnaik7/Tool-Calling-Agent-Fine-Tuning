"""A small, from-scratch decoder-only transformer: the policy model that
LoRA (`lora.py`) and GRPO (`grpo.py`) actually fine-tune in this repo.

**Why not Qwen2.5-7B itself.** The resume line this project is built to
back up names Qwen2.5-7B on an A100. This sandbox has no GPU
(`nvidia-smi` / `torch.cuda.is_available()` both come back negative) and no
network path to the Hugging Face Hub (confirmed via a 403 from the proxy,
same constraint hit and documented in the Two-Stage-Document-Retrieval
project) to download Qwen's weights even for CPU inference. A 7B-parameter
model's forward pass alone is impractical on 2 CPU cores regardless.

So this is a real, from-scratch, from-random-init decoder-only transformer
(multi-head self-attention, pre-norm residual blocks, a feed-forward block
per layer) sized to actually train end-to-end on CPU in minutes --- the
same honesty trade-off as `TfidfLsaEncoder` standing in for a neural
bi-encoder in the retrieval project: every mechanism downstream of it
(LoRA adapters, GRPO's group-relative policy gradient, the deterministic
AST/schema reward) is the real algorithm, just run on a model sized for
what this environment can actually execute rather than against a
7B-parameter checkpoint it has no way to load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyToolCallTransformer(nn.Module):
    """A compact GPT-style causal language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0,
        collect_logprobs: bool = False,
    ):
        """Autoregressively sample up to `max_new_tokens` tokens.

        Returns the full sequence (prompt + generated), and optionally the
        per-step log-probabilities of the tokens actually sampled (needed
        by GRPO for the policy-gradient loss).
        """
        logprobs = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            log_probs_full = F.log_softmax(logits, dim=-1)
            probs = log_probs_full.exp()
            next_id = torch.multinomial(probs, num_samples=1)
            if collect_logprobs:
                logprobs.append(log_probs_full.gather(1, next_id).squeeze(1))
            idx = torch.cat([idx, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        if collect_logprobs:
            return idx, logprobs
        return idx
