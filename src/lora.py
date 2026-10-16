"""Real LoRA (Low-Rank Adaptation), applied to the policy's attention and
MLP projection layers.

For a frozen base weight `W` (shape out x in), LoRA adds a trainable
low-rank update `(alpha / r) * B @ A`, where `A` is (r x in) and `B` is
(out x r), both initialized so the adapter starts as a no-op (`B = 0`).
Only `A` and `B` receive gradients; `W` never does. This is the standard
LoRA parameterization (Hu et al., 2021), not a simplified stand-in --- the
only thing scaled down elsewhere in this project is the base model's size
(see `model.py`), not the fine-tuning *method* itself.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a frozen base + trainable LoRA delta."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B starts at zero so the adapter is an exact no-op at init,
        # matching the reference LoRA implementation.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def apply_lora(model: nn.Module, target_suffixes: List[str], r: int = 8, alpha: int = 16) -> int:
    """Replaces every nn.Linear whose attribute name ends with one of
    `target_suffixes` (e.g. "qkv", "proj") with a LoRALinear wrapper, in
    place. Returns the number of layers wrapped.
    """
    wrapped = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(name == suffix for suffix in target_suffixes):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                wrapped += 1
    return wrapped


def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
