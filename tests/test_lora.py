import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from lora import LoRALinear, apply_lora, count_trainable_parameters, freeze_non_lora, lora_parameters


def test_lora_linear_is_no_op_at_init():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 8)
    torch.testing.assert_close(lora(x), base(x))


def test_lora_linear_base_weights_frozen():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_linear_changes_output_after_training_step():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 8)
    before = lora(x).detach().clone()

    target = torch.randn(2, 8)
    optimizer = torch.optim.SGD([lora.lora_A, lora.lora_B], lr=0.5)
    for _ in range(5):
        optimizer.zero_grad()
        loss = ((lora(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()

    after = lora(x).detach()
    assert not torch.allclose(before, after)
    # Base weights must be untouched by the optimizer step.
    torch.testing.assert_close(lora.base(x), base(x))


def test_apply_lora_wraps_matching_linear_layers_only():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 4)
            self.proj = nn.Linear(4, 4)
            self.head = nn.Linear(4, 4)

    model = Toy()
    n_wrapped = apply_lora(model, target_suffixes=["qkv", "proj"], r=2, alpha=4)
    assert n_wrapped == 2
    assert isinstance(model.qkv, LoRALinear)
    assert isinstance(model.proj, LoRALinear)
    assert isinstance(model.head, nn.Linear)  # untouched


def test_freeze_non_lora_leaves_only_lora_params_trainable():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 4)
            self.head = nn.Linear(4, 4)

    model = Toy()
    apply_lora(model, target_suffixes=["qkv"], r=2, alpha=4)
    freeze_non_lora(model)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {"qkv.lora_A", "qkv.lora_B"}


def test_count_trainable_parameters_matches_lora_parameters():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(16, 16)

    model = Toy()
    apply_lora(model, target_suffixes=["qkv"], r=4, alpha=8)
    freeze_non_lora(model)

    expected = sum(p.numel() for p in lora_parameters(model))
    assert count_trainable_parameters(model) == expected
    assert expected == 4 * 16 + 16 * 4  # r*in + out*r
