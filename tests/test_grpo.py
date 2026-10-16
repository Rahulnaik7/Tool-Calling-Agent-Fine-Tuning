import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import copy

import torch

from grpo import GRPOConfig, _generated_length, grpo_train, group_relative_advantages
from lora import apply_lora, freeze_non_lora, lora_parameters
from model import ModelConfig, TinyToolCallTransformer
from tokenizer import Tokenizer


def test_group_relative_advantages_centers_on_zero():
    rewards = torch.tensor([0.2, 0.5, 0.8, 0.3])
    adv = group_relative_advantages(rewards)
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-5)


def test_group_relative_advantages_ranks_preserved():
    rewards = torch.tensor([0.1, 0.9, 0.5])
    adv = group_relative_advantages(rewards)
    assert adv[1] > adv[2] > adv[0]  # highest reward -> highest advantage


def test_group_relative_advantages_zero_variance_is_safe():
    rewards = torch.tensor([0.5, 0.5, 0.5])
    adv = group_relative_advantages(rewards)
    assert torch.all(adv == 0.0)
    assert not torch.isnan(adv).any()


def test_generated_length_stops_at_first_eos():
    assert _generated_length([3, 4, 9, 5], eos_id=9) == 3
    assert _generated_length([3, 4, 5], eos_id=9) == 3  # no eos -> full length


def _toy_examples(tokenizer: Tokenizer):
    return [
        {
            "prompt": "Instruction: add 1 and 2.\nTools: math_service(operation, x, y)\nCall:",
            "target": "math_service(operation='add', x=1, y=2)",
            "tool": "math_service",
            "arguments": {"operation": "add", "x": 1, "y": 2},
        },
        {
            "prompt": "Instruction: weather in Tokyo.\nTools: weather_api(location, units)\nCall:",
            "target": "weather_api(location='Tokyo')",
            "tool": "weather_api",
            "arguments": {"location": "Tokyo"},
        },
    ]


def _toy_reward(completion: str, tool: str, arguments) -> float:
    return 1.0 if tool in completion else 0.0


def test_grpo_train_only_updates_lora_parameters():
    texts = [
        "Instruction: add 1 and 2.\nTools: math_service(operation, x, y)\nCall: math_service(operation='add', x=1, y=2)",
        "Instruction: weather in Tokyo.\nTools: weather_api(location, units)\nCall: weather_api(location='Tokyo')",
    ]
    tokenizer = Tokenizer.build(texts)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layer=2, n_head=2, d_ff=32, max_seq_len=48)
    model = TinyToolCallTransformer(cfg)
    ref_model = copy.deepcopy(model)

    apply_lora(model, target_suffixes=["qkv", "proj"], r=2, alpha=4)
    freeze_non_lora(model)

    base_weight_before = model.blocks[0].attn.qkv.base.weight.detach().clone()
    lora_A_before = model.blocks[0].attn.qkv.lora_A.detach().clone()

    grpo_cfg = GRPOConfig(steps=2, group_size=3, prompts_per_step=2, lr=1e-2, max_new_tokens=6, kl_coef=0.01, log_every=100)
    grpo_train(model, ref_model, tokenizer, _toy_examples(tokenizer), _toy_reward, grpo_cfg)

    base_weight_after = model.blocks[0].attn.qkv.base.weight.detach()
    lora_A_after = model.blocks[0].attn.qkv.lora_A.detach()

    torch.testing.assert_close(base_weight_before, base_weight_after)  # frozen base never moves
    assert not torch.allclose(lora_A_before, lora_A_after)  # LoRA params did move


def test_grpo_train_does_not_crash_with_single_prompt_group():
    texts = ["Instruction: add 1 and 2.\nTools: math_service(operation, x, y)\nCall: math_service(operation='add', x=1, y=2)"]
    tokenizer = Tokenizer.build(texts)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layer=2, n_head=2, d_ff=32, max_seq_len=48)
    model = TinyToolCallTransformer(cfg)
    ref_model = copy.deepcopy(model)
    apply_lora(model, target_suffixes=["qkv"], r=2, alpha=4)
    freeze_non_lora(model)

    examples = [_toy_examples(tokenizer)[0]]
    grpo_cfg = GRPOConfig(steps=1, group_size=2, prompts_per_step=1, max_new_tokens=4, log_every=100)
    grpo_train(model, ref_model, tokenizer, examples, _toy_reward, grpo_cfg)  # just must not raise
