"""Phase 2: GRPO (Group Relative Policy Optimization) fine-tuning.

For each prompt, the policy samples a *group* of G completions. Each is
scored by the deterministic AST/schema reward (`reward.py`) -- there is no
learned value network or critic anywhere in this file, which is GRPO's
defining difference from PPO: the advantage of each completion is just its
reward normalized against the *other completions sampled for the same
prompt* (group mean/std), rather than a baseline predicted by a trained
critic.

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

The policy-gradient loss is the standard REINFORCE-with-baseline form,
`-A_i * log p(token)`, summed over each completion's generated tokens, plus
a KL penalty against a frozen reference policy (a snapshot of the model
right after the SFT warm-start) so the GRPO phase refines behavior without
drifting arbitrarily far from it.

Honest simplification vs. the original GRPO paper (DeepSeekMath): this
does exactly one gradient step per batch of rollouts (fully on-policy), so
there is no PPO-style importance-sampling ratio or clipping term to add --
those exist to make *multiple* gradient steps safe per rollout batch,
which this trainer doesn't do. The KL term is computed directly as
`E[logp_policy - logp_ref]` over generated tokens rather than the paper's
unbiased k3 estimator. The group-relative-advantage mechanic itself -- the
part of GRPO that replaces the critic, and the part the resume bullet
refers to -- is implemented exactly as described above.

Only LoRA parameters (`lora.py`) receive gradients here; the base policy
weights (set during SFT) are frozen for this whole phase.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from lora import lora_parameters
from model import TinyToolCallTransformer
from tokenizer import Tokenizer


@dataclass
class GRPOConfig:
    steps: int = 150
    group_size: int = 6
    prompts_per_step: int = 4
    lr: float = 5e-3
    max_new_tokens: int = 24
    temperature: float = 1.0
    kl_coef: float = 0.02
    log_every: int = 10


@dataclass
class GRPOStats:
    mean_reward_per_step: List[float] = field(default_factory=list)
    loss_per_step: List[float] = field(default_factory=list)


def _generated_length(seq_tail: List[int], eos_id: int) -> int:
    """How many tokens of the generated tail are "real" (up to and
    including the first eos, or all of them if eos never appears)."""
    for i, tok in enumerate(seq_tail):
        if tok == eos_id:
            return i + 1
    return len(seq_tail)


def grpo_train(
    model: TinyToolCallTransformer,
    ref_model: TinyToolCallTransformer,
    tokenizer: Tokenizer,
    examples: List[Dict],
    reward_fn: Callable[[str, str, Dict], float],
    cfg: GRPOConfig,
    device: torch.device = torch.device("cpu"),
) -> GRPOStats:
    model.to(device)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable = lora_parameters(model)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    stats = GRPOStats()

    for step in range(cfg.steps):
        batch_examples = random.sample(examples, min(cfg.prompts_per_step, len(examples)))

        step_loss = torch.zeros((), device=device)
        step_rewards: List[float] = []
        total_tokens = 0

        optimizer.zero_grad()

        for ex in batch_examples:
            prompt_ids = tokenizer.encode(ex["prompt"], add_bos=True)
            prompt_len = len(prompt_ids)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(cfg.group_size, 1)

            model.eval()
            gen_idx = model.generate(
                prompt_tensor,
                max_new_tokens=cfg.max_new_tokens,
                eos_id=tokenizer.eos_id,
                temperature=cfg.temperature,
            )
            model.train()

            gen_len_total = gen_idx.shape[1] - prompt_len
            if gen_len_total <= 0:
                continue

            rewards = []
            valid_lens = []
            for g in range(cfg.group_size):
                tail = gen_idx[g, prompt_len:].tolist()
                vlen = _generated_length(tail, tokenizer.eos_id)
                valid_lens.append(vlen)
                completion_text = tokenizer.decode(tail[:vlen])
                r = reward_fn(completion_text, ex["tool"], ex["arguments"])
                rewards.append(r)

            reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            mean_r, std_r = reward_t.mean(), reward_t.std(unbiased=False)
            if std_r.item() < 1e-6:
                advantages = torch.zeros_like(reward_t)
            else:
                advantages = (reward_t - mean_r) / (std_r + 1e-6)

            step_rewards.extend(rewards)

            # Score the full rollout with the current (trainable) policy
            # and the frozen reference, in one forward pass each.
            policy_logits = model(gen_idx[:, :-1])
            policy_logp_all = F.log_softmax(policy_logits, dim=-1)
            targets = gen_idx[:, 1:]
            policy_logp_tok = policy_logp_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                ref_logits = ref_model(gen_idx[:, :-1])
                ref_logp_all = F.log_softmax(ref_logits, dim=-1)
                ref_logp_tok = ref_logp_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Build a per-token mask covering only the generated region,
            # truncated to each row's valid (pre-eos) length.
            T = targets.shape[1]
            mask = torch.zeros((cfg.group_size, T), dtype=torch.bool, device=device)
            for g in range(cfg.group_size):
                start = prompt_len - 1  # index into `targets`/shifted logits
                end = min(start + valid_lens[g], T)
                if end > start:
                    mask[g, start:end] = True

            pg_term = -(advantages.unsqueeze(1) * policy_logp_tok) * mask
            kl_term = (policy_logp_tok - ref_logp_tok) * mask
            token_loss = pg_term + cfg.kl_coef * kl_term

            n_valid = mask.sum().clamp(min=1)
            step_loss = step_loss + token_loss.sum() / n_valid
            total_tokens += int(n_valid.item())

        if total_tokens == 0:
            continue

        step_loss = step_loss / len(batch_examples)
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        stats.loss_per_step.append(step_loss.item())
        stats.mean_reward_per_step.append(sum(step_rewards) / max(len(step_rewards), 1))

        if (step + 1) % cfg.log_every == 0:
            print(
                f"[grpo] step {step + 1}/{cfg.steps} "
                f"loss {step_loss.item():.4f} mean_reward {stats.mean_reward_per_step[-1]:.4f}"
            )

    return stats
