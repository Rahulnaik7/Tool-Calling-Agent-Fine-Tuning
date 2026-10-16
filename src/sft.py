"""Phase 1: supervised warm-start (behavior cloning).

A real base LLM like Qwen2.5-7B already has broad instruction-following
and tool-use competence from large-scale pretraining before any
project-specific RL touches it; GRPO in that setting *refines* an already
decent policy rather than teaching one from nothing. This project's policy
starts from random weights (see `model.py`), so to make the GRPO phase a
fair analogue -- refinement, not "learn the task from scratch via sparse
RL reward" -- this script first does a short supervised fine-tune (teacher
forcing, standard cross-entropy next-token loss, masked to target tokens
only) over the training trajectories. This also establishes the pre-GRPO
baseline accuracy that `train.py` reports the GRPO lift against.

Every parameter is trainable in this phase; LoRA is applied only
afterwards, for the GRPO phase (`lora.py`, `grpo.py`), matching the
resume's "fine-tuned ... using LoRA and GRPO" framing: LoRA is the
mechanism used for the RL fine-tune specifically.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from model import TinyToolCallTransformer
from tokenizer import Tokenizer


def _encode_example(tokenizer: Tokenizer, prompt: str, target: str, max_len: int):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    target_ids = tokenizer.encode(" " + target, add_eos=True)
    ids = (prompt_ids + target_ids)[:max_len]
    prompt_len = min(len(prompt_ids), len(ids))
    return ids, prompt_len


def make_batch(tokenizer: Tokenizer, examples: List[Dict], max_len: int, device: torch.device):
    encoded = [_encode_example(tokenizer, ex["prompt"], ex["target"], max_len) for ex in examples]
    max_T = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), max_T), tokenizer.pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(encoded), max_T), dtype=torch.bool)
    for i, (ids, prompt_len) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        loss_mask[i, prompt_len : len(ids)] = True
    return input_ids.to(device), loss_mask.to(device)


def sft_train(
    model: TinyToolCallTransformer,
    tokenizer: Tokenizer,
    examples: List[Dict],
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 3e-4,
    max_len: int = 96,
    device: torch.device = torch.device("cpu"),
    log_every: int = 10,
) -> List[float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(examples)).tolist()
        for start in range(0, len(examples), batch_size):
            batch_idx = perm[start : start + batch_size]
            batch = [examples[i] for i in batch_idx]
            input_ids, loss_mask = make_batch(tokenizer, batch, max_len, device)

            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            mask = loss_mask[:, 1:]

            logp = F.log_softmax(logits, dim=-1)
            token_logp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            loss = -(token_logp * mask).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1
            if step % log_every == 0:
                print(f"[sft] epoch {epoch} step {step} loss {loss.item():.4f}")
    return losses
