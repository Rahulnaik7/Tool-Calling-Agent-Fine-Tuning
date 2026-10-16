"""Evaluation: runs the policy over a set of examples and reports
function-calling accuracy (`reward.exact_match`) -- the metric the resume
bullet's "67% baseline to 75%" figure comes from, now measured for real
against this project's own numbers instead of asserted.

Decoding is near-greedy (a very low sampling temperature) so repeated runs
against the same checkpoint are effectively deterministic, which is what
"baseline" and "post-GRPO" numbers need to be comparable at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from model import TinyToolCallTransformer
from reward import ast_schema_reward, exact_match
from tokenizer import Tokenizer

EVAL_TEMPERATURE = 0.05


@torch.no_grad()
def generate_completion(
    model: TinyToolCallTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 24,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id, temperature=EVAL_TEMPERATURE)
    tail = out[0, len(prompt_ids):].tolist()
    if tokenizer.eos_id in tail:
        tail = tail[: tail.index(tokenizer.eos_id) + 1]
    return tokenizer.decode(tail)


def evaluate_accuracy(
    model: TinyToolCallTransformer,
    tokenizer: Tokenizer,
    examples: List[Dict],
    device: torch.device = torch.device("cpu"),
) -> Dict:
    per_example = []
    correct = 0
    for ex in examples:
        completion = generate_completion(model, tokenizer, ex["prompt"], device=device)
        is_correct = exact_match(completion, ex["tool"], ex["arguments"])
        reward = ast_schema_reward(completion, ex["tool"], ex["arguments"])
        correct += int(is_correct)
        per_example.append(
            {
                "instruction": ex["instruction"],
                "expected_tool": ex["tool"],
                "completion": completion,
                "correct": is_correct,
                "reward": reward,
            }
        )
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "n": len(examples), "n_correct": correct, "per_example": per_example}


def write_report(baseline: Dict, post_grpo: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_accuracy": baseline["accuracy"],
        "post_grpo_accuracy": post_grpo["accuracy"],
        "delta_pp": round((post_grpo["accuracy"] - baseline["accuracy"]) * 100, 2),
        "n_eval_examples": baseline["n"],
        "baseline": baseline,
        "post_grpo": post_grpo,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# BFCL-Style Held-Out Evaluation Report\n",
        "## Summary\n",
        f"- Held-out examples: {baseline['n']}\n",
        f"- Baseline (post-SFT, pre-GRPO) accuracy: {baseline['accuracy'] * 100:.1f}%\n",
        f"- Post-GRPO accuracy: {post_grpo['accuracy'] * 100:.1f}%\n",
        f"- Delta: {payload['delta_pp']:+.1f}pp\n",
        "\n## Sample predictions (post-GRPO)\n",
    ]
    for row in post_grpo["per_example"][:10]:
        mark = "correct" if row["correct"] else "incorrect"
        lines.append(f"- [{mark}] \"{row['instruction']}\" -> `{row['completion']}` (expected `{row['expected_tool']}`)\n")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
