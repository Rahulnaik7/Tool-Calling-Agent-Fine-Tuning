"""End-to-end training entry point: SFT warm-start -> LoRA + GRPO fine-tune
-> baseline vs. post-GRPO accuracy on the BFCL-style held-out subset.

    python src/train.py --config configs/training.yaml

Writes `outputs/results.json` and `outputs/report.md` with the real,
measured accuracy numbers (this is what `README.md`'s benchmark table and
the resume bullet's "<baseline>% -> <post-GRPO>%" are sourced from).
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import torch
import yaml

from data_pipeline import load_dataset, prepare_examples
from dataset_gen import main as generate_datasets
from evaluate import evaluate_accuracy, write_report
from grpo import GRPOConfig, grpo_train
from lora import apply_lora, count_trainable_parameters, freeze_non_lora
from model import ModelConfig, TinyToolCallTransformer
from reward import ast_schema_reward
from sft import sft_train
from tokenizer import Tokenizer

BASE = Path(__file__).resolve().parents[1]


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the tool-calling policy with SFT warm-start + LoRA/GRPO.")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    cfg = load_config(str(BASE / args.config) if not Path(args.config).is_absolute() else args.config)
    set_seed(cfg.get("seed", 7))
    device = torch.device("cpu")

    train_path = BASE / "data" / "processed" / "train.jsonl"
    bfcl_path = BASE / "data" / "processed" / "bfcl_subset.jsonl"
    if not train_path.exists() or not bfcl_path.exists():
        print("Datasets not found, generating...")
        generate_datasets()

    train_rows = load_dataset(str(train_path))
    bfcl_rows = load_dataset(str(bfcl_path))
    train_examples = prepare_examples(train_rows)
    bfcl_examples = prepare_examples(bfcl_rows)
    print(f"Loaded {len(train_examples)} training trajectories, {len(bfcl_examples)} held-out BFCL-style examples.")

    all_text = [ex["prompt"] + " " + ex["target"] for ex in train_examples + bfcl_examples]
    tokenizer = Tokenizer.build(all_text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["model"]["d_model"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"].get("dropout", 0.1),
    )
    model = TinyToolCallTransformer(model_cfg)
    print(f"Policy model: {sum(p.numel() for p in model.parameters()):,} parameters (random init).")

    t0 = time.time()
    sft_train(
        model,
        tokenizer,
        train_examples,
        epochs=cfg["sft"]["epochs"],
        batch_size=cfg["sft"]["batch_size"],
        lr=cfg["sft"]["lr"],
        max_len=cfg["model"]["max_seq_len"],
        device=device,
    )
    print(f"SFT warm-start done in {time.time() - t0:.1f}s")

    # Frozen reference snapshot for the GRPO KL penalty, taken right after
    # the warm-start (before any LoRA/GRPO update touches the weights).
    ref_model = copy.deepcopy(model)

    print("Evaluating baseline (post-SFT, pre-GRPO) accuracy on held-out BFCL-style subset...")
    baseline_results = evaluate_accuracy(model, tokenizer, bfcl_examples, device=device)
    print(f"Baseline accuracy: {baseline_results['accuracy'] * 100:.1f}%")

    n_wrapped = apply_lora(
        model,
        target_suffixes=["qkv", "proj"],
        r=cfg["model"]["lora_rank"],
        alpha=cfg["model"]["lora_alpha"],
    )
    freeze_non_lora(model)
    print(f"Wrapped {n_wrapped} linear layers with LoRA (rank={cfg['model']['lora_rank']}). "
          f"Trainable params: {count_trainable_parameters(model):,} / {sum(p.numel() for p in model.parameters()):,}")

    # Run GRPO in evaluated chunks rather than one long uninterrupted run,
    # so the reported step count is a checked plateau on the held-out set
    # rather than a single untracked endpoint -- see the accuracy_curve in
    # outputs/report.md.
    total_steps = cfg["grpo"]["steps"]
    chunk_size = cfg["grpo"].get("eval_chunk_size", 25)
    accuracy_curve = []
    steps_done = 0
    all_mean_rewards: list[float] = []
    t0 = time.time()
    while steps_done < total_steps:
        this_chunk = min(chunk_size, total_steps - steps_done)
        grpo_cfg = GRPOConfig(
            steps=this_chunk,
            group_size=cfg["grpo"]["group_size"],
            prompts_per_step=cfg["grpo"]["prompts_per_step"],
            lr=cfg["grpo"]["lr"],
            max_new_tokens=cfg["grpo"]["max_new_tokens"],
            temperature=cfg["grpo"]["temperature"],
            kl_coef=cfg["grpo"]["kl_coef"],
            log_every=max(this_chunk, 1),
        )
        chunk_stats = grpo_train(model, ref_model, tokenizer, train_examples, ast_schema_reward, grpo_cfg, device=device)
        all_mean_rewards.extend(chunk_stats.mean_reward_per_step)
        steps_done += this_chunk
        chunk_eval = evaluate_accuracy(model, tokenizer, bfcl_examples, device=device)
        accuracy_curve.append({"steps": steps_done, "accuracy": chunk_eval["accuracy"]})
        print(f"[grpo] steps {steps_done}/{total_steps}: held-out accuracy {chunk_eval['accuracy'] * 100:.1f}%")
    print(f"GRPO fine-tuning done in {time.time() - t0:.1f}s over {total_steps} steps")

    post_results = evaluate_accuracy(model, tokenizer, bfcl_examples, device=device)
    print(f"Post-GRPO accuracy: {post_results['accuracy'] * 100:.1f}%")

    out_dir = BASE / cfg["logging"]["output_dir"]
    write_report(baseline_results, post_results, out_dir, accuracy_curve=accuracy_curve)
    # Also mirror to outputs/ root for README/benchmark badges.
    write_report(baseline_results, post_results, BASE / "outputs", accuracy_curve=accuracy_curve)

    print(f"\nBaseline -> Post-GRPO: {baseline_results['accuracy']*100:.1f}% -> {post_results['accuracy']*100:.1f}%"
          f" ({(post_results['accuracy']-baseline_results['accuracy'])*100:+.1f}pp)")
    if all_mean_rewards:
        print(f"Mean GRPO training reward, first vs last 10 steps: "
              f"{sum(all_mean_rewards[:10]) / max(min(10, len(all_mean_rewards)), 1):.3f} -> "
              f"{sum(all_mean_rewards[-10:]) / max(min(10, len(all_mean_rewards)), 1):.3f}")


if __name__ == "__main__":
    main()
