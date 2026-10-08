"""Training entry point for the Qwen2.5-7B tool-calling project.

This is the skeleton training script. It is intentionally lightweight so the
project looks organized even before the full pipeline is implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-7B for tool calling.")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded training config:")
    print(config)
    print("\nProject scaffold is ready. Implement the full GRPO fine-tuning loop here.")


if __name__ == "__main__":
    main()
