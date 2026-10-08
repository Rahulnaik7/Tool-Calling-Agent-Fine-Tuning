# Tool-Calling Agent Fine-Tuning

Project scaffold for fine-tuning **Qwen2.5-7B** to produce schema-correct tool calls, with a deterministic, rule-based reward (JSON-schema / argument matching) instead of a learned critic.

## Status

This repo currently contains the **project scaffold and a working demo pipeline**, not a completed GRPO/LoRA training run:

- ✅ Reward scaffold (`src/reward.py`) — schema/argument-matching reward function
- ✅ Evaluation + reporting flow (`src/evaluate.py`, `src/demo.py`) — runs a mock policy over sample data and produces `outputs/results.json` / `outputs/report.md`
- ✅ Sample tool-calling dataset (`data/processed/sample.jsonl`) and training config (`configs/training.yaml`)
- ⬜ Actual LoRA + GRPO training loop (`src/train.py` is a skeleton that loads config and prints it — the fine-tuning loop itself is a TODO)
- ⬜ Real dataset loader (`src/data_pipeline.py::load_dataset` is unimplemented)
- ⬜ Integration with a real policy model (`demo.py` uses `mock_policy_emit`, a deterministic stand-in, not an actual model)

See `docs/roadmap.md` for the full phase breakdown.

## Project structure

```
src/
├── train.py          # Training entrypoint (config loading done; GRPO loop is a TODO)
├── reward.py          # Deterministic schema/argument-match reward
├── data_pipeline.py  # Dataset loading/prep utilities (scaffold)
├── demo.py           # Runs a mock policy + reward over sample data
└── evaluate.py       # Turns results.json into a readable report
configs/training.yaml  # Model + LoRA + training hyperparameters
data/processed/        # Sample tool-calling examples
outputs/                # Demo run outputs (results.json, report.md)
docs/roadmap.md         # Phased project plan
scripts/                 # train.sh, run_demo.sh
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the demo

```bash
bash scripts/run_demo.sh
```

This runs a deterministic mock policy against `data/processed/sample.jsonl`, scores it with the reward function in `src/reward.py`, and writes `outputs/results.json` and `outputs/report.md`.

## Training (scaffold)

```bash
bash scripts/train.sh
```

Loads `configs/training.yaml` (Qwen2.5-7B, LoRA rank 16, batch size 4, etc.) and prints the resolved config. The actual GRPO fine-tuning loop still needs to be implemented here.

## Config

`configs/training.yaml` controls the model, LoRA settings (rank 16, alpha 32), training hyperparameters, and which reward checks are enabled (schema validation, AST check, tool-call match).
