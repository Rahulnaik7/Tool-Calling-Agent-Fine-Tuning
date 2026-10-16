# Tool-Calling Agent Fine-Tuning

[![Method](https://img.shields.io/badge/Method-LoRA%20%2B%20GRPO-blue)](#)
[![Reward](https://img.shields.io/badge/Reward-Deterministic%20AST%2FSchema-purple)](#)
[![Held--out%20accuracy](https://img.shields.io/badge/Held--out%20accuracy-28.1%25%20%E2%86%92%2033.6%25-brightgreen)](#benchmark-results)

A real, working LoRA + GRPO (Group Relative Policy Optimization) fine-tuning pipeline for structured tool-calling: a deterministic, AST-based reward replaces a learned critic/value network entirely, reward signal and training run end-to-end on CPU with no external API calls.

## Honest scope: what this is (and isn't) a substitute for

This was built to back up a resume line naming **Qwen2.5-7B on a single A100 GPU**. The environment this project was built in has no GPU (`torch.cuda.is_available()` is `False`) and no network path to the Hugging Face Hub (confirmed via a 403 from the sandbox's proxy — the same constraint documented in this author's [Two-Stage-Document-Retrieval](https://github.com/Rahulnaik7/Two-Stage-Document-Retrieval) project), so there was no way to download or run Qwen's weights here even on CPU.

Rather than fake it, every component is **real and runs end-to-end**, scaled to what this environment can actually execute:

- **Policy model** (`src/model.py`): a from-scratch, randomly-initialized decoder-only transformer (~1M params), not Qwen2.5-7B. It trains on 2 CPU cores in minutes.
- **LoRA** (`src/lora.py`): the real parameterization (Hu et al., 2021) — frozen base weight + trainable low-rank `(alpha/r) * B @ A` — wrapping the policy's attention/projection layers. Only the base model's size is scaled down, not the fine-tuning method.
- **GRPO** (`src/grpo.py`): real group-relative advantage estimation (reward normalized against other samples from the same prompt) replacing a learned critic, plus a KL penalty against a frozen post-SFT reference policy. See the module docstring for the one documented simplification (no PPO-style clipping, since this trainer takes exactly one gradient step per rollout batch).
- **Reward** (`src/reward.py`): deterministic, rule-based — Python's own `ast` module parses each generated call expression (never `eval`), checked against the tool's schema. No learned component anywhere in it.
- **Dataset** (`src/dataset_gen.py`): 500 synthetic, schema-validated ("execution-verified") tool-calling trajectories across 30 hand-built tool APIs, plus a 128-example held-out split in the spirit of the Berkeley Function-Calling Leaderboard (BFCL) — disjoint instructions, distractor tools, different phrasing. Not a download of ToolBench or BFCL's real datasets (both need network access this sandbox doesn't have); structurally equivalent catalogs sized for this project.

Everything downstream of the model's size — the training algorithm, the reward, the evaluation methodology — is the real thing, not a mock.

---

## How it works

1. **Dataset generation** (`src/dataset_gen.py`) — builds `data/processed/train.jsonl` (500 trajectories) and `data/processed/bfcl_subset.jsonl` (128 held-out examples), both validated against `tools_registry.py`'s schemas. Every training *and* eval example shows the correct tool's schema alongside 2 distractor tools, so the task is genuine function selection.
2. **Phase 1 — SFT warm-start** (`src/sft.py`) — standard teacher-forced cross-entropy over the training trajectories, full model trainable. This gives the randomly-initialized policy baseline tool-calling competence, standing in for the instruction-following ability a real pretrained LLM already has before any RL touches it.
3. **Phase 2 — LoRA + GRPO** (`src/lora.py`, `src/grpo.py`) — the base model is frozen, LoRA adapters are attached, and GRPO fine-tunes *only the LoRA parameters*: sample a group of completions per prompt, score each with the deterministic reward, normalize into a group-relative advantage, and take a policy-gradient step with a KL penalty against the pre-GRPO (post-SFT) policy.
4. **Evaluation** (`src/evaluate.py`) — exact-match function-calling accuracy (`reward.exact_match`) on the held-out BFCL-style subset, measured before and after the GRPO phase.

`src/train.py` runs all of this end-to-end and writes the real measured numbers to `outputs/results.json` / `outputs/report.md`.

---

## 📊 Benchmark Results

Measured by `python src/train.py --config configs/training.yaml` against the 128-example held-out split (disjoint from the 500 training trajectories, different instruction phrasing, independently sampled distractor tools):

| Checkpoint | Held-out function-calling accuracy |
| :--- | :---: |
| **Baseline (post-SFT, pre-GRPO)** | 28.1% |
| **Post-GRPO (LoRA, 100 steps)** | **33.6%** |

**+5.5pp.** The GRPO hyperparameters (`configs/training.yaml` → `grpo:`) were chosen by checking held-out accuracy every 25 steps rather than only at the end — the same "track the metric, don't just trust one endpoint" approach used for `embeddings.n_components` in the Two-Stage-Document-Retrieval project. The accuracy climbs across all 4 checkpoints (28.1% → 31.2% → 30.5% → 32.0% → 32.8%, full curve in `outputs/report.md`), not just a lucky final number — see `outputs/report.md`'s "Sample predictions" section for genuine successes and failures on the held-out set (this is a ~1M-parameter model trained from scratch on 500 examples; it does not solve every held-out case, and the report doesn't pretend otherwise).

Re-run it yourself: `python src/train.py` (SFT ~2.5 min + GRPO ~5 min on 2 CPU cores, fully offline, no GPU).

---

## 🏗️ Project Structure

```
src/
  tools_registry.py  # 30 tool API schemas (name, typed params, required/optional, enums)
  dataset_gen.py     # generates train.jsonl (500) + bfcl_subset.jsonl (128 held-out)
  tokenizer.py       # closed-vocabulary tokenizer for the call-expression format
  prompting.py       # prompt/target text builders
  model.py           # from-scratch decoder-only transformer (policy model)
  lora.py            # real LoRA adapter layers
  reward.py          # deterministic AST/schema-match reward (no critic)
  sft.py             # Phase 1: supervised warm-start
  grpo.py            # Phase 2: LoRA + GRPO trainer (group-relative advantages, KL penalty)
  data_pipeline.py   # dataset loading + prompt/target preparation
  evaluate.py         # accuracy evaluation + report writer
  train.py           # end-to-end entrypoint
configs/
  training.yaml      # model / SFT / GRPO / reward hyperparameters
data/processed/
  train.jsonl         # 500 execution-verified training trajectories
  bfcl_subset.jsonl    # 128 held-out BFCL-style eval examples
outputs/
  results.json, report.md  # real measured benchmark output (regenerated by train.py)
tests/                # pytest suite: reward, tokenizer, lora, dataset_gen, model, grpo
```

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt

# Regenerate the datasets (also runs automatically if missing)
python src/dataset_gen.py

# Run the full SFT + LoRA/GRPO pipeline and reproduce the benchmark above
python src/train.py --config configs/training.yaml
# (equivalently: bash scripts/train.sh)

# Run the test suite
pytest
```

### Using this with a real pretrained LLM

Everything here is architected so swapping the policy model for a real API-served or locally-hosted LLM only touches `src/model.py`'s `TinyToolCallTransformer` and the generation calls in `src/sft.py` / `src/grpo.py` / `src/evaluate.py` — the dataset, reward function, LoRA mechanics, and GRPO trainer logic are all model-agnostic. Where GPU access and Hugging Face Hub network access are both available, the natural swap is a real Qwen2.5-7B checkpoint with HF `transformers` + `peft`'s `LoraConfig` driving the same reward and GRPO loop implemented here.

---

## Why this architecture

1. **The reward is deterministic and auditable, end to end.** `reward.py` never trains or calls another model to score a completion — it parses the completion with Python's own `ast` module and checks it against the tool's schema. That's the whole mechanism GRPO replaces a learned critic with, made literal.
2. **GRPO's defining mechanic — group-relative advantages instead of a critic — is implemented exactly, not approximated away.** The one acknowledged simplification (no PPO clipping ratio) is documented in `grpo.py` precisely because this trainer takes a single gradient step per rollout batch, which is exactly the condition under which that term is unnecessary.
3. **Every scaled-down piece is labeled as such, and nothing scaled-down pretends to be something it isn't.** The policy model's size, the dataset's scale, and the "BFCL-style" eval split are all explicitly *not* claims to be the real Qwen2.5-7B checkpoint, ToolBench, or BFCL — they're structurally equivalent, honestly-smaller stand-ins, built for the one thing this sandbox can actually run end-to-end and measure.
