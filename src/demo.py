"""Lightweight demo runner for the Qwen2.5-7B GRPO tool-calling project.

This script is intentionally simple and deterministic: it simulates a model
producing JSON tool-calls, evaluates them using the existing `src/reward`
scaffold, and writes human-readable + machine-readable results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
import sys

# Ensure repo root is on sys.path so `src` imports work when running from repo root.
base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from src.reward import evaluate_batch


def load_sample(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mock_policy_emit(targets: List[Dict]) -> List[str]:
    """Create deterministic predictions that match the targets."""
    preds = []
    for i, t in enumerate(targets):
        # Introduce a controlled error every 4th example to produce realistic metrics
        args = dict(t.get("arguments", {}))
        if i % 4 == 0:
            # change one argument slightly to simulate an incorrect call
            if "x" in args and isinstance(args.get("x"), int):
                args["x"] = args["x"] + 1
            elif "location" in args and isinstance(args.get("location"), str):
                args["location"] = args["location"] + " City"
            else:
                # if no known key, flip target tool to wrong tool name
                preds.append(json.dumps({"tool": "unknown_service", "arguments": {}}))
                continue

        pred = {
            "tool": t.get("tool"),
            "arguments": args,
        }
        preds.append(json.dumps(pred))
    return preds


def summarize_and_write(results: List[float], out_path: Path) -> None:
    avg = float(sum(results) / len(results)) if results else 0.0
    summary = {
        "tool_call_accuracy": avg * 100.0,
        "per_example": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("Demo complete — summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    sample_path = base / "data" / "processed" / "sample.jsonl"
    out_path = base / "outputs" / "results.json"

    targets = load_sample(sample_path)
    if not targets:
        print("No sample data found at", sample_path)
        return

    predictions = mock_policy_emit(targets)
    scores = evaluate_batch(predictions, targets)
    summarize_and_write(scores, out_path)


if __name__ == "__main__":
    main()
