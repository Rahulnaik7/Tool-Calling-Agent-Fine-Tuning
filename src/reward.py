"""Reward functions for tool-calling evaluation.

The final version of this project should validate tool calls against the
expected schema and API contract using deterministic checks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def schema_reward(prediction: str, expected_tool: str, expected_args: Dict[str, Any]) -> float:
    """Placeholder reward function.

    Replace this with AST validation and JSON-schema checking for your
    tool-calling task.
    """
    try:
        parsed = json.loads(prediction)
    except Exception:
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    if parsed.get("tool") == expected_tool:
        args = parsed.get("arguments", {})
        if isinstance(args, dict) and args == expected_args:
            return 1.0

    return 0.0


def evaluate_batch(predictions: List[str], targets: List[Dict[str, Any]]) -> List[float]:
    """Evaluate a batch of predictions against target tool calls."""
    scores: List[float] = []
    for prediction, target in zip(predictions, targets):
        expected_tool = target.get("tool", "")
        expected_args = target.get("arguments", {})
        scores.append(schema_reward(prediction, expected_tool, expected_args))
    return scores
