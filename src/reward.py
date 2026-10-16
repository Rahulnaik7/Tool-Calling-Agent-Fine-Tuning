"""Deterministic, rule-based reward for a generated tool call -- the thing
that replaces a learned critic/value network in this project's GRPO setup.

The policy is trained to emit a Python call expression, e.g.:

    weather_api(location="Tokyo", units="metric")

rather than JSON, specifically so the reward function can validate it with
Python's own `ast` module (`ast.parse(..., mode="eval")`) instead of a
hand-rolled parser -- the literal AST pass the resume bullet refers to,
mirroring the Automated-Code-Review-Agent project's Bandit+AST pattern.
`ast.literal_eval` is used (never `eval`) so a malformed or adversarial
completion can only fail to parse, never execute.

Reward is a weighted sum of three deterministic checks, each independently
verifiable and each worth a fixed, documented amount -- no learned
component anywhere in this function:

1. **AST-valid** (+0.2): the completion parses as a single Python call
   expression at all.
2. **API-match** (+0.3): the called function name matches the tool the
   instruction actually asked for.
3. **Schema-match** (+0.5, scaled by fraction correct): every required
   argument is present with the correct value; unknown keys or stray
   positional arguments are penalized.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

from tools_registry import TOOLS, all_keys

AST_VALID_WEIGHT = 0.2
API_MATCH_WEIGHT = 0.3
SCHEMA_MATCH_WEIGHT = 0.5
EXTRA_ARG_PENALTY = 0.05
POSITIONAL_ARG_PENALTY = 0.1


def parse_call(completion: str) -> Optional[ast.Call]:
    """Parses `completion` as a single Python call expression via the
    standard library `ast` module. Returns None (never raises, never
    executes anything) if it isn't one.
    """
    try:
        tree = ast.parse(completion.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    if not isinstance(tree.body, ast.Call):
        return None
    return tree.body


def extract_call(call: ast.Call) -> Tuple[Optional[str], Dict[str, Any], bool]:
    """Returns (tool_name, kwargs, has_positional_args). Keyword values are
    read with `ast.literal_eval`, so only literal Python values (numbers,
    strings, booleans, lists, dicts) are ever accepted -- no names, calls,
    or attribute access can smuggle code execution through here.
    """
    func = call.func
    tool_name = func.id if isinstance(func, ast.Name) else None
    kwargs: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat -- not a literal key, ignore.
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue
    return tool_name, kwargs, bool(call.args)


def ast_schema_reward(completion: str, expected_tool: str, expected_args: Dict[str, Any]) -> float:
    """The deterministic reward used both at GRPO training time and for the
    BFCL-style held-out accuracy evaluation's correctness checks.
    """
    call = parse_call(completion)
    if call is None:
        return 0.0

    reward = AST_VALID_WEIGHT

    tool_name, kwargs, has_positional = extract_call(call)
    if tool_name != expected_tool:
        return round(reward, 4)
    reward += API_MATCH_WEIGHT

    if has_positional:
        reward -= POSITIONAL_ARG_PENALTY

    spec = TOOLS.get(expected_tool, {})
    required = list(spec.get("required", {}).keys())
    if required:
        correct = sum(1 for key in required if key in kwargs and kwargs[key] == expected_args.get(key))
        match_frac = correct / len(required)
    else:
        match_frac = 1.0 if kwargs == expected_args else 0.0
    reward += SCHEMA_MATCH_WEIGHT * match_frac

    allowed = all_keys(expected_tool)
    extra_keys = set(kwargs) - allowed
    if extra_keys:
        reward -= EXTRA_ARG_PENALTY * len(extra_keys)

    return round(max(0.0, min(1.0, reward)), 4)


def exact_match(completion: str, expected_tool: str, expected_args: Dict[str, Any]) -> bool:
    """Strict pass/fail used for the reported accuracy metric: correct
    tool, every required argument present and correct, no unknown keys.
    """
    call = parse_call(completion)
    if call is None:
        return False
    tool_name, kwargs, has_positional = extract_call(call)
    if tool_name != expected_tool or has_positional:
        return False
    spec = TOOLS.get(expected_tool, {})
    required = list(spec.get("required", {}).keys())
    if any(key not in kwargs or kwargs[key] != expected_args.get(key) for key in required):
        return False
    allowed = all_keys(expected_tool)
    if not set(kwargs.keys()) <= allowed:
        return False
    return True


def evaluate_batch(predictions: List[str], targets: List[Dict[str, Any]]) -> List[float]:
    """Evaluate a batch of predictions against target tool calls."""
    scores: List[float] = []
    for prediction, target in zip(predictions, targets):
        expected_tool = target.get("tool", "")
        expected_args = target.get("arguments", {})
        scores.append(ast_schema_reward(prediction, expected_tool, expected_args))
    return scores


# Alias kept for the original scaffold's naming.
def schema_reward(prediction: str, expected_tool: str, expected_args: Dict[str, Any]) -> float:
    return ast_schema_reward(prediction, expected_tool, expected_args)
