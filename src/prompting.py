"""Builds the prompt/target text the policy model actually sees.

Prompt format (plain text, fed to the tokenizer as-is):

    Instruction: <instruction>
    Tools: tool_a(x, y) | tool_b(z) | tool_c(q, r)
    Call:

Target (what the model must generate, teacher-forced during SFT and
sampled during GRPO):

    tool_a(x=1, y="foo")

Keeping both sides in the same plain-call syntax everywhere, rather than
switching to JSON for one phase, is what lets `reward.py` validate
completions with a single AST-based parser the whole way through.
"""
from __future__ import annotations

from typing import Any, Dict, List


def build_prompt(instruction: str, candidate_schemas: List[str]) -> str:
    tools_str = " | ".join(candidate_schemas)
    return f"Instruction: {instruction}\nTools: {tools_str}\nCall:"


def build_target(tool: str, arguments: Dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={value!r}")
    return f"{tool}({', '.join(parts)})"
