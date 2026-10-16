"""Dataset loading utilities for the tool-calling fine-tuning project.

Loads the execution-verified trajectories and BFCL-style held-out examples
written by `dataset_gen.py`, and converts each row into the
prompt/target text pair the model actually trains and is evaluated on
(see `prompting.py`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from prompting import build_prompt, build_target
from tools_registry import TOOLS, TOOL_NAMES


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL dataset from disk (one JSON object per line)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _schema_str(tool: str) -> str:
    spec = TOOLS[tool]
    keys = list(spec["required"].keys()) + list(spec["optional"].keys())
    return f"{tool}({', '.join(keys)})"


def prepare_examples(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw dataset rows (training format OR bfcl_subset format) into
    training-ready {prompt, target, tool, arguments, candidate_tools}.
    """
    prepared = []
    for row in rows:
        tool = row["tool"]
        arguments = row.get("arguments", {})
        if "candidate_schemas" in row:
            candidate_schemas = row["candidate_schemas"]
            candidate_tools = row["candidate_tools"]
        else:
            candidate_schemas = [row.get("schema", _schema_str(tool))]
            candidate_tools = [tool]
        prepared.append(
            {
                "instruction": row["instruction"],
                "prompt": build_prompt(row["instruction"], candidate_schemas),
                "target": build_target(tool, arguments),
                "tool": tool,
                "arguments": arguments,
                "candidate_tools": candidate_tools,
            }
        )
    return prepared
