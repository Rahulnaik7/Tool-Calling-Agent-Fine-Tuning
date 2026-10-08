"""Dataset preparation utilities for tool-calling fine-tuning.

This file is intentionally kept lightweight as a starting scaffold.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL dataset from disk.

    Replace this with the actual loader for your tool-calling dataset.
    """
    raise NotImplementedError("Implement dataset loading for your project.")


def prepare_examples(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw rows into training-ready prompt/completion samples."""
    prepared = []
    for row in rows:
        prepared.append({
            "prompt": row.get("prompt", ""),
            "response": row.get("response", ""),
            "tools": row.get("tools", []),
        })
    return prepared
