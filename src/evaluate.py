"""Evaluation helper: compute human-friendly metrics from demo outputs.

Reads `data/processed/sample.jsonl` and `outputs/results.json` and writes
`outputs/report.md` with a brief summary and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def load_targets(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_report(targets: List[Dict], results: Dict, out: Path) -> None:
    lines: List[str] = []
    lines.append("# Demo Evaluation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Examples: {len(targets)}\n")
    lines.append(f"- Tool-calling Accuracy: {results.get('tool_call_accuracy', 0.0)}%\n")
    lines.append("\n## Per-example results\n")
    for i, score in enumerate(results.get("per_example", [])):
        lines.append(f"- Example {i+1}: score={score}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    targets = load_targets(base / "data" / "processed" / "sample.jsonl")
    results = load_results(base / "outputs" / "results.json")
    write_report(targets, results, base / "outputs" / "report.md")
    print("Wrote outputs/report.md")


if __name__ == "__main__":
    main()
