"""Generates the tool-calling trajectory dataset.

Each trajectory is (instruction, candidate_tool_schemas, expected_tool,
expected_arguments). "Execution-verified" means every expected call is
checked — not against a live API (ToolBench's real RapidAPI sandbox needs
network access this environment doesn't have), but against this project's
own schema validator (`verify_call`): required arguments present, every
value the correct type, enum fields constrained to a legal value. A
trajectory is only written to the dataset if it passes that check, exactly
the same validator the deterministic AST/API-match reward (`reward.py`)
uses at training time — so "execution-verified" and "reward-checkable" are
the same property by construction.

Two disjoint splits are produced, both showing the correct tool's schema
alongside 2 distractor tools' schemas (so the task is genuine function
*selection*, not "copy the one schema you were shown" — SFT has to learn
to pick a tool out of several candidates either way, and the held-out
split is where that skill is actually measured):

* `data/processed/train.jsonl` — 500 execution-verified training
  trajectories, used for the SFT warm-start and as the prompt pool for
  GRPO.
* `data/processed/bfcl_subset.jsonl` — a held-out evaluation split, in the
  spirit of the Berkeley Function-Calling Leaderboard (BFCL): disjoint
  instructions, a different template/phrasing bias, and independently
  sampled distractor combinations, so accuracy here measures
  generalization, not memorization of the training distribution. (The
  real BFCL dataset is fetched from gorilla-llm/gorilla and evaluated
  against hosted model APIs; this is a structurally equivalent subset
  sized for this project, not a download of that benchmark.)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools_registry import TOOLS, TOOL_NAMES, all_keys, required_keys

BASE = Path(__file__).resolve().parents[1]

# Value pools used to fill in template slots and argument values.
LOCATIONS = ["London", "New York", "Tokyo", "Mumbai", "Berlin", "Sydney", "Toronto", "Cairo"]
NAMES = ["Alice", "Raj", "Wei", "Fatima", "Carlos", "Priya", "Tom", "Yuki"]
TOPICS = ["artificial intelligence", "climate policy", "the stock market", "space exploration", "football", "electric cars"]
DATES = ["2026-01-15", "2026-02-20", "2026-03-10", "2026-04-05", "2026-05-22"]
QUERIES = ["best pizza nearby", "python tutorials", "electric vehicle reviews", "weekend hiking trails", "budget laptops"]
LANGS = ["es", "fr", "de", "ja", "hi"]
TICKERS = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"]


def _sample_value(tool: str, key: str, ptype: type, rng: random.Random) -> Any:
    spec = TOOLS[tool]
    enum_vals = spec.get("enum", {}).get(key)
    if enum_vals:
        return rng.choice(enum_vals)
    if ptype is float:
        return round(rng.uniform(1, 500), 2)
    if ptype is int:
        return rng.randint(1, 20)
    if ptype is bool:
        return rng.choice([True, False])
    # string: pick something contextually plausible by key name
    if "location" in key or key in ("origin", "destination", "city"):
        return rng.choice(LOCATIONS)
    if key in ("name", "user_id", "to"):
        return rng.choice(NAMES) if key != "user_id" else f"u{rng.randint(100, 999)}"
    if key in ("topic",):
        return rng.choice(TOPICS)
    if key in ("date", "when", "check_in", "check_out", "time"):
        return rng.choice(DATES)
    if key in ("query", "pattern", "filter"):
        return rng.choice(QUERIES)
    if key in ("text", "content", "body", "title", "subject"):
        return rng.choice(QUERIES).capitalize()
    if key in ("ticker",):
        return rng.choice(TICKERS)
    if key in ("tracking_number",):
        return f"TRK{rng.randint(100000, 999999)}"
    return rng.choice(QUERIES)


def _sample_args(tool: str, rng: random.Random, include_optional: bool) -> Dict[str, Any]:
    spec = TOOLS[tool]
    args: Dict[str, Any] = {}
    for key, ptype in spec["required"].items():
        args[key] = _sample_value(tool, key, ptype, rng)
    if include_optional:
        for key, ptype in spec["optional"].items():
            if rng.random() < 0.6:
                args[key] = _sample_value(tool, key, ptype, rng)
    return args


# (template, tool, required-arg-keys-used-in-template) -- {key} placeholders
# are filled from the sampled args for readability in the instruction text.
TEMPLATES: Dict[str, List[str]] = {
    "math_service": [
        "What is {x} {operation_word} {y}?",
        "Can you {operation_word2} {x} and {y} for me?",
    ],
    "weather_api": [
        "What's the weather like in {location}?",
        "Give me the current weather for {location} in {units}.",
    ],
    "time_service": [
        "What time is it in {timezone}?",
        "Tell me the current time in the {timezone} timezone.",
    ],
    "search_api": [
        "Search the web for {query}.",
        "Look up {query} online.",
    ],
    "user_profile": [
        "Look up the profile for user {user_id}.",
        "Get me user {user_id}'s profile details.",
    ],
    "translate_api": [
        "Translate '{text}' into {target}.",
        "How do you say '{text}' in {target}?",
    ],
    "calendar_create_event": [
        "Create a calendar event called {title} on {date}.",
        "Schedule an event titled {title} for {date}.",
    ],
    "calendar_list_events": [
        "What events do I have on {date}?",
        "List my calendar events for {date}.",
    ],
    "email_send": [
        "Send an email to {to} with subject {subject}.",
        "Email {to} about {subject}.",
    ],
    "file_search": [
        "Find files matching {pattern}.",
        "Search for files named like {pattern}.",
    ],
    "currency_convert": [
        "Convert {amount} {from_currency} to {to_currency}.",
        "How much is {amount} {from_currency} in {to_currency}?",
    ],
    "unit_convert": [
        "Convert {value} {from_unit} to {to_unit}.",
        "What is {value} {from_unit} in {to_unit}?",
    ],
    "stock_quote": [
        "What's the current price of {ticker}?",
        "Get me a stock quote for {ticker}.",
    ],
    "news_search": [
        "Find recent news about {topic}.",
        "Search news articles on {topic}.",
    ],
    "map_directions": [
        "Get directions from {origin} to {destination}.",
        "How do I get from {origin} to {destination} by {mode}?",
    ],
    "restaurant_search": [
        "Find restaurants near {location}.",
        "Search for places to eat in {location}.",
    ],
    "flight_search": [
        "Search flights from {origin} to {destination} on {date}.",
        "Find a flight from {origin} to {destination} for {date}.",
    ],
    "hotel_search": [
        "Find hotels in {city} from {check_in} to {check_out}.",
        "Search hotels in {city} between {check_in} and {check_out}.",
    ],
    "reminder_set": [
        "Remind me to {text} at {when}.",
        "Set a reminder: {text} on {when}.",
    ],
    "note_create": [
        "Create a note titled {title} with content {content}.",
        "Make a note called {title}: {content}.",
    ],
    "database_query": [
        "Query the {table} table where {filter}.",
        "Look up rows in {table} matching {filter}.",
    ],
    "image_generate": [
        "Generate an image of {prompt}.",
        "Create a picture of {prompt}.",
    ],
    "code_execute": [
        "Run this {language} code: {code}.",
        "Execute the following {language} snippet: {code}.",
    ],
    "calculator_tip": [
        "What's a tip on a {bill_total} bill at {tip_percent} percent?",
        "Calculate the tip for a {bill_total} bill at {tip_percent}%.",
    ],
    "music_play": [
        "Play {query}.",
        "Put on {query} for me.",
    ],
    "alarm_set": [
        "Set an alarm for {time}.",
        "Wake me up at {time}.",
    ],
    "contact_lookup": [
        "Look up contact details for {name}.",
        "Find {name} in my contacts.",
    ],
    "package_track": [
        "Track package {tracking_number}.",
        "Where is my shipment {tracking_number}?",
    ],
    "device_control": [
        "Turn {state} the {device}.",
        "Switch the {device} {state}.",
    ],
    "recipe_search": [
        "Find recipes with {ingredient}.",
        "Search for dishes that use {ingredient}.",
    ],
}

_OP_WORDS = {"add": "plus", "sub": "minus", "mul": "times", "div": "divided by"}
_OP_WORDS2 = {"add": "add", "sub": "subtract", "mul": "multiply", "div": "divide"}


def _fill_instruction(tool: str, template: str, args: Dict[str, Any]) -> str:
    fill = dict(args)
    if tool == "math_service":
        fill["operation_word"] = _OP_WORDS.get(args["operation"], args["operation"])
        fill["operation_word2"] = _OP_WORDS2.get(args["operation"], args["operation"])
    try:
        return template.format(**fill)
    except KeyError:
        return template


def _value_matches_type(value: Any, ptype: type) -> bool:
    if ptype is float:
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if ptype is int:
        return not isinstance(value, bool) and isinstance(value, int)
    if ptype is str:
        return isinstance(value, str)
    if ptype is bool:
        return isinstance(value, bool)
    return False


def verify_call(tool: str, args: Dict[str, Any]) -> bool:
    """The dataset's "execution" check: does this call satisfy the tool's
    schema? Every provided argument (required *or* optional) is checked
    against its declared type and, if it has one, its enum -- not just the
    required ones, since an optional argument with a bad type or an
    out-of-range enum value (e.g. `weather_api(..., units='lightyears')`)
    is just as invalid a call as a missing required one.
    """
    if tool not in TOOLS:
        return False
    spec = TOOLS[tool]
    param_types = {**spec["required"], **spec["optional"]}

    for key, ptype in spec["required"].items():
        if key not in args:
            return False

    for key, value in args.items():
        if key not in param_types:
            return False  # unknown key, checked again below for clarity
        if not _value_matches_type(value, param_types[key]):
            return False
        enum_vals = spec.get("enum", {}).get(key)
        if enum_vals and value not in enum_vals:
            return False

    allowed = all_keys(tool)
    if not set(args.keys()) <= allowed:
        return False
    return True


def format_schema(tool: str) -> str:
    spec = TOOLS[tool]
    keys = list(spec["required"].keys()) + list(spec["optional"].keys())
    return f"{tool}({', '.join(keys)})"


def _unique_key(tool: str, instruction: str) -> str:
    return f"{tool}::{instruction}"


def generate_training_set(target_size: int = 500, seed: int = 7, n_distractors: int = 2) -> List[Dict[str, Any]]:
    """Each training example shows the correct tool's schema ALONGSIDE
    `n_distractors` other tools' schemas (same as the held-out BFCL-style
    split), so the model is trained on genuine function *selection*, not
    just "copy the one schema you're shown" -- without this, SFT has no
    signal at all for picking the right tool out of several candidates,
    which is exactly the skill the held-out split evaluates.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < target_size and attempts < target_size * 50:
        attempts += 1
        tool = TOOL_NAMES[attempts % len(TOOL_NAMES)]
        templates = TEMPLATES[tool]
        template = rng.choice(templates)
        args = _sample_args(tool, rng, include_optional=rng.random() < 0.4)
        if not verify_call(tool, args):
            continue
        instruction = _fill_instruction(tool, template, args)
        key = _unique_key(tool, instruction)
        if key in seen:
            continue
        seen.add(key)

        distractor_pool = [t for t in TOOL_NAMES if t != tool]
        distractors = rng.sample(distractor_pool, k=min(n_distractors, len(distractor_pool)))
        candidate_tools = distractors + [tool]
        rng.shuffle(candidate_tools)

        rows.append(
            {
                "instruction": instruction,
                "tool": tool,
                "schema": format_schema(tool),
                "arguments": args,
                "candidate_tools": candidate_tools,
                "candidate_schemas": [format_schema(t) for t in candidate_tools],
            }
        )
    return rows


def generate_bfcl_subset(size: int = 120, seed: int = 99, n_distractors: int = 2) -> List[Dict[str, Any]]:
    """Held-out eval split: unseen instruction phrasing + distractor tools shown
    alongside the correct one, so accuracy requires real function selection.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < size and attempts < size * 80:
        attempts += 1
        tool = TOOL_NAMES[attempts % len(TOOL_NAMES)]
        templates = TEMPLATES[tool]
        # Use the *second* template (or wrap the first) to bias toward
        # phrasing not emphasized in training, plus light paraphrase noise.
        template = templates[-1]
        args = _sample_args(tool, rng, include_optional=rng.random() < 0.5)
        if not verify_call(tool, args):
            continue
        instruction = _fill_instruction(tool, template, args)
        prefix = rng.choice(["", "Hey, ", "Quick favor -- ", "Could you ", "I need you to "])
        if prefix in ("Could you ", "I need you to "):
            instruction = prefix + instruction[0].lower() + instruction[1:]
        else:
            instruction = prefix + instruction
        key = _unique_key(tool, instruction)
        if key in seen:
            continue
        seen.add(key)

        distractor_pool = [t for t in TOOL_NAMES if t != tool]
        distractors = rng.sample(distractor_pool, k=min(n_distractors, len(distractor_pool)))
        candidate_tools = distractors + [tool]
        rng.shuffle(candidate_tools)

        rows.append(
            {
                "instruction": instruction,
                "tool": tool,
                "arguments": args,
                "candidate_tools": candidate_tools,
                "candidate_schemas": [format_schema(t) for t in candidate_tools],
            }
        )
    return rows


def main() -> None:
    train_rows = generate_training_set(target_size=500)
    bfcl_rows = generate_bfcl_subset(size=140)

    # Guarantee the two splits never share an identical instruction string.
    train_instructions = {r["instruction"] for r in train_rows}
    bfcl_rows = [r for r in bfcl_rows if r["instruction"] not in train_instructions]

    out_dir = BASE / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row) + "\n")

    with (out_dir / "bfcl_subset.jsonl").open("w", encoding="utf-8") as fh:
        for row in bfcl_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"Wrote {len(train_rows)} execution-verified training trajectories -> {out_dir / 'train.jsonl'}")
    print(f"Wrote {len(bfcl_rows)} BFCL-style held-out eval examples -> {out_dir / 'bfcl_subset.jsonl'}")


if __name__ == "__main__":
    main()
