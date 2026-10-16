"""Tool API registry: name, description, and a JSON-schema-style parameter spec
for every callable tool the agent can be asked to invoke.

This stands in for the kind of API catalog ToolBench trains and evaluates
against (thousands of real RapidAPI endpoints). Fetching that catalog or its
live sandboxed execution backend needs network access this sandbox doesn't
have, so this registry is a smaller, hand-built but structurally identical
catalog: every tool has typed, required/optional parameters, and every
training/eval example is checked against this registry before being kept
(see `dataset_gen.py::execution_verify`), which is what "execution-verified"
means in this project.
"""
from __future__ import annotations

from typing import Any, Dict

# Each tool: required params (must appear, in this order for value sampling)
# and optional params. "type" is the Python type `ast.literal_eval` should
# produce for that argument.
TOOLS: Dict[str, Dict[str, Any]] = {
    "math_service": {
        "description": "Perform a basic arithmetic operation on two numbers.",
        "required": {"operation": str, "x": float, "y": float},
        "optional": {},
        "enum": {"operation": ["add", "sub", "mul", "div"]},
    },
    "weather_api": {
        "description": "Get the current weather for a named location.",
        "required": {"location": str},
        "optional": {"units": str},
        "enum": {"units": ["metric", "imperial"]},
    },
    "time_service": {
        "description": "Get the current time in a given timezone.",
        "required": {"timezone": str},
        "optional": {"format": str},
        "enum": {},
    },
    "search_api": {
        "description": "Run a web search query and return the top results.",
        "required": {"query": str},
        "optional": {"top_k": int},
        "enum": {},
    },
    "user_profile": {
        "description": "Look up a user's profile by user id.",
        "required": {"user_id": str},
        "optional": {},
        "enum": {},
    },
    "translate_api": {
        "description": "Translate text into a target language.",
        "required": {"text": str, "target": str},
        "optional": {},
        "enum": {"target": ["es", "fr", "de", "ja", "hi"]},
    },
    "calendar_create_event": {
        "description": "Create a calendar event.",
        "required": {"title": str, "date": str},
        "optional": {"duration_minutes": int},
        "enum": {},
    },
    "calendar_list_events": {
        "description": "List calendar events for a date.",
        "required": {"date": str},
        "optional": {},
        "enum": {},
    },
    "email_send": {
        "description": "Send an email to a recipient.",
        "required": {"to": str, "subject": str},
        "optional": {"body": str},
        "enum": {},
    },
    "file_search": {
        "description": "Search local files by name pattern.",
        "required": {"pattern": str},
        "optional": {"directory": str},
        "enum": {},
    },
    "currency_convert": {
        "description": "Convert an amount from one currency to another.",
        "required": {"amount": float, "from_currency": str, "to_currency": str},
        "optional": {},
        "enum": {"from_currency": ["USD", "EUR", "GBP", "INR"], "to_currency": ["USD", "EUR", "GBP", "INR"]},
    },
    "unit_convert": {
        "description": "Convert a numeric value between measurement units.",
        "required": {"value": float, "from_unit": str, "to_unit": str},
        "optional": {},
        "enum": {"from_unit": ["km", "mi", "kg", "lb", "c", "f"], "to_unit": ["km", "mi", "kg", "lb", "c", "f"]},
    },
    "stock_quote": {
        "description": "Get the latest stock price for a ticker symbol.",
        "required": {"ticker": str},
        "optional": {},
        "enum": {},
    },
    "news_search": {
        "description": "Search recent news articles on a topic.",
        "required": {"topic": str},
        "optional": {"max_results": int},
        "enum": {},
    },
    "map_directions": {
        "description": "Get directions between two locations.",
        "required": {"origin": str, "destination": str},
        "optional": {"mode": str},
        "enum": {"mode": ["driving", "walking", "transit", "bicycling"]},
    },
    "restaurant_search": {
        "description": "Search for restaurants near a location.",
        "required": {"location": str},
        "optional": {"cuisine": str},
        "enum": {},
    },
    "flight_search": {
        "description": "Search for flights between two airports on a date.",
        "required": {"origin": str, "destination": str, "date": str},
        "optional": {},
        "enum": {},
    },
    "hotel_search": {
        "description": "Search for hotels in a city for a date range.",
        "required": {"city": str, "check_in": str, "check_out": str},
        "optional": {},
        "enum": {},
    },
    "reminder_set": {
        "description": "Set a reminder for a future time.",
        "required": {"text": str, "when": str},
        "optional": {},
        "enum": {},
    },
    "note_create": {
        "description": "Create a text note.",
        "required": {"title": str, "content": str},
        "optional": {},
        "enum": {},
    },
    "database_query": {
        "description": "Run a read-only query against a named database table.",
        "required": {"table": str, "filter": str},
        "optional": {"limit": int},
        "enum": {},
    },
    "image_generate": {
        "description": "Generate an image from a text prompt.",
        "required": {"prompt": str},
        "optional": {"style": str},
        "enum": {"style": ["photo", "sketch", "cartoon", "painting"]},
    },
    "code_execute": {
        "description": "Execute a snippet of code in a given language.",
        "required": {"language": str, "code": str},
        "optional": {},
        "enum": {"language": ["python", "javascript", "bash"]},
    },
    "calculator_tip": {
        "description": "Compute a tip amount for a bill.",
        "required": {"bill_total": float, "tip_percent": float},
        "optional": {},
        "enum": {},
    },
    "music_play": {
        "description": "Play a song or artist on the connected music service.",
        "required": {"query": str},
        "optional": {"shuffle": bool},
        "enum": {},
    },
    "alarm_set": {
        "description": "Set an alarm for a given time.",
        "required": {"time": str},
        "optional": {"label": str},
        "enum": {},
    },
    "contact_lookup": {
        "description": "Look up a contact's details by name.",
        "required": {"name": str},
        "optional": {},
        "enum": {},
    },
    "package_track": {
        "description": "Track a shipment by tracking number.",
        "required": {"tracking_number": str},
        "optional": {"carrier": str},
        "enum": {},
    },
    "device_control": {
        "description": "Turn a smart home device on or off.",
        "required": {"device": str, "state": str},
        "optional": {},
        "enum": {"state": ["on", "off"]},
    },
    "recipe_search": {
        "description": "Search for recipes containing a given ingredient.",
        "required": {"ingredient": str},
        "optional": {"max_results": int},
        "enum": {},
    },
}

TOOL_NAMES = sorted(TOOLS.keys())


def schema_for(tool_name: str) -> Dict[str, Any]:
    return TOOLS[tool_name]


def required_keys(tool_name: str) -> list[str]:
    return list(TOOLS[tool_name]["required"].keys())


def all_keys(tool_name: str) -> set[str]:
    spec = TOOLS[tool_name]
    return set(spec["required"].keys()) | set(spec["optional"].keys())
