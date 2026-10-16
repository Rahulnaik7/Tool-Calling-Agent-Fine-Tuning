import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reward import ast_schema_reward, exact_match, parse_call


def test_parse_call_valid_expression():
    call = parse_call("weather_api(location='Tokyo', units='metric')")
    assert call is not None
    assert call.func.id == "weather_api"


def test_parse_call_rejects_non_call():
    assert parse_call("not a call at all !!!") is None
    assert parse_call("1 + 2") is None


def test_parse_call_never_executes_code():
    # A keyword value that looks like a call/attribute (not a literal)
    # must be dropped, not evaluated -- ast.literal_eval raises on it.
    call = parse_call("math_service(x=__import__('os').system('echo hi'))")
    assert call is not None
    from reward import extract_call

    tool, kwargs, _ = extract_call(call)
    assert tool == "math_service"
    assert "x" not in kwargs  # non-literal value silently dropped


def test_exact_match_full_credit():
    completion = "math_service(operation='add', x=1, y=2)"
    assert exact_match(completion, "math_service", {"operation": "add", "x": 1, "y": 2})


def test_exact_match_fails_on_wrong_tool():
    completion = "weather_api(location='Tokyo')"
    assert not exact_match(completion, "time_service", {"timezone": "UTC"})


def test_exact_match_fails_on_wrong_argument_value():
    completion = "math_service(operation='add', x=1, y=999)"
    assert not exact_match(completion, "math_service", {"operation": "add", "x": 1, "y": 2})


def test_exact_match_fails_on_missing_required_argument():
    completion = "math_service(operation='add', x=1)"
    assert not exact_match(completion, "math_service", {"operation": "add", "x": 1, "y": 2})


def test_reward_zero_for_unparseable_completion():
    assert ast_schema_reward("not valid python (((", "math_service", {"x": 1}) == 0.0


def test_reward_partial_credit_for_valid_ast_wrong_tool():
    reward = ast_schema_reward("other_tool(x=1)", "math_service", {"x": 1})
    assert 0.0 < reward < 0.3


def test_reward_higher_for_correct_tool_than_wrong_tool():
    correct_tool = ast_schema_reward("math_service(operation='add', x=1, y=2)", "math_service", {"operation": "add", "x": 1, "y": 2})
    wrong_tool = ast_schema_reward("weather_api(location='Tokyo')", "math_service", {"operation": "add", "x": 1, "y": 2})
    assert correct_tool > wrong_tool


def test_reward_full_credit_for_exact_call():
    reward = ast_schema_reward("math_service(operation='add', x=1, y=2)", "math_service", {"operation": "add", "x": 1, "y": 2})
    assert reward == 1.0


def test_reward_penalizes_extra_unknown_arguments():
    full = ast_schema_reward("weather_api(location='Tokyo')", "weather_api", {"location": "Tokyo"})
    with_extra = ast_schema_reward("weather_api(location='Tokyo', bogus_key=1)", "weather_api", {"location": "Tokyo"})
    assert with_extra < full
