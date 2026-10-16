import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizer import Tokenizer


SAMPLE_TEXTS = [
    "calendar_create_event(title='Best pizza nearby', date='2026-05-22', duration_minutes=3)",
    "math_service(operation='add', x=1, y=2)",
    "hotel_search(city='Tokyo', check_in='2026-02-20', check_out='2026-01-15')",
    "Instruction: What's the weather like in London?",
]


def build_tokenizer():
    return Tokenizer.build(SAMPLE_TEXTS)


def test_encode_decode_preserves_string_argument_value():
    tok = build_tokenizer()
    ids = tok.encode("calendar_create_event(title='Best pizza nearby')")
    decoded = tok.decode(ids)
    import ast

    tree = ast.parse(decoded, mode="eval")
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    assert kwargs["title"] == "Best pizza nearby"


def test_encode_decode_preserves_iso_date():
    tok = build_tokenizer()
    ids = tok.encode("hotel_search(check_in='2026-02-20')")
    decoded = tok.decode(ids)
    import ast

    tree = ast.parse(decoded, mode="eval")
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    assert kwargs["check_in"] == "2026-02-20"


def test_unknown_token_maps_to_unk():
    tok = build_tokenizer()
    ids = tok.encode("some_never_seen_tool_xyz")
    assert tok.unk_id in ids


def test_bos_eos_added_and_stripped_on_decode():
    tok = build_tokenizer()
    ids = tok.encode("math_service(x=1)", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert "<bos>" not in decoded and "<eos>" not in decoded


def test_vocab_is_deterministic_for_same_input():
    tok1 = Tokenizer.build(SAMPLE_TEXTS)
    tok2 = Tokenizer.build(SAMPLE_TEXTS)
    assert tok1.vocab == tok2.vocab


def test_decoded_call_is_valid_python_call_expression():
    tok = build_tokenizer()
    for text in SAMPLE_TEXTS[:3]:
        decoded = tok.decode(tok.encode(text))
        import ast

        tree = ast.parse(decoded, mode="eval")
        assert isinstance(tree.body, ast.Call)
