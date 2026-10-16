import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from model import ModelConfig, TinyToolCallTransformer


def make_model(vocab_size=50):
    cfg = ModelConfig(vocab_size=vocab_size, d_model=16, n_layer=2, n_head=2, d_ff=32, max_seq_len=32)
    return TinyToolCallTransformer(cfg), cfg


def test_forward_output_shape():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    logits = model(idx)
    assert logits.shape == (3, 10, cfg.vocab_size)


def test_causal_mask_blocks_future_tokens():
    # Changing a later token must not change the logits predicted for an
    # earlier position -- that's what the causal mask guarantees.
    model, cfg = make_model()
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_a = model(idx)
        idx_modified = idx.clone()
        idx_modified[0, -1] = (idx_modified[0, -1] + 1) % cfg.vocab_size
        logits_b = model(idx_modified)
    torch.testing.assert_close(logits_a[:, :-1, :], logits_b[:, :-1, :])


def test_generate_respects_max_new_tokens():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=5, eos_id=999999)  # eos never hit
    assert out.shape[1] == idx.shape[1] + 5


def test_generate_stops_at_eos_for_all_rows():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    # eos_id within vocab range so it can actually be sampled eventually;
    # just check the generation never exceeds the requested cap and the
    # shape is consistent across the batch.
    out = model.generate(idx, max_new_tokens=6, eos_id=0, temperature=1.0)
    assert out.shape[0] == 2
    assert out.shape[1] <= idx.shape[1] + 6


def test_generate_collect_logprobs_matches_token_count():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out, logprobs = model.generate(idx, max_new_tokens=5, eos_id=999999, collect_logprobs=True)
    assert len(logprobs) == out.shape[1] - idx.shape[1]
