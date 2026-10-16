"""A small, deterministic tokenizer for the tool-calling prompt/completion
format used by this project.

The policy model (`model.py`) is a from-scratch transformer, not a
pretrained LLM with its own tokenizer (see `model.py`'s docstring for why),
so it needs one. Rather than a generic byte-pair encoder, this is a
regex word/punctuation tokenizer with a vocabulary built directly from the
project's own data (tool names, argument keys, instruction text, and the
literal/punctuation tokens that appear in a Python call expression) --
closed-vocabulary and fully deterministic, which keeps the tiny model's
embedding table small enough to train in minutes on CPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

# No leading-minus branch: none of this project's sampled argument values
# are negative, so "-" is always its own token. That matters for decoding:
# an ISO date like "2026-05-22" must round-trip through encode/decode
# exactly, and that's only possible if "-" tokenizes separately from the
# digits on either side of it (see NO_SPACE_BEFORE/NO_SPACE_AFTER below).
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|\S")

# Punctuation that should never have a space before it, and tokens after
# which the next token should never have a space -- keeps decoded call
# expressions both valid Python and byte-identical to the un-tokenized
# target string (required for exact-match argument comparison).
NO_SPACE_BEFORE = {")", ",", ".", ":", "]", "-"}
NO_SPACE_AFTER = {"(", "-"}


def word_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


class Tokenizer:
    def __init__(self, vocab: List[str]) -> None:
        self.vocab = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in word_tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = []
        for i in ids:
            tok = self.vocab[i] if 0 <= i < len(self.vocab) else UNK
            if tok in (PAD, BOS, EOS):
                continue
            toks.append(tok)
        # Re-join with spaces, but keep common punctuation tight against the
        # previous token, and keep the inside of a '...'-quoted string tight
        # against its quotes, so a decoded string/date argument comes back
        # byte-for-byte identical to what `ast.literal_eval` would read from
        # the un-tokenized target (no "' Best pizza nearby '" or
        # "'2026 -05 -22'" -- those extra spaces would otherwise survive
        # into the literal value and silently break exact-match comparison).
        out = ""
        prev_tok: str | None = None
        in_string = False
        for tok in toks:
            if tok == "'":
                if not in_string:
                    if prev_tok is not None and prev_tok not in NO_SPACE_AFTER:
                        out += " "
                    out += tok
                    in_string = True
                else:
                    out += tok  # closing quote is always tight against the string
                    in_string = False
                prev_tok = tok
                continue

            just_after_open_quote = prev_tok == "'" and in_string
            suppress_space = (
                prev_tok is None
                or just_after_open_quote
                or prev_tok in NO_SPACE_AFTER
                or tok in NO_SPACE_BEFORE
            )
            if not suppress_space:
                out += " "
            out += tok
            prev_tok = tok
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab)

    @classmethod
    def build(cls, texts: Iterable[str]) -> "Tokenizer":
        seen: dict[str, int] = {}
        for text in texts:
            for tok in word_tokenize(text):
                seen[tok] = seen.get(tok, 0) + 1
        # Deterministic ordering: most frequent first, ties broken alphabetically.
        ordered = sorted(seen.keys(), key=lambda t: (-seen[t], t))
        return cls(SPECIAL_TOKENS + ordered)
