"""Section-aware chunking with a contextual header on every chunk.

Rules, each learned from how labels are written:
  * a chunk never crosses a section boundary;
  * "Dogs:" / "Cats:" sub-headings start a new chunk, so a dog dose and a cat
    dose never share one;
  * long tables are split by rows and keep their header row;
  * every chunk carries a header like "Rimadyl (carprofen) chewable tablet | for dogs |
    Warnings". A chunk that just says "Do not use in cats." means nothing without it.
"""

from __future__ import annotations

import re

from furwise.corpus import species as sp
from furwise.settings import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS

_SPECIES_SUBHEAD = re.compile(
    r"^(for |in )?(dogs?|cats?|puppies|kittens|canine|feline)\s*(:|-|\.|only:)", re.I
)


def _words(text: str) -> int:
    return len(text.split())


def _split_long(block: str, limit: int) -> list[str]:
    lines = block.split("\n")
    if len(lines) > 2 and " | " in lines[1]:  # a table: split by rows, repeat the header
        head, rows, parts, cur = lines[0], lines[1:], [], []
        for row in rows:
            if cur and _words("\n".join([head, *cur, row])) > limit:
                parts.append("\n".join([head, *cur]))
                cur = []
            cur.append(row)
        if cur:
            parts.append("\n".join([head, *cur]))
        return parts
    parts, cur = [], []
    for sent in sp.sentences(block):
        if cur and _words(" ".join([*cur, sent])) > limit:
            parts.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        parts.append(" ".join(cur))
    return parts


def split_section(blocks: list[str], limit: int = CHUNK_MAX_WORDS) -> list[str]:
    pieces: list[str] = []
    for block in blocks:
        pieces += _split_long(block, limit) if _words(block) > limit else [block]

    chunks: list[str] = []
    cur: list[str] = []
    for piece in pieces:
        new_species_part = bool(_SPECIES_SUBHEAD.match(piece))
        if cur and (new_species_part or _words("\n".join([*cur, piece])) > limit):
            chunks.append("\n".join(cur))
            cur = []
        cur.append(piece)
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if _words(c) >= CHUNK_MIN_WORDS]


def species_phrase(target: list[str]) -> str:
    if set(target) == {"dog", "cat"}:
        return "for dogs and cats"
    return f"for {target[0]}s" if target else "species not stated"
