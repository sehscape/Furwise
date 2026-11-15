"""Turn a parsed SPL label into canonical sections of clean text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from furwise.corpus import species as sp
from furwise.corpus.sections import SKIP, classify_heading, classify_section
from furwise.ingest.spl import Block, Label, Section

# Boilerplate is removed sentence by sentence, never block by block: the federal
# law line often shares a paragraph with "Do not use in cats."
_BOILERPLATE = re.compile(
    r"federal law (\(u\.?s\.?a?\.?\) )?restricts|^caution:?\s*$|keep (this and all|out of reach)"
    r"|not for (use in )?humans?\b|not for human use|^rx only\.?$|approved by fda under"
    r"|^nada\b|^anada\b|^made in\b|^manufactured (for|by)|^distributed by|^marketed by"
    r"|^revised:?|^iss(ued)?\.?\s|^©|trademark|all rights reserved|^net (contents|wt)",
    re.I,
)
_HEADER_WORD_LIMIT = 90


@dataclass
class LabelText:
    setid: str
    version: int
    title: str
    product: str  # brand / product name as printed
    generic: str
    form: str
    ingredients: list[str]
    labeler: str
    sections: dict[str, list[str]] = field(default_factory=dict)  # canonical -> blocks

    def text(self, section: str) -> str:
        return "\n".join(self.sections.get(section, []))

    def full_text(self) -> str:
        return "\n".join("\n".join(v) for v in self.sections.values())


def strip_boilerplate(text: str) -> str:
    kept = [s for s in sp.sentences(text) if not _BOILERPLATE.search(s.strip())]
    return " ".join(kept).strip() if "\n" not in text else "\n".join(kept).strip()


def _walk(sec: Section, inherited: str | None, out: list[tuple[str, Block]], seen_any: list[bool]):
    own = classify_section(sec.code, sec.title)
    current = own or inherited
    if current is None and not seen_any[0] and not sec.title:
        current = "label_header"  # untitled banner before the first real section
    if own and own != SKIP:
        seen_any[0] = True
    for block in sec.blocks:
        if block.lead:
            heading = classify_heading(block.lead)
            # An inline heading only counts if it is the whole block or ends with a colon.
            if heading and (block.lead == block.text or block.text.startswith(block.lead)):
                current = heading
                if heading != SKIP:
                    seen_any[0] = True
                if block.lead == block.text:
                    continue
        if current and current != SKIP:
            out.append((current, block))
    for child in sec.children:
        _walk(child, current, out, seen_any)


def assemble(label: Label) -> LabelText:
    product = label.products[0] if label.products else None
    ingredients: list[str] = []
    for p in label.products:
        for i in p.ingredients:
            if i not in ingredients:
                ingredients.append(i)
    lt = LabelText(
        setid=label.setid,
        version=label.version,
        title=label.title,
        product=(product.name if product else label.title).strip(),
        generic=product.generic if product else "",
        form=product.form if product else "",
        ingredients=ingredients,
        labeler=label.labeler,
    )
    pairs: list[tuple[str, Block]] = []
    seen_any = [False]
    for sec in label.sections:
        _walk(sec, None, pairs, seen_any)

    header_words = 0
    for canonical, block in pairs:
        text = strip_boilerplate(block.text) if block.kind != "table" else block.text
        if len(text.split()) < 2:
            continue
        if canonical == "label_header":
            if header_words >= _HEADER_WORD_LIMIT:
                continue
            header_words += len(text.split())
        lt.sections.setdefault(canonical, []).append(text)
    return lt
