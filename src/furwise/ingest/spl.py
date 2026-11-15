"""Read a DailyMed SPL document (HL7 v3 XML) into sections of plain text.

The SPL narrative is a small XHTML-like language. Three things matter for us:
  * paragraphs and lists become lines of text;
  * tables become "cell | cell | cell" rows, so a dosing table keeps its shape;
  * a bold or ALL-CAPS lead phrase ("WARNINGS:", "Drug Interactions:") is kept as
    `lead`, because animal labels often hide their real headings inside one big
    unclassified section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

NS = {"v3": "urn:hl7-org:v3"}
_SPACES = re.compile(r"\s+")
_GLUED = re.compile(r"([a-z)]\.)([A-Z][a-z])")


def _clean(text: str) -> str:
    text = _SPACES.sub(" ", text.replace("\xa0", " ").replace("\u200b", "")).strip()
    # Adjacent <content> runs and <br/> lose their space: "therapy.Appropriate".
    return _GLUED.sub(r"\1 \2", text)


def _tag(el) -> str:
    return el.tag.split("}", 1)[-1] if isinstance(el.tag, str) else ""


@dataclass
class Block:
    kind: str  # "p", "list" or "table"
    text: str
    lead: str = ""  # bold / caps opening phrase, a heading candidate


@dataclass
class Section:
    code: str
    code_name: str
    title: str
    blocks: list[Block] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)


@dataclass
class Product:
    name: str
    generic: str
    form: str
    ingredients: list[str]


@dataclass
class Label:
    setid: str
    version: int
    effective: str
    title: str
    labeler: str
    products: list[Product]
    sections: list[Section]


# ------------------------------------------------------------------- narrative
def _lead(el) -> str:
    """Opening bold/underlined phrase, or the whole text if it is short and ALL CAPS."""
    first = None
    if el.text and el.text.strip():
        first = None  # text before any child: no styled lead
    else:
        for child in el:
            if _tag(child) == "content":
                first = child
            break
    if first is not None and re.search(r"bold|underline", first.get("styleCode", ""), re.I):
        return _clean("".join(first.itertext()))
    text = _clean("".join(el.itertext()))
    if 0 < len(text.split()) <= 8 and text.upper() == text and re.search(r"[A-Z]", text):
        return text
    return ""


def _table(el) -> str:
    rows = []
    for tr in el.iter("{urn:hl7-org:v3}tr"):
        cells = [_clean("".join(td.itertext())) for td in tr if _tag(td) in ("td", "th")]
        if any(cells):
            rows.append(" | ".join(cells))
    caption = el.find("v3:caption", NS)
    head = [_clean("".join(caption.itertext()))] if caption is not None else []
    return "\n".join(head + rows)


def _list(el) -> str:
    items = []
    for item in el.findall("v3:item", NS):
        text = _clean("".join(item.itertext()))
        if text:
            items.append(f"- {text}")
    return "\n".join(items)


def _blocks(text_el) -> list[Block]:
    blocks: list[Block] = []
    if text_el is None:
        return blocks
    for br in text_el.iter("{urn:hl7-org:v3}br"):
        br.tail = " " + (br.tail or "")
    if text_el.text and text_el.text.strip():
        blocks.append(Block("p", _clean(text_el.text)))
    for el in text_el:
        kind = _tag(el)
        if kind == "paragraph":
            text = _clean("".join(el.itertext()))
            if text:
                blocks.append(Block("p", text, _lead(el)))
        elif kind == "list":
            text = _list(el)
            if text:
                blocks.append(Block("list", text))
        elif kind == "table":
            text = _table(el)
            if text:
                blocks.append(Block("table", text))
        elif kind in ("content", "br", "sup", "sub", "linkHtml"):
            text = _clean("".join(el.itertext()))
            if text:
                blocks.append(Block("p", text))
        if el.tail and el.tail.strip():
            blocks.append(Block("p", _clean(el.tail)))
    return blocks


def _section(el) -> Section:
    code = el.find("v3:code", NS)
    title = el.find("v3:title", NS)
    sec = Section(
        code=code.get("code", "") if code is not None else "",
        code_name=code.get("displayName", "") if code is not None else "",
        title=_clean("".join(title.itertext())) if title is not None else "",
        blocks=_blocks(el.find("v3:text", NS)),
    )
    for comp in el.findall("v3:component/v3:section", NS):
        sec.children.append(_section(comp))
    return sec


# ---------------------------------------------------------------- products
def _products(root) -> list[Product]:
    out = []
    for prod in root.iterfind(".//v3:manufacturedProduct/v3:manufacturedProduct", NS):
        name = _clean("".join(prod.findtext("v3:name", default="", namespaces=NS)))
        form_el = prod.find("v3:formCode", NS)
        generic = prod.findtext(
            "v3:asEntityWithGeneric/v3:genericMedicine/v3:name", default="", namespaces=NS
        )
        moieties = []
        for ing in prod.iterfind("v3:ingredient", NS):
            if not ing.get("classCode", "").startswith("ACTI"):
                continue
            moiety = ing.findtext(
                "v3:ingredientSubstance/v3:activeMoiety/v3:activeMoiety/v3:name",
                default="",
                namespaces=NS,
            ) or ing.findtext("v3:ingredientSubstance/v3:name", default="", namespaces=NS)
            if moiety and moiety.lower() not in moieties:
                moieties.append(moiety.lower())
        if name:
            out.append(
                Product(
                    name=name,
                    generic=_clean(generic).lower(),
                    form=(form_el.get("displayName", "") if form_el is not None else "").lower(),
                    ingredients=moieties,
                )
            )
    return out


def parse(path: Path | str) -> Label:
    root = etree.parse(str(path)).getroot()
    title = (
        _clean("".join(root.find("v3:title", NS).itertext()))
        if root.find("v3:title", NS) is not None
        else ""
    )
    version = root.find("v3:versionNumber", NS)
    eff = root.find("v3:effectiveTime", NS)
    labeler = root.findtext(
        "v3:author/v3:assignedEntity/v3:representedOrganization/v3:name",
        default="",
        namespaces=NS,
    )
    body = root.find("v3:component/v3:structuredBody", NS)
    sections = (
        [_section(s) for s in body.findall("v3:component/v3:section", NS)]
        if body is not None
        else []
    )
    setid = root.find("v3:setId", NS)
    return Label(
        setid=setid.get("root", "") if setid is not None else Path(path).stem,
        version=int(version.get("value", "0")) if version is not None else 0,
        effective=eff.get("value", "")[:8] if eff is not None else "",
        title=title,
        labeler=_clean(labeler),
        products=_products(root),
        sections=sections,
    )
