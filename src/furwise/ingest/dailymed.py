"""DailyMed animal drug labels: list them, pick the common dog/cat ones, download.

DailyMed has a public REST API (no key). Two endpoints are enough:
    /services/v2/spls.json?doctype=<LOINC>&page=N   every label of one document type
    /services/v2/spls/<setid>.xml                   the full label (SPL XML)

Animal labels come in two document types:
    50578-4  PRESCRIPTION ANIMAL DRUG LABEL
    50577-6  OTC ANIMAL DRUG LABEL
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from furwise.ingest.http import PoliteClient
from furwise.settings import CONFIGS, RAW

log = logging.getLogger(__name__)

API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DOC_TYPES = {"50578-4": "Rx", "50577-6": "OTC"}
INDEX_DIR = RAW / "dailymed" / "index"
SPL_DIR = RAW / "dailymed" / "spl"


def label_url(setid: str) -> str:
    return f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"


@dataclass
class DrugGroup:
    id: str
    cls: str
    ingredients: list[str]
    brands: list[str] = field(default_factory=list)
    brand_only: bool = False


def load_groups(path: Path = CONFIGS / "drugs.yaml") -> list[DrugGroup]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        DrugGroup(
            id=g["id"],
            cls=g.get("class", ""),
            ingredients=[i.lower() for i in g.get("ingredients", [])],
            brands=g.get("brands") or [],
            brand_only=bool(g.get("brand_only", False)),
        )
        for g in raw["groups"]
    ]


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


_TITLE = re.compile(r"^(?P<brand>[^(\[]*)(?:\((?P<generic>[^)]*)\))?\s*(?P<form>[^\[(]*)")


def split_title(title: str) -> dict[str, str]:
    """'RIMADYL (CARPROFEN) TABLET, CHEWABLE [ZOETIS INC.]' -> brand, generic, form."""
    m = _TITLE.match(title.strip())
    parts = {k: (v or "").strip(" ,;") for k, v in m.groupdict().items()} if m else {}
    return {
        "brand": parts.get("brand", title),
        "generic": parts.get("generic", ""),
        "form": parts.get("form", ""),
    }


def match_group(title: str, groups: list[DrugGroup]) -> DrugGroup | None:
    """Most specific group for a label title. Brands beat ingredients, longer brands
    beat shorter ones, and more ingredients beat fewer."""
    t = _norm(title)
    best, best_score = None, 0
    for g in groups:
        for brand in g.brands:
            if _norm(brand) in t:
                score = 100 + 10 * len(brand.split()) + len(g.ingredients)
                if score > best_score:
                    best, best_score = g, score
        if not g.brand_only and all(_norm(i) in t for i in g.ingredients):
            score = 10 * len(g.ingredients)
            if score > best_score:
                best, best_score = g, score
    return best


def fetch_index(client: PoliteClient) -> list[dict]:
    rows = []
    for doctype, kind in DOC_TYPES.items():
        page = 1
        while True:
            dest = INDEX_DIR / f"{doctype}_p{page:03d}.json"
            client.fetch(
                f"{API}/spls.json", dest, params={"doctype": doctype, "pagesize": 100, "page": page}
            )
            payload = json.loads(dest.read_text(encoding="utf-8"))
            rows += [{**r, "kind": kind} for r in payload.get("data", [])]
            if page >= int(payload.get("metadata", {}).get("total_pages") or 0):
                break
            page += 1
    log.info("dailymed index: %d animal labels", len(rows))
    return rows


_PET = re.compile(r" (dogs?|cats?|canine|feline|pupp(y|ies)|kittens?|chewables?|chews?) ")
_LIVESTOCK = re.compile(
    r" (cattle|swine|horses?|equine|sheep|goats?|poultry|chickens?|turkeys?|aquarium|fish|pour on"
    r"|paste|premix|feed|bolus|intramammary|teat|calves|beef|dairy|lactating|deer) "
)


def _published(row: dict) -> float:
    try:
        return datetime.strptime(row.get("published_date", ""), "%b %d, %Y").timestamp()
    except ValueError:
        return 0.0


def candidates(
    index: list[dict], groups: list[DrugGroup], per_group: int
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """(labels to download, every matching label) per group.

    Download order is the original brand first, then the newest label. All matches
    are returned too: their titles are a free source of brand names for the lexicon.
    """
    by_id = {g.id: g for g in groups}
    matched: dict[str, list[dict]] = {g.id: [] for g in groups}
    for row in index:
        group = match_group(row["title"], groups)
        if group:
            matched[group.id].append({**row, "group": group.id, **split_title(row["title"])})

    def rank(row: dict) -> tuple:
        t = _norm(row["title"])
        hits = [i for i, b in enumerate(by_id[row["group"]].brands) if _norm(b) in t]
        # Generic ingredient matches are dominated by livestock products (ivermectin
        # pour-ons, dexamethasone for cattle). Pet-looking titles go first so the
        # download budget is not spent on labels the species filter would drop anyway.
        pet = 0 if _PET.search(t) else 2 if _LIVESTOCK.search(t) else 1
        return (min(hits) if hits else 99, pet, -_published(row))

    return {gid: sorted(rows, key=rank)[:per_group] for gid, rows in matched.items()}, matched


def download(client: PoliteClient, setid: str) -> Path:
    return client.fetch(f"{API}/spls/{setid}.xml", SPL_DIR / f"{setid}.xml")
