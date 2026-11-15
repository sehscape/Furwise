"""ASPCA Toxic and Non-Toxic Plant List for dogs and cats.

Collected lightly and credited:
  * the two list pages (dogs, cats) give names, scientific names and the
    toxic / non-toxic verdict per species;
  * detail pages (toxic principles, clinical signs) are fetched only for plants
    that are toxic to a dog or a cat, one request every 10 s (robots.txt);
  * raw pages stay in data/raw/ (git-ignored). The repository ships code, not content.

Source: ASPCA Animal Poison Control Center, https://www.aspca.org/pet-care/animal-poison-control
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

from furwise.ingest.http import PoliteClient
from furwise.settings import RAW

log = logging.getLogger(__name__)

BASE = "https://www.aspca.org"
LIST_URLS = {
    "dog": f"{BASE}/pet-care/animal-poison-control/dogs-plant-list",
    "cat": f"{BASE}/pet-care/animal-poison-control/cats-plant-list",
}
LIST_DIR = RAW / "aspca" / "lists"
PLANT_DIR = RAW / "aspca" / "plants"
CREDIT = "ASPCA Animal Poison Control Center, Toxic and Non-Toxic Plant List"


@dataclass
class Plant:
    slug: str
    name: str
    url: str
    aliases: list[str] = field(default_factory=list)
    scientific_names: list[str] = field(default_factory=list)
    family: str = ""
    toxic_to: dict[str, bool] = field(default_factory=dict)  # {"dog": True, "cat": False}
    toxic_principles: str = ""
    clinical_signs: str = ""


def _split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", text or "") if p.strip()]


def _add(items: list[str], new: list[str]) -> None:
    for item in new:
        if item and item.lower() not in {i.lower() for i in items}:
            items.append(item)


def parse_list_page(html: str, species: str) -> list[dict]:
    """One row per plant link: name, aliases, scientific names, family, toxic flag."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for view in soup.select("div.view"):
        heading = view.select_one(".view-header h2")
        if heading is None or "toxic" not in heading.get_text().lower():
            continue
        toxic = "non-toxic" not in heading.get_text().lower()
        for span in view.select(".views-field-path .field-content"):
            link = span.find("a")
            if link is None or not link.get("href"):
                continue
            text = span.get_text(" ", strip=True)
            head, *rest = text.split("|")
            aliases = re.search(r"\((.*)\)\s*$", head)
            sci = family = ""
            for part in rest:
                if "Scientific Name" in part:
                    sci = part.split(":", 1)[-1].strip()
                elif "Family" in part:
                    family = part.split(":", 1)[-1].strip()
            href = link["href"]
            rows.append(
                {
                    "slug": href.rstrip("/").rsplit("/", 1)[-1],
                    "url": href if href.startswith("http") else BASE + href,
                    "name": link.get_text(" ", strip=True),
                    "aliases": _split(aliases.group(1)) if aliases else [],
                    "scientific_names": _split(sci),
                    "family": family,
                    "species": species,
                    "toxic": toxic,
                }
            )
    return rows


def parse_detail_page(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    fields = {
        "additional-common-names": "aliases",
        "scientific-name": "scientific",
        "family": "family",
        "toxicity": "toxicity",
        "non-toxicity": "non_toxicity",
        "toxic-principles": "toxic_principles",
        "clinical-signs": "clinical_signs",
    }
    out = {}
    for css, key in fields.items():
        node = soup.select_one(f".field-name-field-{css} .values")
        if node is not None:
            out[key] = node.get_text(" ", strip=True)
    return out


def merge(rows: list[dict]) -> dict[str, Plant]:
    plants: dict[str, Plant] = {}
    for r in rows:
        p = plants.setdefault(r["slug"], Plant(r["slug"], r["name"], r["url"], family=r["family"]))
        _add(p.aliases, r["aliases"])
        _add(p.scientific_names, r["scientific_names"])
        p.toxic_to[r["species"]] = r["toxic"]
    return plants


def apply_detail(plant: Plant, detail: dict[str, str]) -> None:
    _add(plant.aliases, _split(detail.get("aliases", "")))
    if detail.get("scientific"):
        _add(plant.scientific_names, [detail["scientific"]])
    plant.family = detail.get("family") or plant.family
    # The detail page is the more specific statement, so it wins over the lists.
    for sp, word in (("dog", "dogs"), ("cat", "cats")):
        if re.search(rf"\btoxic to {word}\b", detail.get("toxicity", ""), re.I):
            plant.toxic_to[sp] = True
        if re.search(rf"\bnon-toxic to {word}\b", detail.get("non_toxicity", ""), re.I):
            plant.toxic_to[sp] = False
    plant.toxic_principles = detail.get("toxic_principles", "").strip()
    plant.clinical_signs = detail.get("clinical_signs", "").strip()


def fetch(client: PoliteClient, with_details: bool = True) -> dict[str, Plant]:
    rows = []
    for species, url in LIST_URLS.items():
        path = client.fetch(url, LIST_DIR / f"{species}.html")
        rows += parse_list_page(path.read_text(encoding="utf-8", errors="ignore"), species)
    plants = merge(rows)
    log.info("aspca lists: %d plants", len(plants))
    if with_details:
        todo = [p for p in plants.values() if any(p.toxic_to.values())]
        for i, p in enumerate(todo, 1):
            try:
                client.fetch(p.url, PLANT_DIR / f"{p.slug}.html")
            except Exception as exc:  # a missing detail page is not fatal
                log.warning("aspca %s: %s", p.slug, exc)
            if i % 25 == 0:
                log.info("aspca detail pages: %d/%d", i, len(todo))
    return plants


def load() -> list[Plant]:
    """Parse the cached pages (no network)."""
    rows = []
    for species in LIST_URLS:
        html = (LIST_DIR / f"{species}.html").read_text(encoding="utf-8", errors="ignore")
        rows += parse_list_page(html, species)
    plants = merge(rows)
    for p in plants.values():
        page = PLANT_DIR / f"{p.slug}.html"
        if page.exists():
            apply_detail(p, parse_detail_page(page.read_text(encoding="utf-8", errors="ignore")))
    return sorted(plants.values(), key=lambda p: p.slug)


def to_json(plants: list[Plant]) -> str:
    return "\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in plants)
