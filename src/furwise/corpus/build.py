"""Offline corpus build: cached raw files -> labels, plants, chunks, lexicon.

Deterministic and network-free once data/raw/ is filled by `furwise fetch`.
Outputs (git-ignored):           data/processed/{chunks,labels,plants}.jsonl, lexicon.json
Outputs (committed, auditable):  data/manifest/{dailymed,aspca}_manifest.csv, stats.json
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict

import pandas as pd

from furwise.corpus import chunker
from furwise.corpus import species as sp
from furwise.corpus.labels import LabelText, assemble
from furwise.corpus.names import Lexicon
from furwise.corpus.sections import SECTIONS, display
from furwise.ingest import aspca, dailymed, spl
from furwise.settings import MANIFEST, PROCESSED

log = logging.getLogger(__name__)

PER_GROUP_DOWNLOADS = 8
PER_GROUP_KEEP = 3
# Two labels in a group with the same form and species are copies when their
# 5-word-shingle Jaccard similarity is at least this. Generic copies of a brand
# share most of their wording; genuinely different products share very little.
DUPLICATE_JACCARD = 0.30


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 0))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _pretty(name: str) -> str:
    """'RIMADYL' -> 'Rimadyl', keep mixed-case names as printed."""
    name = re.sub(r"\s+", " ", name).strip(" ,;")
    return name.title() if name.isupper() else name


def _brand_names(rows: list[dict]) -> set[str]:
    out = set()
    for r in rows:
        brand = _pretty(r.get("brand", ""))
        brand = re.sub(
            r"\b(chewables?|tablets?|injection|for (dogs|cats).*)$", "", brand, flags=re.I
        )
        if brand and len(brand) <= 40:
            out.add(brand.strip())
    return out


def build_labels():
    """(labels, chunks, drug names per group, display names, manifest rows)."""
    groups = dailymed.load_groups()
    by_id = {g.id: g for g in groups}
    index = []
    for path in sorted(dailymed.INDEX_DIR.glob("*.json")):
        kind = dailymed.DOC_TYPES.get(path.stem.split("_")[0], "")
        index += [{**r, "kind": kind} for r in json.loads(path.read_text("utf-8"))["data"]]
    cand, matched = dailymed.candidates(index, groups, PER_GROUP_DOWNLOADS)

    labels, chunks, manifest = [], [], []
    drug_names: dict[str, set[str]] = {}
    display_names: dict[str, str] = {}

    for gid, rows in cand.items():
        group = by_id[gid]
        names = {gid.replace("-", " "), *group.brands, *_brand_names(matched[gid])}
        if len(group.ingredients) == 1:
            names.add(group.ingredients[0])
        kept: list[tuple[LabelText, sp.SpeciesProfile, set]] = []
        for row in rows:
            path = dailymed.SPL_DIR / f"{row['setid']}.xml"
            status = "kept"
            if not path.exists():
                status = "not downloaded"
            else:
                lt = assemble(spl.parse(path))
                texts = {s: lt.text(s) for s in lt.sections}
                prof = sp.profile(texts, lt.title)
                shingles = _shingles(lt.full_text())
                if not prof.target:
                    status = "no dog/cat target"
                elif not lt.sections.get("indications") and not lt.sections.get("dosage"):
                    status = "no usable sections"
                elif any(
                    k.form == lt.form
                    and kp.target == prof.target
                    and _jaccard(ks, shingles) >= DUPLICATE_JACCARD
                    for k, kp, ks in kept
                ):
                    status = "duplicate"
                elif len(kept) >= PER_GROUP_KEEP:
                    status = "group cap"
                else:
                    kept.append((lt, prof, shingles))
                names.add(_pretty(lt.product))
                names |= set(lt.ingredients) if len(group.ingredients) == 1 else set()
            manifest.append(
                {
                    "group": gid,
                    "setid": row["setid"],
                    "version": row.get("spl_version"),
                    "title": row["title"],
                    "status": status,
                    "url": dailymed.label_url(row["setid"]),
                }
            )

        if not kept:
            continue
        drug_names[gid] = names
        lead = _pretty(kept[0][0].product)
        generic = ", ".join(group.ingredients)
        display_names[gid] = f"{lead} ({generic})" if lead.lower() != generic else lead
        brands = sorted(
            {n for n in names if n.lower() not in group.ingredients and n != gid.replace("-", " ")}
        )

        for lt, prof, _ in kept:
            product = _pretty(lt.product)
            labels.append(
                {
                    "setid": lt.setid,
                    "version": lt.version,
                    "entity_id": gid,
                    "class": group.cls,
                    "product": product,
                    "generic": lt.generic or generic,
                    "form": lt.form,
                    "title": lt.title,
                    "labeler": lt.labeler,
                    "species_target": prof.target,
                    "restricted_species": prof.restricted,
                    "species_evidence": prof.evidence,
                    "species_conflict": prof.conflict,
                    "sections": sorted(lt.sections),
                    "url": dailymed.label_url(lt.setid),
                }
            )
            for section in SECTIONS:
                if section not in lt.sections:
                    continue
                for n, text in enumerate(chunker.split_section(lt.sections[section])):
                    header = (
                        f"{product} ({lt.generic or generic}) {lt.form} | "
                        f"{chunker.species_phrase(prof.target)} | {display(section)}"
                    )
                    chunks.append(
                        {
                            "chunk_id": f"{lt.setid[:8]}-{section}-{n}",
                            "entity_id": gid,
                            "entity_type": "drug",
                            "entity_name": display_names[gid],
                            "product": product,
                            "generic": lt.generic or generic,
                            "brands": brands,
                            "doc_id": lt.setid,
                            "section": section,
                            "section_name": display(section),
                            "species_target": prof.target,
                            "caution_species": sorted(sp.warned_species(text, set(prof.target))),
                            "restricted_species": prof.restricted,
                            "source": "DailyMed",
                            "url": dailymed.label_url(lt.setid),
                            "header": re.sub(r"\s+", " ", header),
                            "text": text,
                        }
                    )
    return labels, chunks, drug_names, display_names, manifest


def plant_text(p: aspca.Plant) -> str:
    lines = [f"{p.name}."]
    if p.aliases:
        lines.append("Additional Common Names: " + ", ".join(p.aliases) + ".")
    if p.scientific_names:
        lines.append("Scientific Name: " + ", ".join(p.scientific_names) + ".")
    if p.family:
        lines.append(f"Family: {p.family}.")
    toxic = [s for s in ("dog", "cat") if p.toxic_to.get(s)]
    safe = [s for s in ("dog", "cat") if p.toxic_to.get(s) is False]
    if toxic:
        lines.append("Toxicity: " + ", ".join(f"Toxic to {s.title()}s" for s in toxic) + ".")
    if safe:
        lines.append("Non-Toxicity: " + ", ".join(f"Non-Toxic to {s.title()}s" for s in safe) + ".")
    if p.toxic_principles:
        lines.append(f"Toxic Principles: {p.toxic_principles}")
    if p.clinical_signs:
        lines.append(f"Clinical Signs: {p.clinical_signs}")
    return "\n".join(lines)


def build_plants() -> tuple[list[dict], list[dict], dict[str, set[str]], dict[str, str]]:
    plants = aspca.load()
    rows, chunks = [], []
    names: dict[str, set[str]] = {}
    display_names: dict[str, str] = {}
    for p in plants:
        eid = f"plant:{p.slug}"
        rows.append(asdict(p))
        aliases = {p.name, *p.aliases, *p.scientific_names}
        for sci in p.scientific_names:  # genus alias: "Lilium" -> every Lilium entry
            genus = sci.split()[0] if sci.split() else ""
            if len(genus) >= 5 and genus[0].isupper():
                aliases.add(genus)
        names[eid] = aliases
        display_names[eid] = p.name
        toxic = [s for s in ("dog", "cat") if p.toxic_to.get(s)]
        chunks.append(
            {
                "chunk_id": f"aspca-{p.slug}",
                "entity_id": eid,
                "entity_type": "plant",
                "entity_name": p.name,
                "product": p.name,
                "generic": ", ".join(p.scientific_names),
                "brands": p.aliases,
                "doc_id": p.slug,
                "section": "plant_toxicity",
                "section_name": display("plant_toxicity"),
                # ASPCA entries state the verdict for both species, so both are covered.
                "species_target": ["cat", "dog"],
                "caution_species": toxic,
                "restricted_species": [],
                "toxic_to_dog": p.toxic_to.get("dog"),
                "toxic_to_cat": p.toxic_to.get("cat"),
                "source": "ASPCA",
                "url": p.url,
                "header": f"{p.name} | ASPCA plant list | dogs and cats",
                "text": plant_text(p),
            }
        )
    return rows, chunks, names, display_names


def _jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")


def build() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    labels, label_chunks, drug_names, drug_display, manifest = build_labels()
    plants, plant_chunks, plant_names, plant_display = build_plants()
    chunks = label_chunks + plant_chunks

    _jsonl(PROCESSED / "labels.jsonl", labels)
    _jsonl(PROCESSED / "plants.jsonl", plants)
    _jsonl(PROCESSED / "chunks.jsonl", chunks)
    lex = Lexicon.build(drug_names, plant_names, drug_display | plant_display)
    lex.save(PROCESSED / "lexicon.json")

    pd.DataFrame(manifest).to_csv(MANIFEST / "dailymed_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "slug": p["slug"],
                "name": p["name"],
                "toxic_to_dog": p["toxic_to"].get("dog"),
                "toxic_to_cat": p["toxic_to"].get("cat"),
                "url": p["url"],
            }
            for p in plants
        ]
    ).to_csv(MANIFEST / "aspca_manifest.csv", index=False)

    statuses = Counter(m["status"] for m in manifest)
    stats = {
        "labels": len(labels),
        "drug_groups": len({lab["entity_id"] for lab in labels}),
        "plants": len(plants),
        "plants_toxic_dog": sum(1 for p in plants if p["toxic_to"].get("dog")),
        "plants_toxic_cat": sum(1 for p in plants if p["toxic_to"].get("cat")),
        "chunks": len(chunks),
        "label_chunks": len(label_chunks),
        "chunks_by_section": dict(Counter(c["section"] for c in chunks).most_common()),
        "labels_by_species": dict(Counter("+".join(lab["species_target"]) for lab in labels)),
        "labels_restricting_other_species": sum(1 for lab in labels if lab["restricted_species"]),
        "species_conflicts": sum(1 for lab in labels if lab["species_conflict"]),
        "label_selection": dict(statuses),
    }
    (MANIFEST / "stats.json").write_text(json.dumps(stats, indent=2), "utf-8")
    log.info("corpus: %s", stats)
    return stats
