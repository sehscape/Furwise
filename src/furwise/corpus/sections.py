"""Map label sections onto a small set of canonical sections.

Human drug labels use clean LOINC section codes. Animal labels often don't: most
of their sections are "SPL UNCLASSIFIED", and the real heading is either the
section title or a bold phrase at the start of a paragraph. So we try, in order:
LOINC code -> title keywords -> inline heading -> parent section.
"""

from __future__ import annotations

import re

# canonical id -> display name (order = reading order in answers)
SECTIONS = {
    "label_header": "Label Header",
    "indications": "Indications",
    "dosage": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "overdose_safety": "Overdosage / Target Animal Safety",
    "owner_info": "Information for Owners",
    "plant_toxicity": "Plant Toxicity",
}

LOINC = {
    "34067-9": "indications",
    "34068-7": "dosage",
    "34070-3": "contraindications",
    "34071-1": "warnings",
    "34066-1": "warnings",  # boxed warning
    "43685-7": "warnings",  # warnings and precautions
    "42232-9": "precautions",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "34088-5": "overdose_safety",
    "34076-0": "owner_info",
    "88436-1": "owner_info",
}

# Sections that never help a pet owner's question.
SKIP = "skip"
SKIP_LOINC = {
    "48780-1",  # product data elements
    "51945-4",  # package label
    "34069-5",  # how supplied
    "44425-7",  # storage
    "34093-5",  # references
    "34090-1",  # clinical pharmacology
    "34092-7",  # clinical studies
    "34089-3",  # description
    "50742-6",  # environmental warning
}

# (pattern on a lower-cased heading, canonical id). First match wins.
TITLE_RULES = [
    (r"user safety|human (safety|warning)|residue|withdrawal|environmental", SKIP),
    (r"contra-?indication", "contraindications"),
    (r"drug interaction|interactions?\b", "drug_interactions"),
    (r"overdos|animal safety|^safety\b|safety stud|margin of safety|toxicology", "overdose_safety"),
    (r"adverse|side effects|post-?approval|reactions?\b", "adverse_reactions"),
    (r"boxed warning|warnings?\b", "warnings"),
    (r"precaution", "precautions"),
    (r"indication|^uses?\b|^purpose", "indications"),
    (r"dosage|dosing|dose\b|administration|directions|how to (use|give|apply)", "dosage"),
    (
        r"(owner|client|patient|pet owner)s? information|information for (dog |cat |pet )?owners?"
        r"|information for (the )?(dog|cat|pet) owner",
        "owner_info",
    ),
    (
        r"how supplied|storage|description|pharmacolog|effectiveness|clinical|reference"
        r"|package|principal display|mode of action|microbiology|chemistry|composition"
        r"|presentation|disposal|manufactured|distributed|questions\?|contact",
        SKIP,
    ),
]
_RULES = [(re.compile(p), c) for p, c in TITLE_RULES]


def classify_heading(heading: str) -> str | None:
    """Canonical id (or "skip") for a heading, or None if it is not a known heading."""
    h = heading.lower().strip(" :.-")
    if not h or len(h.split()) > 9:
        return None
    for rx, canonical in _RULES:
        if rx.search(h):
            return canonical
    return None


def classify_section(code: str, title: str) -> str | None:
    if code in SKIP_LOINC:
        return SKIP
    if code in LOINC:
        # A titled LOINC section can still be more specific ("Precautions" coded as warnings).
        return classify_heading(title) or LOINC[code]
    return classify_heading(title) if title else None


def display(section: str) -> str:
    return SECTIONS.get(section, section.replace("_", " ").title())
