from furwise.corpus import species as sp
from furwise.corpus.chunker import split_section
from furwise.corpus.names import Lexicon, singular
from furwise.corpus.sections import classify_heading, classify_section


# ------------------------------------------------------------------ sections
def test_heading_rules():
    assert classify_heading("WARNINGS:") == "warnings"
    assert classify_heading("Warnings and Precautions") == "warnings"
    assert classify_heading("Precautions") == "precautions"
    assert classify_heading("Drug Interactions:") == "drug_interactions"
    assert classify_heading("Target Animal Safety") == "overdose_safety"
    assert classify_heading("Information for Dog Owners") == "owner_info"
    assert classify_heading("User Safety Warnings") == "skip"
    assert classify_heading("Storage") == "skip"
    assert classify_heading("Do not use in cats.") is None


def test_loinc_code_with_more_specific_title():
    assert classify_section("34071-1", "PRECAUTIONS") == "precautions"
    assert classify_section("34067-9", "") == "indications"
    assert classify_section("51945-4", "Package label") == "skip"


# ------------------------------------------------------------------ species
def test_dog_label_that_forbids_cats():
    prof = sp.profile(
        {
            "indications": "Indicated for the relief of pain in dogs.",
            "warnings": "For use in dogs only. Do not use in cats.",
        },
        "RIMADYL (CARPROFEN) TABLET",
    )
    assert prof.target == ["dog"]
    assert prof.restricted == ["cat"]
    assert prof.evidence == "indications"


def test_only_statement_never_removes_an_indicated_species():
    prof = sp.profile(
        {
            "label_header": "For Oral Use in Dogs Only",
            "indications": "Indicated for skin infections in dogs and cats.",
        },
        "ORBAX",
    )
    assert prof.target == ["cat", "dog"]
    assert prof.conflict


def test_caution_mentions_are_not_targets():
    text = "Safety in cats has not been established. Use in dogs for flea control."
    assert sp.positive_mentions(text) == {"dog"}
    assert sp.caution_mentions(text) == {"cat"}


def test_parasite_names_are_not_pets():
    assert sp.mentions("Treats American dog tick and cat flea infestations.") == set()


def test_warned_species_from_only_statement():
    assert sp.warned_species("For oral use in dogs only.", {"dog"}) == {"cat"}


def test_other_animal_indications_never_fall_back_to_dosage_tables():
    prof = sp.profile(
        {
            "indications": "Indicated for post-procedural pain in captive rodents and ferrets.",
            "dosage": "Prairie dogs | 0.48 mg/kg",
        },
        "ETHIQA XR (BUPRENORPHINE) INJECTION",
    )
    assert prof.target == []


def test_no_species_evidence_means_no_target():
    prof = sp.profile({"indications": "Indicated for horses."}, "EQUI-THING PASTE")
    assert prof.target == []


# ------------------------------------------------------------------ chunker
def test_species_subheadings_split_chunks():
    chunks = split_section(["Dogs: give 2 mg/lb once daily.", "Cats: give a single dose only."])
    assert len(chunks) == 2
    assert chunks[0].startswith("Dogs:") and chunks[1].startswith("Cats:")


def test_long_table_repeats_header():
    rows = "\n".join(f"{i} to {i + 9} lb | {i // 10} tablet" for i in range(0, 600, 10))
    table = "Weight | Dose\n" + rows
    chunks = split_section([table], limit=60)
    assert len(chunks) > 1
    assert all(c.startswith("Weight | Dose") for c in chunks)


def test_short_warning_survives():
    assert split_section(["Do not use in cats."]) == ["Do not use in cats."]


# ------------------------------------------------------------------ names
def _lexicon() -> Lexicon:
    return Lexicon.build(
        {"carprofen": {"Rimadyl", "Novox", "carprofen"}, "afoxolaner": {"NexGard", "afoxolaner"}},
        {
            "plant:tulip": {"Tulip", "Tulipa"},
            "plant:easter-lily": {"Easter Lily", "Lilium"},
            "plant:gardenia": {"Gardenia"},
        },
        {"carprofen": "Rimadyl (carprofen)", "afoxolaner": "NexGard (afoxolaner)"},
    )


def test_exact_brand_and_generic():
    lex = _lexicon()
    assert lex.entities("Can my dog take Novox?") == ["carprofen"]
    assert lex.entities("carprofen dose") == ["carprofen"]


def test_fuzzy_drug_names():
    lex = _lexicon()
    matches = lex.resolve("is rimadil safe")
    assert matches[0].entity_id == "carprofen" and matches[0].kind == "fuzzy"
    assert lex.entities("how often do i give nexguard") == ["afoxolaner"]


def test_plants_plural_and_genus_but_never_fuzzy():
    lex = _lexicon()
    assert lex.entities("are tulips poisonous to cats") == ["plant:tulip"]
    assert lex.entities("my kitten chewed a Lilium") == ["plant:easter-lily"]
    assert lex.entities("my cat was in the garden") == []  # not "gardenia"


def test_singular():
    assert singular("lilies") == "lily"
    assert singular("tulips") == "tulip"
    assert singular("grass") == "grass"
