from pathlib import Path

from furwise.corpus.labels import assemble, strip_boilerplate
from furwise.ingest import aspca
from furwise.ingest.dailymed import DrugGroup, match_group, split_title
from furwise.ingest.spl import parse

FIXTURE = Path(__file__).parent / "fixtures" / "sample_spl.xml"


def test_parse_products_and_metadata():
    label = parse(FIXTURE)
    assert label.setid.startswith("00000000")
    assert label.version == 3
    assert label.products[0].name == "Testofen"
    assert label.products[0].generic == "testoprofen"
    assert label.products[0].ingredients == ["testoprofen"]


def test_assemble_finds_banner_inline_headings_and_skips_package():
    lt = assemble(parse(FIXTURE))
    assert "For oral use in dogs only" in lt.text("label_header")
    assert "Do not use in cats." in lt.text("label_header")
    assert "Not for use in cats." in lt.text("warnings")  # bold inline heading
    assert "corticosteroids" in lt.text("drug_interactions")
    assert "NDC" not in lt.full_text()  # package label section skipped
    assert "Federal law" not in lt.full_text()  # boilerplate removed by sentence
    assert "Keep out of reach" not in lt.text("warnings")


def test_tables_and_line_breaks():
    lt = assemble(parse(FIXTURE))
    dosage = lt.text("dosage")
    assert "Weight (lb) | Tablets" in dosage
    assert "once daily. Give with food." in dosage  # <br/> keeps a space


def test_strip_boilerplate_keeps_the_warning():
    text = "Federal law restricts this drug to use by a veterinarian. Do not use in cats."
    assert strip_boilerplate(text) == "Do not use in cats."


def test_title_split_and_group_matching():
    parts = split_title("RIMADYL (CARPROFEN) TABLET, CHEWABLE [ZOETIS INC.]")
    assert parts["brand"] == "RIMADYL" and parts["generic"] == "CARPROFEN"
    groups = [
        DrugGroup("ivermectin", "", ["ivermectin"], ["Heartgard"]),
        DrugGroup("ivermectin-pyrantel", "", ["ivermectin", "pyrantel"], ["Heartgard Plus"]),
    ]
    assert match_group("HEARTGARD PLUS (IVERMECTIN AND PYRANTEL) CHEW", groups).id == (
        "ivermectin-pyrantel"
    )
    assert match_group("IVOMEC (IVERMECTIN) INJECTION", groups).id == "ivermectin"


ASPCA_LIST = """
<div class="view"><div class="view-header"><h2>Plants Toxic to Cats</h2></div>
<div class="views-field views-field-path"><span class="field-content">
<a href="/pet-care/x/easter-lily">Easter Lily</a> (White Lily) | <b>Scientific Names:</b>
<i>Lilium longiflorum</i> | <b>Family:</b> Liliaceae</span></div></div>
<div class="view"><div class="view-header"><h2>Plants Non-Toxic to Cats</h2></div>
<div class="views-field views-field-path"><span class="field-content">
<a href="/pet-care/x/spider-plant">Spider Plant</a> () | <b>Scientific Names:</b>
<i>Chlorophytum comosum</i> | <b>Family:</b> </span></div></div>
"""


def test_aspca_list_parsing_and_merge():
    rows = aspca.parse_list_page(ASPCA_LIST, "cat")
    plants = aspca.merge(rows)
    assert plants["easter-lily"].toxic_to == {"cat": True}
    assert plants["easter-lily"].scientific_names == ["Lilium longiflorum"]
    assert plants["easter-lily"].aliases == ["White Lily"]
    assert plants["spider-plant"].toxic_to == {"cat": False}


def test_aspca_detail_overrides_list():
    plant = aspca.Plant("easter-lily", "Easter Lily", "u", toxic_to={"dog": True})
    aspca.apply_detail(
        plant,
        {
            "toxicity": "Toxic to Cats",
            "non_toxicity": "Non-Toxic to Dogs",
            "clinical_signs": "Vomiting",
        },
    )
    assert plant.toxic_to == {"dog": False, "cat": True}
    assert plant.clinical_signs == "Vomiting"
