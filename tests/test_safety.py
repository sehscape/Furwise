import pytest

from furwise.generate import parse_reply, species_note
from furwise.safety import (
    DECLINE_PREFIX,
    EMERGENCY_PREFIX,
    check_scope,
    decline_message,
    is_emergency,
    verify,
)


@pytest.mark.parametrize(
    "q",
    [
        "My dog ate a whole bottle of Rimadyl chewables",
        "My cat chewed an Easter lily leaf and now she is vomiting",
        "I accidentally gave my dog two Simparica chews today",
        "My dog is having seizures after his NexGard dose",
        "my dog ate my tylenol, help!",
    ],
)
def test_emergencies_are_detected(q):
    assert is_emergency(q)


@pytest.mark.parametrize(
    "q",
    [
        "How often is the Bravecto chew given to dogs?",
        "Is NexGard safe for a dog with a history of seizures?",
        "What if my dog ate a lily?",
        "Is chocolate toxic to dogs?",
        "How much Rimadyl can a 30 lb dog take?",
    ],
)
def test_non_emergencies(q):
    assert not is_emergency(q)


def test_scope_gate():
    assert not check_scope("Can I give my dog Benadryl?", [], ["dog"]).allowed
    assert not check_scope("Can I give Rimadyl to my horse?", ["carprofen"], []).allowed
    assert not check_scope("What is wrong with my dog, he is limping?", [], ["dog"]).allowed
    assert not check_scope("What vaccines does my puppy need?", [], ["dog"]).allowed
    assert check_scope("Can my dog take Rimadyl?", ["carprofen"], ["dog"]).allowed
    assert decline_message("x").startswith(DECLINE_PREFIX)


CHUNKS = [
    {
        "chunk_id": "a",
        "header": "Rimadyl (carprofen) | for dogs | Warnings",
        "section": "warnings",
        "text": "For use in dogs only. Do not use in cats.",
    },
    {
        "chunk_id": "b",
        "header": "Rimadyl (carprofen) | for dogs | Dosage",
        "section": "dosage",
        "text": "The recommended dosage for oral administration to dogs is 2 mg/lb of body weight daily.",
    },
    {
        "chunk_id": "c",
        "header": "Rimadyl (carprofen) | for dogs | Precautions",
        "section": "precautions",
        "text": "Concomitant use with other NSAIDs or corticosteroids should be avoided.",
    },
]


def test_verifier_keeps_supported_sentences():
    out = verify(
        "Rimadyl is for use in dogs only and should not be used in cats [1]. "
        "The recommended oral dosage for dogs is 2 mg/lb daily [2].",
        CHUNKS,
    )
    assert out.citations == [1, 2] and not out.dropped


def test_verifier_drops_uncited_and_invented_numbers():
    out = verify("Rimadyl is great for cats. The dosage for dogs is 5 mg/lb daily [2].", CHUNKS)
    assert out.answer == ""
    reasons = " ".join(d["reason"] for d in out.dropped)
    assert "no valid citation" in reasons and "number not in cited text" in reasons


def test_verifier_needs_interaction_text_for_interaction_claims():
    bad = verify("Rimadyl interacts with insulin in dogs [2].", CHUNKS)
    assert bad.answer == ""
    good = verify("Concomitant use of Rimadyl with corticosteroids should be avoided [3].", CHUNKS)
    assert good.citations == [3]


def test_verifier_removes_model_written_policy_lines():
    out = verify(f"{EMERGENCY_PREFIX} right away. Do not use Rimadyl in cats [1].", CHUNKS)
    assert not out.answer.startswith(EMERGENCY_PREFIX)
    assert out.citations == [1]


def test_parse_reply_json_and_fallback():
    assert parse_reply('{"answer": "Yes [1].", "citations": [1]}') == ("Yes [1].", [1], True)
    answer, nums, ok = parse_reply("Plain text answer [2] and [3].")
    assert nums == [2, 3] and not ok


def test_species_note_flags_wrong_species_label():
    chunk = {"source": "DailyMed", "species_target": ["dog"], "caution_species": ["cat"]}
    note = species_note([chunk], ["cat"])
    assert "about a cat" in note and "for dogs only" in note and "warns about cats" in note
