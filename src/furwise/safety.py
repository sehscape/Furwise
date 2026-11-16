"""Safety rules that run as code around the LLM, so they are guaranteed rather than
"usually followed".

1. Emergency gate: a rule-based, high-recall detector. When it fires, the answer
   starts with EMERGENCY_PREFIX, prepended in code (never generated).
2. Scope gate: questions Furwise must not answer (human medicines, other species,
   diagnosis, products outside the sources) get a templated decline before retrieval.
3. Citation verifier: every sentence must cite a retrieved chunk that supports it;
   dose numbers must appear in the cited text; an interaction claim needs a label
   section that actually names the interaction. Anything else is removed, and if
   nothing survives the answer becomes a decline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMERGENCY_PREFIX = "Contact your vet or a pet poison helpline"
EMERGENCY_LINE = f"{EMERGENCY_PREFIX} right away."
DECLINE_PREFIX = "I can't answer that from my sources"
# Shown with emergency answers. The number is printed on the ASPCA plant pages.
POISON_CONTROL = "ASPCA Animal Poison Control Center: (888) 426-4435"

# ------------------------------------------------------------------ emergencies
# An exposure that already happened: "ate", "got into", "licked", "two chews today".
_EXPOSURE = re.compile(
    r"\b(ate|eaten|swallowed|ingested|chewed|chewing on|chewed up|got into|licked|drank|"
    r"consumed|gobbled|nibbled|bit into|took|double[- ]?dosed|overdosed|"
    r"accidentally (gave|given|put)|gave .{0,20}\b(two|three|2|3)\b|"
    r"whole (box|bottle|pack|tube|bag))\b",
    re.I,
)
_SYMPTOMS = (
    r"(seiz(ure|ing)s?|convuls\w*|collaps\w*|unconscious|unresponsive|not breathing|"
    r"trouble breathing|can'?t breathe|vomit(ing|ed)?|throwing up|bloody (stool|diarrh\w+)|"
    r"trembl\w+|tremor(s|ing)?|shaking|twitching|drooling|foaming|pale gums|stumbling|"
    r"wobbly|paraly\w+|swollen (face|muzzle)|lethargic|won'?t wake|bleeding)"
)
# A symptom counts when the pet has it now ("is vomiting", "seizures after his dose"),
# not when it is mentioned ("a dog with a history of seizures").
_SYMPTOM_NOW = re.compile(
    rf"\b(is|are|was|keeps|kept|started|starting|has been|have been|having|had|began|now)\b"
    rf"[^.?!]{{0,25}}\b{_SYMPTOMS}|\b{_SYMPTOMS}\b[^.?!]{{0,40}}\b(after|since|right now|now)\b",
    re.I,
)
_URGENT = re.compile(r"\b(emergency|urgent|asap|dying|poisoned|going to die)\b|\bhelp\s*!", re.I)
# Hypotheticals and prevention questions are not emergencies on their own.
_HYPOTHETICAL = re.compile(r"\b(what if|if (my|a) (dog|cat) (ate|eats)|in case)\b", re.I)


def is_emergency(question: str) -> bool:
    q = question.strip()
    if _URGENT.search(q) or _SYMPTOM_NOW.search(q):
        return True
    return bool(_EXPOSURE.search(q)) and not _HYPOTHETICAL.search(q)


# ------------------------------------------------------------------ scope
_HUMAN_MEDS = re.compile(
    r"\b(tylenol|acetaminophen|paracetamol|ibuprofen|advil|motrin|aspirin|naproxen|aleve|"
    r"benadryl|diphenhydramine|zyrtec|cetirizine|claritin|loratadine|pepto|imodium|loperamide|"
    r"xanax|alprazolam|melatonin|tramadol|gabapentin|prozac|zoloft|sertraline|adderall|"
    r"ambien|nyquil|dayquil|sudafed|pseudoephedrine|metformin|lisinopril|omeprazole|prilosec|"
    r"famotidine|pepcid|vicks|neosporin|hydrocortisone cream|cbd|ivermectin paste)\b",
    re.I,
)
_OTHER_SPECIES = re.compile(
    r"\b(horses?|ponies|pony|cattle|cows?|calves|pigs?|swine|sheep|goats?|chickens?|poultry|"
    r"birds?|parrots?|budgies?|rabbits?|bunn(y|ies)|ferrets?|hamsters?|guinea pigs?|rats?|mice|"
    r"reptiles?|snakes?|lizards?|turtles?|tortoises?|fish|humans?|my (son|daughter|child|baby|kid)|"
    r"myself|people)\b",
    re.I,
)
_DIAGNOSIS = re.compile(
    r"\b(diagnos\w+|what (disease|illness|condition) does|what'?s wrong with|"
    r"does my (dog|cat) have|is (this|it) cancer|should i euthani[sz]e|"
    r"how long (does|will) (my|he|she) (have|live))\b",
    re.I,
)
_OUT_OF_SCOPE_TOPICS = re.compile(
    r"\b(vaccin\w+|rabies shot|spay|neuter|microchip|pet insurance|grooming|training|"
    r"diet plan|raw food|kibble|price|cost|buy|cheapest|where can i get|prescription without)\b",
    re.I,
)


@dataclass
class ScopeDecision:
    allowed: bool
    reason: str = ""


def check_scope(question: str, entities: list[str], species: list[str]) -> ScopeDecision:
    """Decide before retrieval whether Furwise may answer at all."""
    if _HUMAN_MEDS.search(question) and not entities:
        return ScopeDecision(
            False, "it is a human or unlabeled medicine with no FDA animal label in DailyMed"
        )
    if _OTHER_SPECIES.search(question) and not species:
        return ScopeDecision(False, "Furwise only covers dogs and cats")
    if _DIAGNOSIS.search(question):
        return ScopeDecision(False, "diagnosing a pet needs a veterinary examination")
    if _OUT_OF_SCOPE_TOPICS.search(question) and not entities:
        return ScopeDecision(False, "that topic is outside the drug labels and plant list I use")
    if not entities:
        return ScopeDecision(
            False,
            "I could not find that medicine or plant in my sources "
            "(FDA animal drug labels on DailyMed and the ASPCA plant list)",
        )
    return ScopeDecision(True)


def decline_message(reason: str) -> str:
    return (
        f"{DECLINE_PREFIX}: {reason}. Please ask your veterinarian, who can look at "
        "your pet's full history."
    )


# ------------------------------------------------------------------ verification
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")
_CITE = re.compile(r"\[(\d{1,2})\]")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|mL|g|kg|lb|lbs|%|iu|units?)\b", re.I)
_INTERACTION = re.compile(
    r"\b(interact\w*|concurrent\w*|concomitant\w*|together with|"
    r"combined with|in combination|co-?administ\w+|with other)\b",
    re.I,
)
_INTERACTION_SECTIONS = {"drug_interactions", "precautions", "warnings", "contraindications"}
_STOPWORDS = set(
    """the a an and or of to in on for with is are be was were this that it its as at by
from not no do does can may should your you if than then there these those has have had will
would could also any all into per each which who when what""".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if w not in _STOPWORDS}


@dataclass
class Verified:
    answer: str
    citations: list[int]  # source numbers kept, in order
    dropped: list[dict] = field(default_factory=list)


def verify(answer: str, chunks: list[dict], min_overlap: float = 0.35) -> Verified:
    """Keep only sentences whose [n] citations point at retrieved chunks that support them.
    Emergency / decline lines written by the model are removed: the app adds its own."""
    kept, cited, dropped = [], [], []
    for sent in _SENT_SPLIT.split(answer.strip()):
        if not sent.strip() or sent.startswith((EMERGENCY_PREFIX, DECLINE_PREFIX)):
            continue
        numbers = [int(n) for n in _CITE.findall(sent)]
        valid = [n for n in numbers if 1 <= n <= len(chunks)]
        reason = ""
        if not valid:
            reason = "no valid citation"
        else:
            evidence = " ".join(f"{chunks[n - 1]['header']} {chunks[n - 1]['text']}" for n in valid)
            words = _content_words(_CITE.sub("", sent))
            overlap = len(words & _content_words(evidence)) / max(len(words), 1)
            missing_numbers = [
                x
                for x in _NUMBER.findall(sent)
                if re.sub(r"\s+", "", x.lower()) not in re.sub(r"\s+", "", evidence.lower())
            ]
            if overlap < min_overlap:
                reason = f"weak support ({overlap:.0%} word overlap)"
            elif missing_numbers:
                reason = f"number not in cited text: {missing_numbers}"
            elif _INTERACTION.search(sent) and not any(
                chunks[n - 1]["section"] in _INTERACTION_SECTIONS
                and _INTERACTION.search(chunks[n - 1]["text"])
                for n in valid
            ):
                reason = "interaction claim without an interaction statement in the label"
        if reason:
            dropped.append({"sentence": sent, "reason": reason})
            continue
        kept.append(sent)
        cited += [n for n in valid if n not in cited]
    return Verified(" ".join(kept).strip(), cited, dropped)
