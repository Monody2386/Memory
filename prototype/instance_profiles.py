from typing import Optional, Tuple

PRONOUN_LIST = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those",
}

DEFAULT_ENTITY_KIND_BY_NOUN = {
    # people
    "man": "person", "boy": "person", "father": "person", "son": "person",
    "brother": "person", "uncle": "person", "king": "person", "prince": "person",
    "husband": "person", "actor": "person", "waiter": "person", "gentleman": "person",
    "woman": "person", "girl": "person", "mother": "person", "daughter": "person",
    "sister": "person", "aunt": "person", "queen": "person", "princess": "person",
    "wife": "person", "actress": "person", "waitress": "person", "lady": "person",
    "person": "person", "people": "person", "child": "person", "children": "person",
    "student": "person", "teacher": "person", "doctor": "person", "nurse": "person",
    "farmer": "person", "driver": "person", "friend": "person", "tom": "person",
    "mary": "person", "john": "person", "anna": "person",
    # objects / animals / things
    "apple": "object", "fruit": "object", "banana": "object", "orange": "object",
    "galaxy": "object", "tree": "object", "flower": "object", "book": "object",
    "car": "object", "bus": "object", "bike": "object", "chair": "object",
    "table": "object", "bed": "object", "cup": "object", "bottle": "object",
    "box": "object", "toy": "object", "ball": "object", "food": "object",
    "water": "object", "soup": "object", "house": "object", "room": "object",
    "school": "object", "city": "object", "dog": "object", "cat": "object",
    "bird": "object", "fish": "object", "horse": "object", "cow": "object",
}

DEFAULT_GENDER_BY_NOUN = {
    "man": "male", "boy": "male", "father": "male", "son": "male",
    "brother": "male", "uncle": "male", "king": "male", "prince": "male",
    "husband": "male", "actor": "male", "waiter": "male", "gentleman": "male",
    "tom": "male", "john": "male",
    "woman": "female", "girl": "female", "mother": "female", "daughter": "female",
    "sister": "female", "aunt": "female", "queen": "female", "princess": "female",
    "wife": "female", "actress": "female", "waitress": "female", "lady": "female",
    "mary": "female", "anna": "female",
}


def default_entity_kind(noun_text: Optional[str]) -> str:
    noun_key = None if noun_text is None else noun_text.lower()
    return DEFAULT_ENTITY_KIND_BY_NOUN.get(noun_key, "unknown")


def default_gender(noun_text: Optional[str]) -> str:
    noun_key = None if noun_text is None else noun_text.lower()
    return DEFAULT_GENDER_BY_NOUN.get(noun_key, "unknown")


def pronoun_filters(pronoun: str) -> Tuple[Optional[str], Optional[str]]:
    pronoun_key = pronoun.lower()
    if pronoun_key in {"he", "him", "his"}:
        return "person", "male"
    if pronoun_key in {"she", "her", "hers"}:
        return "person", "female"
    if pronoun_key in {"it", "its"}:
        return "object", None
    return None, None
