"""Grammar utilities organized as layered stages:

LAYER 1. TOKENIZER
LAYER 2. PART-OF-SPEECH ANALYSIS
LAYER 3. REDUCTION RULES
LAYER 4. INFORMATION EXTRACTION BY SENTENCE PATTERN
LAYER 5. INSTANCE DECISION RULES
LAYER 6. TIME-STEP RULES

At the current stage the parser is intentionally rule-based. The grammar core is
not responsible for learning arbitrary sentence semantics. Instead, each known
sentence structure gets an explicit extraction strategy so the project can grow
pattern by pattern over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from world.action_vocab import ensure_action
from prototype.instance_metadata import ARTICLE_LIST, PRONOUN_LIST, POSSESSIVE_LIST, POSSESSIVE_NOUN_LIST, possessive_owner_role, pronoun_filters, is_named_person_noun
from .grammar_routes import DEFAULT_EXTRACTOR_ROUTES


@dataclass
class TaggedToken:
    text: str
    pos: str
    index: int
    source: str = "lexicon"
    question_prompt: Optional[str] = None
    instance_id: Optional[str] = None
    entity_text: Optional[str] = None
    owner_instance_id: Optional[str] = None
    owner_role: Optional[str] = None


@dataclass
class ActionTuple:
    noun: str
    action: str
    role: str
    position: int
    noun_instance_id: Optional[str] = None
    owner_instance_id: Optional[str] = None
    owner_role: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)
    polarity: int = 1
    accept_label: str = "none"
    question_label: str = "none"


@dataclass
class RelationTuple:
    source: str
    relation: str
    target: str
    kind: str
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    owner_instance_id: Optional[str] = None
    owner_role: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)
    polarity: int = 1
    accept_label: str = "none"
    question_label: str = "none"


@dataclass
class RewardTuple:
    subject: str
    reward_word: str
    reward_value: float
    action: Optional[str] = None
    object: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)
    polarity: int = 1
    accept_label: str = "none"
    question_label: str = "none"


@dataclass
class SurpriseTuple:
    subject: str
    surprise_word: str
    surprise_value: float
    action: Optional[str] = None
    object: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)
    polarity: int = 1
    accept_label: str = "none"
    question_label: str = "none"


@dataclass
class InstanceAttributeUpdate:
    instance_id: str
    attribute_name: str
    attribute_value: str
    noun_text: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)
    polarity: int = 1
    question_label: str = "none"


@dataclass
class QuestionInfo:
    is_question: bool = False
    marker: Optional[str] = None
    question_type: Optional[str] = None
    original_structure: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ParsedSentence:
    sentence: str
    tokens: List[str]
    tagged_tokens: List[TaggedToken]
    structure: Tuple[str, ...]
    pattern_name: str
    sentence_type: str = "unknown_sentence"
    action_tuples: List[ActionTuple] = field(default_factory=list)
    relation_tuples: List[RelationTuple] = field(default_factory=list)
    reward_tuples: List[RewardTuple] = field(default_factory=list)
    surprise_tuples: List[SurpriseTuple] = field(default_factory=list)
    instance_updates: List[InstanceAttributeUpdate] = field(default_factory=list)
    question_info: QuestionInfo = field(default_factory=QuestionInfo)


@dataclass
class NounActionPair:
    noun: str
    action: str
    role: str
    position: int
    noun_instance_id: Optional[str] = None
    noun_index: Optional[int] = None
    noun_embedding: Optional[torch.Tensor] = None
    adjectives: Optional[List[str]] = None


@dataclass
class ShortMemoryState:
    noun: str
    action: str
    noun_instance_id: Optional[str]
    noun_index: Optional[int]
    action_type: int
    noun_embedding: torch.Tensor
    action_embedding: torch.Tensor
    time_position: int
    pair_index: int
    role: str
    adjectives: List[str]
    pair_kind: str = "noun_action"
    polarity: int = 1


@dataclass
class ShortMemoryRelationState:
    source: str
    relation: str
    target: str
    relation_kind: str
    source_instance_id: Optional[str]
    target_instance_id: Optional[str]
    source_index: Optional[int]
    target_index: Optional[int]
    source_embedding: Optional[torch.Tensor]
    target_embedding: Optional[torch.Tensor]
    time_position: int
    pair_index: int
    polarity: int = 1
    accept_label: str = "none"
    question_label: str = "none"


@dataclass
class NounNounRelationSample:
    source_noun: str
    target_noun: str
    relation_type: int
    source_idx: Optional[int] = None
    target_idx: Optional[int] = None


@dataclass
class AdjNounRelationSample:
    noun: str
    adjective: str
    relation_type: int
    noun_idx: Optional[int] = None
    adjective_idx: Optional[int] = None


@dataclass
class KnowledgeTrainingSamples:
    noun_noun_samples: List[NounNounRelationSample] = field(default_factory=list)
    adj_noun_samples: List[AdjNounRelationSample] = field(default_factory=list)


ACTION_FORM_PAIRS = [
    ("eat", "eaten"),
    ("see", "seen"),
    ("write", "written"),
    ("take", "taken"),
    ("drive", "driven"),
]
ACTIVE_TO_ACTIONED = dict(ACTION_FORM_PAIRS)
ACTIONED_TO_ACTIVE = {actioned: active for active, actioned in ACTION_FORM_PAIRS}
# Backward-compatible name: object-side events use the actioned/passive form.
IRREGULAR_OBJECT_ACTIONS = ACTIVE_TO_ACTIONED

ADJECTIVE_RELATION_HINTS = {
    "red": "color",
    "blue": "color",
    "green": "color",
    "round": "shape",
    "square": "shape",
    "sweet": "taste",
    "sour": "taste",
    "large": "size",
    "small": "size",
    "smooth": "texture",
    "rough": "texture",
    "warm": "temperature",
    "cold": "temperature",
}

WORD_RE = re.compile(r"[A-Za-z']+")
RelationOverride = Union[int, str]
STANDALONE_POSSESSIVE_EXPANSIONS = {
    "mine": "my",
    "yours": "your",
    "hers": "her",
    "ours": "our",
    "theirs": "their",
}
BE_VERB_SET = {"am", "is", "are", "was", "were", "be", "been", "being"}
HELPER_WORD_SET = {"by", "do", "does", "did", "can", "could"}
QUESTION_HELPER_WORD_SET = {"do", "does", "did", "can", "could"}
DO_QUESTION_HELPER_WORD_SET = {"do", "does", "did"}
ABILITY_HELPER_WORD_SET = {"can", "could"}
NEGATIVE_WORD_SET = {"not", "no", "never", "n't"}
REWARD_WORD_VALUE_MAP = {
    "hate": -50.0,
    "dislike": -30.0,
    "like": 30.0,
    "love": 50.0,
}
SURPRISE_WORD_VALUE_MAP = {
    "can": -50.0,
    "could": -50.0,
}
NEGATED_REWARD_SCALE = 0.5
NEGATED_SURPRISE_SCALE = 0.5


def negate_reward_value(value: float) -> float:
    return -float(value) * NEGATED_REWARD_SCALE


def negate_surprise_value(value: float) -> float:
    return -float(value) * NEGATED_SURPRISE_SCALE


# ===========================================================================
# LAYER 1. TOKENIZER
# ===========================================================================

def split_event(event_tokens):
    if len(event_tokens) != 5:
        raise ValueError("event_tokens must contain exactly 5 items")
    return tuple(event_tokens)


def tokenize_sentence(
    sentence: str,
    instance_context=None,
    short_memory=None,
) -> List[str]:
    del instance_context, short_memory
    tokens = WORD_RE.findall(sentence)
    if not tokens:
        raise ValueError("sentence does not contain any word tokens")
    return [token.lower() for token in tokens]


def object_action_form(verb: str) -> str:
    """Return the actioned form used by object-side event pairs."""
    verb = verb.lower()
    if verb in ACTIVE_TO_ACTIONED:
        return ACTIVE_TO_ACTIONED[verb]
    # Productive default for new actions. Irregular forms can be added to ACTION_FORM_PAIRS.
    return verb + "ed"


def subject_action_form(actioned: str) -> str:
    """Return the active action form that corresponds to an actioned/passive token."""
    actioned = actioned.lower()
    if actioned in ACTIONED_TO_ACTIVE:
        return ACTIONED_TO_ACTIVE[actioned]
    if actioned.endswith("ed") and len(actioned) > 2:
        return actioned[:-2]
    return actioned


# ---------------------------------------------------------------------------
# Shared language context
# ---------------------------------------------------------------------------

def _load_language_context():
    rm = importlib.import_module("knowledge.relation_map")
    arm = importlib.import_module("knowledge.adj_relation_map")
    kt = importlib.import_module("knowledge.training")

    rm.load_relation_data()
    arm.load_adj_relation_data()

    if os.path.exists(kt.MODEL_PATH):
        kt.knowledge_map_one.load_state_dict(torch.load(kt.MODEL_PATH, map_location="cpu"))
    if os.path.exists(kt.ADJ_MODEL_PATH):
        kt.adj_map_one.load_state_dict(
            torch.load(kt.ADJ_MODEL_PATH, map_location="cpu"), strict=False
        )

    return rm, arm, kt


def _ensure_noun_embedding(noun: str):
    rm, _, kt = _load_language_context()
    noun_key = noun.lower()
    noun_idx = rm._ensure_noun(noun_key)
    noun_embedding = kt.knowledge_map_one.embedding.weight.detach()[noun_idx].clone()
    return noun_idx, noun_embedding


def _normalize_relation_override(
    relation_override: Optional[RelationOverride],
    relation_names: Sequence[str],
    *,
    relation_label: str,
) -> Optional[int]:
    if relation_override is None:
        return None
    if isinstance(relation_override, str):
        if relation_override not in relation_names:
            raise ValueError(f"Unknown {relation_label}: {relation_override}")
        return relation_names.index(relation_override) + 1
    return int(relation_override)


def _resolve_adjective_relation_overrides(
    adjectives: Sequence[str],
    overrides: Optional[Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]],
    relation_names: Sequence[str],
) -> List[Optional[int]]:
    if overrides is None:
        return [None for _ in adjectives]

    if isinstance(overrides, Mapping):
        return [
            _normalize_relation_override(
                overrides.get(adjective.lower()),
                relation_names,
                relation_label="adjective relation",
            )
            for adjective in adjectives
        ]

    if len(overrides) != len(adjectives):
        raise ValueError("adjective_relation_types must match the number of adjectives")
    return [
        _normalize_relation_override(value, relation_names, relation_label="adjective relation")
        for value in overrides
    ]


def infer_adj_relation_type(noun: str, adjective: str) -> Optional[int]:
    rm, arm, _ = _load_language_context()
    noun = noun.lower()
    adjective = adjective.lower()

    if noun in rm.noun_list and adjective in arm.adjective_list:
        noun_idx = rm.noun_list.index(noun)
        adjective_idx = arm.adjective_list.index(adjective)
        relation_type = int(arm.adj_relation_map[noun_idx, adjective_idx])
        if relation_type > 0:
            return relation_type

    hinted_relation = ADJECTIVE_RELATION_HINTS.get(adjective)
    if hinted_relation and hinted_relation in arm.adj_relation_list:
        return arm.adj_relation_list.index(hinted_relation) + 1

    return None


# ===========================================================================
# LAYER 2. PART-OF-SPEECH ANALYSIS
# ===========================================================================

def _build_pos_lexicons():
    rm, arm, _ = _load_language_context()
    action_vocab = importlib.import_module("world.action_vocab")
    relation_tokens = set()
    for relation in rm.relation_list:
        relation_tokens.update(relation.lower().split())
    return {
        "nouns": {noun.lower() for noun in rm.noun_list},
        "adjectives": set(arm.adjective_list) | set(ADJECTIVE_RELATION_HINTS.keys()),
        "pronouns": set(PRONOUN_LIST),
        "possessives": set(POSSESSIVE_LIST),
        "possessive_nouns": set(POSSESSIVE_NOUN_LIST),
        "articles": set(ARTICLE_LIST),
        "helpers": set(HELPER_WORD_SET),
        "negative_words": set(NEGATIVE_WORD_SET),
        "actions": {action.lower() for action in action_vocab.action_list},
        "actioned_words": {object_action_form(action.lower()) for action in action_vocab.action_list}
        | set(ACTIONED_TO_ACTIVE.keys()),
        "be_verbs": set(BE_VERB_SET),
        "reward_words": set(REWARD_WORD_VALUE_MAP),
        "relations": [relation.lower() for relation in rm.relation_list],
        "relation_tokens": relation_tokens,
    }


def _resolve_relation_phrase(tokens: Sequence[str], relation_list: Sequence[str]) -> Optional[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            phrase = " ".join(tokens[start:end]).lower()
            if phrase in relation_list:
                candidates.append((phrase, start, end))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2] - item[1], reverse=True)
    return candidates[0]


def _coerce_instance_context(instance_context=None, short_memory=None):
    if instance_context is not None:
        return instance_context
    if short_memory is None:
        return None
    return build_instance_context_from_memory(short_memory)



def tag_tokens(
    sentence: str,
    instance_context=None,
    short_memory=None,
) -> List[TaggedToken]:
    tokens = tokenize_sentence(
        sentence,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    lexicons = _build_pos_lexicons()
    relation_match = _resolve_relation_phrase(tokens, lexicons["relations"])
    relation_span = set(range(relation_match[1], relation_match[2])) if relation_match else set()
    question_engine = importlib.import_module("prototype.question").QuestionEngine()

    tagged_tokens: List[TaggedToken] = []
    for index, token in enumerate(tokens):
        if index in relation_span:
            tagged_tokens.append(TaggedToken(text=token, pos="relation", index=index, source="relation_phrase"))
            continue
        if token in lexicons["possessive_nouns"]:
            tagged_tokens.append(TaggedToken(text=token, pos="possessive_noun", index=index, source="possessive_noun_list", entity_text=token))
            continue
        if token in lexicons["possessives"]:
            tagged_tokens.append(TaggedToken(text=token, pos="possessive", index=index, source="possessive_list", entity_text=token))
            continue
        if token in lexicons["articles"]:
            tagged_tokens.append(TaggedToken(text=token, pos="article", index=index, source="article_list", entity_text=token))
            continue
        if token in lexicons["be_verbs"]:
            tagged_tokens.append(TaggedToken(text=token, pos="be", index=index, source="be_verb_list", entity_text=token))
            continue
        if token in lexicons["helpers"]:
            tagged_tokens.append(TaggedToken(text=token, pos="helper", index=index, source="helper_word_list", entity_text=token))
            continue
        if token in lexicons["negative_words"]:
            tagged_tokens.append(TaggedToken(text=token, pos="negative", index=index, source="negative_word_list", entity_text=token))
            continue
        if token in lexicons["pronouns"]:
            tagged_tokens.append(TaggedToken(text=token, pos="pronoun", index=index, source="pronoun_list", entity_text=token))
            continue
        if token in lexicons["reward_words"]:
            tagged_tokens.append(TaggedToken(text=token, pos="reward", index=index, source="reward_word_list", entity_text=token))
            continue
        if token in lexicons["actioned_words"]:
            tagged_tokens.append(TaggedToken(text=token, pos="actioned", index=index, source="actioned_form_list", entity_text=token))
            continue
        if token in lexicons["actions"]:
            tagged_tokens.append(TaggedToken(text=token, pos="action", index=index, source="action_list", entity_text=token))
            continue
        if is_named_person_noun(token):
            tagged_tokens.append(TaggedToken(text=token, pos="noun", index=index, source="named_person_list", entity_text=token))
            continue

        token_info = question_engine.what_is_token(token, position=index, tokens=tokens)
        pos = token_info.predicted_pos if token_info.predicted_pos != "unknown" else "noun"
        tagged_tokens.append(
            TaggedToken(
                text=token,
                pos=pos,
                index=index,
                source=token_info.source,
                question_prompt=None if token_info.status == "known" else token_info.prompt,
                entity_text=token,
            )
        )

    return tagged_tokens


def sentence_structure(
    sentence: str,
    instance_context=None,
    short_memory=None,
) -> Tuple[str, ...]:
    return tuple(
        token.pos for token in tag_tokens(
            sentence,
            instance_context=instance_context,
            short_memory=short_memory,
        )
    )


def build_instance_context_from_memory(short_memory) -> Dict[str, object]:
    context: Dict[str, object] = {
        "focus_instance_id": None,
        "focus_noun_text": None,
        "by_noun": {},
        "recent_instances": [],
        "by_instance": {},
    }
    if short_memory is None:
        return context

    focus_entry = short_memory.get_focus_entry() if hasattr(short_memory, "get_focus_entry") else None
    if focus_entry is not None:
        context["focus_instance_id"] = focus_entry.noun_instance_id
        context["focus_noun_text"] = None if focus_entry.noun_text is None else focus_entry.noun_text.lower()

    def register(noun_text: Optional[str], instance_id: Optional[str], time_position: int, pair_index: int, score: float = 0.0):
        if noun_text is None or instance_id is None:
            return
        noun_key = noun_text.lower()
        bucket = context["by_noun"].setdefault(noun_key, [])
        for existing_time, existing_pair, existing_instance_id in bucket:
            if existing_instance_id == instance_id:
                break
        else:
            bucket.append((int(time_position), int(pair_index), instance_id))

        metadata = None
        if hasattr(short_memory, "get_noun_instance_metadata"):
            metadata = short_memory.get_noun_instance_metadata(instance_id)
        current = context["by_instance"].get(instance_id)
        candidate = {
            "instance_id": instance_id,
            "noun_text": noun_key,
            "time_position": int(time_position),
            "pair_index": int(pair_index),
            "score": float(score),
            "entity_kind": "unknown" if metadata is None else metadata.get("entity_kind", "unknown"),
            "gender": "unknown" if metadata is None else metadata.get("gender", "unknown"),
            "owner_instance_id": None if metadata is None else metadata.get("owner_instance_id"),
            "owner_role": None if metadata is None else metadata.get("owner_role"),
            "instance_scope": "scene" if metadata is None else metadata.get("instance_scope", "scene"),
        }
        if current is None or (candidate["time_position"], candidate["pair_index"]) >= (current["time_position"], current["pair_index"]):
            context["by_instance"][instance_id] = candidate

    for entry in getattr(short_memory, "short_memory_event", []):
        register(entry.noun_text, entry.noun_instance_id, entry.time_position, entry.pair_index, getattr(entry, "score", 0.0))

    for entry in getattr(short_memory, "short_memory_relation", []):
        register(entry.source_text, entry.source_instance_id, entry.time_position, entry.pair_index, getattr(entry, "score", 0.0))
        register(entry.target_text, entry.target_instance_id, entry.time_position, entry.pair_index, getattr(entry, "score", 0.0))

    for instance_id, metadata in getattr(short_memory, "noun_instance_metadata", {}).items():
        if instance_id in context["by_instance"]:
            continue
        noun_text = metadata.get("noun_text")
        if noun_text is None:
            continue
        register(noun_text, instance_id, -1, -1, 0.0)

    for noun_key, bucket in context["by_noun"].items():
        bucket.sort(key=lambda item: (item[0], item[1]))
        context["by_noun"][noun_key] = [instance_id for _, _, instance_id in bucket]

    context["recent_instances"] = sorted(
        context["by_instance"].values(),
        key=lambda item: (item["time_position"], item["pair_index"], item["score"]),
    )

    return context


def _resolve_existing_instance_id(
    noun_text: str,
    instance_context,
    sentence_assignments: Dict[str, str],
) -> Optional[str]:
    noun_key = noun_text.lower()
    if noun_key in sentence_assignments:
        return sentence_assignments[noun_key]
    if not instance_context:
        return None

    focus_noun_text = instance_context.get("focus_noun_text")
    focus_instance_id = instance_context.get("focus_instance_id")
    if focus_noun_text == noun_key and focus_instance_id is not None:
        return focus_instance_id

    candidates = instance_context.get("by_noun", {}).get(noun_key, [])
    if candidates:
        return candidates[-1]
    return None


def _select_instance_record_for_pronoun(pronoun: str, instance_context) -> Optional[Dict[str, object]]:
    if not instance_context:
        return None

    candidates = list(instance_context.get("recent_instances", []))
    if not candidates:
        return None

    required_kind, preferred_gender = pronoun_filters(pronoun)
    if required_kind is not None:
        filtered = [item for item in candidates if item.get("entity_kind", "unknown") in {required_kind, "unknown"}]
        if filtered:
            candidates = filtered
    if preferred_gender is not None:
        filtered = [item for item in candidates if item.get("gender", "unknown") in {preferred_gender, "unknown"}]
        if filtered:
            candidates = filtered

    focus_instance_id = instance_context.get("focus_instance_id")
    pronoun_key = pronoun.lower()
    demonstrative_pronouns = {"it", "this", "that", "these", "those"}
    personal_pronouns = {"he", "she", "him", "her", "they", "them", "we", "us", "i", "me", "you"}

    best_item = None
    best_score = None
    for item in candidates:
        score = 0.0
        if item["instance_id"] == focus_instance_id:
            score += 100.0
        if pronoun_key in demonstrative_pronouns and item["instance_id"] == focus_instance_id:
            score += 25.0
        if pronoun_key in personal_pronouns:
            score += 5.0
        if required_kind is not None and item.get("entity_kind") == required_kind:
            score += 20.0
        if preferred_gender is not None and item.get("gender") == preferred_gender:
            score += 20.0
        score += float(item.get("score", 0.0))
        score += float(item.get("time_position", 0)) * 10.0
        score += float(item.get("pair_index", 0))
        if best_score is None or score > best_score:
            best_score = score
            best_item = item

    return best_item


def _select_instance_for_pronoun(pronoun: str, instance_context) -> Optional[str]:
    item = _select_instance_record_for_pronoun(pronoun, instance_context)
    return None if item is None else str(item["instance_id"])


def _resolve_existing_possessive_instance_id(
    noun_text: str,
    owner_instance_id: Optional[str],
    owner_role: Optional[str],
    instance_context,
) -> Optional[str]:
    if not instance_context:
        return None

    candidates = instance_context.get("by_noun", {}).get(noun_text.lower(), [])
    by_instance = instance_context.get("by_instance", {})
    for candidate_instance_id in reversed(candidates):
        metadata = by_instance.get(candidate_instance_id, {})
        if metadata.get("owner_instance_id") == owner_instance_id and metadata.get("owner_role") == owner_role:
            return candidate_instance_id
    return None


def classify_sentence_type_from_parsed(parsed: ParsedSentence) -> str:
    if getattr(parsed.question_info, "is_question", False):
        return "question_sentence"

    has_action = bool(parsed.action_tuples)
    has_noun_relation = any(
        relation_tuple.kind == "noun_noun_relation" for relation_tuple in parsed.relation_tuples
    )
    has_adj_relation = any(
        relation_tuple.kind == "adj_noun_relation" for relation_tuple in parsed.relation_tuples
    )
    has_instance_update = bool(parsed.instance_updates)

    if has_action and has_noun_relation:
        return "mixed_sentence"
    if has_action:
        return "action_sentence"
    if has_noun_relation:
        return "relation_sentence"
    if has_adj_relation:
        return "attribute_sentence"
    if has_instance_update:
        return "instance_update_sentence"
    if parsed.reward_tuples:
        return "reward_sentence"
    if parsed.surprise_tuples:
        return "surprise_sentence"
    return "unknown_sentence"


def classify_sentence_type(sentence: str) -> str:
    parsed = parse_sentence(sentence)
    return parsed.sentence_type


# ===========================================================================
# LAYER 3. REDUCTION RULES
# ===========================================================================


def _token_entity_text(token: str, tag: TaggedToken) -> str:
    return str(tag.entity_text or token)


def _resolve_adj_relation_name(
    noun: str,
    adjective: str,
    relation_override: Optional[RelationOverride],
    *,
    infer_missing: bool,
    relation_names: Sequence[str],
) -> Optional[str]:
    if relation_override is not None:
        relation_type = _normalize_relation_override(
            relation_override,
            relation_names,
            relation_label="adjective relation",
        )
        if relation_type is None:
            return None
        return relation_names[int(relation_type) - 1]

    if infer_missing:
        inferred_type = infer_adj_relation_type(noun, adjective)
        if inferred_type is not None:
            return relation_names[int(inferred_type) - 1]
    return None


def _resolve_owner_for_possessive(word: str, instance_context) -> Tuple[Optional[str], Optional[str]]:
    role = possessive_owner_role(word)
    if role is None:
        return None, None
    lookup_word = word.lower()
    if lookup_word == "his":
        instance_id = _select_instance_for_pronoun("he", instance_context)
        return (instance_id or "male_owner#core", role)
    if lookup_word in {"her", "hers"}:
        instance_id = _select_instance_for_pronoun("she", instance_context)
        return (instance_id or "female_owner#core", role)
    if lookup_word in {"their", "theirs"}:
        instance_id = _select_instance_for_pronoun("they", instance_context)
        return (instance_id or "group_owner#core", role)
    if lookup_word == "its":
        instance_id = _select_instance_for_pronoun("it", instance_context)
        return (instance_id or "object_owner#core", role)
    if lookup_word in {"my", "mine"}:
        return ("speaker#core", role)
    if lookup_word in {"your", "yours"}:
        return ("listener#core", role)
    if lookup_word in {"our", "ours"}:
        return ("speaker_group#core", role)
    return None, role


def _expand_possessive_noun_tokens(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    instance_context=None,
    short_memory=None,
) -> Tuple[List[str], List[TaggedToken]]:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    focus_noun_text = None if not resolved_context else resolved_context.get("focus_noun_text")
    if not focus_noun_text:
        return list(tokens), list(tagged_tokens)

    expanded_tokens: List[str] = []
    expanded_tags: List[TaggedToken] = []
    for tag in tagged_tokens:
        token_text = tag.text
        if tag.pos == "possessive_noun":
            replacement = STANDALONE_POSSESSIVE_EXPANSIONS.get(token_text.lower())
            if replacement is not None:
                expanded_tokens.append(replacement)
                expanded_tags.append(
                    TaggedToken(
                        text=replacement,
                        pos="possessive",
                        index=tag.index,
                        source="expansion_possessive_noun",
                        question_prompt=tag.question_prompt,
                        entity_text=replacement,
                    )
                )
                expanded_tokens.append(str(focus_noun_text))
                expanded_tags.append(
                    TaggedToken(
                        text=str(focus_noun_text),
                        pos="noun",
                        index=tag.index,
                        source="expansion_possessive_noun_focus",
                        entity_text=str(focus_noun_text),
                    )
                )
                continue
        expanded_tokens.append(token_text)
        expanded_tags.append(tag)
    return expanded_tokens, expanded_tags


def _reduce_possessive_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    instance_context=None,
    short_memory=None,
) -> Tuple[List[str], List[TaggedToken]]:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens: List[str] = []
    reduced_tags: List[TaggedToken] = []
    index = 0
    create_counters: Dict[str, int] = {}

    while index < len(tagged_tokens):
        if (
            index + 1 < len(tagged_tokens)
            and tagged_tokens[index].pos == "possessive"
            and tagged_tokens[index + 1].pos == "noun"
        ):
            possessive_word = tokens[index]
            head_tag = tagged_tokens[index + 1]
            head_noun = _token_entity_text(tokens[index + 1], head_tag)
            owner_instance_id, owner_role = _resolve_owner_for_possessive(possessive_word, resolved_context)
            noun_instance_id = head_tag.instance_id
            if noun_instance_id is None:
                noun_instance_id = _resolve_existing_possessive_instance_id(
                    head_noun,
                    owner_instance_id,
                    owner_role,
                    resolved_context,
                )
            if noun_instance_id is None:
                noun_instance_id = _build_instance_id(
                    head_noun,
                    create_counters,
                    instance_context=resolved_context,
                )
            reduced_tokens.append(noun_instance_id)
            reduced_tags.append(
                TaggedToken(
                    text=noun_instance_id,
                    pos="noun",
                    index=head_tag.index,
                    source="reduction_possessive_noun",
                    question_prompt=None,
                    instance_id=noun_instance_id,
                    entity_text=head_noun,
                    owner_instance_id=owner_instance_id,
                    owner_role=owner_role,
                )
            )
            index += 2
            continue

        reduced_tokens.append(tokens[index])
        reduced_tags.append(tagged_tokens[index])
        index += 1

    return reduced_tokens, reduced_tags


def _reduce_pronoun_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    instance_context=None,
    short_memory=None,
) -> Tuple[List[str], List[TaggedToken]]:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens: List[str] = []
    reduced_tags: List[TaggedToken] = []
    index = 0
    create_counters: Dict[str, int] = {}

    while index < len(tagged_tokens):
        if (
            index + 1 < len(tagged_tokens)
            and tagged_tokens[index].pos == "pronoun"
            and tagged_tokens[index + 1].pos in {"noun", "pronoun"}
        ):
            head_tag = tagged_tokens[index + 1]
            head_token = tokens[index + 1]
            noun_instance_id = head_tag.instance_id
            reduced_text = head_token
            reduced_pos = "noun" if head_tag.pos == "noun" else head_tag.pos

            if head_tag.pos == "noun":
                noun_instance_id = _resolve_existing_instance_id(
                    head_token,
                    resolved_context,
                    {},
                ) or noun_instance_id
                if noun_instance_id is None:
                    noun_instance_id = _build_instance_id(head_token, create_counters, instance_context=resolved_context)
            else:
                selected_item = _select_instance_record_for_pronoun(head_token, resolved_context)
                if selected_item is not None:
                    noun_instance_id = str(selected_item["instance_id"])
                    reduced_text = str(selected_item.get("noun_text") or head_token)
                    reduced_pos = "noun"
                elif noun_instance_id is None:
                    noun_instance_id = _build_instance_id(head_token, create_counters, instance_context=resolved_context)

            reduced_surface = noun_instance_id if noun_instance_id is not None else reduced_text
            reduced_tokens.append(reduced_surface)
            reduced_tags.append(
                TaggedToken(
                    text=reduced_surface,
                    pos=reduced_pos,
                    index=head_tag.index,
                    source="reduction_pronoun_noun",
                    question_prompt=None,
                    instance_id=noun_instance_id,
                    entity_text=reduced_text,
                )
            )
            index += 2
            continue

        current_tag = tagged_tokens[index]
        current_token = tokens[index]
        if current_tag.pos == "pronoun":
            selected_item = _select_instance_record_for_pronoun(current_tag.text, resolved_context)
            if selected_item is not None:
                current_token = str(selected_item.get("noun_text") or current_tag.text)
                current_tag = TaggedToken(
                    text=str(selected_item["instance_id"]),
                    pos="noun",
                    index=current_tag.index,
                    source="reduction_pronoun_instance",
                    question_prompt=current_tag.question_prompt,
                    instance_id=str(selected_item["instance_id"]),
                    entity_text=current_token,
                )
                current_token = current_tag.text
            elif current_tag.instance_id is None:
                new_instance_id = _build_instance_id(current_tag.text, create_counters, instance_context=resolved_context)
                current_tag = TaggedToken(
                    text=new_instance_id,
                    pos="noun",
                    index=current_tag.index,
                    source="reduction_pronoun_instance",
                    question_prompt=current_tag.question_prompt,
                    instance_id=new_instance_id,
                    entity_text=current_tag.text,
                )
                current_token = current_tag.text

        reduced_tokens.append(current_token)
        reduced_tags.append(current_tag)
        index += 1

    return reduced_tokens, reduced_tags


def _reduce_adj_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
) -> Tuple[List[str], List[TaggedToken], List[RelationTuple]]:
    _, arm, _ = _load_language_context()
    reduced_tokens: List[str] = []
    reduced_tags: List[TaggedToken] = []
    extracted_relations: List[RelationTuple] = []
    index = 0

    while index < len(tagged_tokens):
        if (
            index + 1 < len(tagged_tokens)
            and tagged_tokens[index].pos == "adj"
            and tagged_tokens[index + 1].pos == "noun"
        ):
            phrase_tokens: List[str] = []
            phrase_tags: List[TaggedToken] = []
            while index < len(tagged_tokens) and tagged_tokens[index].pos == "adj":
                phrase_tokens.append(tokens[index])
                phrase_tags.append(tagged_tokens[index])
                index += 1
            if index < len(tagged_tokens) and tagged_tokens[index].pos == "noun":
                phrase_tokens.append(tokens[index])
                phrase_tags.append(tagged_tokens[index])
                noun, adjectives, noun_instance_id, owner_instance_id, owner_role = _parse_noun_phrase(phrase_tokens, phrase_tags)
                relation_overrides = _resolve_adjective_relation_overrides(
                    adjectives,
                    adjective_relation_types,
                    arm.adj_relation_list,
                )
                for adjective, relation_override in zip(adjectives, relation_overrides):
                    relation_name = _resolve_adj_relation_name(
                        noun,
                        adjective,
                        relation_override,
                        infer_missing=infer_missing,
                        relation_names=arm.adj_relation_list,
                    )
                    if relation_name is None:
                        continue
                    extracted_relations.append(
                        RelationTuple(
                            source=noun,
                            relation=relation_name,
                            target=adjective,
                            kind="adj_noun_relation",
                            source_instance_id=noun_instance_id,
                            target_instance_id=None,
                            owner_instance_id=owner_instance_id,
                            owner_role=owner_role,
                            source_tokens=list(phrase_tokens),
                        )
                    )
                reduced_tokens.append(noun)
                reduced_tags.append(
                    TaggedToken(
                        text=noun,
                        pos="noun",
                        index=phrase_tags[-1].index,
                        source="reduction_adj_noun",
                        question_prompt=None,
                        instance_id=noun_instance_id,
                    )
                )
                index += 1
                continue

            reduced_tokens.extend(phrase_tokens)
            reduced_tags.extend(phrase_tags)
            continue

        reduced_tokens.append(tokens[index])
        reduced_tags.append(tagged_tokens[index])
        index += 1

    return reduced_tokens, reduced_tags, extracted_relations


def _reduce_article_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    instance_context=None,
    short_memory=None,
) -> Tuple[List[str], List[TaggedToken]]:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens: List[str] = []
    reduced_tags: List[TaggedToken] = []
    index = 0
    create_counters: Dict[str, int] = {}

    while index < len(tagged_tokens):
        if (
            index + 1 < len(tagged_tokens)
            and tagged_tokens[index].pos == "article"
            and tagged_tokens[index + 1].pos == "noun"
        ):
            article_text = tokens[index].lower()
            head_tag = tagged_tokens[index + 1]
            head_noun = _token_entity_text(tokens[index + 1], head_tag)
            noun_instance_id = head_tag.instance_id
            if noun_instance_id is None:
                if article_text == "the":
                    noun_instance_id = _resolve_existing_instance_id(head_noun, resolved_context, {})
                elif article_text in {"a", "an"}:
                    noun_instance_id = _build_instance_id(
                        head_noun,
                        create_counters,
                        instance_context=resolved_context,
                    )
                else:
                    noun_instance_id = _resolve_existing_instance_id(head_noun, resolved_context, {})
            if noun_instance_id is None:
                noun_instance_id = _build_instance_id(head_noun, create_counters, instance_context=resolved_context)
            reduced_tokens.append(noun_instance_id)
            reduced_tags.append(
                TaggedToken(
                    text=noun_instance_id,
                    pos="noun",
                    index=head_tag.index,
                    source="reduction_article_noun",
                    question_prompt=None,
                    instance_id=noun_instance_id,
                    entity_text=head_noun,
                    owner_instance_id=head_tag.owner_instance_id,
                    owner_role=head_tag.owner_role,
                )
            )
            index += 2
            continue

        reduced_tokens.append(tokens[index])
        reduced_tags.append(tagged_tokens[index])
        index += 1

    return reduced_tokens, reduced_tags


def _detect_and_reorder_question_helper(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
) -> Tuple[List[str], List[TaggedToken], QuestionInfo]:
    """Move leading question markers behind the following noun before helper deletion."""
    if not tokens or not tagged_tokens:
        return list(tokens), list(tagged_tokens), QuestionInfo()

    first_tag = tagged_tokens[0]
    first_token = tokens[0]
    first_token_key = first_token.lower()
    is_helper_question = first_tag.pos == "helper" and first_token_key in QUESTION_HELPER_WORD_SET
    is_be_question = first_tag.pos == "be"
    if not (is_helper_question or is_be_question):
        return list(tokens), list(tagged_tokens), QuestionInfo()

    noun_index: Optional[int] = None
    if is_be_question and len(tagged_tokens) > 1:
        noun_index = 1
    else:
        for index in range(1, len(tagged_tokens)):
            if tagged_tokens[index].pos == "noun":
                noun_index = index
                break

    question_info = QuestionInfo(
        is_question=True,
        marker=first_token_key,
        question_type="yes_no",
        original_structure=tuple(tag.pos for tag in tagged_tokens),
    )
    if noun_index is None:
        return list(tokens), list(tagged_tokens), question_info

    reordered_tokens = list(tokens)
    reordered_tags = list(tagged_tokens)
    if is_be_question and reordered_tags[noun_index].pos not in {"noun", "pronoun"}:
        subject_tag = reordered_tags[noun_index]
        reordered_tags[noun_index] = TaggedToken(
            text=subject_tag.text,
            pos="noun",
            index=subject_tag.index,
            source="be_question_subject",
            question_prompt=subject_tag.question_prompt,
            instance_id=subject_tag.instance_id,
            entity_text=subject_tag.entity_text or subject_tag.text,
            owner_instance_id=subject_tag.owner_instance_id,
            owner_role=subject_tag.owner_role,
        )
    reordered_tokens[0], reordered_tokens[noun_index] = reordered_tokens[noun_index], reordered_tokens[0]
    reordered_tags[0], reordered_tags[noun_index] = reordered_tags[noun_index], reordered_tags[0]
    return reordered_tokens, reordered_tags, question_info


def _reduce_helper_tokens(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
) -> Tuple[List[str], List[TaggedToken]]:
    """Drop grammar helper tokens after POS tagging and before extractor routing."""
    reduced_tokens: List[str] = []
    reduced_tags: List[TaggedToken] = []
    for token, tag in zip(tokens, tagged_tokens):
        if tag.pos == "helper" and token.lower() not in ABILITY_HELPER_WORD_SET:
            continue
        reduced_tokens.append(token)
        reduced_tags.append(tag)
    return reduced_tokens, reduced_tags


def _bind_existing_noun_instances(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    instance_context=None,
    short_memory=None,
) -> Tuple[List[str], List[TaggedToken]]:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    bound_tokens = list(tokens)
    bound_tags: List[TaggedToken] = []

    for token, tag in zip(tokens, tagged_tokens):
        if tag.pos != "noun":
            bound_tags.append(tag)
            continue

        noun_instance_id = tag.instance_id
        if noun_instance_id is None:
            noun_key = _token_entity_text(token, tag)
            noun_instance_id = _resolve_existing_instance_id(noun_key, resolved_context, {})
            if noun_instance_id is None and is_named_person_noun(noun_key):
                noun_instance_id = _build_instance_id(noun_key, {}, instance_context=resolved_context)
        bound_surface = noun_instance_id if noun_instance_id is not None else tag.text
        bound_tokens[len(bound_tags)] = bound_surface
        bound_tags.append(
            TaggedToken(
                text=bound_surface,
                pos=tag.pos,
                index=tag.index,
                source=tag.source,
                question_prompt=tag.question_prompt,
                instance_id=noun_instance_id,
                entity_text=tag.entity_text or token,
                owner_instance_id=tag.owner_instance_id,
                owner_role=tag.owner_role,
            )
        )

    return bound_tokens, bound_tags


# ===========================================================================
# LAYER 4. INFORMATION EXTRACTION BY SENTENCE PATTERN
# ===========================================================================


def _parse_noun_phrase(tokens: Sequence[str], tags: Sequence[TaggedToken]) -> Tuple[str, List[str], Optional[str], Optional[str], Optional[str]]:
    if not tokens:
        raise ValueError("noun phrase cannot be empty")
    head_tag = tags[-1]
    if head_tag.pos not in {"noun", "pronoun"}:
        raise ValueError("noun phrase must end with noun or pronoun")
    adjectives = [token for token, tag in zip(tokens[:-1], tags[:-1]) if tag.pos == "adj"]
    noun = _token_entity_text(tokens[-1], tags[-1])
    noun_instance_id = tags[-1].instance_id if tags else None
    owner_instance_id = tags[-1].owner_instance_id if tags else None
    owner_role = tags[-1].owner_role if tags else None
    return noun, adjectives, noun_instance_id, owner_instance_id, owner_role


def _has_noun_relation(source_noun: str, relation_name: str, target_noun: str) -> bool:
    rm, _, _ = _load_language_context()
    source_key = source_noun.lower()
    target_key = target_noun.lower()
    if relation_name not in rm.relation_list:
        return False
    if source_key not in rm.noun_list or target_key not in rm.noun_list:
        return False
    relation_type = rm.relation_list.index(relation_name) + 1
    source_idx = rm.noun_list.index(source_key)
    target_idx = rm.noun_list.index(target_key)
    return int(rm.relation_map[source_idx, target_idx]) == relation_type


def _lookup_belong_to_slot(noun: str) -> Optional[str]:
    rm, _, _ = _load_language_context()
    relation_name = "belong to"
    if relation_name not in rm.relation_list:
        return None
    noun_key = noun.lower()
    if noun_key not in rm.noun_list:
        return None
    relation_type = rm.relation_list.index(relation_name) + 1
    noun_idx = rm.noun_list.index(noun_key)
    for target_idx, stored_relation_type in enumerate(rm.relation_map[noun_idx]):
        if int(stored_relation_type) == relation_type and target_idx < len(rm.noun_list):
            return rm.noun_list[target_idx]
    return None


def _extract_pattern_be_noun_core(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    polarity: int = 1,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing
    if len(tokens) != 3:
        raise ValueError("be_noun sentence requires noun + be + noun")

    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="be_noun" if polarity == 1 else "negative_be_noun",
        sentence_type="relation_sentence",
    )

    left_noun, _, left_instance_id, left_owner_instance_id, left_owner_role = _parse_noun_phrase([tokens[0]], [tagged_tokens[0]])
    right_noun, _, right_instance_id, _, _ = _parse_noun_phrase([tokens[2]], [tagged_tokens[2]])

    if left_instance_id is not None and left_owner_instance_id is not None:
        include_match = _has_noun_relation(left_noun, "include", right_noun)
        if include_match:
            parsed.instance_updates.append(
                InstanceAttributeUpdate(
                    instance_id=left_owner_instance_id,
                    attribute_name=left_noun,
                    attribute_value=right_noun,
                    noun_text=None,
                    source_tokens=list(tokens),
                    polarity=int(polarity),
                )
            )
            parsed.sentence_type = "instance_update_sentence"
            return parsed

    if left_instance_id is not None and left_owner_instance_id is None and left_owner_role is None:
        slot_name = _lookup_belong_to_slot(right_noun)
        if slot_name is not None:
            parsed.instance_updates.append(
                InstanceAttributeUpdate(
                    instance_id=left_instance_id,
                    attribute_name=slot_name,
                    attribute_value=right_noun,
                    noun_text=left_noun,
                    source_tokens=list(tokens),
                    polarity=int(polarity),
                )
            )
            parsed.relation_tuples.append(
                RelationTuple(
                    source=left_noun,
                    relation=f"{slot_name}_relation",
                    target=right_noun,
                    kind="noun_noun_relation",
                    source_instance_id=left_instance_id,
                    target_instance_id=right_instance_id,
                    owner_instance_id=left_owner_instance_id,
                    owner_role=left_owner_role,
                    source_tokens=list(tokens),
                    polarity=int(polarity),
                )
            )
            parsed.sentence_type = "mixed_sentence"
            return parsed

    if left_instance_id is None:
        resolved_context = _coerce_instance_context(instance_context=instance_context, short_memory=short_memory)
        left_instance_id = _build_instance_id(left_noun, {}, instance_context=resolved_context)

    parsed.relation_tuples.append(
        RelationTuple(
            source=left_noun,
            relation="belong to",
            target=right_noun,
            kind="noun_noun_relation",
            source_instance_id=left_instance_id,
            target_instance_id=right_instance_id,
            owner_instance_id=left_owner_instance_id,
            owner_role=left_owner_role,
            source_tokens=list(tokens),
            polarity=int(polarity),
        )
    )
    return parsed


def _extract_pattern_be_noun(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    return _extract_pattern_be_noun_core(
        tokens,
        tagged_tokens,
        polarity=1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )


def _extract_pattern_negative_be_noun(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    reduced_tokens = [tokens[0], tokens[1], tokens[3]]
    reduced_tags = [tagged_tokens[0], tagged_tokens[1], tagged_tokens[3]]
    return _extract_pattern_be_noun_core(
        reduced_tokens,
        reduced_tags,
        polarity=-1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )


def _extract_pattern_reward_sentence(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    polarity: int = 1,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) not in {3, 4}:
        raise ValueError("reward sentence requires noun + reward + action/noun")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    reward_word = tokens[1].lower()
    reward_value = REWARD_WORD_VALUE_MAP.get(reward_word)
    if reward_value is None:
        raise ValueError(f"Unknown reward word: {reward_word}")
    if int(polarity) == -1:
        reward_value = negate_reward_value(reward_value)

    action_text = None
    object_text = None
    object_instance_id = None

    if tagged_tokens[2].pos == "action":
        action_text = tokens[2].lower()
        if len(tokens) == 4:
            object_text = _token_entity_text(tokens[3], tagged_tokens[3])
            object_instance_id = tagged_tokens[3].instance_id
    elif tagged_tokens[2].pos == "noun" and len(tokens) == 3:
        object_text = _token_entity_text(tokens[2], tagged_tokens[2])
        object_instance_id = tagged_tokens[2].instance_id
    else:
        raise ValueError(f"Unsupported reward sentence structure: {tuple(tag.pos for tag in tagged_tokens)}")

    return ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_reward" if int(polarity) == 1 else "negative_noun_reward",
        sentence_type="reward_sentence",
        reward_tuples=[
            RewardTuple(
                subject=subject,
                reward_word=reward_word,
                reward_value=float(reward_value),
                action=action_text,
                object=object_text,
                subject_instance_id=subject_instance_id,
                object_instance_id=object_instance_id,
                source_tokens=list(tokens),
                polarity=1,
            )
        ],
    )


def _extract_pattern_do_question_action_reward(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    question_info: QuestionInfo,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 3:
        raise ValueError("do-question action reward requires noun + action + noun")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    verb = tokens[1].lower()
    object_noun, _, object_instance_id, _, _ = _parse_noun_phrase([tokens[2]], [tagged_tokens[2]])
    question_word = "question"

    return ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="do_question_noun_action_object_reward",
        sentence_type="question_sentence",
        reward_tuples=[
            RewardTuple(
                subject=subject,
                reward_word=question_word,
                reward_value=0.0,
                action=verb,
                object=object_noun,
                subject_instance_id=subject_instance_id,
                object_instance_id=object_instance_id,
                source_tokens=list(tokens),
                polarity=1,
                accept_label="none",
                question_label="question",
            )
        ],
        question_info=question_info,
    )


def _extract_pattern_negative_reward_sentence(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    if len(tokens) not in {4, 5}:
        raise ValueError("negative reward sentence requires noun + negative + reward + action/noun")
    reduced_tokens = [tokens[0], *tokens[2:]]
    reduced_tags = [tagged_tokens[0], *tagged_tokens[2:]]
    parsed = _extract_pattern_reward_sentence(
        reduced_tokens,
        reduced_tags,
        polarity=-1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.tokens = list(tokens)
    parsed.tagged_tokens = list(tagged_tokens)
    parsed.structure = tuple(tag.pos for tag in tagged_tokens)
    for reward_tuple in parsed.reward_tuples:
        reward_tuple.source_tokens = list(tokens)
        reward_tuple.polarity = 1
    return parsed


def _extract_pattern_modal_surprise_action_with_object(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
    polarity: int = 1,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 4:
        raise ValueError("modal surprise sentence requires noun + can/could + action + noun")
    modal = tokens[1].lower()
    if modal not in ABILITY_HELPER_WORD_SET:
        raise ValueError(f"Unsupported surprise modal: {modal}")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    action_text = tokens[2].lower()
    object_noun, _, object_instance_id, _, _ = _parse_noun_phrase([tokens[3]], [tagged_tokens[3]])
    surprise_value = SURPRISE_WORD_VALUE_MAP.get(modal)
    if surprise_value is None:
        raise ValueError(f"Unknown surprise word: {modal}")
    if int(polarity) == -1:
        surprise_value = negate_surprise_value(surprise_value)

    return ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_can_action_object_surprise",
        sentence_type="surprise_sentence",
        surprise_tuples=[
            SurpriseTuple(
                subject=subject,
                surprise_word=modal,
                surprise_value=float(surprise_value),
                action=action_text,
                object=object_noun,
                subject_instance_id=subject_instance_id,
                object_instance_id=object_instance_id,
                source_tokens=list(tokens),
                polarity=1,
            )
        ],
    )


def _extract_pattern_can_question_action_surprise(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    question_info: QuestionInfo,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    parsed = _extract_pattern_modal_surprise_action_with_object(
        tokens,
        tagged_tokens,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.pattern_name = "can_question_noun_action_object_surprise"
    parsed.sentence_type = "question_sentence"
    parsed.question_info = question_info
    return parsed


def _extract_pattern_negative_modal_surprise_action_with_object(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    if len(tokens) != 5:
        raise ValueError("negative modal surprise sentence requires noun + can/could + negative + action + noun")

    if tagged_tokens[1].pos == "helper" and tagged_tokens[2].pos == "negative":
        reduced_tokens = [tokens[0], tokens[1], tokens[3], tokens[4]]
        reduced_tags = [tagged_tokens[0], tagged_tokens[1], tagged_tokens[3], tagged_tokens[4]]
    elif tagged_tokens[1].pos == "negative" and tagged_tokens[2].pos == "helper":
        reduced_tokens = [tokens[0], tokens[2], tokens[3], tokens[4]]
        reduced_tags = [tagged_tokens[0], tagged_tokens[2], tagged_tokens[3], tagged_tokens[4]]
    else:
        raise ValueError(f"Unsupported negative modal surprise structure: {tuple(tag.pos for tag in tagged_tokens)}")

    parsed = _extract_pattern_modal_surprise_action_with_object(
        reduced_tokens,
        reduced_tags,
        polarity=-1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.tokens = list(tokens)
    parsed.tagged_tokens = list(tagged_tokens)
    parsed.structure = tuple(tag.pos for tag in tagged_tokens)
    parsed.pattern_name = "negative_noun_can_action_object_surprise"
    for surprise_tuple in parsed.surprise_tuples:
        surprise_tuple.source_tokens = list(tokens)
        surprise_tuple.polarity = 1
    return parsed


def _extract_pattern_intransitive_action(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 2:
        raise ValueError("noun_action sentence requires exactly noun + action")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    verb = tokens[1]

    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_action",
        sentence_type="action_sentence",
        action_tuples=[
            ActionTuple(
                noun=subject,
                action=verb,
                role="subject",
                position=0,
                noun_instance_id=subject_instance_id,
                owner_instance_id=tagged_tokens[0].owner_instance_id,
                owner_role=tagged_tokens[0].owner_role,
                source_tokens=[subject, verb],
            )
        ],
    )
    return parsed


def _extract_pattern_action_with_object(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_action_object_phrase",
        sentence_type="action_sentence",
    )

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    verb = tokens[1]
    object_tokens = list(tokens[2:])
    object_tags = list(tagged_tokens[2:])

    if object_tokens:
        object_noun, _, object_instance_id, object_owner_instance_id, object_owner_role = _parse_noun_phrase(object_tokens, object_tags)
        parsed.action_tuples.append(
            ActionTuple(
                noun=object_noun,
                action=object_action_form(verb),
                role="object",
                position=1,
                noun_instance_id=object_instance_id,
                owner_instance_id=object_owner_instance_id,
                owner_role=object_owner_role,
                source_tokens=object_tokens,
            )
        )

    parsed.action_tuples.append(
        ActionTuple(
            noun=subject,
            action=verb,
            role="subject",
            position=0,
            noun_instance_id=subject_instance_id,
            owner_instance_id=tagged_tokens[0].owner_instance_id,
            owner_role=tagged_tokens[0].owner_role,
            source_tokens=[subject, verb],
        )
    )

    return parsed


def _extract_pattern_negative_intransitive_action(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 3:
        raise ValueError("negative noun_action sentence requires noun + negative + action")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    verb = tokens[2].lower()
    return ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="negative_noun_action",
        sentence_type="action_sentence",
        action_tuples=[
            ActionTuple(
                noun=subject,
                action=verb,
                role="subject",
                position=0,
                noun_instance_id=tagged_tokens[0].instance_id,
                owner_instance_id=tagged_tokens[0].owner_instance_id,
                owner_role=tagged_tokens[0].owner_role,
                source_tokens=[subject, tokens[1].lower(), verb],
                polarity=-1,
            )
        ],
    )


def _extract_pattern_negative_action_with_object(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) < 4:
        raise ValueError("negative action sentence requires noun + negative + action + object")

    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="negative_noun_action_object_phrase",
        sentence_type="action_sentence",
    )

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    verb = tokens[2].lower()
    object_noun, _, object_instance_id, object_owner_instance_id, object_owner_role = _parse_noun_phrase(
        list(tokens[3:]),
        list(tagged_tokens[3:]),
    )
    parsed.action_tuples.append(
        ActionTuple(
            noun=object_noun,
            action=object_action_form(verb),
            role="object",
            position=1,
            noun_instance_id=object_instance_id,
            owner_instance_id=object_owner_instance_id,
            owner_role=object_owner_role,
            source_tokens=list(tokens[3:]),
            polarity=-1,
        )
    )
    parsed.action_tuples.append(
        ActionTuple(
            noun=subject,
            action=verb,
            role="subject",
            position=0,
            noun_instance_id=tagged_tokens[0].instance_id,
            owner_instance_id=tagged_tokens[0].owner_instance_id,
            owner_role=tagged_tokens[0].owner_role,
            source_tokens=[subject, tokens[1].lower(), verb],
            polarity=-1,
        )
    )
    return parsed


def _extract_pattern_passive_action_without_subject(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 3:
        raise ValueError("passive action sentence requires noun + be + actioned")

    object_noun = _token_entity_text(tokens[0], tagged_tokens[0])
    actioned = tokens[2].lower()
    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="passive_action_without_subject",
        sentence_type="action_sentence",
    )
    parsed.action_tuples.append(
        ActionTuple(
            noun=object_noun,
            action=actioned,
            role="object",
            position=1,
            noun_instance_id=tagged_tokens[0].instance_id,
            owner_instance_id=tagged_tokens[0].owner_instance_id,
            owner_role=tagged_tokens[0].owner_role,
            source_tokens=[object_noun, actioned],
            polarity=1,
        )
    )
    return parsed


def _extract_pattern_negative_passive_action_without_subject(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    if len(tokens) != 4:
        raise ValueError("negative passive action sentence requires noun + be + negative + actioned")
    reduced_tokens = [tokens[0], tokens[1], tokens[3]]
    reduced_tags = [tagged_tokens[0], tagged_tokens[1], tagged_tokens[3]]
    parsed = _extract_pattern_passive_action_without_subject(
        reduced_tokens,
        reduced_tags,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.pattern_name = "negative_passive_action_without_subject"
    parsed.tokens = list(tokens)
    parsed.tagged_tokens = list(tagged_tokens)
    parsed.structure = tuple(tag.pos for tag in tagged_tokens)
    for action_tuple in parsed.action_tuples:
        action_tuple.polarity = -1
    return parsed


def _extract_pattern_passive_action_with_subject(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    if len(tokens) != 4:
        raise ValueError("passive action sentence requires noun + be + actioned + noun")

    object_noun = _token_entity_text(tokens[0], tagged_tokens[0])
    object_instance_id = tagged_tokens[0].instance_id
    actioned = tokens[2].lower()
    active_action = subject_action_form(actioned)
    subject_noun = _token_entity_text(tokens[3], tagged_tokens[3])
    subject_instance_id = tagged_tokens[3].instance_id

    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="passive_action_with_subject",
        sentence_type="action_sentence",
    )
    parsed.action_tuples.append(
        ActionTuple(
            noun=subject_noun,
            action=active_action,
            role="subject",
            position=0,
            noun_instance_id=subject_instance_id,
            owner_instance_id=tagged_tokens[3].owner_instance_id,
            owner_role=tagged_tokens[3].owner_role,
            source_tokens=[subject_noun, active_action],
            polarity=1,
        )
    )
    # Passive sentences focus the affected object, so keep the object-side event last.
    parsed.action_tuples.append(
        ActionTuple(
            noun=object_noun,
            action=actioned,
            role="object",
            position=1,
            noun_instance_id=object_instance_id,
            owner_instance_id=tagged_tokens[0].owner_instance_id,
            owner_role=tagged_tokens[0].owner_role,
            source_tokens=[object_noun, actioned],
            polarity=1,
        )
    )
    return parsed


def _extract_pattern_negative_passive_action_with_subject(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    if len(tokens) != 5:
        raise ValueError("negative passive sentence requires noun + be + negative + actioned + noun")
    reduced_tokens = [tokens[0], tokens[1], tokens[3], tokens[4]]
    reduced_tags = [tagged_tokens[0], tagged_tokens[1], tagged_tokens[3], tagged_tokens[4]]
    parsed = _extract_pattern_passive_action_with_subject(
        reduced_tokens,
        reduced_tags,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.pattern_name = "negative_passive_action_with_subject"
    parsed.tokens = list(tokens)
    parsed.tagged_tokens = list(tagged_tokens)
    parsed.structure = tuple(tag.pos for tag in tagged_tokens)
    for action_tuple in parsed.action_tuples:
        action_tuple.polarity = -1
    return parsed


def _extract_pattern_be_attribute_core(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    polarity: int = 1,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del instance_context, short_memory
    _, arm, _ = _load_language_context()
    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="be_attribute" if polarity == 1 else "negative_be_attribute",
        sentence_type="attribute_sentence",
    )

    if len(tokens) < 3:
        raise ValueError("be_attribute sentence requires at least noun + be + adj")

    subject = _token_entity_text(tokens[0], tagged_tokens[0])
    subject_instance_id = tagged_tokens[0].instance_id
    adjective_tokens = list(tokens[2:])
    adjective_tags = list(tagged_tokens[2:])
    if not adjective_tags or any(tag.pos != "adj" for tag in adjective_tags):
        raise ValueError("be_attribute sentence requires adjective tokens after be")

    relation_overrides = _resolve_adjective_relation_overrides(
        adjective_tokens,
        adjective_relation_types,
        arm.adj_relation_list,
    )
    for adjective, relation_override in zip(adjective_tokens, relation_overrides):
        relation_name = _resolve_adj_relation_name(
            subject,
            adjective,
            relation_override,
            infer_missing=infer_missing,
            relation_names=arm.adj_relation_list,
        )
        if relation_name is None:
            continue
        parsed.relation_tuples.append(
            RelationTuple(
                source=subject,
                relation=relation_name,
                target=adjective,
                kind="adj_noun_relation",
                source_instance_id=subject_instance_id,
                target_instance_id=None,
                owner_instance_id=tagged_tokens[0].owner_instance_id,
                owner_role=tagged_tokens[0].owner_role,
                source_tokens=list(tokens),
                polarity=int(polarity),
            )
        )

    return parsed


def _extract_pattern_be_attribute(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    return _extract_pattern_be_attribute_core(
        tokens,
        tagged_tokens,
        polarity=1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )


def _extract_pattern_negative_be_attribute(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    reduced_tokens = [tokens[0], tokens[1], *tokens[3:]]
    reduced_tags = [tagged_tokens[0], tagged_tokens[1], *tagged_tokens[3:]]
    return _extract_pattern_be_attribute_core(
        reduced_tokens,
        reduced_tags,
        polarity=-1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )


def _extract_pattern_relation_between_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    polarity: int = 1,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del adjective_relation_types, infer_missing, instance_context, short_memory
    rm, _, _ = _load_language_context()
    relation_match = _resolve_relation_phrase(tokens, [relation.lower() for relation in rm.relation_list])
    if relation_match is None:
        raise ValueError("No noun_noun relation phrase found in relation-pattern sentence")

    relation_name, rel_start, rel_end = relation_match
    left_tokens = list(tokens[:rel_start])
    right_tokens = list(tokens[rel_end:])
    left_tags = list(tagged_tokens[:rel_start])
    right_tags = list(tagged_tokens[rel_end:])
    if not left_tokens or not right_tokens:
        raise ValueError("relation sentence requires noun phrases on both sides")

    left_noun, _, left_instance_id, left_owner_instance_id, left_owner_role = _parse_noun_phrase(left_tokens, left_tags)
    right_noun, _, right_instance_id, _, _ = _parse_noun_phrase(right_tokens, right_tags)

    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_phrase_relation_noun_phrase",
        sentence_type="relation_sentence",
    )
    parsed.relation_tuples.append(
        RelationTuple(
            source=left_noun,
            relation=relation_name,
            target=right_noun,
            kind="noun_noun_relation",
            source_instance_id=left_instance_id,
            target_instance_id=right_instance_id,
            owner_instance_id=left_owner_instance_id,
            owner_role=left_owner_role,
            source_tokens=list(tokens),
            polarity=int(polarity),
        )
    )

    return parsed


def _extract_pattern_negative_relation_between_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    reduced_tokens = [tokens[0], *tokens[2:]]
    reduced_tags = [tagged_tokens[0], *tagged_tokens[2:]]
    parsed = _extract_pattern_relation_between_noun_phrases(
        reduced_tokens,
        reduced_tags,
        polarity=-1,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.pattern_name = "negative_noun_phrase_relation_noun_phrase"
    parsed.tokens = list(tokens)
    parsed.tagged_tokens = list(tagged_tokens)
    parsed.structure = tuple(tag.pos for tag in tagged_tokens)
    return parsed


def _mark_parsed_question_label(parsed: ParsedSentence, label: str = "question") -> None:
    for action_tuple in parsed.action_tuples:
        action_tuple.question_label = label
    for relation_tuple in parsed.relation_tuples:
        relation_tuple.question_label = label
    for reward_tuple in parsed.reward_tuples:
        reward_tuple.question_label = label
    for surprise_tuple in parsed.surprise_tuples:
        surprise_tuple.question_label = label
    for instance_update in parsed.instance_updates:
        instance_update.question_label = label


def _build_reduction_only_parse(
    sentence: str,
    reduced_tokens: Sequence[str],
    reduced_tags: Sequence[TaggedToken],
    reduced_relations: Sequence[RelationTuple],
) -> ParsedSentence:
    parsed = ParsedSentence(
        sentence=sentence,
        tokens=list(reduced_tokens),
        tagged_tokens=list(reduced_tags),
        structure=tuple(tag.pos for tag in reduced_tags),
        pattern_name="reduction_only",
        sentence_type="attribute_sentence",
    )
    parsed.relation_tuples.extend(list(reduced_relations))
    parsed.sentence_type = classify_sentence_type_from_parsed(parsed)
    return parsed


def _structure_for_extractor_routing(tagged_tokens: Sequence[TaggedToken]) -> Tuple[str, ...]:
    structure: List[str] = []
    for tag in tagged_tokens:
        pos = tag.pos
        if pos == "relation" and structure and structure[-1] == "relation":
            continue
        structure.append(pos)
    return tuple(structure)


# Extractor routing is driven by structure signatures so new simple sentence
# patterns can be added in grammar_layer/grammar_routes.py without rewriting the
# selection logic here.
def _match_extractor_route(structure: Tuple[str, ...], route) -> bool:
    route_structure = tuple(route.structure)
    if route.match_mode == "exact":
        return structure == route_structure
    if route.match_mode == "prefix":
        if len(structure) < len(route_structure):
            return False
        return structure[: len(route_structure)] == route_structure
    raise ValueError(f"Unsupported extractor route match_mode: {route.match_mode}")


def _select_extractor(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    question_info: Optional[QuestionInfo] = None,
):
    structure = _structure_for_extractor_routing(tagged_tokens)
    if question_info is not None and question_info.is_question:
        if question_info.marker in DO_QUESTION_HELPER_WORD_SET and structure == ("noun", "action", "noun"):
            def _do_question_extractor(*args, **kwargs):
                return _extract_pattern_do_question_action_reward(
                    *args,
                    question_info=question_info,
                    **kwargs,
                )
            return _do_question_extractor
        if question_info.marker in ABILITY_HELPER_WORD_SET and structure == ("noun", "helper", "action", "noun"):
            def _can_question_extractor(*args, **kwargs):
                return _extract_pattern_can_question_action_surprise(
                    *args,
                    question_info=question_info,
                    **kwargs,
                )
            return _can_question_extractor

    extractor_map = {
        "_extract_pattern_be_attribute": _extract_pattern_be_attribute,
        "_extract_pattern_negative_be_attribute": _extract_pattern_negative_be_attribute,
        "_extract_pattern_be_noun": _extract_pattern_be_noun,
        "_extract_pattern_negative_be_noun": _extract_pattern_negative_be_noun,
        "_extract_pattern_reward_sentence": _extract_pattern_reward_sentence,
        "_extract_pattern_negative_reward_sentence": _extract_pattern_negative_reward_sentence,
        "_extract_pattern_modal_surprise_action_with_object": _extract_pattern_modal_surprise_action_with_object,
        "_extract_pattern_negative_modal_surprise_action_with_object": _extract_pattern_negative_modal_surprise_action_with_object,
        "_extract_pattern_intransitive_action": _extract_pattern_intransitive_action,
        "_extract_pattern_action_with_object": _extract_pattern_action_with_object,
        "_extract_pattern_negative_intransitive_action": _extract_pattern_negative_intransitive_action,
        "_extract_pattern_negative_action_with_object": _extract_pattern_negative_action_with_object,
        "_extract_pattern_passive_action_without_subject": _extract_pattern_passive_action_without_subject,
        "_extract_pattern_negative_passive_action_without_subject": _extract_pattern_negative_passive_action_without_subject,
        "_extract_pattern_passive_action_with_subject": _extract_pattern_passive_action_with_subject,
        "_extract_pattern_negative_passive_action_with_subject": _extract_pattern_negative_passive_action_with_subject,
        "_extract_pattern_relation_between_noun_phrases": _extract_pattern_relation_between_noun_phrases,
        "_extract_pattern_negative_relation_between_noun_phrases": _extract_pattern_negative_relation_between_noun_phrases,
    }

    for route in DEFAULT_EXTRACTOR_ROUTES:
        if not _match_extractor_route(structure, route):
            continue
        extractor = extractor_map.get(route.extractor_name)
        if extractor is None:
            raise ValueError(f"Unknown extractor registered for pattern {route.pattern_name}: {route.extractor_name}")
        return extractor

    raise ValueError(f"No extraction strategy defined for structure {structure}")


def parse_sentence(
    sentence: str,
    noun_relation_type: Optional[RelationOverride] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    del noun_relation_type  # grammar parses structure; routing decides how to use relation configs.
    tokens = tokenize_sentence(
        sentence,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    tagged_tokens = tag_tokens(
        " ".join(tokens),
        instance_context=instance_context,
        short_memory=short_memory,
    )

    reduced_tokens, reduced_tags = _expand_possessive_noun_tokens(
        tokens,
        tagged_tokens,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens, reduced_tags, reduced_relations = _reduce_adj_noun_phrases(
        reduced_tokens,
        reduced_tags,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    reduced_tokens, reduced_tags = _reduce_pronoun_noun_phrases(
        reduced_tokens,
        reduced_tags,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens, reduced_tags = _reduce_possessive_noun_phrases(
        reduced_tokens,
        reduced_tags,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens, reduced_tags = _reduce_article_noun_phrases(
        reduced_tokens,
        reduced_tags,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens, reduced_tags, question_info = _detect_and_reorder_question_helper(
        reduced_tokens,
        reduced_tags,
    )
    reduced_tokens, reduced_tags = _bind_existing_noun_instances(
        reduced_tokens,
        reduced_tags,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    reduced_tokens, reduced_tags = _reduce_helper_tokens(
        reduced_tokens,
        reduced_tags,
    )

    try:
        extractor = _select_extractor(
            reduced_tokens,
            reduced_tags,
            question_info=question_info,
        )
    except ValueError:
        if reduced_relations or reduced_tokens != tokens or tuple(tag.pos for tag in reduced_tags) != tuple(tag.pos for tag in tagged_tokens):
            parsed = _build_reduction_only_parse(sentence, reduced_tokens, reduced_tags, reduced_relations)
        else:
            raise
    else:
        parsed = extractor(
            reduced_tokens,
            reduced_tags,
            adjective_relation_types=adjective_relation_types,
            infer_missing=infer_missing,
            instance_context=instance_context,
            short_memory=short_memory,
        )
        parsed.relation_tuples = list(reduced_relations) + list(parsed.relation_tuples)

    parsed.question_info = question_info
    if question_info.is_question:
        _mark_parsed_question_label(parsed)
    parsed = resolve_instances_for_parsed_sentence(
        parsed,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    parsed.sentence_type = classify_sentence_type_from_parsed(parsed)
    return parsed


# ===========================================================================
# LAYER 5. INSTANCE DECISION RULES
# ===========================================================================


def _build_instance_id(
    noun_text: str,
    create_counters: Dict[str, int],
    instance_context=None,
) -> str:
    noun_key = noun_text.lower()
    existing_max = 0
    if instance_context:
        for instance_id in instance_context.get("by_noun", {}).get(noun_key, []):
            prefix = f"{noun_key}#new:"
            if instance_id.startswith(prefix):
                suffix = instance_id[len(prefix):]
                if suffix.isdigit():
                    existing_max = max(existing_max, int(suffix))
    next_counter = max(int(create_counters.get(noun_key, 0)), existing_max) + 1
    create_counters[noun_key] = next_counter
    return f"{noun_key}#new:{next_counter}"


def _resolve_instance_candidate(
    noun_text: str,
    *,
    instance_context,
    sentence_assignments: Dict[str, str],
) -> Optional[str]:
    noun_key = noun_text.lower()
    if noun_key in PRONOUN_LIST:
        return _select_instance_for_pronoun(noun_key, instance_context)
    return _resolve_existing_instance_id(noun_key, instance_context, sentence_assignments)


def _resolve_action_tuple_instance_id(
    action_tuple: ActionTuple,
    *,
    instance_context,
    sentence_assignments: Dict[str, str],
    create_counters: Dict[str, int],
) -> str:
    candidate = action_tuple.noun_instance_id
    if candidate is None:
        candidate = _resolve_instance_candidate(
            action_tuple.noun,
            instance_context=instance_context,
            sentence_assignments=sentence_assignments,
        )
    if candidate is None:
        candidate = _build_instance_id(action_tuple.noun, create_counters, instance_context=instance_context)
    sentence_assignments[action_tuple.noun.lower()] = candidate
    return candidate


def _resolve_relation_tuple_source_instance_id(
    relation_tuple: RelationTuple,
    *,
    instance_context,
    sentence_assignments: Dict[str, str],
    create_counters: Dict[str, int],
) -> Optional[str]:
    del create_counters
    source_key = relation_tuple.source.lower()
    if source_key in sentence_assignments:
        return sentence_assignments[source_key]

    candidate = relation_tuple.source_instance_id
    if candidate is None:
        candidate = _resolve_instance_candidate(
            relation_tuple.source,
            instance_context=instance_context,
            sentence_assignments=sentence_assignments,
        )
    if candidate is not None:
        sentence_assignments[source_key] = candidate
        return candidate

    return None


def _resolve_relation_tuple_target_instance_id(
    relation_tuple: RelationTuple,
    *,
    instance_context,
    sentence_assignments: Dict[str, str],
) -> Optional[str]:
    if relation_tuple.kind != "noun_noun_relation":
        return None

    target_key = relation_tuple.target.lower()
    if target_key in sentence_assignments:
        return sentence_assignments[target_key]

    candidate = relation_tuple.target_instance_id
    if candidate is None:
        candidate = _resolve_instance_candidate(
            relation_tuple.target,
            instance_context=instance_context,
            sentence_assignments=sentence_assignments,
        )
    if candidate is not None:
        sentence_assignments[target_key] = candidate
    return candidate


def resolve_instances_for_parsed_sentence(
    parsed: ParsedSentence,
    *,
    instance_context=None,
    short_memory=None,
) -> ParsedSentence:
    resolved_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    sentence_assignments: Dict[str, str] = {}
    create_counters: Dict[str, int] = {}

    resolved_actions: List[ActionTuple] = []
    for action_tuple in parsed.action_tuples:
        noun_instance_id = _resolve_action_tuple_instance_id(
            action_tuple,
            instance_context=resolved_context,
            sentence_assignments=sentence_assignments,
            create_counters=create_counters,
        )
        resolved_actions.append(
            ActionTuple(
                noun=action_tuple.noun,
                action=action_tuple.action,
                role=action_tuple.role,
                position=action_tuple.position,
                noun_instance_id=noun_instance_id,
                owner_instance_id=action_tuple.owner_instance_id,
                owner_role=action_tuple.owner_role,
                source_tokens=list(action_tuple.source_tokens),
                polarity=action_tuple.polarity,
                accept_label=action_tuple.accept_label,
                question_label=action_tuple.question_label,
            )
        )

    resolved_relations: List[RelationTuple] = []
    for relation_tuple in parsed.relation_tuples:
        source_instance_id = _resolve_relation_tuple_source_instance_id(
            relation_tuple,
            instance_context=resolved_context,
            sentence_assignments=sentence_assignments,
            create_counters=create_counters,
        )
        target_instance_id = _resolve_relation_tuple_target_instance_id(
            relation_tuple,
            instance_context=resolved_context,
            sentence_assignments=sentence_assignments,
        )
        resolved_relations.append(
            RelationTuple(
                source=relation_tuple.source,
                relation=relation_tuple.relation,
                target=relation_tuple.target,
                kind=relation_tuple.kind,
                source_instance_id=source_instance_id,
                target_instance_id=target_instance_id,
                owner_instance_id=relation_tuple.owner_instance_id,
                owner_role=relation_tuple.owner_role,
                source_tokens=list(relation_tuple.source_tokens),
                polarity=relation_tuple.polarity,
                accept_label=relation_tuple.accept_label,
                question_label=relation_tuple.question_label,
            )
        )

    resolved_rewards: List[RewardTuple] = []
    for reward_tuple in parsed.reward_tuples:
        subject_instance_id = reward_tuple.subject_instance_id
        if subject_instance_id is None:
            subject_instance_id = sentence_assignments.get(reward_tuple.subject)
        if subject_instance_id is None:
            subject_instance_id = _resolve_existing_instance_id(reward_tuple.subject, resolved_context, sentence_assignments)
        if subject_instance_id is None:
            subject_instance_id = _build_instance_id(reward_tuple.subject, create_counters, instance_context=resolved_context)
        sentence_assignments[reward_tuple.subject] = subject_instance_id

        object_instance_id = reward_tuple.object_instance_id
        if reward_tuple.object is not None and object_instance_id is None:
            object_instance_id = sentence_assignments.get(reward_tuple.object)
            if object_instance_id is None:
                object_instance_id = _resolve_existing_instance_id(reward_tuple.object, resolved_context, sentence_assignments)
            if object_instance_id is None:
                object_instance_id = _build_instance_id(reward_tuple.object, create_counters, instance_context=resolved_context)
            sentence_assignments[reward_tuple.object] = object_instance_id

        resolved_rewards.append(
            RewardTuple(
                subject=reward_tuple.subject,
                reward_word=reward_tuple.reward_word,
                reward_value=reward_tuple.reward_value,
                action=reward_tuple.action,
                object=reward_tuple.object,
                subject_instance_id=subject_instance_id,
                object_instance_id=object_instance_id,
                source_tokens=list(reward_tuple.source_tokens),
                polarity=reward_tuple.polarity,
                accept_label=reward_tuple.accept_label,
                question_label=reward_tuple.question_label,
            )
        )

    resolved_surprises: List[SurpriseTuple] = []
    for surprise_tuple in parsed.surprise_tuples:
        subject_instance_id = surprise_tuple.subject_instance_id
        if subject_instance_id is None:
            subject_instance_id = sentence_assignments.get(surprise_tuple.subject)
        if subject_instance_id is None:
            subject_instance_id = _resolve_existing_instance_id(surprise_tuple.subject, resolved_context, sentence_assignments)
        if subject_instance_id is None:
            subject_instance_id = _build_instance_id(surprise_tuple.subject, create_counters, instance_context=resolved_context)
        sentence_assignments[surprise_tuple.subject] = subject_instance_id

        object_instance_id = surprise_tuple.object_instance_id
        if surprise_tuple.object is not None and object_instance_id is None:
            object_instance_id = sentence_assignments.get(surprise_tuple.object)
            if object_instance_id is None:
                object_instance_id = _resolve_existing_instance_id(surprise_tuple.object, resolved_context, sentence_assignments)
            if object_instance_id is None:
                object_instance_id = _build_instance_id(surprise_tuple.object, create_counters, instance_context=resolved_context)
            sentence_assignments[surprise_tuple.object] = object_instance_id

        resolved_surprises.append(
            SurpriseTuple(
                subject=surprise_tuple.subject,
                surprise_word=surprise_tuple.surprise_word,
                surprise_value=surprise_tuple.surprise_value,
                action=surprise_tuple.action,
                object=surprise_tuple.object,
                subject_instance_id=subject_instance_id,
                object_instance_id=object_instance_id,
                source_tokens=list(surprise_tuple.source_tokens),
                polarity=surprise_tuple.polarity,
                accept_label=surprise_tuple.accept_label,
                question_label=surprise_tuple.question_label,
            )
        )

    return ParsedSentence(
        sentence=parsed.sentence,
        tokens=list(parsed.tokens),
        tagged_tokens=list(parsed.tagged_tokens),
        structure=tuple(parsed.structure),
        pattern_name=parsed.pattern_name,
        sentence_type=parsed.sentence_type,
        action_tuples=resolved_actions,
        relation_tuples=resolved_relations,
        reward_tuples=resolved_rewards,
        surprise_tuples=resolved_surprises,
        instance_updates=list(parsed.instance_updates),
        question_info=parsed.question_info,
    )


# ===========================================================================
# LAYER 6. TIME-STEP RULES
# ===========================================================================

def _memory_last_time_position(short_memory) -> int:
    if short_memory is None:
        return -1
    time_positions = [entry.time_position for entry in getattr(short_memory, "short_memory_event", [])]
    time_positions.extend(entry.time_position for entry in getattr(short_memory, "short_memory_relation", []))
    if not time_positions:
        return -1
    return int(max(time_positions))


def _parsed_action_uses_existing_instance(parsed: ParsedSentence, short_memory) -> bool:
    if short_memory is None:
        return False
    existing_instance_ids = {
        entry.noun_instance_id
        for entry in getattr(short_memory, "short_memory_event", [])
        if entry.noun_instance_id is not None
    }
    for action_tuple in parsed.action_tuples:
        if action_tuple.noun_instance_id in existing_instance_ids:
            return True
    return False


def determine_time_position(
    parsed: ParsedSentence,
    *,
    short_memory=None,
    explicit_time_position: Optional[int] = None,
    default_time_position: int = 0,
    rule_name: str = "existing_action_instance_advances",
) -> int:
    if explicit_time_position is not None:
        return int(explicit_time_position)

    last_time_position = _memory_last_time_position(short_memory)
    if last_time_position < 0:
        return int(default_time_position)

    if rule_name == "existing_action_instance_advances":
        if _parsed_action_uses_existing_instance(parsed, short_memory):
            return last_time_position + 1
        return last_time_position

    raise ValueError(f"Unknown time-step rule: {rule_name}")


def sentence_to_knowledge_samples(
    sentence: str,
    noun_relation_type: Optional[RelationOverride] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = False,
    instance_context=None,
    short_memory=None,
) -> KnowledgeTrainingSamples:
    parsed = parse_sentence(
        sentence,
        noun_relation_type=noun_relation_type,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    rm, arm, _ = _load_language_context()
    samples = KnowledgeTrainingSamples()

    for relation_tuple in parsed.relation_tuples:
        if relation_tuple.kind == "noun_noun_relation":
            relation_type = _normalize_relation_override(
                relation_tuple.relation,
                rm.relation_list,
                relation_label="noun relation",
            )
            if relation_type is None:
                continue
            source_idx = rm._ensure_noun(relation_tuple.source)
            target_idx = rm._ensure_noun(relation_tuple.target)
            samples.noun_noun_samples.append(
                NounNounRelationSample(
                    source_noun=relation_tuple.source,
                    target_noun=relation_tuple.target,
                    relation_type=relation_type,
                    source_idx=source_idx,
                    target_idx=target_idx,
                )
            )
        elif relation_tuple.kind == "adj_noun_relation":
            relation_type = _normalize_relation_override(
                relation_tuple.relation,
                arm.adj_relation_list,
                relation_label="adjective relation",
            )
            if relation_type is None:
                continue
            noun_idx = rm._ensure_noun(relation_tuple.source)
            if relation_tuple.target not in arm.adjective_list:
                arm.adjective_list.append(relation_tuple.target)
            adjective_idx = arm.adjective_list.index(relation_tuple.target)
            samples.adj_noun_samples.append(
                AdjNounRelationSample(
                    noun=relation_tuple.source,
                    adjective=relation_tuple.target,
                    relation_type=relation_type,
                    noun_idx=noun_idx,
                    adjective_idx=adjective_idx,
                )
            )

    return samples


def inject_adjective_into_noun_embedding(
    noun: str,
    adjective: str,
    relation_type: Optional[int] = None,
    step_scale: float = 0.1,
) -> Tuple[int, torch.Tensor]:
    rm, arm, kt = _load_language_context()

    noun_key = noun.lower()
    adjective_key = adjective.lower()

    noun_idx = rm._ensure_noun(noun_key)
    if adjective_key not in arm.adjective_list:
        arm.adjective_list.append(adjective_key)
    adjective_idx = arm.adjective_list.index(adjective_key)

    if relation_type is None:
        relation_type = infer_adj_relation_type(noun_key, adjective_key)
    if relation_type is None:
        raise ValueError(
            f"Cannot infer adjective relation type for ({adjective_key}, {noun_key})"
        )

    rel_idx = int(relation_type) - 1
    if rel_idx < 0 or rel_idx >= len(kt.adj_map_one.relations):
        raise ValueError(f"relation_type must be in [1, {len(kt.adj_map_one.relations)}]")

    noun_embedding = (
        kt.knowledge_map_one.embedding.weight.data[noun_idx]
        .clone()
        .detach()
        .requires_grad_(True)
    )
    adjective_target = kt.adj_map_one.adjective_embedding.weight.data[adjective_idx]
    relation_weight = kt.adj_map_one.relations[rel_idx].weight.data
    adjective_pred = relation_weight @ noun_embedding
    loss = F.mse_loss(adjective_pred, adjective_target)
    loss.backward()

    with torch.no_grad():
        lr = float(rm.lr_per_embedding[noun_idx]) * float(step_scale)
        adjusted_noun_embedding = noun_embedding - lr * noun_embedding.grad

    noun_embedding.grad.zero_()
    return noun_idx, adjusted_noun_embedding.detach()


def sentence_to_noun_action_pairs(
    sentence: str,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> List[NounActionPair]:
    parsed = parse_sentence(
        sentence,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    pairs: List[NounActionPair] = []
    object_adjectives_map: Dict[str, List[str]] = {}
    for relation_tuple in parsed.relation_tuples:
        if relation_tuple.kind == "adj_noun_relation":
            object_adjectives_map.setdefault(relation_tuple.source, []).append(relation_tuple.target)

    for action_tuple in parsed.action_tuples:
        ensure_action(action_tuple.action)
        noun_idx, noun_embedding = _ensure_noun_embedding(action_tuple.noun)
        adjectives: List[str] = []

        if action_tuple.role == "object":
            adjectives = object_adjectives_map.get(action_tuple.noun, [])

        pairs.append(
            NounActionPair(
                noun=action_tuple.noun,
                action=action_tuple.action,
                role=action_tuple.role,
                position=action_tuple.position,
                noun_instance_id=action_tuple.noun_instance_id,
                noun_index=noun_idx,
                noun_embedding=noun_embedding,
                adjectives=adjectives,
            )
        )

    return pairs


def sentence_to_pair_tuples(sentence: str) -> List[Tuple[str, str]]:
    return [(pair.noun, pair.action) for pair in sentence_to_noun_action_pairs(sentence)]


def sentences_to_pair_dataset(sentences: Sequence[str]) -> List[List[NounActionPair]]:
    return [sentence_to_noun_action_pairs(sentence) for sentence in sentences]


def build_action_type_map(world_model) -> Dict[str, int]:
    action_type_map = {}
    for action_type, action_name in enumerate(world_model.action_list):
        if action_type == 0:
            continue
        action_type_map[action_name.lower()] = action_type
    return action_type_map


def noun_action_pairs_to_short_memory_states(
    pairs: Sequence[NounActionPair],
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    time_position: int = 0,
) -> List[ShortMemoryState]:
    action_type_map = action_type_map or build_action_type_map(world_model)
    states = []

    for pair_index, pair in enumerate(pairs):
        action_key = pair.action.lower()
        if action_key not in action_type_map:
            raise ValueError(f"Action '{pair.action}' not found in action_type_map")
        if pair.noun_embedding is None:
            raise ValueError(f"Noun embedding missing for '{pair.noun}'")

        noun_embedding = pair.noun_embedding.view(-1).clone()
        if noun_embedding.numel() != world_model.noun_dim:
            raise ValueError(
                f"Expected noun embedding dim {world_model.noun_dim}, got {noun_embedding.numel()}"
            )

        action_type = int(action_type_map[action_key])
        action_embedding = world_model.get_action_embedding(action_type).detach().clone()
        states.append(
            ShortMemoryState(
                noun=pair.noun,
                action=pair.action,
                noun_instance_id=pair.noun_instance_id,
                noun_index=pair.noun_index,
                action_type=action_type,
                noun_embedding=noun_embedding,
                action_embedding=action_embedding,
                time_position=int(time_position),
                pair_index=pair_index,
                role=pair.role,
                adjectives=list(pair.adjectives or []),
            )
        )

    return states


def parsed_sentence_to_relation_memory_states(
    parsed: ParsedSentence,
    time_position: int = 0,
) -> List[ShortMemoryRelationState]:
    relation_states: List[ShortMemoryRelationState] = []
    for pair_index, relation_tuple in enumerate(parsed.relation_tuples):
        source_index = None
        target_index = None
        source_embedding = None
        target_embedding = None

        if relation_tuple.source_instance_id is not None:
            source_index, source_embedding = _ensure_noun_embedding(relation_tuple.source)

        if relation_tuple.target_instance_id is not None:
            target_index, target_embedding = _ensure_noun_embedding(relation_tuple.target)

        relation_states.append(
            ShortMemoryRelationState(
                source=relation_tuple.source,
                relation=relation_tuple.relation,
                target=relation_tuple.target,
                relation_kind=relation_tuple.kind,
                source_instance_id=relation_tuple.source_instance_id,
                target_instance_id=relation_tuple.target_instance_id,
                source_index=source_index,
                target_index=target_index,
                source_embedding=source_embedding,
                target_embedding=target_embedding,
                time_position=int(time_position),
                pair_index=pair_index,
                polarity=relation_tuple.polarity,
                accept_label=relation_tuple.accept_label,
                question_label=relation_tuple.question_label,
            )
        )

    return relation_states


def sentence_to_relation_memory_states(
    sentence: str,
    time_position: Optional[int] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
    instance_context=None,
    short_memory=None,
) -> List[ShortMemoryRelationState]:
    parsed = parse_sentence(
        sentence,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    resolved_time_position = determine_time_position(
        parsed,
        short_memory=short_memory,
        explicit_time_position=time_position,
    )
    return parsed_sentence_to_relation_memory_states(
        parsed=parsed,
        time_position=resolved_time_position,
    )


def sentence_to_short_memory_states(
    sentence: str,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    time_position: Optional[int] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    instance_context=None,
    short_memory=None,
) -> List[ShortMemoryState]:
    parsed = parse_sentence(
        sentence,
        adjective_relation_types=adjective_relation_types,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    resolved_time_position = determine_time_position(
        parsed,
        short_memory=short_memory,
        explicit_time_position=time_position,
    )
    pairs = sentence_to_noun_action_pairs(
        sentence,
        adjective_relation_types=adjective_relation_types,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    return noun_action_pairs_to_short_memory_states(
        pairs=pairs,
        world_model=world_model,
        action_type_map=action_type_map,
        time_position=resolved_time_position,
    )


def append_sentence_to_short_memory(
    sentence: str,
    short_memory,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    time_position: Optional[int] = None,
    base_score: float = 1.0,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    sentence_label: Optional[str] = None,
):
    parsed = parse_sentence(
        sentence,
        adjective_relation_types=adjective_relation_types,
        short_memory=short_memory,
    )
    resolved_time_position = determine_time_position(
        parsed,
        short_memory=short_memory,
        explicit_time_position=time_position,
    )
    action_type_map = action_type_map or build_action_type_map(world_model)
    if sentence_label is None:
        if hasattr(short_memory, "next_sentence_label"):
            sentence_label = short_memory.next_sentence_label()
        else:
            sentence_label = f"sentence:{int(resolved_time_position)}"
    sentence_label = str(sentence_label)
    sentence_event_index = short_memory.next_event_index() if parsed.action_tuples else None

    relation_states = parsed_sentence_to_relation_memory_states(
        parsed=parsed,
        time_position=resolved_time_position,
    )

    for instance_update in parsed.instance_updates:
        if instance_update.question_label == "question":
            continue
        update_kwargs = {
            "extra_attributes": {instance_update.attribute_name: instance_update.attribute_value},
            "attribute_polarities": {instance_update.attribute_name: instance_update.polarity},
        }
        if instance_update.noun_text is not None:
            update_kwargs["noun_text"] = instance_update.noun_text
        short_memory.update_noun_instance_metadata(
            instance_update.instance_id,
            **update_kwargs,
        )

    for relation_tuple in parsed.relation_tuples:
        if relation_tuple.source_instance_id is not None and (
            relation_tuple.owner_instance_id is not None or relation_tuple.owner_role is not None
        ):
            short_memory.update_noun_instance_metadata(
                relation_tuple.source_instance_id,
                noun_text=relation_tuple.source,
                owner_instance_id=relation_tuple.owner_instance_id,
                owner_role=relation_tuple.owner_role,
            )
        if relation_tuple.source_instance_id is not None and relation_tuple.relation == "belong to":
            slot_name = _lookup_belong_to_slot(relation_tuple.target)
            if slot_name is not None:
                short_memory.update_noun_instance_metadata(
                    relation_tuple.source_instance_id,
                    noun_text=relation_tuple.source,
                    extra_attributes={slot_name: relation_tuple.target},
                )

    for pair_index, reward_tuple in enumerate(parsed.reward_tuples):
        _, _, subject_instance_id = short_memory.ensure_noun_instance(
            reward_tuple.subject,
            reward_tuple.subject_instance_id,
        )
        object_instance_id = None
        if reward_tuple.object is not None:
            _, _, object_instance_id = short_memory.ensure_noun_instance(
                reward_tuple.object,
                reward_tuple.object_instance_id,
            )
        short_memory.append_reward(
            subject_text=reward_tuple.subject,
            subject_instance_id=subject_instance_id,
            reward_word=reward_tuple.reward_word,
            reward_value=reward_tuple.reward_value,
            action_text=reward_tuple.action,
            object_text=reward_tuple.object,
            object_instance_id=object_instance_id,
            polarity=reward_tuple.polarity,
            accept_label=reward_tuple.accept_label,
            question_label=reward_tuple.question_label,
            sentence_label=sentence_label,
            score=base_score,
            time_position=int(resolved_time_position),
            pair_index=pair_index,
            info_pair={
                "pair_kind": "subject_event_reward",
                "subject": reward_tuple.subject,
                "subject_instance_id": subject_instance_id,
                "reward_word": reward_tuple.reward_word,
                "reward_value": reward_tuple.reward_value,
                "polarity": reward_tuple.polarity,
                "action": reward_tuple.action,
                "object": reward_tuple.object,
                "object_instance_id": object_instance_id,
                "time_position": int(resolved_time_position),
                "pair_index": pair_index,
                "accept_label": reward_tuple.accept_label,
                "question_label": reward_tuple.question_label,
                "sentence_label": sentence_label,
            },
        )

    for pair_index, surprise_tuple in enumerate(parsed.surprise_tuples):
        _, _, subject_instance_id = short_memory.ensure_noun_instance(
            surprise_tuple.subject,
            surprise_tuple.subject_instance_id,
        )
        object_instance_id = None
        if surprise_tuple.object is not None:
            _, _, object_instance_id = short_memory.ensure_noun_instance(
                surprise_tuple.object,
                surprise_tuple.object_instance_id,
            )
        short_memory.append_surprise(
            subject_text=surprise_tuple.subject,
            subject_instance_id=subject_instance_id,
            surprise_word=surprise_tuple.surprise_word,
            surprise_value=surprise_tuple.surprise_value,
            action_text=surprise_tuple.action,
            object_text=surprise_tuple.object,
            object_instance_id=object_instance_id,
            polarity=surprise_tuple.polarity,
            accept_label=surprise_tuple.accept_label,
            question_label=surprise_tuple.question_label,
            sentence_label=sentence_label,
            score=base_score,
            time_position=int(resolved_time_position),
            pair_index=pair_index,
            info_pair={
                "pair_kind": "subject_event_surprise",
                "subject": surprise_tuple.subject,
                "subject_instance_id": subject_instance_id,
                "surprise_word": surprise_tuple.surprise_word,
                "surprise_value": surprise_tuple.surprise_value,
                "polarity": surprise_tuple.polarity,
                "action": surprise_tuple.action,
                "object": surprise_tuple.object,
                "object_instance_id": object_instance_id,
                "time_position": int(resolved_time_position),
                "pair_index": pair_index,
                "accept_label": surprise_tuple.accept_label,
                "question_label": surprise_tuple.question_label,
                "sentence_label": sentence_label,
            },
        )

    for relation_state in relation_states:
        short_memory.append_relation(
            relation_name=relation_state.relation,
            source_text=relation_state.source,
            target_text=relation_state.target,
            relation_kind=relation_state.relation_kind,
            score=base_score,
            time_position=relation_state.time_position,
            pair_index=relation_state.pair_index,
            source_instance_id=relation_state.source_instance_id,
            target_instance_id=relation_state.target_instance_id,
            source_type=relation_state.source_index,
            target_type=relation_state.target_index,
            polarity=relation_state.polarity,
            accept_label=relation_state.accept_label,
            question_label=relation_state.question_label,
            sentence_label=sentence_label,
            source_embedding=None,
            target_embedding=None,
            info_pair={
                "pair_kind": relation_state.relation_kind,
                "relation_name": relation_state.relation,
                "source": relation_state.source,
                "target": relation_state.target,
                "source_instance_id": relation_state.source_instance_id,
                "target_instance_id": relation_state.target_instance_id,
                "time_position": relation_state.time_position,
                "pair_index": relation_state.pair_index,
                "polarity": relation_state.polarity,
                "accept_label": relation_state.accept_label,
                "question_label": relation_state.question_label,
                "sentence_label": sentence_label,
            },
        )

    object_adjectives_map: Dict[str, List[str]] = {}
    for relation_tuple in parsed.relation_tuples:
        if relation_tuple.kind == "adj_noun_relation":
            object_adjectives_map.setdefault(relation_tuple.source, []).append(relation_tuple.target)

    states: List[ShortMemoryState] = []
    for pair_index, action_tuple in enumerate(parsed.action_tuples):
        ensure_action(action_tuple.action)
        noun_idx, noun_embedding, noun_instance_id = short_memory.ensure_noun_instance(
            action_tuple.noun,
            action_tuple.noun_instance_id,
        )
        if action_tuple.owner_instance_id is not None or action_tuple.owner_role is not None:
            short_memory.update_noun_instance_metadata(
                noun_instance_id,
                noun_text=action_tuple.noun,
                owner_instance_id=action_tuple.owner_instance_id,
                owner_role=action_tuple.owner_role,
            )
        action_key = action_tuple.action.lower()
        if action_key not in action_type_map:
            raise ValueError(f"Action '{action_tuple.action}' not found in action_type_map")
        action_type = int(action_type_map[action_key])
        action_embedding = world_model.get_action_embedding(action_type).detach().clone()
        adjectives = object_adjectives_map.get(action_tuple.noun, []) if action_tuple.role == "object" else []

        state = ShortMemoryState(
            noun=action_tuple.noun,
            action=action_tuple.action,
            noun_instance_id=noun_instance_id,
            noun_index=noun_idx,
            action_type=action_type,
            noun_embedding=noun_embedding.view(-1).clone(),
            action_embedding=action_embedding,
            time_position=int(resolved_time_position),
            pair_index=pair_index,
            role=action_tuple.role,
            adjectives=list(adjectives),
            polarity=action_tuple.polarity,
        )
        states.append(state)

        short_memory.append_event(
            noun_embedding=state.noun_embedding,
            action_embedding=state.action_embedding,
            score=base_score,
            event_index=sentence_event_index,
            noun_type=state.noun_index,
            action_type=state.action_type,
            time_position=state.time_position,
            pair_index=state.pair_index,
            instance_id=state.noun_instance_id,
            noun_text=state.noun,
            action_text=state.action,
            role=state.role,
            adjectives=list(state.adjectives),
            pair_kind=state.pair_kind,
            polarity=state.polarity,
            accept_label=action_tuple.accept_label,
            question_label=action_tuple.question_label,
            sentence_label=sentence_label,
            info_pair={
                "pair_kind": state.pair_kind,
                "noun": state.noun,
                "noun_instance_id": state.noun_instance_id,
                "action": state.action,
                "role": state.role,
                "polarity": state.polarity,
                "adjectives": list(state.adjectives),
                "time_position": state.time_position,
                "pair_index": state.pair_index,
                "event_index": sentence_event_index,
                "accept_label": action_tuple.accept_label,
                "question_label": action_tuple.question_label,
                "sentence_label": sentence_label,
            },
        )

    return states


def sentences_to_short_memory(
    sentences: Sequence[str],
    short_memory,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    start_time_position: Optional[int] = None,
    base_score: float = 1.0,
    adjective_relation_types: Optional[Sequence[Optional[Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]]]] = None,
):
    all_states = []
    adjective_relation_types = adjective_relation_types or [None] * len(sentences)
    if len(adjective_relation_types) != len(sentences):
        raise ValueError("adjective_relation_types must align with sentences")

    next_explicit_time_position = start_time_position
    for sentence, relation_overrides in zip(sentences, adjective_relation_types):
        states = append_sentence_to_short_memory(
            sentence=sentence,
            short_memory=short_memory,
            world_model=world_model,
            action_type_map=action_type_map,
            time_position=next_explicit_time_position,
            base_score=base_score,
            adjective_relation_types=relation_overrides,
        )
        all_states.append(states)
        next_explicit_time_position = None
    return all_states

