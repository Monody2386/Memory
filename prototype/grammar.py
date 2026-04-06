"""Grammar utilities organized as:

1. tokenizer
2. part-of-speech analysis
3. instance resolution
4. information extraction by sentence pattern
5. time-step rules

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

from knowledge.noun_action_map import add_noun_action
from world.action_vocab import ensure_action


@dataclass
class TaggedToken:
    text: str
    pos: str
    index: int
    source: str = "lexicon"
    question_prompt: Optional[str] = None
    instance_id: Optional[str] = None


@dataclass
class ActionTuple:
    noun: str
    action: str
    role: str
    position: int
    noun_instance_id: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)


@dataclass
class RelationTuple:
    source: str
    relation: str
    target: str
    kind: str
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    source_tokens: List[str] = field(default_factory=list)


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


IRREGULAR_OBJECT_ACTIONS = {
    "eat": "eaten",
    "see": "seen",
    "write": "written",
    "take": "taken",
    "drive": "driven",
}

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


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------

def split_event(event_tokens):
    if len(event_tokens) != 5:
        raise ValueError("event_tokens must contain exactly 5 items")
    return tuple(event_tokens)


def tokenize_sentence(sentence: str) -> List[str]:
    tokens = WORD_RE.findall(sentence)
    if not tokens:
        raise ValueError("sentence does not contain any word tokens")
    return [token.lower() for token in tokens]


def object_action_form(verb: str) -> str:
    verb = verb.lower()
    if verb in IRREGULAR_OBJECT_ACTIONS:
        return IRREGULAR_OBJECT_ACTIONS[verb]
    if verb.endswith("e"):
        return verb + "d"
    if len(verb) >= 3 and verb[-1] not in "aeiou" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        return verb + verb[-1] + "ed"
    return verb + "ed"


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


# ---------------------------------------------------------------------------
# 2. Part-of-speech analysis
# ---------------------------------------------------------------------------

def _build_pos_lexicons():
    rm, arm, _ = _load_language_context()
    action_vocab = importlib.import_module("world.action_vocab")
    relation_tokens = set()
    for relation in rm.relation_list:
        relation_tokens.update(relation.lower().split())
    return {
        "nouns": {noun.lower() for noun in rm.noun_list},
        "adjectives": set(arm.adjective_list) | set(ADJECTIVE_RELATION_HINTS.keys()),
        "actions": {action.lower() for action in action_vocab.action_list},
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
    tokens = tokenize_sentence(sentence)
    lexicons = _build_pos_lexicons()
    relation_match = _resolve_relation_phrase(tokens, lexicons["relations"])
    relation_span = set(range(relation_match[1], relation_match[2])) if relation_match else set()
    question_engine = importlib.import_module("prototype.question").QuestionEngine()

    tagged_tokens: List[TaggedToken] = []
    for index, token in enumerate(tokens):
        if index in relation_span:
            tagged_tokens.append(TaggedToken(text=token, pos="relation", index=index, source="relation_phrase"))
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
            )
        )

    if len(tagged_tokens) >= 2 and relation_match is None:
        tagged_tokens[1].pos = "action"
        if tagged_tokens[1].source == "default_guess":
            tagged_tokens[1].source = "position_heuristic"

    resolved_instance_context = _coerce_instance_context(
        instance_context=instance_context,
        short_memory=short_memory,
    )
    _assign_noun_instance_ids(tagged_tokens, instance_context=resolved_instance_context)
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
    }
    if short_memory is None:
        return context

    focus_entry = short_memory.get_focus_entry() if hasattr(short_memory, "get_focus_entry") else None
    if focus_entry is not None:
        context["focus_instance_id"] = focus_entry.noun_instance_id
        context["focus_noun_text"] = None if focus_entry.noun_text is None else focus_entry.noun_text.lower()

    def register(noun_text: Optional[str], instance_id: Optional[str], time_position: int, pair_index: int):
        if noun_text is None or instance_id is None:
            return
        noun_key = noun_text.lower()
        bucket = context["by_noun"].setdefault(noun_key, [])
        for existing_time, existing_pair, existing_instance_id in bucket:
            if existing_instance_id == instance_id:
                return
        bucket.append((int(time_position), int(pair_index), instance_id))

    for entry in getattr(short_memory, "short_memory_event", []):
        register(entry.noun_text, entry.noun_instance_id, entry.time_position, entry.pair_index)

    for entry in getattr(short_memory, "short_memory_relation", []):
        register(entry.source_text, entry.source_instance_id, entry.time_position, entry.pair_index)
        register(entry.target_text, entry.target_instance_id, entry.time_position, entry.pair_index)

    for noun_key, bucket in context["by_noun"].items():
        bucket.sort(key=lambda item: (item[0], item[1]))
        context["by_noun"][noun_key] = [instance_id for _, _, instance_id in bucket]

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


def _assign_noun_instance_ids(
    tagged_tokens: Sequence[TaggedToken],
    instance_context=None,
) -> None:
    noun_occurrence = 0
    sentence_assignments: Dict[str, str] = {}
    for token in tagged_tokens:
        if token.pos != "noun":
            continue
        noun_occurrence += 1
        resolved_instance_id = _resolve_existing_instance_id(
            token.text,
            instance_context,
            sentence_assignments,
        )
        if resolved_instance_id is None:
            resolved_instance_id = f"{token.text}#{token.index}:{noun_occurrence}"
        sentence_assignments.setdefault(token.text.lower(), resolved_instance_id)
        token.instance_id = resolved_instance_id


def classify_sentence_type_from_parsed(parsed: ParsedSentence) -> str:
    has_action = bool(parsed.action_tuples)
    has_noun_relation = any(
        relation_tuple.kind == "noun_noun_relation" for relation_tuple in parsed.relation_tuples
    )
    has_adj_relation = any(
        relation_tuple.kind == "adj_noun_relation" for relation_tuple in parsed.relation_tuples
    )

    if has_action and has_noun_relation:
        return "mixed_sentence"
    if has_action:
        return "action_sentence"
    if has_noun_relation:
        return "relation_sentence"
    if has_adj_relation:
        return "attribute_sentence"
    return "unknown_sentence"


def classify_sentence_type(sentence: str) -> str:
    parsed = parse_sentence(sentence)
    return parsed.sentence_type


# ---------------------------------------------------------------------------
# 3. Instance resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Information extraction by sentence pattern
# ---------------------------------------------------------------------------

def _parse_noun_phrase(tokens: Sequence[str], tags: Sequence[TaggedToken]) -> Tuple[str, List[str], Optional[str]]:
    if not tokens:
        raise ValueError("noun phrase cannot be empty")
    adjectives = [token for token, tag in zip(tokens[:-1], tags[:-1]) if tag.pos == "adj"]
    noun = tokens[-1]
    noun_instance_id = tags[-1].instance_id if tags else None
    return noun, adjectives, noun_instance_id


def _extract_pattern_action_with_object(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
) -> ParsedSentence:
    parsed = ParsedSentence(
        sentence=" ".join(tokens),
        tokens=list(tokens),
        tagged_tokens=list(tagged_tokens),
        structure=tuple(tag.pos for tag in tagged_tokens),
        pattern_name="noun_action_object_phrase",
        sentence_type="action_sentence",
    )

    subject = tokens[0]
    subject_instance_id = tagged_tokens[0].instance_id
    verb = tokens[1]
    object_tokens = list(tokens[2:])
    object_tags = list(tagged_tokens[2:])

    parsed.action_tuples.append(
        ActionTuple(
            noun=subject,
            action=verb,
            role="subject",
            position=0,
            noun_instance_id=subject_instance_id,
            source_tokens=[subject, verb],
        )
    )

    if object_tokens:
        object_noun, adjectives, object_instance_id = _parse_noun_phrase(object_tokens, object_tags)
        parsed.action_tuples.append(
            ActionTuple(
                noun=object_noun,
                action=object_action_form(verb),
                role="object",
                position=1,
                noun_instance_id=object_instance_id,
                source_tokens=object_tokens,
            )
        )

        _, arm, _ = _load_language_context()
        relation_types = _resolve_adjective_relation_overrides(
            adjectives,
            adjective_relation_types,
            arm.adj_relation_list,
        )
        for adjective, relation_type in zip(adjectives, relation_types):
            relation_name = None
            if relation_type is not None:
                relation_name = arm.adj_relation_list[int(relation_type) - 1]
            elif infer_missing:
                inferred_type = infer_adj_relation_type(object_noun, adjective)
                if inferred_type is not None:
                    relation_name = arm.adj_relation_list[int(inferred_type) - 1]
            if relation_name is None:
                continue
            parsed.relation_tuples.append(
                RelationTuple(
                    source=object_noun,
                    relation=relation_name,
                    target=adjective,
                    kind="adj_noun_relation",
                    source_instance_id=object_instance_id,
                    target_instance_id=None,
                    source_tokens=[adjective, object_noun],
                )
            )

    return parsed


def _extract_pattern_relation_between_noun_phrases(
    tokens: Sequence[str],
    tagged_tokens: Sequence[TaggedToken],
    *,
    adjective_relation_types=None,
    infer_missing: bool = True,
) -> ParsedSentence:
    rm, arm, _ = _load_language_context()
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

    left_noun, left_adjectives, left_instance_id = _parse_noun_phrase(left_tokens, left_tags)
    right_noun, right_adjectives, right_instance_id = _parse_noun_phrase(right_tokens, right_tags)

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
            source_tokens=list(tokens),
        )
    )

    all_adj_nouns = [
        (left_noun, left_adjectives, left_instance_id),
        (right_noun, right_adjectives, right_instance_id),
    ]
    for noun, adjectives, noun_instance_id in all_adj_nouns:
        relation_types = _resolve_adjective_relation_overrides(
            adjectives,
            adjective_relation_types,
            arm.adj_relation_list,
        )
        for adjective, relation_type in zip(adjectives, relation_types):
            relation_label = None
            if relation_type is not None:
                relation_label = arm.adj_relation_list[int(relation_type) - 1]
            elif infer_missing:
                inferred_type = infer_adj_relation_type(noun, adjective)
                if inferred_type is not None:
                    relation_label = arm.adj_relation_list[int(inferred_type) - 1]
            if relation_label is None:
                continue
            parsed.relation_tuples.append(
                RelationTuple(
                    source=noun,
                    relation=relation_label,
                    target=adjective,
                    kind="adj_noun_relation",
                    source_instance_id=noun_instance_id,
                    target_instance_id=None,
                    source_tokens=[adjective, noun],
                )
            )

    return parsed


def _select_extractor(tokens: Sequence[str], tagged_tokens: Sequence[TaggedToken]):
    rm, _, _ = _load_language_context()
    relation_match = _resolve_relation_phrase(tokens, [relation.lower() for relation in rm.relation_list])
    if relation_match is not None:
        return _extract_pattern_relation_between_noun_phrases
    if len(tokens) >= 2 and tagged_tokens[1].pos == "action":
        return _extract_pattern_action_with_object
    raise ValueError(f"No extraction strategy defined for structure {tuple(tag.pos for tag in tagged_tokens)}")


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
    tokens = tokenize_sentence(sentence)
    tagged_tokens = tag_tokens(
        sentence,
        instance_context=instance_context,
        short_memory=short_memory,
    )
    extractor = _select_extractor(tokens, tagged_tokens)
    parsed = extractor(
        tokens,
        tagged_tokens,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    parsed.sentence_type = classify_sentence_type_from_parsed(parsed)
    return parsed


# ---------------------------------------------------------------------------
# 5. Time-step rules
# ---------------------------------------------------------------------------

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
        add_noun_action(action_tuple.noun, action_tuple.action)

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

    relation_states = parsed_sentence_to_relation_memory_states(
        parsed=parsed,
        time_position=resolved_time_position,
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
        )
        states.append(state)
        add_noun_action(action_tuple.noun, action_tuple.action)

        short_memory.append_event(
            noun_embedding=state.noun_embedding,
            action_embedding=state.action_embedding,
            score=base_score,
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
            info_pair={
                "pair_kind": state.pair_kind,
                "noun": state.noun,
                "noun_instance_id": state.noun_instance_id,
                "action": state.action,
                "role": state.role,
                "adjectives": list(state.adjectives),
                "time_position": state.time_position,
                "pair_index": state.pair_index,
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
