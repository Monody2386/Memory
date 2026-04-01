"""Grammar utilities for generating noun-action training pairs.

This module provides a small rule-based path from plain sentences to the
noun-action pairs used by short memory and world-model training.
It also supports adjective-conditioned noun embeddings via the shared adj_map.
"""

from dataclasses import dataclass, field
import importlib
import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from world.action_vocab import ensure_action


@dataclass
class NounActionPair:
    noun: str
    action: str
    role: str
    position: int
    noun_index: Optional[int] = None
    noun_embedding: Optional[torch.Tensor] = None
    adjectives: Optional[List[str]] = None


@dataclass
class ShortMemoryState:
    noun: str
    action: str
    noun_index: Optional[int]
    action_type: int
    noun_embedding: torch.Tensor
    action_embedding: torch.Tensor
    time_position: int
    pair_index: int
    role: str
    adjectives: List[str]


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


def split_event(event_tokens):
    if len(event_tokens) != 5:
        raise ValueError("event_tokens must contain exactly 5 items")
    return tuple(event_tokens)


def tokenize_sentence(sentence: str) -> List[str]:
    tokens = WORD_RE.findall(sentence)
    if not tokens:
        raise ValueError("sentence does not contain any word tokens")
    return tokens


def object_action_form(verb: str) -> str:
    verb = verb.lower()
    if verb in IRREGULAR_OBJECT_ACTIONS:
        return IRREGULAR_OBJECT_ACTIONS[verb]
    if verb.endswith("e"):
        return verb + "d"
    if len(verb) >= 3 and verb[-1] not in "aeiou" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        return verb + verb[-1] + "ed"
    return verb + "ed"


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
        resolved = []
        for adjective in adjectives:
            resolved.append(
                _normalize_relation_override(
                    overrides.get(adjective.lower()),
                    relation_names,
                    relation_label="adjective relation",
                )
            )
        return resolved

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


def sentence_to_knowledge_samples(
    sentence: str,
    noun_relation_type: Optional[RelationOverride] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = False,
) -> KnowledgeTrainingSamples:
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 2:
        raise ValueError("sentence must contain at least a noun and a verb")

    rm, arm, _ = _load_language_context()
    subject = tokens[0].lower()
    object_tokens = tokens[2:]
    samples = KnowledgeTrainingSamples()

    if object_tokens:
        object_noun, adjectives = _parse_object_phrase(object_tokens)
        object_noun = object_noun.lower()
        subject_idx = rm._ensure_noun(subject)
        object_idx = rm._ensure_noun(object_noun)

        noun_relation_type_idx = _normalize_relation_override(
            noun_relation_type,
            rm.relation_list,
            relation_label="noun relation",
        )
        if noun_relation_type_idx is None and infer_missing:
            noun_relation_type_idx = 1
        if noun_relation_type_idx is not None:
            samples.noun_noun_samples.append(
                NounNounRelationSample(
                    source_noun=subject,
                    target_noun=object_noun,
                    relation_type=noun_relation_type_idx,
                    source_idx=subject_idx,
                    target_idx=object_idx,
                )
            )

        adj_relation_types = _resolve_adjective_relation_overrides(
            adjectives,
            adjective_relation_types,
            arm.adj_relation_list,
        )
        for adjective, relation_type in zip(adjectives, adj_relation_types):
            adjective = adjective.lower()
            if relation_type is None and infer_missing:
                relation_type = infer_adj_relation_type(object_noun, adjective)
            if relation_type is None:
                continue
            if adjective not in arm.adjective_list:
                arm.adjective_list.append(adjective)
            adjective_idx = arm.adjective_list.index(adjective)
            samples.adj_noun_samples.append(
                AdjNounRelationSample(
                    noun=object_noun,
                    adjective=adjective,
                    relation_type=int(relation_type),
                    noun_idx=object_idx,
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


def _parse_object_phrase(tokens: Sequence[str]) -> Tuple[str, List[str]]:
    if not tokens:
        return "", []
    if len(tokens) == 1:
        return tokens[0], []
    return tokens[-1], [token.lower() for token in tokens[:-1]]


def sentence_to_noun_action_pairs(
    sentence: str,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
) -> List[NounActionPair]:
    tokens = tokenize_sentence(sentence)
    if len(tokens) < 2:
        raise ValueError("sentence must contain at least a noun and a verb")

    _, arm, _ = _load_language_context()
    subject = tokens[0]
    verb = tokens[1]
    object_tokens = tokens[2:]

    subject_action = verb.lower()
    ensure_action(subject_action)

    subject_idx, subject_embedding = _ensure_noun_embedding(subject)

    pairs = [
        NounActionPair(
            noun=subject,
            action=subject_action,
            role="subject",
            position=0,
            noun_index=subject_idx,
            noun_embedding=subject_embedding,
            adjectives=[],
        )
    ]

    if object_tokens:
        object_noun, adjectives = _parse_object_phrase(object_tokens)
        object_action = object_action_form(verb)
        ensure_action(object_action)
        relation_types = _resolve_adjective_relation_overrides(
            adjectives,
            adjective_relation_types,
            arm.adj_relation_list,
        )

        if adjectives:
            current_embedding = None
            object_idx = None
            for adjective, relation_type in zip(adjectives, relation_types):
                if relation_type is None and infer_missing:
                    relation_type = infer_adj_relation_type(object_noun, adjective)
                if relation_type is None:
                    continue
                object_idx, current_embedding = inject_adjective_into_noun_embedding(
                    noun=object_noun,
                    adjective=adjective,
                    relation_type=relation_type,
                )
            if current_embedding is None:
                object_idx, object_embedding = _ensure_noun_embedding(object_noun)
            else:
                object_embedding = current_embedding
        else:
            object_idx, object_embedding = _ensure_noun_embedding(object_noun)

        pairs.append(
            NounActionPair(
                noun=object_noun,
                action=object_action,
                role="object",
                position=1,
                noun_index=object_idx,
                noun_embedding=object_embedding,
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


def sentence_to_short_memory_states(
    sentence: str,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    time_position: int = 0,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
) -> List[ShortMemoryState]:
    pairs = sentence_to_noun_action_pairs(
        sentence,
        adjective_relation_types=adjective_relation_types,
    )
    return noun_action_pairs_to_short_memory_states(
        pairs=pairs,
        world_model=world_model,
        action_type_map=action_type_map,
        time_position=time_position,
    )


def append_sentence_to_short_memory(
    sentence: str,
    short_memory,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    time_position: int = 0,
    base_score: float = 1.0,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
):
    states = sentence_to_short_memory_states(
        sentence=sentence,
        world_model=world_model,
        action_type_map=action_type_map,
        time_position=time_position,
        adjective_relation_types=adjective_relation_types,
    )

    for state in states:
        short_memory.append_state(
            noun_embedding=state.noun_embedding,
            action_embedding=state.action_embedding,
            score=base_score,
            noun_type=state.noun_index,
            action_type=state.action_type,
            time_position=state.time_position,
            pair_index=state.pair_index,
        )

    return states


def sentences_to_short_memory(
    sentences: Sequence[str],
    short_memory,
    world_model,
    action_type_map: Optional[Dict[str, int]] = None,
    start_time_position: int = 0,
    base_score: float = 1.0,
    adjective_relation_types: Optional[Sequence[Optional[Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]]]] = None,
):
    all_states = []
    adjective_relation_types = adjective_relation_types or [None] * len(sentences)
    if len(adjective_relation_types) != len(sentences):
        raise ValueError("adjective_relation_types must align with sentences")

    for offset, (sentence, relation_overrides) in enumerate(zip(sentences, adjective_relation_types)):
        states = append_sentence_to_short_memory(
            sentence=sentence,
            short_memory=short_memory,
            world_model=world_model,
            action_type_map=action_type_map,
            time_position=start_time_position + offset,
            base_score=base_score,
            adjective_relation_types=relation_overrides,
        )
        all_states.append(states)
    return all_states
