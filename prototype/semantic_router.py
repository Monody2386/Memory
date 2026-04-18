"""Semantic routing for sentence processing.

This module keeps the responsibilities of the knowledge layer and the world
model separate:

- ``noun_noun`` relations update symbolic relation memory and can optionally
  trigger knowledge-map training.
- ``adj_noun`` relations update adjective memory and can optionally trigger
  adjective-map training.
- ``noun_action`` pairs update ``noun_action_map`` and can optionally be turned
  into short-memory states for world-model training or inference.

The goal is to provide one explicit entry point for sentence handling without
forcing every sentence through every subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from knowledge import adj_relation_map as arm
from knowledge import relation_map as rm
from knowledge.training import apply_language_training_samples
from grammar_layer import (
    AdjNounRelationSample,
    KnowledgeTrainingSamples,
    NounActionPair,
    NounNounRelationSample,
    ShortMemoryState,
    sentence_to_knowledge_samples,
    sentence_to_noun_action_pairs,
    append_sentence_to_short_memory,
    sentence_to_short_memory_states,
)

RelationOverride = Union[int, str]


@dataclass
class RoutedSentence:
    sentence: str
    noun_noun_relations: List[NounNounRelationSample] = field(default_factory=list)
    adj_noun_relations: List[AdjNounRelationSample] = field(default_factory=list)
    noun_action_pairs: List[NounActionPair] = field(default_factory=list)

    def as_training_samples(self) -> KnowledgeTrainingSamples:
        return KnowledgeTrainingSamples(
            noun_noun_samples=list(self.noun_noun_relations),
            adj_noun_samples=list(self.adj_noun_relations),
        )


@dataclass
class ProcessingResult:
    routed: RoutedSentence
    relation_memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    adj_memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    noun_action_updates: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_training_results: Optional[Dict[str, Any]] = None
    short_memory_states: List[ShortMemoryState] = field(default_factory=list)


def route_sentence(
    sentence: str,
    *,
    noun_relation_type: Optional[RelationOverride] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
) -> RoutedSentence:
    """Parse one sentence into relation- and action-oriented units."""
    knowledge_samples = sentence_to_knowledge_samples(
        sentence,
        noun_relation_type=noun_relation_type,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    noun_action_pairs = sentence_to_noun_action_pairs(
        sentence,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    return RoutedSentence(
        sentence=sentence,
        noun_noun_relations=list(knowledge_samples.noun_noun_samples),
        adj_noun_relations=list(knowledge_samples.adj_noun_samples),
        noun_action_pairs=noun_action_pairs,
    )


def _store_noun_noun_relations(samples: Sequence[NounNounRelationSample]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for sample in samples:
        created, source_idx, target_idx, relation_type = rm.add_relation_by_type(
            sample.source_noun,
            sample.target_noun,
            sample.relation_type,
        )
        relation_name = (
            rm.relation_list[int(relation_type) - 1]
            if int(relation_type) - 1 < len(rm.relation_list)
            else f"relation_{int(relation_type)}"
        )
        updates.append(
            {
                "created": bool(created),
                "source_noun": sample.source_noun,
                "target_noun": sample.target_noun,
                "source_idx": int(source_idx),
                "target_idx": int(target_idx),
                "relation_type": int(relation_type),
                "relation_name": relation_name,
            }
        )
    return updates


def _store_adj_noun_relations(samples: Sequence[AdjNounRelationSample]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for sample in samples:
        created, noun_idx, adjective_idx, relation_type = arm.add_adj_relation_by_type(
            sample.noun,
            sample.adjective,
            sample.relation_type,
        )
        relation_name = (
            arm.adj_relation_list[int(relation_type) - 1]
            if int(relation_type) - 1 < len(arm.adj_relation_list)
            else f"relation_{int(relation_type)}"
        )
        updates.append(
            {
                "created": bool(created),
                "noun": sample.noun,
                "adjective": sample.adjective,
                "noun_idx": int(noun_idx),
                "adjective_idx": int(adjective_idx),
                "relation_type": int(relation_type),
                "relation_name": relation_name,
            }
        )
    return updates


def _store_noun_action_pairs(pairs: Sequence[NounActionPair]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for pair in pairs:
        updates.append(
            {
                "noun": pair.noun,
                "action": pair.action,
                "role": pair.role,
                "value": 0,
            }
        )
    return updates


def process_sentence(
    sentence: str,
    *,
    world_model=None,
    short_memory=None,
    noun_relation_type: Optional[RelationOverride] = None,
    adjective_relation_types: Optional[
        Union[Sequence[RelationOverride], Mapping[str, RelationOverride]]
    ] = None,
    infer_missing: bool = True,
    store_noun_noun: bool = False,
    store_adj_noun: bool = False,
    store_noun_action: bool = True,
    train_knowledge: bool = False,
    save_knowledge: bool = False,
    build_short_memory: bool = False,
    time_position: int = 0,
    base_score: float = 1.0,
) -> ProcessingResult:
    """Route one sentence and optionally apply each routed unit.

    Recommended usage patterns:

    - Relation sentence: ``store_noun_noun=True`` and optionally
      ``train_knowledge=True``.
    - Attribute sentence: ``store_adj_noun=True`` and optionally
      ``train_knowledge=True``.
    - Action sentence: ``store_noun_action=True`` and
      ``build_short_memory=True`` with a provided ``world_model``.
    """
    routed = route_sentence(
        sentence,
        noun_relation_type=noun_relation_type,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    result = ProcessingResult(routed=routed)

    if store_noun_noun:
        result.relation_memory_updates = _store_noun_noun_relations(
            routed.noun_noun_relations
        )

    if store_adj_noun:
        result.adj_memory_updates = _store_adj_noun_relations(
            routed.adj_noun_relations
        )

    if store_noun_action:
        result.noun_action_updates = _store_noun_action_pairs(routed.noun_action_pairs)

    if train_knowledge:
        result.knowledge_training_results = apply_language_training_samples(
            routed.as_training_samples(),
            save=save_knowledge,
        )

    if build_short_memory:
        if world_model is None:
            raise ValueError("world_model is required when build_short_memory=True")
        if short_memory is None:
            raise ValueError("short_memory is required when build_short_memory=True")
        result.short_memory_states = append_sentence_to_short_memory(
            sentence=sentence,
            short_memory=short_memory,
            world_model=world_model,
            time_position=time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
        )

    return result


__all__ = [
    "ProcessingResult",
    "RoutedSentence",
    "process_sentence",
    "route_sentence",
]

