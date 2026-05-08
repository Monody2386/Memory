from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


RELATION_FAMILY: Dict[str, str] = {
    "in": "location",
    "inside": "location",
    "on": "location",
    "under": "location",
    "above": "location",
    "below": "location",
    "held_by": "location",
    "near": "proximity",
    "left_of": "direction",
    "right_of": "direction",
    "front_of": "direction",
    "behind": "direction",
    "contain": "containment",
    "intersect": "overlap",
    "contain": "containment",
    "support": "support",
    "hold": "holding",
}

DUAL_RELATION: Dict[str, str] = {
    "in": "contain",
    "inside": "contain",
    "contain": "in",
    "on": "support",
    "support": "on",
    "under": "above",
    "above": "below",
    "below": "above",
    "held_by": "hold",
    "hold": "held_by",
    "near": "near",
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
    "intersect": "intersect",
}


@dataclass(frozen=True)
class SpatialFact:
    source_instance_id: str
    relation: str
    target_instance_id: str
    time_position: Optional[int] = None
    confidence: float = 1.0
    active: bool = True

    @property
    def relation_family(self) -> str:
        return RELATION_FAMILY.get(self.relation, self.relation)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_instance_id": self.source_instance_id,
            "relation": self.relation,
            "target_instance_id": self.target_instance_id,
            "relation_family": self.relation_family,
            "time_position": self.time_position,
            "confidence": self.confidence,
            "active": self.active,
        }


@dataclass
class SpatialPatch:
    add_facts: List[SpatialFact] = field(default_factory=list)
    remove_facts: List[SpatialFact] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.add_facts and not self.remove_facts


class SpaceState:
    def __init__(self) -> None:
        self.facts: Set[SpatialFact] = set()
        self.out_edges_by_source: Dict[str, Set[SpatialFact]] = {}
        self.in_edges_by_target: Dict[str, Set[SpatialFact]] = {}
        self.edges_by_relation: Dict[str, Set[SpatialFact]] = {}

    def __len__(self) -> int:
        return len(self.facts)

    def clear(self) -> None:
        self.facts.clear()
        self.out_edges_by_source.clear()
        self.in_edges_by_target.clear()
        self.edges_by_relation.clear()

    def _register_fact(self, fact: SpatialFact) -> None:
        self.out_edges_by_source.setdefault(fact.source_instance_id, set()).add(fact)
        self.in_edges_by_target.setdefault(fact.target_instance_id, set()).add(fact)
        self.edges_by_relation.setdefault(fact.relation, set()).add(fact)

    def _unregister_fact(self, fact: SpatialFact) -> None:
        outgoing = self.out_edges_by_source.get(fact.source_instance_id)
        if outgoing is not None:
            outgoing.discard(fact)
            if not outgoing:
                self.out_edges_by_source.pop(fact.source_instance_id, None)
        incoming = self.in_edges_by_target.get(fact.target_instance_id)
        if incoming is not None:
            incoming.discard(fact)
            if not incoming:
                self.in_edges_by_target.pop(fact.target_instance_id, None)
        related = self.edges_by_relation.get(fact.relation)
        if related is not None:
            related.discard(fact)
            if not related:
                self.edges_by_relation.pop(fact.relation, None)

    def _dual_fact(self, fact: SpatialFact) -> Optional[SpatialFact]:
        dual_relation = DUAL_RELATION.get(fact.relation)
        if dual_relation is None:
            return None
        return SpatialFact(
            source_instance_id=fact.target_instance_id,
            relation=dual_relation,
            target_instance_id=fact.source_instance_id,
            time_position=fact.time_position,
            confidence=fact.confidence,
            active=fact.active,
        )

    def _add_fact_single(self, fact: SpatialFact) -> None:
        if fact in self.facts:
            return
        self.facts.add(fact)
        self._register_fact(fact)

    def _remove_fact_single(self, fact: SpatialFact) -> None:
        if fact not in self.facts:
            return
        self.facts.remove(fact)
        self._unregister_fact(fact)

    def add_fact(self, fact: SpatialFact, replace_family: bool = False, include_dual: bool = True) -> None:
        if replace_family:
            self.clear_instance_relations(fact.source_instance_id, family=fact.relation_family)
        self._add_fact_single(fact)
        if include_dual:
            dual_fact = self._dual_fact(fact)
            if dual_fact is not None:
                self._add_fact_single(dual_fact)

    def remove_fact(self, fact: SpatialFact, include_dual: bool = True) -> None:
        dual_fact = self._dual_fact(fact) if include_dual else None
        self._remove_fact_single(fact)
        if dual_fact is not None:
            self._remove_fact_single(dual_fact)

    def apply_patch(self, patch: SpatialPatch, replace_families: bool = False) -> None:
        for fact in patch.remove_facts:
            self.remove_fact(fact)
        for fact in patch.add_facts:
            self.add_fact(fact, replace_family=replace_families)

    def clear_instance_relations(self, instance_id: str, family: Optional[str] = None) -> List[SpatialFact]:
        outgoing = list(self.out_edges_by_source.get(instance_id, set()))
        if family is not None:
            outgoing = [fact for fact in outgoing if fact.relation_family == family]
        for fact in outgoing:
            self.remove_fact(fact)
        return outgoing

    def get_outgoing(self, instance_id: str, relation: Optional[str] = None) -> List[SpatialFact]:
        facts = list(self.out_edges_by_source.get(instance_id, set()))
        if relation is not None:
            facts = [fact for fact in facts if fact.relation == relation]
        return sorted(facts, key=lambda fact: (fact.relation, fact.target_instance_id))

    def get_incoming(self, instance_id: str, relation: Optional[str] = None) -> List[SpatialFact]:
        facts = list(self.in_edges_by_target.get(instance_id, set()))
        if relation is not None:
            facts = [fact for fact in facts if fact.relation == relation]
        return sorted(facts, key=lambda fact: (fact.relation, fact.source_instance_id))

    def get_neighbors(self, instance_id: str) -> List[str]:
        neighbors = {fact.target_instance_id for fact in self.out_edges_by_source.get(instance_id, set())}
        neighbors.update(fact.source_instance_id for fact in self.in_edges_by_target.get(instance_id, set()))
        return sorted(neighbors)

    def has_relation(self, source_instance_id: str, relation: str, target_instance_id: str) -> bool:
        probe = SpatialFact(source_instance_id, relation, target_instance_id)
        return probe in self.facts

    def referenced_instance_ids(self) -> Set[str]:
        referenced: Set[str] = set()
        for fact in self.facts:
            referenced.add(fact.source_instance_id)
            referenced.add(fact.target_instance_id)
        return referenced

    def get_container(self, instance_id: str) -> Optional[str]:
        for relation in ("in", "inside"):
            facts = self.get_outgoing(instance_id, relation=relation)
            if facts:
                return facts[0].target_instance_id
        return None

    def get_support(self, instance_id: str) -> Optional[str]:
        facts = self.get_outgoing(instance_id, relation="on")
        if facts:
            return facts[0].target_instance_id
        return None

    def get_region(self, instance_id: str, max_hops: int = 4) -> Optional[str]:
        current = instance_id
        visited = {instance_id}
        for _ in range(max_hops):
            container = self.get_container(current)
            if container is None or container in visited:
                return container
            visited.add(container)
            current = container
        return current if current != instance_id else None

    def get_nearby(self, instance_id: str) -> List[str]:
        nearby = []
        for fact in self.get_outgoing(instance_id, relation="near"):
            nearby.append(fact.target_instance_id)
        for fact in self.get_incoming(instance_id, relation="near"):
            nearby.append(fact.source_instance_id)
        return sorted(set(nearby))

    def build_summary(self, focus_instance_id: str) -> Dict[str, object]:
        directional = {
            relation: [fact.target_instance_id for fact in self.get_outgoing(focus_instance_id, relation=relation)]
            for relation in ("left_of", "right_of", "front_of", "behind")
        }
        container = self.get_container(focus_instance_id)
        support = self.get_support(focus_instance_id)
        nearby = self.get_nearby(focus_instance_id)
        held_by = None
        held_facts = self.get_outgoing(focus_instance_id, relation="held_by")
        if held_facts:
            held_by = held_facts[0].target_instance_id
        return {
            "focus_instance_id": focus_instance_id,
            "container": container,
            "support": support,
            "region": self.get_region(focus_instance_id),
            "nearby": nearby,
            "directional": directional,
            "held_by": held_by,
            "flags": {
                "is_held": int(held_by is not None),
                "is_in_container": int(container is not None),
                "has_support": int(support is not None),
                "has_nearby": int(bool(nearby)),
            },
        }

    def content_view(self) -> List[Dict[str, object]]:
        return [
            fact.to_dict()
            for fact in sorted(
                self.facts,
                key=lambda fact: (
                    -int(fact.active),
                    fact.time_position if fact.time_position is not None else -1,
                    fact.source_instance_id,
                    fact.relation,
                    fact.target_instance_id,
                ),
            )
        ]
