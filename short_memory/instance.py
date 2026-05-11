from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class MemoryInstance:
    instance_id: str
    noun_text: Optional[str] = None
    noun_type: Optional[int] = None
    embedding: Optional[torch.Tensor] = None
    entity_kind: str = "unknown"
    gender: str = "unknown"
    owner_instance_id: Optional[str] = None
    owner_role: Optional[str] = None
    instance_scope: str = "scene"
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)
    rewards: list[Any] = field(default_factory=list)
    surprises: list[Any] = field(default_factory=list)
    score: float = 1.0
    created_at: int = 0
    last_seen_at: int = 0

    def update_embedding(self, embedding: torch.Tensor) -> None:
        self.embedding = embedding.detach().clone()

    def add_event(self, event: Any) -> None:
        self.events.append(event)
        self._touch_from_entry(event)

    def add_relation(self, relation: Any) -> None:
        existing_index = self._find_entry_index(self.relations, relation)
        if existing_index is not None:
            existing = self.relations[existing_index]
            if getattr(relation, "info_pair", None) and not getattr(existing, "info_pair", None):
                self.relations[existing_index] = relation
                self._touch_from_entry(relation)
            return
        self.relations.append(relation)
        self._touch_from_entry(relation)

    def add_reward(self, reward: Any) -> None:
        self.rewards.append(reward)
        self._touch_from_entry(reward)

    def add_surprise(self, surprise: Any) -> None:
        self.surprises.append(surprise)
        self._touch_from_entry(surprise)

    def update_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is not None:
                self.metadata[key] = value
        self._sync_fields_from_metadata()

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata = dict(metadata)
        self._sync_fields_from_metadata()

    def boost_score(self, amount: float) -> None:
        self.score += float(amount)

    def set_score_floor(self, score: float) -> None:
        self.score = max(self.score, float(score))

    def embedding_norm(self) -> Optional[float]:
        if self.embedding is None:
            return None
        return float(self.embedding.norm().item())

    def summary(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "noun_text": self.noun_text,
            "noun_type": self.noun_type,
            "entity_kind": self.entity_kind,
            "gender": self.gender,
            "owner_instance_id": self.owner_instance_id,
            "owner_role": self.owner_role,
            "instance_scope": self.instance_scope,
            "embedding_norm": self.embedding_norm(),
            "event_count": len(self.events),
            "relation_count": len(self.relations),
            "reward_count": len(self.rewards),
            "surprise_count": len(self.surprises),
            "score": float(self.score),
            "created_at": int(self.created_at),
            "last_seen_at": int(self.last_seen_at),
            "metadata": dict(self.metadata),
        }

    def _touch_from_entry(self, entry: Any) -> None:
        time_position = getattr(entry, "time_position", None)
        if time_position is not None:
            self.last_seen_at = max(self.last_seen_at, int(time_position))

    def _has_entry(self, entries: list[Any], candidate: Any) -> bool:
        return self._find_entry_index(entries, candidate) is not None

    def _find_entry_index(self, entries: list[Any], candidate: Any) -> Optional[int]:
        candidate_key = self._entry_key(candidate)
        for index, entry in enumerate(entries):
            if self._entry_key(entry) == candidate_key:
                return index
        return None

    def _entry_key(self, entry: Any) -> Any:
        info_pair = getattr(entry, "info_pair", None)
        info = dict(info_pair) if info_pair else {}
        relation = (
            getattr(entry, "relation", None)
            or getattr(entry, "relation_name", None)
            or info.get("relation")
            or info.get("relation_name")
        )
        relation_kind = (
            getattr(entry, "kind", None)
            or getattr(entry, "relation_kind", None)
            or info.get("kind")
            or info.get("pair_kind")
        )
        if relation is not None or relation_kind is not None:
            return (
                "relation",
                relation_kind,
                relation,
                getattr(entry, "source", None) or getattr(entry, "source_text", None) or info.get("source") or info.get("source_text"),
                getattr(entry, "target", None) or getattr(entry, "target_text", None) or info.get("target") or info.get("target_text"),
                getattr(entry, "source_instance_id", None) or info.get("source_instance_id"),
                getattr(entry, "target_instance_id", None) or info.get("target_instance_id"),
                getattr(entry, "polarity", None) or info.get("polarity"),
                getattr(entry, "question_label", None) or info.get("question_label"),
            )
        if info_pair:
            return tuple(sorted(info.items()))
        fields = (
            "kind",
            "relation_kind",
            "relation",
            "relation_name",
            "source",
            "source_text",
            "target",
            "target_text",
            "source_instance_id",
            "target_instance_id",
            "polarity",
            "question_label",
        )
        return tuple((field, getattr(entry, field, None)) for field in fields)

    def _sync_fields_from_metadata(self) -> None:
        if self.metadata.get("noun_text") is not None:
            self.noun_text = str(self.metadata["noun_text"]).lower()
        self.entity_kind = str(self.metadata.get("entity_kind", self.entity_kind))
        self.gender = str(self.metadata.get("gender", self.gender))
        self.owner_instance_id = self.metadata.get("owner_instance_id", self.owner_instance_id)
        self.owner_role = self.metadata.get("owner_role", self.owner_role)
        self.instance_scope = str(self.metadata.get("instance_scope", self.instance_scope))
