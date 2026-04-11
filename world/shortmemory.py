

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from prototype.instance_metadata import default_entity_kind, default_gender, normalize_instance_scope

import torch
import torch.nn.functional as F

from knowledge.relation_map import noun_dim


@dataclass
class EventMemoryEntry:
    score: float
    noun_type: Optional[int]
    action_type: Optional[int]
    time_position: int
    pair_index: int
    event_index: Optional[int]
    noun_instance_id: Optional[str]
    action_instance_id: Optional[str]
    noun_text: Optional[str] = None
    action_text: Optional[str] = None
    role: Optional[str] = None
    polarity: int = 1
    accept_label: str = "none"
    diff_value: Any = "none"
    question_label: str = "none"
    sentence_label: str = "none"
    pair_kind: str = "noun_action"
    adjectives: List[str] = field(default_factory=list)
    info_pair: Dict[str, Any] = field(default_factory=dict)

    @property
    def instance_id(self) -> Optional[str]:
        return self.noun_instance_id


@dataclass
class RelationMemoryEntry:
    score: float
    time_position: int
    pair_index: int
    relation_kind: str
    relation_name: str
    source_text: str
    target_text: str
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    polarity: int = 1
    accept_label: str = "none"
    diff_value: Any = "none"
    question_label: str = "none"
    sentence_label: str = "none"
    source_type: Optional[int] = None
    target_type: Optional[int] = None
    info_pair: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardMemoryEntry:
    score: float
    reward_word: str
    reward_value: float
    time_position: int
    pair_index: int
    subject_text: str
    subject_instance_id: str
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    object_instance_id: Optional[str] = None
    polarity: int = 1
    accept_label: str = "none"
    diff_value: Any = "none"
    question_label: str = "none"
    sentence_label: str = "none"
    info_pair: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurpriseMemoryEntry:
    score: float
    surprise_word: str
    surprise_value: float
    time_position: int
    pair_index: int
    subject_text: str
    subject_instance_id: str
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    object_instance_id: Optional[str] = None
    polarity: int = 1
    accept_label: str = "none"
    diff_value: Any = "none"
    question_label: str = "none"
    sentence_label: str = "none"
    info_pair: Dict[str, Any] = field(default_factory=dict)


MemoryEntry = EventMemoryEntry


class ShortMemory:
    def __init__(
        self,
        maxlen=100,
        device="cpu",
        state_dim=None,
        relation_update_mode: str = "average",
        relation_update_frequency: str = "per_relation",
        relation_step_scale: float = 0.1,
        relation_clone_update_mode: str = "average",
        relation_clone_update_frequency: str = "per_relation",
        relation_clone_step_scale: float = 0.1,
    ):
        self.maxlen = maxlen
        self.device = device
        self.state_dim = state_dim
        self.relation_update_mode = relation_update_mode
        self.relation_update_frequency = relation_update_frequency
        self.relation_step_scale = float(relation_step_scale)
        self.relation_clone_update_mode = relation_clone_update_mode
        self.relation_clone_update_frequency = relation_clone_update_frequency
        self.relation_clone_step_scale = float(relation_clone_step_scale)
        self.event_entries: List[EventMemoryEntry] = []
        self.relation_entries: List[RelationMemoryEntry] = []
        self.reward_entries: List[RewardMemoryEntry] = []
        self.surprise_entries: List[SurpriseMemoryEntry] = []
        self.noun_instance_memory: Dict[str, torch.Tensor] = {}
        self.noun_instance_metadata: Dict[str, Dict[str, Any]] = {}
        self.action_instance_memory: Dict[str, torch.Tensor] = {}
        self.noun_relation_memory: Dict[str, torch.Tensor] = {}
        self.adj_relation_memory: Dict[str, torch.Tensor] = {}
        self._insert_counter = 0
        self._event_counter = 0
        self._sentence_counter = 0

    def next_sentence_label(self) -> str:
        sentence_label = f"sentence:{self._sentence_counter}"
        self._sentence_counter += 1
        return sentence_label

    def next_event_index(self) -> int:
        event_index = int(self._event_counter)
        self._event_counter += 1
        return event_index

    @property
    def entries(self) -> List[EventMemoryEntry]:
        return self.event_entries

    @property
    def short_memory_event(self) -> List[EventMemoryEntry]:
        return self.event_entries

    @property
    def short_memory_relation(self) -> List[RelationMemoryEntry]:
        return self.relation_entries

    @property
    def short_memory_reward(self) -> List[RewardMemoryEntry]:
        return self.reward_entries

    @property
    def short_memory_surprise(self) -> List[SurpriseMemoryEntry]:
        return self.surprise_entries

    def reward_list(self) -> List[RewardMemoryEntry]:
        return list(self.reward_entries)

    def surprise_list(self) -> List[SurpriseMemoryEntry]:
        return list(self.surprise_entries)

    def _resolve_state_dim(self, noun_embedding: torch.Tensor, action_embedding: torch.Tensor) -> int:
        candidate_dim = noun_embedding.view(-1).numel() + action_embedding.view(-1).numel()
        if self.state_dim is None:
            self.state_dim = candidate_dim
        return self.state_dim

    def _sinusoidal_position_encoding(self, time_position: int, pair_index: int) -> torch.Tensor:
        if self.state_dim is None:
            raise ValueError("state_dim is unknown; append at least one state first")
        position_value = float(time_position) * 1000.0 + float(pair_index)
        encoding = torch.zeros(self.state_dim, device=self.device, dtype=torch.float32)
        div_term = torch.exp(
            torch.arange(0, self.state_dim, 2, device=self.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=self.device)) / self.state_dim)
        )
        encoding[0::2] = torch.sin(position_value * div_term)
        encoding[1::2] = torch.cos(position_value * div_term[: encoding[1::2].shape[0]])
        return encoding

    def _compose_state(self, noun_embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        noun_embedding = noun_embedding.to(self.device).view(-1)
        action_embedding = action_embedding.to(self.device).view(-1)
        expected_dim = self._resolve_state_dim(noun_embedding, action_embedding)
        state_tensor = torch.cat([noun_embedding, action_embedding], dim=0)
        if state_tensor.numel() != expected_dim:
            raise ValueError(
                f"state tensor must have {expected_dim} values, got {state_tensor.numel()}"
            )
        return state_tensor

    def _encode_entry(self, entry: EventMemoryEntry) -> torch.Tensor:
        noun_embedding = self.get_noun_embedding(entry.noun_instance_id)
        action_embedding = self.get_action_embedding(entry.action_instance_id)
        if noun_embedding is None or action_embedding is None:
            raise ValueError("missing noun/action instance embedding for event entry")
        state_tensor = self._compose_state(noun_embedding, action_embedding)
        return state_tensor + self._sinusoidal_position_encoding(entry.time_position, entry.pair_index)

    def _sort_event_entries(self):
        self.event_entries.sort(key=lambda entry: (entry.time_position, entry.pair_index))

    def _reorder_event_entries_for_world_model(self):
        reordered = []
        current_time = None
        current_group = []
        for entry in self.event_entries:
            if current_time is None or entry.time_position == current_time:
                current_group.append(entry)
                current_time = entry.time_position
                continue
            reordered.extend(sorted(current_group, key=lambda item: (item.score, item.pair_index)))
            current_group = [entry]
            current_time = entry.time_position

        if current_group:
            reordered.extend(sorted(current_group, key=lambda item: (item.score, item.pair_index)))

        self.event_entries = reordered

    def _sort_relation_entries(self):
        self.relation_entries.sort(key=lambda entry: (entry.time_position, entry.pair_index))

    def _event_entries_by_attention(self) -> List[EventMemoryEntry]:
        return sorted(
            self.event_entries,
            key=lambda entry: (-entry.score, entry.time_position, entry.pair_index),
        )

    def _relation_entries_by_attention(self) -> List[RelationMemoryEntry]:
        return sorted(
            self.relation_entries,
            key=lambda entry: (-entry.score, entry.time_position, entry.pair_index),
        )

    def _prune_instance_stores(self):
        referenced_nouns = {
            entry.noun_instance_id for entry in self.event_entries if entry.noun_instance_id is not None
        }
        referenced_nouns.update(
            entry.source_instance_id for entry in self.relation_entries if entry.source_instance_id is not None
        )
        referenced_nouns.update(
            entry.target_instance_id for entry in self.relation_entries if entry.target_instance_id is not None
        )
        referenced_nouns.update(
            entry.subject_instance_id for entry in self.reward_entries if entry.subject_instance_id is not None
        )
        referenced_nouns.update(
            entry.object_instance_id for entry in self.reward_entries if entry.object_instance_id is not None
        )
        referenced_nouns.update(
            entry.subject_instance_id for entry in self.surprise_entries if entry.subject_instance_id is not None
        )
        referenced_nouns.update(
            entry.object_instance_id for entry in self.surprise_entries if entry.object_instance_id is not None
        )
        referenced_actions = {
            entry.action_instance_id for entry in self.event_entries if entry.action_instance_id is not None
        }
        self.noun_instance_memory = {
            key: value for key, value in self.noun_instance_memory.items() if key in referenced_nouns
        }
        self.noun_instance_metadata = {
            key: value for key, value in self.noun_instance_metadata.items() if key in referenced_nouns
        }
        self.action_instance_memory = {
            key: value for key, value in self.action_instance_memory.items() if key in referenced_actions
        }

    def _trim(self):
        while len(self.event_entries) > self.maxlen:
            self.event_entries.pop(0)
        while len(self.relation_entries) > self.maxlen:
            self.relation_entries.pop(0)
        while len(self.reward_entries) > self.maxlen:
            self.reward_entries.pop(0)
        while len(self.surprise_entries) > self.maxlen:
            self.surprise_entries.pop(0)
        self._prune_instance_stores()

    def _default_noun_instance_id(self, noun_type=None) -> str:
        noun_label = "noun" if noun_type is None else f"noun{int(noun_type)}"
        return f"{noun_label}@{self._insert_counter}"

    def _default_action_instance_id(self, action_type=None, time_position: int = 0, pair_index: int = 0) -> str:
        action_label = "action" if action_type is None else f"action{int(action_type)}"
        return f"{action_label}@t{int(time_position)}:p{int(pair_index)}:{self._insert_counter}"

    def store_noun_instance(
        self,
        instance_id: str,
        noun_embedding: torch.Tensor,
        noun_text: Optional[str] = None,
        instance_scope: Optional[str] = None,
    ) -> None:
        self.noun_instance_memory[instance_id] = noun_embedding.to(self.device).view(-1).detach().clone()
        self.ensure_noun_instance_metadata(instance_id, noun_text=noun_text, instance_scope=instance_scope)

    def _default_instance_metadata(self, noun_text: Optional[str], instance_scope: Optional[str] = None) -> Dict[str, Any]:
        noun_key = None if noun_text is None else noun_text.lower()
        return {
            "noun_text": noun_key,
            "entity_kind": default_entity_kind(noun_key),
            "gender": default_gender(noun_key),
            "owner_instance_id": None,
            "owner_role": None,
            "instance_scope": normalize_instance_scope(instance_scope),
            "attribute_polarity": {},
        }

    def ensure_noun_instance_metadata(
        self,
        instance_id: str,
        noun_text: Optional[str] = None,
        instance_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = self.noun_instance_metadata.get(instance_id)
        if existing is None:
            existing = self._default_instance_metadata(noun_text, instance_scope=instance_scope)
            self.noun_instance_metadata[instance_id] = dict(existing)
        else:
            if noun_text is not None and not existing.get("noun_text"):
                existing["noun_text"] = noun_text.lower()
            if instance_scope is not None and not existing.get("instance_scope"):
                existing["instance_scope"] = normalize_instance_scope(instance_scope)
            self.noun_instance_metadata[instance_id] = dict(existing)
        return dict(self.noun_instance_metadata[instance_id])

    def get_noun_instance_metadata(self, instance_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if instance_id is None:
            return None
        metadata = self.noun_instance_metadata.get(instance_id)
        return None if metadata is None else dict(metadata)

    def update_noun_instance_metadata(
        self,
        instance_id: str,
        *,
        noun_text: Optional[str] = None,
        entity_kind: Optional[str] = None,
        gender: Optional[str] = None,
        owner_instance_id: Optional[str] = None,
        owner_role: Optional[str] = None,
        instance_scope: Optional[str] = None,
        extra_attributes: Optional[Dict[str, Any]] = None,
        attribute_polarities: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        metadata = self.ensure_noun_instance_metadata(instance_id, noun_text=noun_text, instance_scope=instance_scope)
        if noun_text is not None:
            metadata["noun_text"] = noun_text.lower()
        if entity_kind is not None:
            metadata["entity_kind"] = entity_kind
        if gender is not None:
            metadata["gender"] = gender
        if owner_instance_id is not None:
            metadata["owner_instance_id"] = owner_instance_id
        if owner_role is not None:
            metadata["owner_role"] = owner_role
        if instance_scope is not None:
            metadata["instance_scope"] = normalize_instance_scope(instance_scope)
        if extra_attributes:
            metadata.update(dict(extra_attributes))
        if attribute_polarities:
            polarity_map = dict(metadata.get("attribute_polarity") or {})
            for key, polarity in attribute_polarities.items():
                polarity_map[str(key)] = int(polarity)
            metadata["attribute_polarity"] = polarity_map
        self.noun_instance_metadata[instance_id] = dict(metadata)
        return dict(metadata)

    def store_action_instance(self, instance_id: str, action_embedding: torch.Tensor) -> None:
        self.action_instance_memory[instance_id] = action_embedding.to(self.device).view(-1).detach().clone()

    def _relation_store(self, relation_kind: str) -> Dict[str, torch.Tensor]:
        if relation_kind == "noun_noun_relation":
            return self.noun_relation_memory
        if relation_kind == "adj_noun_relation":
            return self.adj_relation_memory
        raise ValueError(f"unsupported relation kind: {relation_kind}")

    def _relation_lr(self, relation_kind: str, relation_index: int, rm, arm) -> float:
        if relation_kind == "noun_noun_relation":
            return float(rm.lr_relation[relation_index])
        if relation_kind == "adj_noun_relation":
            return float(arm.lr_adj_relation[relation_index])
        raise ValueError(f"unsupported relation kind: {relation_kind}")

    def _resolve_relation_index(self, relation_kind: str, relation_name: str, rm, arm) -> Optional[int]:
        if relation_kind == "noun_noun_relation":
            if relation_name not in rm.relation_list:
                return None
            return rm.relation_list.index(relation_name)
        if relation_kind == "adj_noun_relation":
            if relation_name not in arm.adj_relation_list:
                return None
            return arm.adj_relation_list.index(relation_name)
        return None

    def ensure_relation_clone(self, relation_kind: str, relation_name: str):
        store = self._relation_store(relation_kind)
        existing = store.get(relation_name)
        if existing is not None:
            return existing.detach().clone()

        rm, arm, kt = self._load_language_context()
        relation_index = self._resolve_relation_index(relation_kind, relation_name, rm, arm)
        if relation_index is None:
            return None

        if relation_kind == "noun_noun_relation":
            relation_weight = kt.knowledge_map_one.relations[relation_index].weight.detach().clone()
        else:
            relation_weight = kt.adj_map_one.relations[relation_index].weight.detach().clone()

        store[relation_name] = relation_weight.to(self.device).detach().clone()
        return store[relation_name].detach().clone()

    def store_relation_clone(self, relation_kind: str, relation_name: str, relation_weight: torch.Tensor) -> None:
        store = self._relation_store(relation_kind)
        store[relation_name] = relation_weight.to(self.device).detach().clone()

    def get_relation_clone(self, relation_kind: str, relation_name: str) -> Optional[torch.Tensor]:
        store = self._relation_store(relation_kind)
        relation_weight = store.get(relation_name)
        return None if relation_weight is None else relation_weight.detach().clone()

    def _load_language_context(self):
        import importlib
        import os

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

    def ensure_noun_instance(
        self,
        noun_text: str,
        instance_id: Optional[str],
        noun_type: Optional[int] = None,
        instance_scope: Optional[str] = None,
    ):
        instance_id = instance_id or self._default_noun_instance_id(noun_type)
        existing_embedding = self.get_noun_embedding(instance_id)
        if existing_embedding is not None:
            self.ensure_noun_instance_metadata(instance_id, noun_text=noun_text, instance_scope=instance_scope)
            resolved_type = noun_type
            if resolved_type is None:
                rm, _, _ = self._load_language_context()
                resolved_type = rm._ensure_noun(noun_text.lower())
            return int(resolved_type) if resolved_type is not None else None, existing_embedding, instance_id

        rm, _, kt = self._load_language_context()
        noun_key = noun_text.lower()
        noun_idx = rm._ensure_noun(noun_key) if noun_type is None else int(noun_type)
        noun_embedding = kt.knowledge_map_one.embedding.weight.detach()[noun_idx].clone()
        self.store_noun_instance(instance_id, noun_embedding, noun_text=noun_text, instance_scope=instance_scope)
        return noun_idx, noun_embedding.detach().clone(), instance_id

    def _weighted_mean_losses(self, losses: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        if not losses:
            raise ValueError("losses must not be empty")
        stacked_losses = torch.stack(losses)
        if weights is None:
            return stacked_losses.mean()
        weight_tensor = torch.tensor(
            [max(0.0, float(weight)) for weight in weights],
            dtype=stacked_losses.dtype,
            device=stacked_losses.device,
        )
        if float(weight_tensor.sum().item()) <= 1e-6:
            return stacked_losses.mean()
        return (stacked_losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)

    def _relation_loss_for_entry(
        self,
        entry: RelationMemoryEntry,
        noun_embedding: torch.Tensor,
        rm,
        arm,
        kt,
    ) -> Optional[torch.Tensor]:
        if entry.relation_kind == "adj_noun_relation":
            relation_weight = self.ensure_relation_clone(entry.relation_kind, entry.relation_name)
            if relation_weight is None:
                return None
            adjective_key = entry.target_text.lower()
            if adjective_key not in arm.adjective_list:
                arm.adjective_list.append(adjective_key)
            adjective_idx = arm.adjective_list.index(adjective_key)
            target_embedding = kt.adj_map_one.adjective_embedding.weight.data[adjective_idx]
            predicted_target = relation_weight @ noun_embedding
            return F.mse_loss(predicted_target, target_embedding)

        if entry.relation_kind == "noun_noun_relation":
            relation_weight = self.ensure_relation_clone(entry.relation_kind, entry.relation_name)
            if relation_weight is None:
                return None
            target_embedding = self.get_noun_embedding(entry.target_instance_id)
            if target_embedding is None:
                _, target_embedding, _ = self.ensure_noun_instance(entry.target_text, entry.target_instance_id)
            else:
                target_embedding = target_embedding.detach().clone()
            predicted_target = relation_weight @ noun_embedding
            return F.mse_loss(predicted_target, target_embedding)

        return None

    def _collect_source_relations(self, source_instance_id: str) -> List[RelationMemoryEntry]:
        return [
            entry
            for entry in self.relation_entries
            if entry.source_instance_id == source_instance_id
            and entry.relation_kind in {"adj_noun_relation", "noun_noun_relation"}
        ]

    def _collect_relation_entries(self, relation_kind: str, relation_name: str) -> List[RelationMemoryEntry]:
        return [
            entry
            for entry in self.relation_entries
            if entry.relation_kind == relation_kind and entry.relation_name == relation_name
        ]

    def rebuild_instance_embedding(self, source_instance_id: Optional[str], step_scale: Optional[float] = None):
        if source_instance_id is None:
            return None
        return self.rebuild_instance_embedding_from_relation_and_reward(
            source_instance_id,
            step_scale=step_scale,
        )

    def rebuild_relation_clone(
        self,
        relation_kind: str,
        relation_name: str,
        step_scale: Optional[float] = None,
    ):
        relevant_entries = self._collect_relation_entries(relation_kind, relation_name)
        if not relevant_entries:
            return self.get_relation_clone(relation_kind, relation_name)

        rm, arm, kt = self._load_language_context()
        relation_index = self._resolve_relation_index(relation_kind, relation_name, rm, arm)
        if relation_index is None:
            return None

        base_relation_weight = self.ensure_relation_clone(relation_kind, relation_name)
        if base_relation_weight is None:
            return None

        relation_weight = base_relation_weight.clone().detach().requires_grad_(True)
        losses = []
        loss_weights = []

        for entry in relevant_entries:
            source_embedding = self.get_noun_embedding(entry.source_instance_id)
            if source_embedding is None:
                _, source_embedding, _ = self.ensure_noun_instance(
                    entry.source_text,
                    entry.source_instance_id,
                    noun_type=entry.source_type,
                )
            else:
                source_embedding = source_embedding.detach().clone()

            if relation_kind == "noun_noun_relation":
                target_embedding = self.get_noun_embedding(entry.target_instance_id)
                if target_embedding is None:
                    _, target_embedding, _ = self.ensure_noun_instance(
                        entry.target_text,
                        entry.target_instance_id,
                        noun_type=entry.target_type,
                    )
                else:
                    target_embedding = target_embedding.detach().clone()
            else:
                adjective_key = entry.target_text.lower()
                if adjective_key not in arm.adjective_list:
                    arm.adjective_list.append(adjective_key)
                adjective_idx = arm.adjective_list.index(adjective_key)
                target_embedding = kt.adj_map_one.adjective_embedding.weight.data[adjective_idx].detach().clone()

            predicted_target = relation_weight @ source_embedding
            relation_loss = F.mse_loss(predicted_target, target_embedding)
            if int(getattr(entry, "polarity", 1)) == -1:
                relation_loss = -relation_loss
            losses.append(relation_loss)
            loss_weights.append(float(entry.score))

        if not losses:
            return relation_weight.detach().clone()

        total_loss = self._weighted_mean_losses(losses, loss_weights)
        total_loss.backward()

        scale = self.relation_clone_step_scale if step_scale is None else float(step_scale)
        lr = self._relation_lr(relation_kind, relation_index, rm, arm) * scale
        with torch.no_grad():
            adjusted_relation_weight = relation_weight - lr * relation_weight.grad

        relation_weight.grad.zero_()
        self.store_relation_clone(relation_kind, relation_name, adjusted_relation_weight.detach())
        return adjusted_relation_weight.detach().clone()

    def update_relation_clone(
        self,
        relation_kind: str,
        relation_name: str,
        step_scale: Optional[float] = None,
    ):
        if self.relation_clone_update_mode != "average":
            raise ValueError("unsupported relation_clone_update_mode")
        return self.rebuild_relation_clone(
            relation_kind=relation_kind,
            relation_name=relation_name,
            step_scale=step_scale,
        )

    def update_all_relation_clones(
        self,
        relation_kind: Optional[str] = None,
        step_scale: Optional[float] = None,
    ):
        updated = []
        seen = set()
        for entry in self.relation_entries:
            if relation_kind is not None and entry.relation_kind != relation_kind:
                continue
            key = (entry.relation_kind, entry.relation_name)
            if key in seen:
                continue
            seen.add(key)
            clone = self.update_relation_clone(
                relation_kind=entry.relation_kind,
                relation_name=entry.relation_name,
                step_scale=step_scale,
            )
            updated.append(
                {
                    "relation_kind": entry.relation_kind,
                    "relation_name": entry.relation_name,
                    "updated": clone is not None,
                }
            )
        return updated

    def get_noun_embedding(self, instance_id: Optional[str]) -> Optional[torch.Tensor]:
        if instance_id is None:
            return None
        embedding = self.noun_instance_memory.get(instance_id)
        return None if embedding is None else embedding.detach().clone()

    def get_action_embedding(self, instance_id: Optional[str]) -> Optional[torch.Tensor]:
        if instance_id is None:
            return None
        embedding = self.action_instance_memory.get(instance_id)
        return None if embedding is None else embedding.detach().clone()

    def get_entry_tensor(self, entry: EventMemoryEntry) -> torch.Tensor:
        return self._encode_entry(entry)

    def append(self, tensor, score=0.0, noun_type=None, action_type=None):
        return self.append_state(
            noun_embedding=tensor[:noun_dim],
            action_embedding=tensor[noun_dim:],
            score=score,
            noun_type=noun_type,
            action_type=action_type,
            time_position=len(self.event_entries),
        )

    def append_state(self, *args, **kwargs):
        return self.append_event(*args, **kwargs)

    def append_event(
        self,
        noun_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        score=0.0,
        noun_type=None,
        action_type=None,
        time_position: int = 0,
        pair_index: Optional[int] = None,
        event_index: Optional[int] = None,
        noun_text: Optional[str] = None,
        action_text: Optional[str] = None,
        instance_id: Optional[str] = None,
        noun_instance_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
        role: Optional[str] = None,
        polarity: int = 1,
        accept_label: str = "none",
        diff_value: Any = "none",
        question_label: str = "none",
        sentence_label: str = "none",
        adjectives: Optional[List[str]] = None,
        pair_kind: str = "noun_action",
        info_pair: Optional[Dict[str, Any]] = None,
    ):
        if pair_index is None:
            pair_index = self._insert_counter
        noun_instance_id = noun_instance_id or instance_id or self._default_noun_instance_id(noun_type)
        action_instance_id = action_instance_id or self._default_action_instance_id(
            action_type,
            time_position=time_position,
            pair_index=pair_index,
        )

        noun_embedding = noun_embedding.to(self.device).view(-1)
        action_embedding = action_embedding.to(self.device).view(-1)
        self._resolve_state_dim(noun_embedding, action_embedding)
        self.store_noun_instance(noun_instance_id, noun_embedding, noun_text=noun_text)
        self.store_action_instance(action_instance_id, action_embedding)

        base_info_pair = {
            "pair_kind": pair_kind,
            "noun_instance_id": noun_instance_id,
            "action_instance_id": action_instance_id,
            "noun_type": None if noun_type is None else int(noun_type),
            "action_type": None if action_type is None else int(action_type),
            "time_position": int(time_position),
            "pair_index": int(pair_index),
            "event_index": None if event_index is None else int(event_index),
        }
        base_info_pair["polarity"] = int(polarity)
        base_info_pair["accept_label"] = str(accept_label)
        base_info_pair["diff_value"] = diff_value
        base_info_pair["question_label"] = str(question_label)
        base_info_pair["sentence_label"] = str(sentence_label)
        if role is not None:
            base_info_pair["role"] = role
        if info_pair:
            base_info_pair.update(dict(info_pair))

        entry = EventMemoryEntry(
            score=float(score),
            noun_type=None if noun_type is None else int(noun_type),
            action_type=None if action_type is None else int(action_type),
            time_position=int(time_position),
            pair_index=int(pair_index),
            event_index=None if event_index is None else int(event_index),
            noun_instance_id=noun_instance_id,
            action_instance_id=action_instance_id,
            noun_text=noun_text,
            action_text=action_text,
            role=role,
            polarity=int(polarity),
            accept_label=str(accept_label),
            diff_value=diff_value,
            question_label=str(question_label),
            sentence_label=str(sentence_label),
            pair_kind=pair_kind,
            adjectives=list(adjectives or []),
            info_pair=base_info_pair,
        )
        self.event_entries.append(entry)
        self._insert_counter += 1
        self._sort_event_entries()
        self._trim()
        return self.get_entry_tensor(entry)

    def append_relation(
        self,
        relation_name: str,
        source_text: str,
        target_text: str,
        relation_kind: str,
        score: float = 0.0,
        time_position: int = 0,
        pair_index: Optional[int] = None,
        source_instance_id: Optional[str] = None,
        target_instance_id: Optional[str] = None,
        source_type: Optional[int] = None,
        target_type: Optional[int] = None,
        polarity: int = 1,
        accept_label: str = "none",
        diff_value: Any = "none",
        question_label: str = "none",
        sentence_label: str = "none",
        source_embedding: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
        info_pair: Optional[Dict[str, Any]] = None,
    ):
        if pair_index is None:
            pair_index = self._insert_counter

        if source_instance_id is not None:
            if source_embedding is not None:
                self.store_noun_instance(source_instance_id, source_embedding)
            else:
                source_type, _, source_instance_id = self.ensure_noun_instance(
                    source_text,
                    source_instance_id,
                    noun_type=source_type,
                )
        if target_instance_id is not None:
            if target_embedding is not None:
                self.store_noun_instance(target_instance_id, target_embedding)
            else:
                target_type, _, target_instance_id = self.ensure_noun_instance(
                    target_text,
                    target_instance_id,
                    noun_type=target_type,
                )

        base_info_pair = {
            "pair_kind": relation_kind,
            "relation_name": relation_name,
            "source_text": source_text,
            "target_text": target_text,
            "source_instance_id": source_instance_id,
            "target_instance_id": target_instance_id,
            "source_type": source_type,
            "target_type": target_type,
            "time_position": int(time_position),
            "pair_index": int(pair_index),
            "polarity": int(polarity),
            "accept_label": str(accept_label),
            "diff_value": diff_value,
            "question_label": str(question_label),
            "sentence_label": str(sentence_label),
        }
        if info_pair:
            base_info_pair.update(dict(info_pair))

        entry = RelationMemoryEntry(
            score=float(score),
            time_position=int(time_position),
            pair_index=int(pair_index),
            relation_kind=relation_kind,
            relation_name=relation_name,
            source_text=source_text,
            target_text=target_text,
            source_instance_id=source_instance_id,
            target_instance_id=target_instance_id,
            polarity=int(polarity),
            accept_label=str(accept_label),
            diff_value=diff_value,
            question_label=str(question_label),
            sentence_label=str(sentence_label),
            source_type=None if source_type is None else int(source_type),
            target_type=None if target_type is None else int(target_type),
            info_pair=base_info_pair,
        )
        self.relation_entries.append(entry)
        self._insert_counter += 1
        self._sort_relation_entries()
        self.ensure_relation_clone(relation_kind, relation_name)

        if (
            self.relation_update_mode == "average"
            and self.relation_update_frequency == "per_relation"
            and source_instance_id is not None
        ):
            self.rebuild_instance_embedding(source_instance_id)

        self._trim()
        return dict(entry.info_pair)

    def append_reward(
        self,
        *,
        subject_text: str,
        subject_instance_id: str,
        reward_word: str,
        reward_value: float,
        action_text: Optional[str] = None,
        object_text: Optional[str] = None,
        object_instance_id: Optional[str] = None,
        score: float = 1.0,
        time_position: int = 0,
        pair_index: Optional[int] = None,
        polarity: int = 1,
        accept_label: str = "none",
        diff_value: Any = "none",
        question_label: str = "none",
        sentence_label: str = "none",
        info_pair: Optional[Dict[str, Any]] = None,
    ):
        if pair_index is None:
            pair_index = len(self.reward_entries)

        base_info_pair = {
            "pair_kind": "subject_event_reward",
            "subject_text": subject_text,
            "subject_instance_id": subject_instance_id,
            "reward_word": reward_word,
            "reward_value": float(reward_value),
            "polarity": int(polarity),
            "action_text": action_text,
            "object_text": object_text,
            "object_instance_id": object_instance_id,
            "time_position": int(time_position),
            "pair_index": int(pair_index),
            "polarity": int(polarity),
            "accept_label": str(accept_label),
            "diff_value": diff_value,
            "question_label": str(question_label),
            "sentence_label": str(sentence_label),
        }
        if info_pair:
            base_info_pair.update(dict(info_pair))

        entry = RewardMemoryEntry(
            score=float(score),
            reward_word=reward_word,
            reward_value=float(reward_value),
            time_position=int(time_position),
            pair_index=int(pair_index),
            subject_text=subject_text,
            subject_instance_id=subject_instance_id,
            action_text=action_text,
            object_text=object_text,
            object_instance_id=object_instance_id,
            polarity=int(polarity),
            accept_label=str(accept_label),
            diff_value=diff_value,
            question_label=str(question_label),
            sentence_label=str(sentence_label),
            info_pair=base_info_pair,
        )
        self.reward_entries.append(entry)
        self.reward_entries.sort(key=lambda item: (item.time_position, item.pair_index))
        self._trim()
        return dict(entry.info_pair)

    def append_surprise(
        self,
        *,
        subject_text: str,
        subject_instance_id: str,
        surprise_word: str,
        surprise_value: float,
        action_text: Optional[str] = None,
        object_text: Optional[str] = None,
        object_instance_id: Optional[str] = None,
        score: float = 1.0,
        time_position: int = 0,
        pair_index: Optional[int] = None,
        polarity: int = 1,
        accept_label: str = "none",
        diff_value: Any = "none",
        question_label: str = "none",
        sentence_label: str = "none",
        info_pair: Optional[Dict[str, Any]] = None,
    ):
        if pair_index is None:
            pair_index = len(self.surprise_entries)

        base_info_pair = {
            "pair_kind": "subject_event_surprise",
            "subject_text": subject_text,
            "subject_instance_id": subject_instance_id,
            "surprise_word": surprise_word,
            "surprise_value": float(surprise_value),
            "polarity": int(polarity),
            "action_text": action_text,
            "object_text": object_text,
            "object_instance_id": object_instance_id,
            "time_position": int(time_position),
            "pair_index": int(pair_index),
            "accept_label": str(accept_label),
            "diff_value": diff_value,
            "question_label": str(question_label),
            "sentence_label": str(sentence_label),
        }
        if info_pair:
            base_info_pair.update(dict(info_pair))

        entry = SurpriseMemoryEntry(
            score=float(score),
            surprise_word=surprise_word,
            surprise_value=float(surprise_value),
            time_position=int(time_position),
            pair_index=int(pair_index),
            subject_text=subject_text,
            subject_instance_id=subject_instance_id,
            action_text=action_text,
            object_text=object_text,
            object_instance_id=object_instance_id,
            polarity=int(polarity),
            accept_label=str(accept_label),
            diff_value=diff_value,
            question_label=str(question_label),
            sentence_label=str(sentence_label),
            info_pair=base_info_pair,
        )
        self.surprise_entries.append(entry)
        self.surprise_entries.sort(key=lambda item: (item.time_position, item.pair_index))
        self._trim()
        return dict(entry.info_pair)

    def _reward_loss_for_subject_entry(
        self,
        entry: RewardMemoryEntry,
        instance_id: str,
        subject_embedding: torch.Tensor,
        reward_model,
        reward_encoder,
    ) -> Optional[torch.Tensor]:
        if entry.subject_instance_id != instance_id:
            return None
        action_embedding = reward_encoder.encode_action(action_text=entry.action_text)
        object_embedding = reward_encoder.encode_noun(
            noun_text=entry.object_text,
            noun_instance_id=entry.object_instance_id,
        )
        prediction = reward_model(
            subject_embedding=subject_embedding,
            action_embedding=action_embedding,
            object_embedding=object_embedding,
        ).view(-1)[0]
        target = torch.tensor(
            float(entry.reward_value),
            dtype=prediction.dtype,
            device=prediction.device,
        )
        return F.smooth_l1_loss(prediction, target)

    def rebuild_instance_embedding_from_relation_and_reward(
        self,
        instance_id: str,
        *,
        reward_model=None,
        reward_encoder=None,
        step_scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        metadata = self.get_noun_instance_metadata(instance_id)
        noun_text = None if metadata is None else metadata.get("noun_text")
        relation_entries = self._collect_source_relations(instance_id)
        reward_entries = [
            entry for entry in self.reward_entries
            if entry.subject_instance_id == instance_id
        ]
        if not relation_entries and not reward_entries:
            return self.get_noun_embedding(instance_id)

        rm, arm, kt = self._load_language_context()
        source_text = relation_entries[-1].source_text if relation_entries else noun_text
        if source_text is None:
            return self.get_noun_embedding(instance_id)
        source_type = relation_entries[-1].source_type if relation_entries else None
        noun_idx, base_embedding, resolved_instance_id = self.ensure_noun_instance(
            source_text,
            instance_id,
            noun_type=source_type,
        )
        noun_embedding = base_embedding.clone().detach().requires_grad_(True)

        losses = []
        loss_weights = []
        for entry in relation_entries:
            loss = self._relation_loss_for_entry(entry, noun_embedding, rm, arm, kt)
            if loss is not None:
                if int(getattr(entry, "polarity", 1)) == -1:
                    loss = -loss
                losses.append(loss)
                loss_weights.append(float(entry.score))

        if reward_model is not None and reward_encoder is not None:
            for entry in reward_entries:
                loss = self._reward_loss_for_subject_entry(
                    entry,
                    instance_id,
                    noun_embedding,
                    reward_model,
                    reward_encoder,
                )
                if loss is not None:
                    losses.append(loss)
                    loss_weights.append(float(entry.score))

        if not losses:
            return noun_embedding.detach().clone()

        total_loss = self._weighted_mean_losses(losses, loss_weights)
        total_loss.backward()
        if noun_embedding.grad is None:
            return noun_embedding.detach().clone()

        scale = self.relation_step_scale if step_scale is None else float(step_scale)
        with torch.no_grad():
            lr = float(rm.lr_per_embedding[noun_idx]) * scale
            adjusted_noun_embedding = noun_embedding - lr * noun_embedding.grad

        noun_embedding.grad.zero_()
        self.store_noun_instance(resolved_instance_id, adjusted_noun_embedding.detach(), noun_text=source_text)
        return adjusted_noun_embedding.detach().clone()

    def rebuild_instance_embedding_from_reward(
        self,
        instance_id: str,
        *,
        reward_model,
        reward_encoder,
        step_scale: float = 0.01,
    ) -> Optional[torch.Tensor]:
        involved_entries = [
            entry for entry in self.reward_entries
            if entry.subject_instance_id == instance_id
        ]
        if not involved_entries:
            return None

        base_embedding = self.get_noun_embedding(instance_id)
        if base_embedding is None:
            return None

        trainable_embedding = base_embedding.detach().clone().requires_grad_(True)
        losses = []
        loss_weights = []
        for entry in involved_entries:
            subject_embedding = reward_encoder.encode_noun(
                noun_text=entry.subject_text,
                noun_instance_id=entry.subject_instance_id,
            )
            object_embedding = reward_encoder.encode_noun(
                noun_text=entry.object_text,
                noun_instance_id=entry.object_instance_id,
            )
            if entry.subject_instance_id == instance_id:
                subject_embedding = trainable_embedding
            action_embedding = reward_encoder.encode_action(action_text=entry.action_text)

            prediction = reward_model(
                subject_embedding=subject_embedding,
                action_embedding=action_embedding,
                object_embedding=object_embedding,
            ).view(-1)[0]
            target = torch.tensor(
                float(entry.reward_value),
                dtype=prediction.dtype,
                device=prediction.device,
            )
            losses.append(F.smooth_l1_loss(prediction, target))
            loss_weights.append(float(entry.score))

        if not losses:
            return None

        total_loss = self._weighted_mean_losses(losses, loss_weights)
        total_loss.backward()
        if trainable_embedding.grad is None:
            return None

        with torch.no_grad():
            updated_embedding = trainable_embedding - float(step_scale) * trainable_embedding.grad
        metadata = self.get_noun_instance_metadata(instance_id)
        noun_text = None if metadata is None else metadata.get("noun_text")
        self.store_noun_instance(instance_id, updated_embedding.detach(), noun_text=noun_text)
        return updated_embedding.detach().clone()

    def set_maxlen(self, new_maxlen):
        self.maxlen = new_maxlen
        self._trim()

    def filter_by_score(self, threshold, mode="ge"):
        comparator = (lambda score: score >= threshold) if mode == "ge" else (lambda score: score <= threshold)
        if mode not in {"ge", "le"}:
            raise ValueError("mode must be 'ge' or 'le'")
        self.event_entries = [entry for entry in self.event_entries if comparator(entry.score)]
        self.relation_entries = [entry for entry in self.relation_entries if comparator(entry.score)]
        self._sort_event_entries()
        self._sort_relation_entries()
        self._prune_instance_stores()

    def boost_related(self, noun_type=None, action_type=None, amount=1.0):
        if noun_type is None and action_type is None:
            raise ValueError("Provide noun_type index or action_type to boost")

        noun_type = None if noun_type is None else int(noun_type)
        action_type = None if action_type is None else int(action_type)

        for entry in self.event_entries:
            matched = False
            if noun_type is not None and entry.noun_type == noun_type:
                matched = True
            if action_type is not None and entry.action_type == action_type:
                matched = True
            if matched:
                entry.score += float(amount)

        for entry in self.relation_entries:
            matched = False
            if noun_type is not None and (
                entry.source_type == noun_type or entry.target_type == noun_type
            ):
                matched = True
            if action_type is not None and entry.relation_kind == "noun_action":
                matched = True
            if matched:
                entry.score += float(amount)

        self._sort_event_entries()
        self._sort_relation_entries()

    def focus_instance(self, instance_id: str, target_score: float = 100.0):
        event_count = 0
        relation_count = 0

        for entry in self.event_entries:
            if entry.noun_instance_id == instance_id:
                entry.score = max(float(entry.score), float(target_score))
                event_count += 1

        for entry in self.relation_entries:
            if entry.source_instance_id == instance_id or entry.target_instance_id == instance_id:
                entry.score = max(float(entry.score), float(target_score))
                relation_count += 1

        self._reorder_event_entries_for_world_model()
        self._sort_relation_entries()
        return {
            "instance_id": instance_id,
            "event_count": event_count,
            "relation_count": relation_count,
            "target_score": float(target_score),
        }

    def get_stack(self):
        if len(self.event_entries) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                [],
                [],
            )

        tensors = [self.get_entry_tensor(entry) for entry in self.event_entries]
        scores = [entry.score for entry in self.event_entries]
        noun_types = [entry.noun_type for entry in self.event_entries]
        action_types = [entry.action_type for entry in self.event_entries]
        return (
            torch.stack(tensors),
            torch.tensor(scores, device=self.device),
            noun_types,
            action_types,
        )

    def get_latest_n(self, n):
        if not self.event_entries:
            return torch.empty(0, device=self.device)
        latest = self.event_entries[-n:]
        return torch.stack([self.get_entry_tensor(entry) for entry in latest])

    def get_event_entries(self, steps=None, order_by: str = "time") -> List[EventMemoryEntry]:
        if order_by == "time":
            entries = self.event_entries
        elif order_by == "attention":
            entries = self._event_entries_by_attention()
        else:
            raise ValueError("order_by must be 'time' or 'attention'")
        if steps is None:
            return list(entries)
        return list(entries[-steps:]) if order_by == "time" else list(entries[:steps])

    def _world_model_selected_event_entries(self, steps=None) -> List[EventMemoryEntry]:
        self._reorder_event_entries_for_world_model()
        selected = list(self.event_entries if steps is None else self.event_entries[-steps:])
        if len(selected) <= 1:
            return selected

        focus = selected[-1]
        focus_event_key = (focus.time_position, focus.event_index)
        selected_for_input = []
        for entry in selected:
            same_focus_event = (entry.time_position, entry.event_index) == focus_event_key
            is_negative = int(getattr(entry, "polarity", 1)) == -1
            if is_negative and not same_focus_event:
                continue
            selected_for_input.append(entry)
        return selected_for_input

    def build_world_model_event_input(self, steps=None):
        selected = self._world_model_selected_event_entries(steps=steps)
        if not selected:
            return torch.empty((0, 0), device=self.device)
        return torch.stack([self.get_entry_tensor(entry) for entry in selected], dim=1)

    def build_world_model_input(self, steps=None):
        return self.build_world_model_event_input(steps=steps)

    def get_focus_entry(self, steps=None) -> Optional[EventMemoryEntry]:
        if len(self.event_entries) == 0:
            return None
        selected = self.event_entries if steps is None else self.event_entries[-steps:]
        if not selected:
            return None
        return selected[-1]

    def get_info_pairs(self):
        return [dict(entry.info_pair) for entry in self.event_entries]

    def get_relation_pairs(self):
        return [dict(entry.info_pair) for entry in self.relation_entries]

    def get_event_content_view(self, order_by: str = "time"):
        if order_by == "time":
            entries = self.event_entries
        elif order_by == "attention":
            entries = self._event_entries_by_attention()
        else:
            raise ValueError("order_by must be 'time' or 'attention'")
        return [
            {
                "pair_kind": entry.pair_kind,
                "noun_instance_id": entry.noun_instance_id,
                "action_instance_id": entry.action_instance_id,
                "noun_type": entry.noun_type,
                "action_type": entry.action_type,
                "noun_text": entry.noun_text,
                "action_text": entry.action_text,
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "event_index": entry.event_index,
                "score": entry.score,
                "role": entry.role,
                "polarity": entry.polarity,
                "accept_label": entry.accept_label,
                "diff_value": entry.diff_value,
                "question_label": entry.question_label,
                "sentence_label": entry.sentence_label,
            }
            for entry in entries
        ]

    def get_relation_content_view(self, order_by: str = "time"):
        if order_by == "time":
            entries = self.relation_entries
        elif order_by == "attention":
            entries = self._relation_entries_by_attention()
        else:
            raise ValueError("order_by must be 'time' or 'attention'")
        return [
            {
                "pair_kind": entry.relation_kind,
                "relation_name": entry.relation_name,
                "source_text": entry.source_text,
                "target_text": entry.target_text,
                "source_instance_id": entry.source_instance_id,
                "target_instance_id": entry.target_instance_id,
                "source_type": entry.source_type,
                "target_type": entry.target_type,
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "score": entry.score,
                "polarity": entry.polarity,
                "accept_label": entry.accept_label,
                "diff_value": entry.diff_value,
                "question_label": entry.question_label,
                "sentence_label": entry.sentence_label,
            }
            for entry in entries
        ]

    def get_reward_content_view(self, order_by: str = "time"):
        if order_by == "time":
            entries = self.reward_entries
        elif order_by == "attention":
            entries = sorted(
                self.reward_entries,
                key=lambda entry: (-entry.score, entry.time_position, entry.pair_index),
            )
        else:
            raise ValueError("order_by must be 'time' or 'attention'")
        return [
            {
                "pair_kind": "subject_event_reward",
                "subject_text": entry.subject_text,
                "subject_instance_id": entry.subject_instance_id,
                "reward_word": entry.reward_word,
                "reward_value": entry.reward_value,
                "action_text": entry.action_text,
                "object_text": entry.object_text,
                "object_instance_id": entry.object_instance_id,
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "event_index": getattr(entry, "event_index", None),
                "score": entry.score,
                "polarity": entry.polarity,
                "accept_label": entry.accept_label,
                "diff_value": entry.diff_value,
                "question_label": entry.question_label,
                "sentence_label": entry.sentence_label,
            }
            for entry in entries
        ]

    def get_surprise_content_view(self, order_by: str = "time"):
        if order_by == "time":
            entries = self.surprise_entries
        elif order_by == "attention":
            entries = sorted(
                self.surprise_entries,
                key=lambda entry: (-entry.score, entry.time_position, entry.pair_index),
            )
        else:
            raise ValueError("order_by must be 'time' or 'attention'")
        return [
            {
                "pair_kind": "subject_event_surprise",
                "subject_text": entry.subject_text,
                "subject_instance_id": entry.subject_instance_id,
                "surprise_word": entry.surprise_word,
                "surprise_value": entry.surprise_value,
                "action_text": entry.action_text,
                "object_text": entry.object_text,
                "object_instance_id": entry.object_instance_id,
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "event_index": getattr(entry, "event_index", None),
                "score": entry.score,
                "polarity": entry.polarity,
                "accept_label": entry.accept_label,
                "diff_value": entry.diff_value,
                "question_label": entry.question_label,
                "sentence_label": entry.sentence_label,
            }
            for entry in entries
        ]

    def get_content_view(self, order_by: str = "time"):
        return {
            "event": self.get_event_content_view(order_by=order_by),
            "relation": self.get_relation_content_view(order_by=order_by),
            "reward": self.get_reward_content_view(order_by=order_by),
            "surprise": self.get_surprise_content_view(order_by=order_by),
        }

    def latest_state(self):
        if not self.event_entries:
            return None
        entry = self.event_entries[-1]
        return (self.get_entry_tensor(entry), entry.score, entry.noun_type, entry.action_type)

    def __len__(self):
        return len(self.event_entries)

    def filter_by_type(self, noun_type=None, action_type=None):
        noun_type = None if noun_type is None else int(noun_type)
        action_type = None if action_type is None else int(action_type)
        kept = []
        for entry in self.event_entries:
            if noun_type is not None and entry.noun_type != noun_type:
                continue
            if action_type is not None and entry.action_type != action_type:
                continue
            kept.append(entry)
        self.event_entries = kept
        self._sort_event_entries()
        self._prune_instance_stores()

    def clear(self):
        self.event_entries.clear()
        self.relation_entries.clear()
        self.reward_entries.clear()
        self.surprise_entries.clear()
        self.noun_instance_memory.clear()
        self.noun_instance_metadata.clear()
        self.action_instance_memory.clear()
        self.noun_relation_memory.clear()
        self.adj_relation_memory.clear()
        self._insert_counter = 0
        self._event_counter = 0
        self._sentence_counter = 0


ScoredTensorQueue = ShortMemory
short_memory = ShortMemory(maxlen=50, device="cpu")



