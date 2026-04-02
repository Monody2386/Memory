from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from knowledge.relation_map import noun_dim


@dataclass
class EventMemoryEntry:
    score: float
    noun_type: Optional[int]
    action_type: Optional[int]
    time_position: int
    pair_index: int
    noun_instance_id: Optional[str]
    action_instance_id: Optional[str]
    noun_text: Optional[str] = None
    action_text: Optional[str] = None
    role: Optional[str] = None
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
    source_type: Optional[int] = None
    target_type: Optional[int] = None
    info_pair: Dict[str, Any] = field(default_factory=dict)


MemoryEntry = EventMemoryEntry


class ShortMemory:
    def __init__(self, maxlen=100, device="cpu", state_dim=None):
        self.maxlen = maxlen
        self.device = device
        self.state_dim = state_dim
        self.event_entries: List[EventMemoryEntry] = []
        self.relation_entries: List[RelationMemoryEntry] = []
        self.noun_instance_memory: Dict[str, torch.Tensor] = {}
        self.action_instance_memory: Dict[str, torch.Tensor] = {}
        self._insert_counter = 0

    @property
    def entries(self) -> List[EventMemoryEntry]:
        return self.event_entries

    @property
    def short_memory_event(self) -> List[EventMemoryEntry]:
        return self.event_entries

    @property
    def short_memory_relation(self) -> List[RelationMemoryEntry]:
        return self.relation_entries

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
        referenced_actions = {
            entry.action_instance_id for entry in self.event_entries if entry.action_instance_id is not None
        }
        self.noun_instance_memory = {
            key: value for key, value in self.noun_instance_memory.items() if key in referenced_nouns
        }
        self.action_instance_memory = {
            key: value for key, value in self.action_instance_memory.items() if key in referenced_actions
        }

    def _trim(self):
        while len(self.event_entries) > self.maxlen:
            self.event_entries.pop(0)
        while len(self.relation_entries) > self.maxlen:
            self.relation_entries.pop(0)
        self._prune_instance_stores()

    def _default_noun_instance_id(self, noun_type=None) -> str:
        noun_label = "noun" if noun_type is None else f"noun{int(noun_type)}"
        return f"{noun_label}@{self._insert_counter}"

    def _default_action_instance_id(self, action_type=None, time_position: int = 0, pair_index: int = 0) -> str:
        action_label = "action" if action_type is None else f"action{int(action_type)}"
        return f"{action_label}@t{int(time_position)}:p{int(pair_index)}:{self._insert_counter}"

    def store_noun_instance(self, instance_id: str, noun_embedding: torch.Tensor) -> None:
        self.noun_instance_memory[instance_id] = noun_embedding.to(self.device).view(-1).detach().clone()

    def store_action_instance(self, instance_id: str, action_embedding: torch.Tensor) -> None:
        self.action_instance_memory[instance_id] = action_embedding.to(self.device).view(-1).detach().clone()

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
    ):
        instance_id = instance_id or self._default_noun_instance_id(noun_type)
        existing_embedding = self.get_noun_embedding(instance_id)
        if existing_embedding is not None:
            resolved_type = noun_type
            if resolved_type is None:
                rm, _, _ = self._load_language_context()
                resolved_type = rm._ensure_noun(noun_text.lower())
            return int(resolved_type) if resolved_type is not None else None, existing_embedding, instance_id

        rm, _, kt = self._load_language_context()
        noun_key = noun_text.lower()
        noun_idx = rm._ensure_noun(noun_key) if noun_type is None else int(noun_type)
        noun_embedding = kt.knowledge_map_one.embedding.weight.detach()[noun_idx].clone()
        self.store_noun_instance(instance_id, noun_embedding)
        return noun_idx, noun_embedding.detach().clone(), instance_id

    def _apply_adjective_relation_update(
        self,
        relation_name: str,
        target_text: str,
        noun_idx: int,
        noun_embedding: torch.Tensor,
        arm,
        kt,
    ) -> Optional[torch.Tensor]:
        if relation_name not in arm.adj_relation_list:
            return None
        adjective_key = target_text.lower()
        if adjective_key not in arm.adjective_list:
            arm.adjective_list.append(adjective_key)
        adjective_idx = arm.adjective_list.index(adjective_key)
        relation_type = arm.adj_relation_list.index(relation_name) + 1
        rel_idx = int(relation_type) - 1
        target_embedding = kt.adj_map_one.adjective_embedding.weight.data[adjective_idx]
        relation_weight = kt.adj_map_one.relations[rel_idx].weight.data
        predicted_target = relation_weight @ noun_embedding
        return torch.nn.functional.mse_loss(predicted_target, target_embedding)

    def _apply_noun_relation_update(
        self,
        relation_name: str,
        target_text: str,
        target_instance_id: Optional[str],
        noun_embedding: torch.Tensor,
        rm,
        kt,
    ) -> Optional[torch.Tensor]:
        if relation_name not in rm.relation_list:
            return None
        relation_type = rm.relation_list.index(relation_name) + 1
        rel_idx = int(relation_type) - 1
        target_embedding = self.get_noun_embedding(target_instance_id) if target_instance_id is not None else None
        if target_embedding is None:
            _, target_embedding, _ = self.ensure_noun_instance(target_text, target_instance_id)
        else:
            target_embedding = target_embedding.detach().clone()
        relation_weight = kt.knowledge_map_one.relations[rel_idx].weight.data
        predicted_target = relation_weight @ noun_embedding
        return torch.nn.functional.mse_loss(predicted_target, target_embedding)

    def apply_relation_to_noun_instance(
        self,
        relation_name: str,
        relation_kind: str,
        source_text: str,
        target_text: str,
        source_instance_id: Optional[str],
        target_instance_id: Optional[str] = None,
        step_scale: float = 0.1,
    ):
        if source_instance_id is None:
            return None

        rm, arm, kt = self._load_language_context()
        noun_idx, noun_embedding, resolved_instance_id = self.ensure_noun_instance(
            source_text,
            source_instance_id,
        )
        noun_embedding = noun_embedding.clone().detach().requires_grad_(True)

        if relation_kind == "adj_noun_relation":
            loss = self._apply_adjective_relation_update(
                relation_name=relation_name,
                target_text=target_text,
                noun_idx=noun_idx,
                noun_embedding=noun_embedding,
                arm=arm,
                kt=kt,
            )
        elif relation_kind == "noun_noun_relation":
            loss = self._apply_noun_relation_update(
                relation_name=relation_name,
                target_text=target_text,
                target_instance_id=target_instance_id,
                noun_embedding=noun_embedding,
                rm=rm,
                kt=kt,
            )
        else:
            return noun_embedding.detach().clone()

        if loss is None:
            return noun_embedding.detach().clone()

        loss.backward()
        with torch.no_grad():
            lr = float(rm.lr_per_embedding[noun_idx]) * float(step_scale)
            adjusted_noun_embedding = noun_embedding - lr * noun_embedding.grad

        noun_embedding.grad.zero_()
        self.store_noun_instance(resolved_instance_id, adjusted_noun_embedding.detach())
        return adjusted_noun_embedding.detach().clone()

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
        noun_text: Optional[str] = None,
        action_text: Optional[str] = None,
        instance_id: Optional[str] = None,
        noun_instance_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
        role: Optional[str] = None,
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
        self.store_noun_instance(noun_instance_id, noun_embedding)
        self.store_action_instance(action_instance_id, action_embedding)

        base_info_pair = {
            "pair_kind": pair_kind,
            "noun_instance_id": noun_instance_id,
            "action_instance_id": action_instance_id,
            "noun_type": None if noun_type is None else int(noun_type),
            "action_type": None if action_type is None else int(action_type),
            "time_position": int(time_position),
            "pair_index": int(pair_index),
        }
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
            noun_instance_id=noun_instance_id,
            action_instance_id=action_instance_id,
            noun_text=noun_text,
            action_text=action_text,
            role=role,
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

        if relation_kind in {"adj_noun_relation", "noun_noun_relation"}:
            updated_embedding = self.apply_relation_to_noun_instance(
                relation_name=relation_name,
                relation_kind=relation_kind,
                source_text=source_text,
                target_text=target_text,
                source_instance_id=source_instance_id,
                target_instance_id=target_instance_id,
            )
            if updated_embedding is not None:
                self.store_noun_instance(source_instance_id, updated_embedding)

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
            source_type=None if source_type is None else int(source_type),
            target_type=None if target_type is None else int(target_type),
            info_pair=base_info_pair,
        )
        self.relation_entries.append(entry)
        self._insert_counter += 1
        self._sort_relation_entries()
        self._trim()
        return dict(entry.info_pair)

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

    def build_world_model_event_input(self, steps=None):
        self._reorder_event_entries_for_world_model()
        selected = self.event_entries if steps is None else self.event_entries[-steps:]
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
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "score": entry.score,
                "role": entry.role,
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
                "source_instance_id": entry.source_instance_id,
                "target_instance_id": entry.target_instance_id,
                "source_type": entry.source_type,
                "target_type": entry.target_type,
                "time_position": entry.time_position,
                "pair_index": entry.pair_index,
                "score": entry.score,
            }
            for entry in entries
        ]

    def get_content_view(self, order_by: str = "time"):
        return {
            "event": self.get_event_content_view(order_by=order_by),
            "relation": self.get_relation_content_view(order_by=order_by),
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
        self.noun_instance_memory.clear()
        self.action_instance_memory.clear()


ScoredTensorQueue = ShortMemory
short_memory = ShortMemory(maxlen=50, device="cpu")
