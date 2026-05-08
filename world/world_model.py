from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_vocab import get_full_action_type_list, set_action_list

attention_dim = 100
action_dim = 80
noun_dim = 50
value_dim = 100
hidden_dim = 50


class WorldModelResetError(NotImplementedError):
    """Raised when code tries to use the removed legacy world-model pipeline."""


def _reset_message(method_name: str) -> str:
    return (
        f"Legacy world-model logic has been cleared. `{method_name}` is intentionally unavailable "
        "until the new world_state/event_state world model is implemented."
    )


@dataclass
class PredictedEvent:
    noun_instance_id: Optional[str]
    noun_type: Optional[int]
    noun_embedding: Optional[torch.Tensor]
    action_instance_id: Optional[str]
    action_type: int
    action_embedding: torch.Tensor
    source_action_type: Optional[int]
    source_time_position: Optional[int]
    source_pair_index: Optional[int]
    source_event_index: Optional[int]
    time_position: int
    event_index: Optional[int]
    score: float
    polarity: int = 1

    def as_dict(self):
        return {
            "noun_instance_id": self.noun_instance_id,
            "noun_type": self.noun_type,
            "noun_embedding": self.noun_embedding,
            "action_instance_id": self.action_instance_id,
            "action_type": self.action_type,
            "action_embedding": self.action_embedding,
            "source_action_type": self.source_action_type,
            "source_time_position": self.source_time_position,
            "source_pair_index": self.source_pair_index,
            "source_event_index": self.source_event_index,
            "time_position": self.time_position,
            "event_index": self.event_index,
            "score": self.score,
            "polarity": self.polarity,
        }


@dataclass
class InstanceUpdateResult:
    noun_instance_id: str
    noun_type: Optional[int]
    action_type: int
    action_instance_id: Optional[str]
    old_embedding: torch.Tensor
    new_embedding: torch.Tensor
    time_position: Optional[int]
    event_index: Optional[int]
    pair_index: Optional[int]
    score: float

    def as_dict(self):
        return {
            "noun_instance_id": self.noun_instance_id,
            "noun_type": self.noun_type,
            "action_type": self.action_type,
            "action_instance_id": self.action_instance_id,
            "old_embedding": self.old_embedding,
            "new_embedding": self.new_embedding,
            "time_position": self.time_position,
            "event_index": self.event_index,
            "pair_index": self.pair_index,
            "score": self.score,
        }


@dataclass
class SpaceUpdateResult:
    source_instance_id: str
    target_instance_id: str
    action_space_type: int
    action_space_name: str
    relation: str
    time_position: Optional[int]
    event_index: Optional[int]
    pair_index: Optional[int]
    score: float
    replace_family: bool

    def as_dict(self):
        return {
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
            "action_space_type": self.action_space_type,
            "action_space_name": self.action_space_name,
            "relation": self.relation,
            "time_position": self.time_position,
            "event_index": self.event_index,
            "pair_index": self.pair_index,
            "score": self.score,
            "replace_family": self.replace_family,
        }


class ActionModel(nn.Module):
    """Legacy placeholder kept only so old imports fail clearly."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise WorldModelResetError(_reset_message("ActionModel.__init__"))


class WorldModel(nn.Module):
    """Reset baseline for the next world-model redesign.

    The old action-sequence predictor has been removed on purpose. This class
    currently preserves only action-vocabulary utilities and compatibility
    fields so the rest of the project can evolve toward the new
    world_state/event_state design without inheriting the previous logic.
    """

    def __init__(
        self,
        noun_dim: int,
        action_dim: int,
        attention_dim: int,
        value_dim: int,
        hidden_dim: int,
        action_names: Optional[Sequence[str]] = None,
        action_space_names: Optional[Sequence[str]] = None,
        action_space_relations: Optional[Dict[str, str]] = None,
        action_lrs: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.noun_dim = int(noun_dim)
        self.action_dim = int(action_dim)
        self.attention_dim = int(attention_dim)
        self.value_dim = int(value_dim)
        self.hidden_dim = int(hidden_dim)

        if action_names is not None:
            set_action_list(action_names)

        self.action_list = list(get_full_action_type_list())
        self.model_count = len(self.action_list) - 1
        self.action_embeddings = nn.Embedding(len(self.action_list), self.action_dim)
        self.action_space_list = self._build_action_space_list(action_space_names)
        self.action_space_count = len(self.action_space_list) - 1
        self.action_space_embeddings = nn.Embedding(len(self.action_space_list), self.action_dim)
        self.action_space_relation_map = self._build_action_space_relation_map(action_space_relations)
        self.instance_transforms = nn.ModuleList(
            [nn.Linear(self.noun_dim, self.noun_dim) for _ in range(self.model_count)]
        )
        with torch.no_grad():
            self.action_embeddings.weight[0].zero_()
            self.action_space_embeddings.weight[0].zero_()

        if action_lrs is None:
            self.action_learning_rates = [1e-3] * len(self.action_list)
        else:
            self.action_learning_rates = [float(lr) for lr in action_lrs]

        self.action_models = nn.ModuleList()

    def _build_action_space_list(self, action_space_names: Optional[Sequence[str]]) -> list[str]:
        if action_space_names is None:
            return ["no_space_action"]
        normalized = []
        for action_name in action_space_names:
            action_name = str(action_name).lower()
            if action_name == "no_space_action":
                continue
            if action_name not in normalized:
                normalized.append(action_name)
        return ["no_space_action"] + normalized

    def _build_action_space_relation_map(
        self,
        action_space_relations: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for action_name in self.action_space_list[1:]:
            mapping[action_name] = action_name
        if action_space_relations:
            for action_name, relation_name in action_space_relations.items():
                mapping[str(action_name).lower()] = str(relation_name).lower()
        return mapping

    def sync_action_list(self, action_names: Sequence[str]) -> None:
        old_action_list = list(self.action_list)
        old_embeddings = self.action_embeddings
        old_transforms = self.instance_transforms
        set_action_list(action_names)
        self.action_list = list(get_full_action_type_list())
        self.model_count = len(self.action_list) - 1
        self.action_embeddings = nn.Embedding(len(self.action_list), self.action_dim)
        self.instance_transforms = nn.ModuleList(
            [nn.Linear(self.noun_dim, self.noun_dim) for _ in range(self.model_count)]
        )

        old_index = {name.lower(): idx for idx, name in enumerate(old_action_list)}
        with torch.no_grad():
            for new_idx, action_name in enumerate(self.action_list):
                old_idx = old_index.get(action_name.lower())
                if old_idx is not None and old_idx < old_embeddings.weight.shape[0]:
                    self.action_embeddings.weight[new_idx].copy_(old_embeddings.weight[old_idx])
                if (
                    old_idx is not None
                    and old_idx != 0
                    and (old_idx - 1) < len(old_transforms)
                    and (new_idx - 1) < len(self.instance_transforms)
                ):
                    self.instance_transforms[new_idx - 1].load_state_dict(
                        old_transforms[old_idx - 1].state_dict()
                    )
            self.action_embeddings.weight[0].zero_()

    def sync_action_space_list(
        self,
        action_space_names: Sequence[str],
        action_space_relations: Optional[Dict[str, str]] = None,
    ) -> None:
        old_action_space_list = list(self.action_space_list)
        old_embeddings = self.action_space_embeddings
        self.action_space_list = self._build_action_space_list(action_space_names)
        self.action_space_count = len(self.action_space_list) - 1
        self.action_space_embeddings = nn.Embedding(len(self.action_space_list), self.action_dim)
        self.action_space_relation_map = self._build_action_space_relation_map(action_space_relations)

        old_index = {name.lower(): idx for idx, name in enumerate(old_action_space_list)}
        with torch.no_grad():
            for new_idx, action_name in enumerate(self.action_space_list):
                old_idx = old_index.get(action_name.lower())
                if old_idx is not None and old_idx < old_embeddings.weight.shape[0]:
                    self.action_space_embeddings.weight[new_idx].copy_(old_embeddings.weight[old_idx])
            self.action_space_embeddings.weight[0].zero_()

    def build_optimizer(self):
        return torch.optim.Adam(
            [
                {"params": self.action_embeddings.parameters(), "lr": 1e-3},
                {"params": self.action_space_embeddings.parameters(), "lr": 1e-3},
                {"params": self.instance_transforms.parameters(), "lr": 1e-3},
            ]
        )

    def action_type_to_idx(self, action_type: int) -> int:
        action_type = int(action_type)
        if action_type < 1 or action_type > self.model_count:
            raise ValueError(
                f"action_type {action_type} out of range [1, {self.model_count}] with 0 reserved for no_action"
            )
        return action_type - 1

    def get_action_embedding(self, action_type: int) -> torch.Tensor:
        action_type = int(action_type)
        if action_type < 0 or action_type >= len(self.action_list):
            raise ValueError(
                f"action_type {action_type} out of range [0, {len(self.action_list) - 1}]"
            )
        device = self.action_embeddings.weight.device
        action_type_tensor = torch.tensor(action_type, dtype=torch.long, device=device)
        return self.action_embeddings(action_type_tensor)

    def action_space_type_to_idx(self, action_space_type: int) -> int:
        action_space_type = int(action_space_type)
        if action_space_type < 1 or action_space_type > self.action_space_count:
            raise ValueError(
                f"action_space_type {action_space_type} out of range [1, {self.action_space_count}] "
                "with 0 reserved for no_space_action"
            )
        return action_space_type - 1

    def get_action_space_embedding(self, action_space_type: int) -> torch.Tensor:
        action_space_type = int(action_space_type)
        if action_space_type < 0 or action_space_type >= len(self.action_space_list):
            raise ValueError(
                f"action_space_type {action_space_type} out of range [0, {len(self.action_space_list) - 1}]"
            )
        device = self.action_space_embeddings.weight.device
        action_space_tensor = torch.tensor(action_space_type, dtype=torch.long, device=device)
        return self.action_space_embeddings(action_space_tensor)

    def get_action_space_name(self, action_space_type: int) -> str:
        action_space_type = int(action_space_type)
        if action_space_type < 0 or action_space_type >= len(self.action_space_list):
            raise ValueError(
                f"action_space_type {action_space_type} out of range [0, {len(self.action_space_list) - 1}]"
            )
        return self.action_space_list[action_space_type]

    def resolve_action_space_relation(
        self,
        action_space_type: int,
        relation: Optional[str] = None,
    ) -> str:
        if relation is not None:
            return str(relation).lower()
        action_space_name = self.get_action_space_name(action_space_type)
        if action_space_name == "no_space_action":
            raise ValueError("no_space_action does not map to a spatial relation")
        return self.action_space_relation_map.get(action_space_name, action_space_name)

    def infer_action_type(self, action_tensor: torch.Tensor, top_k: int = 1, include_no_action: bool = False):
        if action_tensor.dim() > 1:
            action_tensor = action_tensor.view(-1)

        if action_tensor.numel() != self.action_dim:
            raise ValueError(
                f"action_tensor must have {self.action_dim} values, got {action_tensor.numel()}"
            )

        all_embeddings = self.action_embeddings.weight
        if include_no_action:
            embeddings = all_embeddings
            index_offset = 0
        else:
            embeddings = all_embeddings[1:]
            index_offset = 1

        if embeddings.shape[0] == 0:
            raise ValueError("no trainable action embeddings are registered")

        action_tensor = action_tensor.to(embeddings.device, dtype=embeddings.dtype)
        query = F.normalize(action_tensor.unsqueeze(0), dim=1)
        normalized_embeddings = F.normalize(embeddings, dim=1)

        top_k = min(int(top_k), embeddings.shape[0])
        similarity = torch.matmul(query, normalized_embeddings.t()).squeeze(0)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        top_indices = top_indices + index_offset
        return top_indices, top_scores

    def nearest_action_type(self, action_tensor: torch.Tensor, include_no_action: bool = False) -> int:
        top_indices, _ = self.infer_action_type(
            action_tensor, top_k=1, include_no_action=include_no_action
        )
        return int(top_indices[0].item())

    def forward(self, noun_embedding: torch.Tensor, action_type: int) -> torch.Tensor:
        return self.transform_instance_embedding(noun_embedding, action_type)

    def transform_instance_embedding(self, noun_embedding: torch.Tensor, action_type: int) -> torch.Tensor:
        if noun_embedding is None:
            raise ValueError("noun_embedding is required")
        action_idx = self.action_type_to_idx(action_type)
        transform = self.instance_transforms[action_idx]
        device = next(transform.parameters()).device
        noun_embedding = noun_embedding.to(device=device, dtype=transform.weight.dtype).view(-1)
        if noun_embedding.numel() != self.noun_dim:
            raise ValueError(
                f"noun_embedding must have {self.noun_dim} values, got {noun_embedding.numel()}"
            )
        return transform(noun_embedding.unsqueeze(0)).squeeze(0)

    def update_instance_embedding(
        self,
        short_memory,
        noun_instance_id: str,
        action_type: int,
        *,
        noun_type: Optional[int] = None,
        noun_text: Optional[str] = None,
        action_text: Optional[str] = None,
        score: float = 0.5,
        time_position: Optional[int] = None,
        pair_index: Optional[int] = None,
        event_index: Optional[int] = None,
        action_instance_id: Optional[str] = None,
        append_event: bool = True,
    ) -> InstanceUpdateResult:
        old_embedding = short_memory.get_noun_embedding(noun_instance_id)
        if old_embedding is None:
            raise ValueError(f"noun instance `{noun_instance_id}` is not stored in short_memory")

        if noun_type is None:
            focus_entry = short_memory.get_focus_entry()
            if focus_entry is not None and focus_entry.noun_instance_id == noun_instance_id:
                noun_type = focus_entry.noun_type

        if noun_text is None:
            metadata = short_memory.get_noun_instance_metadata(noun_instance_id)
            if metadata is not None:
                noun_text = metadata.get("noun_text")

        new_embedding = self.transform_instance_embedding(old_embedding, action_type).detach()
        short_memory.store_noun_instance(noun_instance_id, new_embedding, noun_text=noun_text)

        stored_action_instance_id = action_instance_id
        if append_event:
            if time_position is None:
                focus_entry = short_memory.get_focus_entry()
                if focus_entry is None or focus_entry.time_position is None:
                    time_position = len(short_memory.event_entries)
                else:
                    time_position = int(focus_entry.time_position) + 1
            if event_index is None:
                event_index = short_memory.next_event_index()
            action_embedding = self.get_action_embedding(action_type).detach().clone()
            short_memory.append_event(
                noun_embedding=new_embedding,
                action_embedding=action_embedding,
                score=score,
                noun_type=noun_type,
                action_type=action_type,
                time_position=int(time_position),
                pair_index=pair_index,
                event_index=event_index,
                noun_text=noun_text,
                action_text=action_text,
                noun_instance_id=noun_instance_id,
                action_instance_id=action_instance_id,
            )
            focus_entry = short_memory.get_focus_entry()
            stored_action_instance_id = None if focus_entry is None else focus_entry.action_instance_id

        return InstanceUpdateResult(
            noun_instance_id=noun_instance_id,
            noun_type=noun_type,
            action_type=int(action_type),
            action_instance_id=stored_action_instance_id,
            old_embedding=old_embedding.detach().clone(),
            new_embedding=new_embedding.detach().clone(),
            time_position=time_position,
            event_index=event_index,
            pair_index=pair_index,
            score=float(score),
        )

    def update_focus_instance_embedding(
        self,
        short_memory,
        action_type: int,
        *,
        steps=None,
        score: float = 0.5,
        action_text: Optional[str] = None,
        append_event: bool = True,
    ) -> InstanceUpdateResult:
        focus_entry = short_memory.get_focus_entry(steps=steps)
        if focus_entry is None or focus_entry.noun_instance_id is None:
            raise ValueError("short_memory has no focus noun instance to update")
        return self.update_instance_embedding(
            short_memory,
            focus_entry.noun_instance_id,
            action_type,
            noun_type=focus_entry.noun_type,
            noun_text=focus_entry.noun_text,
            action_text=action_text,
            score=score,
            append_event=append_event,
        )

    def update_space_state(
        self,
        short_memory,
        source_instance_id: str,
        target_instance_id: str,
        action_space_type: int,
        *,
        relation: Optional[str] = None,
        noun_type: Optional[int] = None,
        noun_text: Optional[str] = None,
        action_text: Optional[str] = None,
        score: float = 0.5,
        time_position: Optional[int] = None,
        pair_index: Optional[int] = None,
        event_index: Optional[int] = None,
        replace_family: bool = True,
        append_event: bool = True,
    ) -> SpaceUpdateResult:
        source_embedding = short_memory.get_noun_embedding(source_instance_id)
        if source_embedding is None:
            raise ValueError(f"source instance `{source_instance_id}` is not stored in short_memory")

        if noun_type is None:
            focus_entry = short_memory.get_focus_entry()
            if focus_entry is not None and focus_entry.noun_instance_id == source_instance_id:
                noun_type = focus_entry.noun_type

        if noun_text is None:
            metadata = short_memory.get_noun_instance_metadata(source_instance_id)
            if metadata is not None:
                noun_text = metadata.get("noun_text")

        resolved_relation = self.resolve_action_space_relation(action_space_type, relation=relation)
        if time_position is None:
            focus_entry = short_memory.get_focus_entry()
            if focus_entry is None or focus_entry.time_position is None:
                time_position = len(short_memory.event_entries)
            else:
                time_position = int(focus_entry.time_position) + 1
        if event_index is None:
            event_index = short_memory.next_event_index()

        short_memory.append_spatial_fact(
            source_instance_id=source_instance_id,
            relation=resolved_relation,
            target_instance_id=target_instance_id,
            time_position=int(time_position),
            replace_family=replace_family,
        )

        if append_event:
            space_action_embedding = self.get_action_space_embedding(action_space_type).detach().clone()
            effective_action_text = action_text or self.get_action_space_name(action_space_type)
            short_memory.append_event(
                noun_embedding=source_embedding.detach().clone(),
                action_embedding=space_action_embedding,
                score=score,
                noun_type=noun_type,
                action_type=None,
                time_position=int(time_position),
                pair_index=pair_index,
                event_index=event_index,
                noun_text=noun_text,
                action_text=effective_action_text,
                noun_instance_id=source_instance_id,
                action_instance_id=None,
                pair_kind="noun_space_action",
                info_pair={
                    "event_channel": "space",
                    "action_space_type": int(action_space_type),
                    "action_space_name": self.get_action_space_name(action_space_type),
                    "space_relation": resolved_relation,
                    "space_target_instance_id": target_instance_id,
                },
            )

        return SpaceUpdateResult(
            source_instance_id=source_instance_id,
            target_instance_id=target_instance_id,
            action_space_type=int(action_space_type),
            action_space_name=self.get_action_space_name(action_space_type),
            relation=resolved_relation,
            time_position=time_position,
            event_index=event_index,
            pair_index=pair_index,
            score=float(score),
            replace_family=bool(replace_family),
        )

    def update_focus_space_state(
        self,
        short_memory,
        target_instance_id: str,
        action_space_type: int,
        *,
        relation: Optional[str] = None,
        steps=None,
        score: float = 0.5,
        action_text: Optional[str] = None,
        replace_family: bool = True,
        append_event: bool = True,
    ) -> SpaceUpdateResult:
        focus_entry = short_memory.get_focus_entry(steps=steps)
        if focus_entry is None or focus_entry.noun_instance_id is None:
            raise ValueError("short_memory has no focus noun instance to update in space_state")
        return self.update_space_state(
            short_memory,
            focus_entry.noun_instance_id,
            target_instance_id,
            action_space_type,
            relation=relation,
            noun_type=focus_entry.noun_type,
            noun_text=focus_entry.noun_text,
            action_text=action_text,
            score=score,
            replace_family=replace_family,
            append_event=append_event,
        )

    def encode_sequence_context(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.encode_sequence_context"))

    def predict_action_from_context(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.predict_action_from_context"))

    def predict_from_short_memory(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.predict_from_short_memory"))

    def predict_next_event(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.predict_next_event"))

    def append_predicted_event(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.append_predicted_event"))

    def training_step_next_event(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.training_step_next_event"))

    def training_step_from_short_memory(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.training_step_from_short_memory"))

    def autoregressive_step(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.autoregressive_step"))

    def autoregressive_rollout(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.autoregressive_rollout"))

    def train_action_model(self, *args, **kwargs):
        raise WorldModelResetError(_reset_message("WorldModel.train_action_model"))


Action_list = get_full_action_type_list()


def train_action_model(*args, **kwargs):
    raise WorldModelResetError(_reset_message("train_action_model"))
