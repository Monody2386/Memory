from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_vocab import get_full_action_type_list
from .world_model import action_dim, attention_dim, hidden_dim, noun_dim, value_dim


@dataclass
class AdaLNWorldConfig:
    noun_dim: int = noun_dim
    action_dim: int = action_dim
    attention_dim: int = attention_dim
    value_dim: int = value_dim
    hidden_dim: int = hidden_dim
    action_names: Optional[Sequence[str]] = None


class World_AdaLN(nn.Module):
    """Skeleton for a future AdaLN-based world model.

    This class intentionally mirrors the minimal protocol used by Consciousness
    and grammar_layer so it can be injected without changing short memory.
    Prediction and training are left as explicit extension points.
    """

    def __init__(
        self,
        noun_dim: int = noun_dim,
        action_dim: int = action_dim,
        attention_dim: int = attention_dim,
        value_dim: int = value_dim,
        hidden_dim: int = hidden_dim,
        action_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.noun_dim = int(noun_dim)
        self.action_dim = int(action_dim)
        self.attention_dim = int(attention_dim)
        self.value_dim = int(value_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_list = self._build_action_list(action_names)
        self.model_count = len(self.action_list) - 1
        self.action_embeddings = nn.Embedding(len(self.action_list), self.action_dim)
        with torch.no_grad():
            self.action_embeddings.weight[0].zero_()

    @classmethod
    def from_config(cls, config: AdaLNWorldConfig) -> "World_AdaLN":
        return cls(
            noun_dim=config.noun_dim,
            action_dim=config.action_dim,
            attention_dim=config.attention_dim,
            value_dim=config.value_dim,
            hidden_dim=config.hidden_dim,
            action_names=config.action_names,
        )

    def _build_action_list(self, action_names: Optional[Sequence[str]]) -> list[str]:
        if action_names is None:
            return list(get_full_action_type_list())
        return ["no_action"] + [str(action).lower() for action in action_names]

    def sync_action_list(self, action_names: Sequence[str]) -> None:
        """Resize action embeddings while preserving known action rows."""
        old_action_list = list(self.action_list)
        old_embeddings = self.action_embeddings
        self.action_list = self._build_action_list(action_names)
        self.model_count = len(self.action_list) - 1
        self.action_embeddings = nn.Embedding(len(self.action_list), self.action_dim)

        old_index = {name.lower(): idx for idx, name in enumerate(old_action_list)}
        with torch.no_grad():
            for new_idx, action_name in enumerate(self.action_list):
                old_idx = old_index.get(action_name.lower())
                if old_idx is None or old_idx >= old_embeddings.weight.shape[0]:
                    continue
                self.action_embeddings.weight[new_idx].copy_(old_embeddings.weight[old_idx])
            self.action_embeddings.weight[0].zero_()

    def build_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _normalize_action_type(self, action_type: int) -> int:
        action_type = int(action_type)
        if action_type < 0 or action_type > self.model_count:
            raise ValueError(
                f"action_type {action_type} out of range [0, {self.model_count}]"
            )
        return action_type

    def get_action_embedding(self, action_type: int) -> torch.Tensor:
        action_type = self._normalize_action_type(action_type)
        device = self.action_embeddings.weight.device
        action_type_tensor = torch.tensor(action_type, dtype=torch.long, device=device)
        return self.action_embeddings(action_type_tensor)

    def infer_action_type(
        self,
        action_tensor: torch.Tensor,
        top_k: int = 1,
        include_no_action: bool = False,
    ):
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

        top_k = min(int(top_k), embeddings.shape[0])
        query = F.normalize(action_tensor.to(embeddings.device).unsqueeze(0), dim=1)
        normalized_embeddings = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(query, normalized_embeddings.t()).squeeze(0)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        return top_indices + index_offset, top_scores

    def nearest_action_type(self, action_tensor: torch.Tensor, include_no_action: bool = False) -> int:
        top_indices, _ = self.infer_action_type(
            action_tensor,
            top_k=1,
            include_no_action=include_no_action,
        )
        return int(top_indices[0].item())

    def encode_sequence_context(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.encode_sequence_context is waiting for the AdaLN sequence encoder design.")

    def predict_action_from_context(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.predict_action_from_context is waiting for the AdaLN prediction head design.")

    def forward(self, input_2d: torch.Tensor, action_type: int) -> torch.Tensor:
        context = self.encode_sequence_context(input_2d, action_type)
        return self.predict_action_from_context(context, action_type)

    def predict_next_event(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.predict_next_event is not implemented yet.")

    def training_step_next_event(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.training_step_next_event is not implemented yet.")

    def append_predicted_event(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.append_predicted_event is not implemented yet.")

    def autoregressive_step(self, *args, **kwargs):
        raise NotImplementedError("World_AdaLN.autoregressive_step is not implemented yet.")


WorldAdaLN = World_AdaLN
