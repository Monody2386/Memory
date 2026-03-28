from dataclasses import dataclass
from typing import Optional

import torch

from knowledge.relation_map import (
    lr_per_embedding,
    noun_dim,
    noun_number,
    relation_map,
    relation_num,
)


def search_relation(relation_map, i, j, relation_type):
    return relation_map[i][j] == relation_type


def update_relation_map(relation_map, i, j, relation_type):
    relation_map[i][j] = relation_type


@dataclass
class MemoryEntry:
    tensor: torch.Tensor
    score: float
    noun_type: Optional[int]
    action_type: Optional[int]
    time_position: int
    pair_index: int
    noun_embedding: torch.Tensor
    action_embedding: torch.Tensor


class ShortMemory:
    def __init__(self, maxlen=100, device="cpu", state_dim=None):
        self.maxlen = maxlen
        self.device = device
        self.state_dim = state_dim
        self.entries = []
        self._insert_counter = 0

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

    def _encode_state(
        self,
        noun_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        time_position: int,
        pair_index: int,
    ) -> torch.Tensor:
        noun_embedding = noun_embedding.to(self.device).view(-1)
        action_embedding = action_embedding.to(self.device).view(-1)
        expected_dim = self._resolve_state_dim(noun_embedding, action_embedding)
        state_tensor = torch.cat([noun_embedding, action_embedding], dim=0)
        if state_tensor.numel() != expected_dim:
            raise ValueError(
                f"state tensor must have {expected_dim} values, got {state_tensor.numel()}"
            )
        return state_tensor + self._sinusoidal_position_encoding(time_position, pair_index)

    def _sort_entries(self):
        self.entries.sort(
            key=lambda entry: (entry.time_position, entry.score, entry.pair_index)
        )

    def _trim(self):
        while len(self.entries) > self.maxlen:
            self.entries.pop(0)

    def append(self, tensor, score=0.0, noun_type=None, action_type=None):
        return self.append_state(
            noun_embedding=tensor[:noun_dim],
            action_embedding=tensor[noun_dim:],
            score=score,
            noun_type=noun_type,
            action_type=action_type,
            time_position=len(self.entries),
        )

    def append_state(
        self,
        noun_embedding: torch.Tensor,
        action_embedding: torch.Tensor,
        score=0.0,
        noun_type=None,
        action_type=None,
        time_position: int = 0,
        pair_index: Optional[int] = None,
    ):
        if pair_index is None:
            pair_index = self._insert_counter
        encoded_tensor = self._encode_state(
            noun_embedding, action_embedding, int(time_position), int(pair_index)
        )
        entry = MemoryEntry(
            tensor=encoded_tensor,
            score=float(score),
            noun_type=None if noun_type is None else int(noun_type),
            action_type=None if action_type is None else int(action_type),
            time_position=int(time_position),
            pair_index=int(pair_index),
            noun_embedding=noun_embedding.to(self.device).view(-1),
            action_embedding=action_embedding.to(self.device).view(-1),
        )
        self.entries.append(entry)
        self._insert_counter += 1
        self._sort_entries()
        self._trim()
        return encoded_tensor

    def set_maxlen(self, new_maxlen):
        self.maxlen = new_maxlen
        self._trim()

    def filter_by_score(self, threshold, mode="ge"):
        if mode == "ge":
            self.entries = [entry for entry in self.entries if entry.score >= threshold]
        elif mode == "le":
            self.entries = [entry for entry in self.entries if entry.score <= threshold]
        else:
            raise ValueError("mode must be 'ge' or 'le'")
        self._sort_entries()

    def boost_related(self, noun_type=None, action_type=None, amount=1.0):
        if noun_type is None and action_type is None:
            raise ValueError("Provide noun_type index or action_type to boost")

        noun_type = None if noun_type is None else int(noun_type)
        action_type = None if action_type is None else int(action_type)

        for entry in self.entries:
            matched = False
            if noun_type is not None and entry.noun_type == noun_type:
                matched = True
            if action_type is not None and entry.action_type == action_type:
                matched = True
            if matched:
                entry.score += float(amount)

        self._sort_entries()

    def get_stack(self):
        if len(self.entries) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                [],
                [],
            )

        tensors = [entry.tensor for entry in self.entries]
        scores = [entry.score for entry in self.entries]
        noun_types = [entry.noun_type for entry in self.entries]
        action_types = [entry.action_type for entry in self.entries]
        return (
            torch.stack(tensors),
            torch.tensor(scores, device=self.device),
            noun_types,
            action_types,
        )

    def get_latest_n(self, n):
        if not self.entries:
            return torch.empty(0, device=self.device)
        latest = self.entries[-n:]
        return torch.stack([entry.tensor for entry in latest])

    def build_world_model_input(self, steps=None):
        if len(self.entries) == 0:
            return torch.empty((0, 0), device=self.device)
        selected = self.entries if steps is None else self.entries[-steps:]
        return torch.stack([entry.tensor for entry in selected], dim=1)

    def latest_state(self):
        if not self.entries:
            return None
        entry = self.entries[-1]
        return (entry.tensor, entry.score, entry.noun_type, entry.action_type)

    def __len__(self):
        return len(self.entries)

    def filter_by_type(self, noun_type=None, action_type=None):
        noun_type = None if noun_type is None else int(noun_type)
        action_type = None if action_type is None else int(action_type)
        kept = []
        for entry in self.entries:
            if noun_type is not None and entry.noun_type != noun_type:
                continue
            if action_type is not None and entry.action_type != action_type:
                continue
            kept.append(entry)
        self.entries = kept
        self._sort_entries()

    def clear(self):
        self.entries.clear()


ScoredTensorQueue = ShortMemory
short_memory = ShortMemory(maxlen=50, device="cpu")
