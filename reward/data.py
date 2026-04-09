from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch

from .reward_types import RewardSample, SubjectEventRewardSample


@dataclass
class RewardBatch:
    noun_embeddings: List[Optional[torch.Tensor]]
    action_embeddings: List[Optional[torch.Tensor]]
    targets: torch.Tensor
    weights: torch.Tensor
    samples: List[RewardSample]


@dataclass
class SubjectEventRewardBatch:
    subject_embeddings: List[Optional[torch.Tensor]]
    action_embeddings: List[Optional[torch.Tensor]]
    object_embeddings: List[Optional[torch.Tensor]]
    targets: torch.Tensor
    weights: torch.Tensor
    samples: List[SubjectEventRewardSample]


class RewardDataset:
    def __init__(self, samples: Optional[Sequence[RewardSample]] = None):
        self.samples: List[RewardSample] = list(samples or [])

    def add(self, sample: RewardSample) -> None:
        self.samples.append(sample)

    def extend(self, samples: Iterable[RewardSample]) -> None:
        self.samples.extend(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


class RewardEncoder:
    def __init__(self, consciousness=None):
        self.consciousness = consciousness

    def _short_memory(self):
        return None if self.consciousness is None else self.consciousness.short_memory

    def _world_model(self):
        return None if self.consciousness is None else self.consciousness.world_model

    def encode_noun(
        self,
        *,
        noun_text: Optional[str] = None,
        noun_instance_id: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        short_memory = self._short_memory()
        if noun_instance_id is not None and short_memory is not None:
            embedding = short_memory.get_noun_embedding(noun_instance_id)
            if embedding is not None:
                return embedding.detach().clone().view(-1)

        if noun_text is None:
            return None

        from knowledge import knowledge_map_one
        from knowledge.relation_map import noun_list

        noun_key = noun_text.lower()
        if noun_key not in noun_list:
            return None

        noun_index = noun_list.index(noun_key)
        embedding = knowledge_map_one.embedding.weight.detach()[noun_index].clone()
        return embedding.view(-1)

    def encode_action(self, *, action_text: Optional[str] = None) -> Optional[torch.Tensor]:
        if action_text is None:
            return None

        from world.action_vocab import get_action_list

        action_key = action_text.lower()
        action_list = get_action_list()
        if action_key not in action_list:
            return None

        world_model = self._world_model()
        if world_model is None:
            return None

        action_type = action_list.index(action_key) + 1
        embedding = world_model.get_action_embedding(action_type).detach().clone()
        return embedding.view(-1)

    def encode_sample(self, sample: RewardSample) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        noun_embedding = self.encode_noun(
            noun_text=sample.noun_text,
            noun_instance_id=sample.noun_instance_id,
        )
        action_embedding = self.encode_action(action_text=sample.action_text)
        return noun_embedding, action_embedding

    def collate(self, samples: Sequence[RewardSample]) -> RewardBatch:
        noun_embeddings: List[Optional[torch.Tensor]] = []
        action_embeddings: List[Optional[torch.Tensor]] = []
        targets = []
        weights = []

        for sample in samples:
            noun_embedding, action_embedding = self.encode_sample(sample)
            noun_embeddings.append(noun_embedding)
            action_embeddings.append(action_embedding)
            targets.append(float(sample.reward_value))
            weights.append(float(sample.weight))

        return RewardBatch(
            noun_embeddings=noun_embeddings,
            action_embeddings=action_embeddings,
            targets=torch.tensor(targets, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32),
            samples=list(samples),
        )

    def encode_subject_event_sample(
        self,
        sample: SubjectEventRewardSample,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        subject_embedding = self.encode_noun(
            noun_text=sample.subject_text,
            noun_instance_id=sample.subject_instance_id,
        )
        action_embedding = self.encode_action(action_text=sample.action_text)
        object_embedding = self.encode_noun(
            noun_text=sample.object_text,
            noun_instance_id=sample.object_instance_id,
        )
        return subject_embedding, action_embedding, object_embedding

    def collate_subject_event_samples(
        self,
        samples: Sequence[SubjectEventRewardSample],
    ) -> SubjectEventRewardBatch:
        subject_embeddings: List[Optional[torch.Tensor]] = []
        action_embeddings: List[Optional[torch.Tensor]] = []
        object_embeddings: List[Optional[torch.Tensor]] = []
        targets = []
        weights = []

        for sample in samples:
            subject_embedding, action_embedding, object_embedding = self.encode_subject_event_sample(sample)
            subject_embeddings.append(subject_embedding)
            action_embeddings.append(action_embedding)
            object_embeddings.append(object_embedding)
            targets.append(float(sample.reward_value))
            weights.append(float(sample.weight))

        return SubjectEventRewardBatch(
            subject_embeddings=subject_embeddings,
            action_embeddings=action_embeddings,
            object_embeddings=object_embeddings,
            targets=torch.tensor(targets, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32),
            samples=list(samples),
        )
