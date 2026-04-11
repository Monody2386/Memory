from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from reward import RewardEncoder
from .surprise_types import SubjectEventSurpriseSample


@dataclass
class SubjectEventSurpriseBatch:
    subject_embeddings: List[Optional[torch.Tensor]]
    action_embeddings: List[Optional[torch.Tensor]]
    object_embeddings: List[Optional[torch.Tensor]]
    targets: torch.Tensor
    weights: torch.Tensor
    samples: List[SubjectEventSurpriseSample]


class SurpriseEncoder(RewardEncoder):
    """Uses the same subject/action/object embedding lookup as RewardEncoder."""

    def collate_subject_event_samples(
        self,
        samples: Sequence[SubjectEventSurpriseSample],
    ) -> SubjectEventSurpriseBatch:
        subject_embeddings: List[Optional[torch.Tensor]] = []
        action_embeddings: List[Optional[torch.Tensor]] = []
        object_embeddings: List[Optional[torch.Tensor]] = []
        targets = []
        weights = []

        for sample in samples:
            subject_embedding = self.encode_noun(
                noun_text=sample.subject_text,
                noun_instance_id=sample.subject_instance_id,
            )
            action_embedding = self.encode_action(action_text=sample.action_text)
            object_embedding = self.encode_noun(
                noun_text=sample.object_text,
                noun_instance_id=sample.object_instance_id,
            )
            subject_embeddings.append(subject_embedding)
            action_embeddings.append(action_embedding)
            object_embeddings.append(object_embedding)
            targets.append(float(sample.surprise_value))
            weights.append(float(sample.weight))

        return SubjectEventSurpriseBatch(
            subject_embeddings=subject_embeddings,
            action_embeddings=action_embeddings,
            object_embeddings=object_embeddings,
            targets=torch.tensor(targets, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32),
            samples=list(samples),
        )
