from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SubjectEventSurpriseNet(nn.Module):
    """Predict surprise for a subject/action/object event in the range [-100, 100]."""

    def __init__(self, noun_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.noun_dim = int(noun_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.null_subject = nn.Parameter(torch.zeros(self.noun_dim))
        self.null_action = nn.Parameter(torch.zeros(self.action_dim))
        self.null_object = nn.Parameter(torch.zeros(self.noun_dim))

        input_dim = self.noun_dim + self.action_dim + self.noun_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _coerce_batch(
        self,
        embedding: Optional[torch.Tensor],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if embedding is None:
            embedding = fallback.unsqueeze(0)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return embedding

    def forward(
        self,
        *,
        subject_embedding: Optional[torch.Tensor] = None,
        action_embedding: Optional[torch.Tensor] = None,
        object_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        subject_embedding = self._coerce_batch(subject_embedding, self.null_subject)
        action_embedding = self._coerce_batch(action_embedding, self.null_action)
        object_embedding = self._coerce_batch(object_embedding, self.null_object)

        batch_sizes = {subject_embedding.shape[0], action_embedding.shape[0], object_embedding.shape[0]}
        if len(batch_sizes) != 1:
            raise ValueError("subject/action/object embedding batch sizes must match")

        x = torch.cat([subject_embedding, action_embedding, object_embedding], dim=-1)
        logits = self.mlp(x)
        return (torch.tanh(logits) * 100.0).squeeze(-1)
