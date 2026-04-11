from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .data import SurpriseEncoder
from .model import SubjectEventSurpriseNet
from .surprise_types import SubjectEventSurpriseSample


class SubjectEventSurpriseTrainer:
    def __init__(self, model: SubjectEventSurpriseNet, encoder: SurpriseEncoder, lr: float = 1e-3):
        self.model = model
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

    def train_step(self, samples: Sequence[SubjectEventSurpriseSample]) -> dict:
        if not samples:
            return {"loss": 0.0, "count": 0}

        batch = self.encoder.collate_subject_event_samples(samples)
        self.model.train()

        preds = []
        for subject_embedding, action_embedding, object_embedding in zip(
            batch.subject_embeddings,
            batch.action_embeddings,
            batch.object_embeddings,
        ):
            pred = self.model(
                subject_embedding=subject_embedding,
                action_embedding=action_embedding,
                object_embedding=object_embedding,
            )
            preds.append(pred.view(-1)[0])

        pred_tensor = torch.stack(preds)
        targets = batch.targets.to(pred_tensor.device).clamp(-100.0, 100.0)
        weights = batch.weights.to(pred_tensor.device)

        base_loss = F.smooth_l1_loss(pred_tensor, targets, reduction="none")
        weighted_loss = (base_loss * weights).sum() / weights.clamp_min(1e-6).sum()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return {
            "loss": float(weighted_loss.item()),
            "count": len(samples),
            "mean_prediction": float(pred_tensor.mean().item()),
            "mean_target": float(targets.mean().item()),
        }

    def train_epochs(self, samples: Sequence[SubjectEventSurpriseSample], epochs: int = 1) -> list[dict]:
        history = []
        for _ in range(int(epochs)):
            history.append(self.train_step(samples))
        return history
