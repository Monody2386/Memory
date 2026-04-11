from __future__ import annotations

from .data import SurpriseEncoder
from .model import SubjectEventSurpriseNet
from .surprise_types import SubjectEventSurpriseInput, SubjectEventSurprisePrediction


class SubjectEventSurpriseEngine:
    def __init__(self, model: SubjectEventSurpriseNet, encoder: SurpriseEncoder):
        self.model = model
        self.encoder = encoder

    def predict(self, surprise_input: SubjectEventSurpriseInput) -> SubjectEventSurprisePrediction:
        subject_embedding = self.encoder.encode_noun(
            noun_text=surprise_input.subject_text,
            noun_instance_id=surprise_input.subject_instance_id,
        )
        action_embedding = self.encoder.encode_action(action_text=surprise_input.action_text)
        object_embedding = self.encoder.encode_noun(
            noun_text=surprise_input.object_text,
            noun_instance_id=surprise_input.object_instance_id,
        )

        self.model.eval()
        with __import__("torch").no_grad():
            score = self.model(
                subject_embedding=subject_embedding,
                action_embedding=action_embedding,
                object_embedding=object_embedding,
            )

        return SubjectEventSurprisePrediction(
            score=float(score.view(-1)[0].item()),
            subject_text=surprise_input.subject_text,
            action_text=surprise_input.action_text,
            object_text=surprise_input.object_text,
            subject_instance_id=surprise_input.subject_instance_id,
            object_instance_id=surprise_input.object_instance_id,
        )
