from __future__ import annotations

from .data import RewardEncoder
from .model import RewardNet, SubjectEventRewardNet
from .reward_types import ObservedRewardPrediction, RewardInput, RewardPrediction, SubjectEventRewardInput, SubjectEventRewardPrediction


class RewardEngine:
    def __init__(self, model: RewardNet, encoder: RewardEncoder):
        self.model = model
        self.encoder = encoder

    def predict(self, reward_input: RewardInput) -> RewardPrediction:
        noun_embedding = self.encoder.encode_noun(
            noun_text=reward_input.noun_text,
            noun_instance_id=reward_input.noun_instance_id,
        )
        action_embedding = self.encoder.encode_action(action_text=reward_input.action_text)

        self.model.eval()
        with __import__('torch').no_grad():
            score = self.model(
                noun_embedding=noun_embedding,
                action_embedding=action_embedding,
            )

        return RewardPrediction(
            score=float(score.view(-1)[0].item()),
            noun_text=reward_input.noun_text,
            action_text=reward_input.action_text,
            noun_instance_id=reward_input.noun_instance_id,
        )

    def predict_observed_reward(
        self,
        reward_input: RewardInput,
        *,
        observer_instance_id: str,
        target_instance_id: str,
        empathy: float,
        relation: float,
    ) -> ObservedRewardPrediction:
        base_prediction = self.predict(reward_input)
        empathy = max(0.0, min(1.0, float(empathy)))
        relation = max(-1.0, min(1.0, float(relation)))
        final_score = base_prediction.score * empathy * relation
        final_score = max(-100.0, min(100.0, final_score))
        return ObservedRewardPrediction(
            base_score=float(base_prediction.score),
            empathy=empathy,
            relation=relation,
            final_score=float(final_score),
            observer_instance_id=observer_instance_id,
            target_instance_id=target_instance_id,
            noun_text=reward_input.noun_text,
            action_text=reward_input.action_text,
        )

class SubjectEventRewardEngine:
    def __init__(self, model: SubjectEventRewardNet, encoder: RewardEncoder):
        self.model = model
        self.encoder = encoder

    def predict(self, reward_input: SubjectEventRewardInput) -> SubjectEventRewardPrediction:
        subject_embedding = self.encoder.encode_noun(
            noun_text=reward_input.subject_text,
            noun_instance_id=reward_input.subject_instance_id,
        )
        action_embedding = self.encoder.encode_action(action_text=reward_input.action_text)
        object_embedding = self.encoder.encode_noun(
            noun_text=reward_input.object_text,
            noun_instance_id=reward_input.object_instance_id,
        )

        self.model.eval()
        with __import__('torch').no_grad():
            score = self.model(
                subject_embedding=subject_embedding,
                action_embedding=action_embedding,
                object_embedding=object_embedding,
            )

        return SubjectEventRewardPrediction(
            score=float(score.view(-1)[0].item()),
            subject_text=reward_input.subject_text,
            action_text=reward_input.action_text,
            object_text=reward_input.object_text,
            subject_instance_id=reward_input.subject_instance_id,
            object_instance_id=reward_input.object_instance_id,
        )

