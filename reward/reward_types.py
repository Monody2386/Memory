from dataclasses import dataclass
from typing import Optional


@dataclass
class SubjectEvent:
    subject_instance_id: Optional[str] = None
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_instance_id: Optional[str] = None
    object_text: Optional[str] = None
    time_position: Optional[int] = None
    subject_pair_index: Optional[int] = None
    object_pair_index: Optional[int] = None


@dataclass
class SubjectEventRewardInput:
    subject_instance_id: Optional[str] = None
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_instance_id: Optional[str] = None
    object_text: Optional[str] = None


@dataclass
class SubjectEventRewardSample:
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None
    reward_value: float = 0.0
    weight: float = 1.0
    source: str = "manual"


@dataclass
class SubjectEventRewardPrediction:
    score: float
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None

    @property
    def label(self) -> str:
        if self.score <= -60.0:
            return "very_dislike"
        if self.score < -20.0:
            return "dislike"
        if self.score <= 20.0:
            return "neutral"
        if self.score < 60.0:
            return "like"
        return "love"


@dataclass
class RewardInput:
    noun_text: Optional[str] = None
    action_text: Optional[str] = None
    noun_instance_id: Optional[str] = None


@dataclass
class RewardSample:
    noun_text: Optional[str] = None
    action_text: Optional[str] = None
    noun_instance_id: Optional[str] = None
    reward_value: float = 0.0
    weight: float = 1.0
    source: str = "manual"


@dataclass
class ObservedRewardPrediction:
    base_score: float
    empathy: float
    relation: float
    final_score: float
    observer_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    noun_text: Optional[str] = None
    action_text: Optional[str] = None

    @property
    def label(self) -> str:
        if self.final_score <= -60.0:
            return "very_dislike"
        if self.final_score < -20.0:
            return "dislike"
        if self.final_score <= 20.0:
            return "neutral"
        if self.final_score < 60.0:
            return "like"
        return "love"


@dataclass
class RewardPrediction:
    score: float
    noun_text: Optional[str] = None
    action_text: Optional[str] = None
    noun_instance_id: Optional[str] = None

    @property
    def label(self) -> str:
        if self.score <= -60.0:
            return "very_dislike"
        if self.score < -20.0:
            return "dislike"
        if self.score <= 20.0:
            return "neutral"
        if self.score < 60.0:
            return "like"
        return "love"
