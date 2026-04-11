from dataclasses import dataclass
from typing import Optional


@dataclass
class SubjectEventSurpriseInput:
    subject_instance_id: Optional[str] = None
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_instance_id: Optional[str] = None
    object_text: Optional[str] = None


@dataclass
class SubjectEventSurpriseSample:
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None
    surprise_value: float = 0.0
    weight: float = 1.0
    source: str = "manual"


@dataclass
class SubjectEventSurprisePrediction:
    score: float
    subject_text: Optional[str] = None
    action_text: Optional[str] = None
    object_text: Optional[str] = None
    subject_instance_id: Optional[str] = None
    object_instance_id: Optional[str] = None

    @property
    def label(self) -> str:
        if self.score <= -60.0:
            return "strongly_expected"
        if self.score < -20.0:
            return "expected"
        if self.score <= 20.0:
            return "neutral"
        if self.score < 60.0:
            return "surprise"
        return "high_surprise"
