from __future__ import annotations

from typing import List

from reward import SubjectEvent, subject_events_from_short_memory
from .surprise_types import SubjectEventSurpriseInput, SubjectEventSurpriseSample


def surprise_input_from_subject_event(event: SubjectEvent) -> SubjectEventSurpriseInput:
    return SubjectEventSurpriseInput(
        subject_instance_id=event.subject_instance_id,
        subject_text=event.subject_text,
        action_text=event.action_text,
        object_instance_id=event.object_instance_id,
        object_text=event.object_text,
    )


def surprise_sample_from_subject_event(
    event: SubjectEvent,
    *,
    surprise_value: float,
    weight: float = 1.0,
    source: str = "manual",
) -> SubjectEventSurpriseSample:
    return SubjectEventSurpriseSample(
        subject_text=event.subject_text,
        action_text=event.action_text,
        object_text=event.object_text,
        subject_instance_id=event.subject_instance_id,
        object_instance_id=event.object_instance_id,
        surprise_value=float(surprise_value),
        weight=float(weight),
        source=source,
    )


def subject_events_from_surprise_memory(short_memory, *, include_intransitive: bool = True) -> List[SubjectEvent]:
    return subject_events_from_short_memory(short_memory, include_intransitive=include_intransitive)
