from __future__ import annotations

from typing import List, Optional, Set

from .reward_types import SubjectEvent, SubjectEventRewardInput, SubjectEventRewardSample


def reward_input_from_subject_event(event: SubjectEvent) -> SubjectEventRewardInput:
    return SubjectEventRewardInput(
        subject_instance_id=event.subject_instance_id,
        subject_text=event.subject_text,
        action_text=event.action_text,
        object_instance_id=event.object_instance_id,
        object_text=event.object_text,
    )


def reward_sample_from_subject_event(
    event: SubjectEvent,
    *,
    reward_value: float,
    weight: float = 1.0,
    source: str = "manual",
) -> SubjectEventRewardSample:
    return SubjectEventRewardSample(
        subject_text=event.subject_text,
        action_text=event.action_text,
        object_text=event.object_text,
        subject_instance_id=event.subject_instance_id,
        object_instance_id=event.object_instance_id,
        reward_value=float(reward_value),
        weight=float(weight),
        source=source,
    )


def _best_object_entry(subject_entry, object_entries, used_object_ids: Set[int]):
    candidates = [
        entry for entry in object_entries
        if id(entry) not in used_object_ids
        and entry.time_position == subject_entry.time_position
        and entry.pair_index < subject_entry.pair_index
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry.pair_index, reverse=True)
    return candidates[0]


def subject_events_from_short_memory(short_memory, *, include_intransitive: bool = True) -> List[SubjectEvent]:
    entries = list(getattr(short_memory, "short_memory_event", []))
    subject_entries = [entry for entry in entries if entry.role == "subject"]
    object_entries = [entry for entry in entries if entry.role == "object"]
    used_object_ids: Set[int] = set()
    events: List[SubjectEvent] = []

    for subject_entry in sorted(subject_entries, key=lambda entry: (entry.time_position, entry.pair_index)):
        object_entry = _best_object_entry(subject_entry, object_entries, used_object_ids)
        if object_entry is not None:
            used_object_ids.add(id(object_entry))
        elif not include_intransitive:
            continue

        events.append(
            SubjectEvent(
                subject_instance_id=subject_entry.noun_instance_id,
                subject_text=subject_entry.noun_text,
                action_text=subject_entry.action_text,
                object_instance_id=None if object_entry is None else object_entry.noun_instance_id,
                object_text=None if object_entry is None else object_entry.noun_text,
                time_position=subject_entry.time_position,
                subject_pair_index=subject_entry.pair_index,
                object_pair_index=None if object_entry is None else object_entry.pair_index,
            )
        )

    return events


def reward_sample_from_reward_memory_entry(entry) -> SubjectEventRewardSample:
    return SubjectEventRewardSample(
        subject_text=entry.subject_text,
        action_text=entry.action_text,
        object_text=entry.object_text,
        subject_instance_id=entry.subject_instance_id,
        object_instance_id=entry.object_instance_id,
        reward_value=float(entry.reward_value),
        weight=float(getattr(entry, "score", 1.0)),
        source="short_memory_reward",
    )


def reward_samples_from_short_memory(short_memory) -> List[SubjectEventRewardSample]:
    return [
        reward_sample_from_reward_memory_entry(entry)
        for entry in getattr(short_memory, "short_memory_reward", [])
    ]
