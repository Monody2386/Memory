from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExtractorRoute:
    pattern_name: str
    structure: Tuple[str, ...]
    extractor_name: str
    match_mode: str = "exact"


DEFAULT_EXTRACTOR_ROUTES = (
    ExtractorRoute(
        pattern_name="negative_passive_action_with_subject",
        structure=("noun", "be", "negative", "actioned", "noun"),
        extractor_name="_extract_pattern_negative_passive_action_with_subject",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_passive_action_without_subject",
        structure=("noun", "be", "negative", "actioned"),
        extractor_name="_extract_pattern_negative_passive_action_without_subject",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_be_attribute",
        structure=("noun", "be", "negative", "adj"),
        extractor_name="_extract_pattern_negative_be_attribute",
        match_mode="prefix",
    ),
    ExtractorRoute(
        pattern_name="negative_be_noun",
        structure=("noun", "be", "negative", "noun"),
        extractor_name="_extract_pattern_negative_be_noun",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="be_attribute",
        structure=("noun", "be", "adj"),
        extractor_name="_extract_pattern_be_attribute",
        match_mode="prefix",
    ),
    ExtractorRoute(
        pattern_name="passive_action_with_subject",
        structure=("noun", "be", "actioned", "noun"),
        extractor_name="_extract_pattern_passive_action_with_subject",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="passive_action_without_subject",
        structure=("noun", "be", "actioned"),
        extractor_name="_extract_pattern_passive_action_without_subject",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="be_noun",
        structure=("noun", "be", "noun"),
        extractor_name="_extract_pattern_be_noun",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_phrase_relation_noun_phrase",
        structure=("noun", "negative", "relation", "noun"),
        extractor_name="_extract_pattern_negative_relation_between_noun_phrases",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_phrase_relation_noun_phrase",
        structure=("noun", "relation", "noun"),
        extractor_name="_extract_pattern_relation_between_noun_phrases",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_reward_action_object",
        structure=("noun", "negative", "reward", "action", "noun"),
        extractor_name="_extract_pattern_negative_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_reward_action",
        structure=("noun", "negative", "reward", "action"),
        extractor_name="_extract_pattern_negative_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_reward_noun",
        structure=("noun", "negative", "reward", "noun"),
        extractor_name="_extract_pattern_negative_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_reward_action_object",
        structure=("noun", "reward", "action", "noun"),
        extractor_name="_extract_pattern_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_reward_action",
        structure=("noun", "reward", "action"),
        extractor_name="_extract_pattern_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_reward_noun",
        structure=("noun", "reward", "noun"),
        extractor_name="_extract_pattern_reward_sentence",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_action_object_phrase",
        structure=("noun", "negative", "action", "noun"),
        extractor_name="_extract_pattern_negative_action_with_object",
        match_mode="prefix",
    ),
    ExtractorRoute(
        pattern_name="negative_noun_action",
        structure=("noun", "negative", "action"),
        extractor_name="_extract_pattern_negative_intransitive_action",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_action",
        structure=("noun", "action"),
        extractor_name="_extract_pattern_intransitive_action",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_action_object_phrase",
        structure=("noun", "action", "noun"),
        extractor_name="_extract_pattern_action_with_object",
        match_mode="prefix",
    ),
)
