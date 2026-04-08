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
        pattern_name="be_attribute",
        structure=("noun", "be", "adj"),
        extractor_name="_extract_pattern_be_attribute",
        match_mode="prefix",
    ),
    ExtractorRoute(
        pattern_name="be_noun",
        structure=("noun", "be", "noun"),
        extractor_name="_extract_pattern_be_noun",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_phrase_relation_noun_phrase",
        structure=("noun", "relation", "noun"),
        extractor_name="_extract_pattern_relation_between_noun_phrases",
        match_mode="exact",
    ),
    ExtractorRoute(
        pattern_name="noun_action_object_phrase",
        structure=("noun", "action"),
        extractor_name="_extract_pattern_action_with_object",
        match_mode="prefix",
    ),
)
