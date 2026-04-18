__all__ = [
    "Episode",
    "EpisodeMemory",
    "available_reasoning_modes",
    "has_relation",
    "split_event",
]


def __getattr__(name):
    if name in {"available_reasoning_modes", "has_relation"}:
        from .consciousness import available_reasoning_modes, has_relation

        return {
            "available_reasoning_modes": available_reasoning_modes,
            "has_relation": has_relation,
        }[name]
    if name in {"Episode", "EpisodeMemory"}:
        from .episode_memory import Episode, EpisodeMemory

        return {
            "Episode": Episode,
            "EpisodeMemory": EpisodeMemory,
        }[name]
    if name == "split_event":
        from .grammar import split_event

        return split_event
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
