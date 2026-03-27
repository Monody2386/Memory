"""Grammar and event parsing placeholder utilities.

This module is intentionally small for now: it defines the expected shape for
future event parsing without pretending to provide a full grammar system yet.
"""


def split_event(event_tokens):
    """Return a normalized five-slot event tuple when possible."""
    if len(event_tokens) != 5:
        raise ValueError("event_tokens must contain exactly 5 items")
    return tuple(event_tokens)
