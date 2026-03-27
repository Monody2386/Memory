"""High-level orchestration ideas for retrieval and question handling.

This module is still a prototype. The project currently uses the knowledge
graph modules and the world-model modules directly.
"""

from knowledge.relation_map import relation_map


def available_reasoning_modes():
    return {
        "what": "similarity search over concept embeddings",
        "where": "type or relation lookup",
        "when": "type or relation lookup",
    }


def has_relation(i: int, j: int, relation_type: int) -> bool:
    return int(relation_map[i][j]) == int(relation_type)

