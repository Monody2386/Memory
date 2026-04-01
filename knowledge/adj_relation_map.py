import os

import numpy as np

import knowledge.relation_map as noun_rm

from ._storage_utils import ensure_vector_capacity

adjective_number = 200
adjective_dim = noun_rm.noun_dim
adj_relation_num = 7

DEFAULT_ADJECTIVES = [
    "red",
    "round",
    "sweet",
    "large",
    "smooth",
]

DEFAULT_ADJ_RELATIONS = [
    "color",
    "shape",
    "taste",
    "size",
    "texture",
    "personality",
    "temperature",
]

adj_relation_map = np.zeros((noun_rm.noun_number, adjective_number), dtype=np.int64)
adjective_list = list(DEFAULT_ADJECTIVES)
adj_relation_list = list(DEFAULT_ADJ_RELATIONS)
lr_per_adjective = np.ones(adjective_number)
lr_adj_relation = np.ones(adj_relation_num)


def _ensure_adj_capacity() -> None:
    global lr_per_adjective, lr_adj_relation
    lr_per_adjective = ensure_vector_capacity(lr_per_adjective, adjective_number)
    lr_adj_relation = ensure_vector_capacity(lr_adj_relation, adj_relation_num)


def _placeholder_name(index: int) -> str:
    return f"adj_{int(index)}"


def is_defined_adjective_index(index: int) -> bool:
    index = int(index)
    return 0 <= index < len(adjective_list) and adjective_list[index] != _placeholder_name(index)


def bind_adjective_to_index(adjective: str, index: int) -> int:
    adjective = adjective.lower()
    index = int(index)
    if index < 0 or index >= adjective_number:
        raise ValueError(f"adjective index must be in [0, {adjective_number - 1}]")

    if adjective in adjective_list:
        return adjective_list.index(adjective)

    while len(adjective_list) <= index:
        adjective_list.append(_placeholder_name(len(adjective_list)))

    current_value = adjective_list[index]
    if current_value == adjective or current_value == _placeholder_name(index):
        adjective_list[index] = adjective
        return index

    raise ValueError(f"adjective slot {index} is already bound to '{current_value}'")


def _ensure_adjective(adjective: str) -> int:
    adjective = adjective.lower()
    if adjective in adjective_list:
        return adjective_list.index(adjective)
    if len(adjective_list) >= adjective_number:
        raise ValueError("adjective_list is full; cannot register a new adjective")
    adjective_list.append(adjective)
    return len(adjective_list) - 1


def _normalize_relation_type(relation_type: int) -> int:
    relation_type = int(relation_type)
    if relation_type < 1 or relation_type > adj_relation_num:
        raise ValueError(f"relation_type must be in [1, {adj_relation_num}]")
    return relation_type


def _add_adj_relation_entry(noun: str, adjective: str, relation_type: int):
    noun_idx = noun_rm._ensure_noun(noun)
    adjective_idx = _ensure_adjective(adjective)
    relation_type = _normalize_relation_type(relation_type)
    existing_relation = int(adj_relation_map[noun_idx, adjective_idx])

    if existing_relation == 0:
        adj_relation_map[noun_idx, adjective_idx] = relation_type
        return True, noun_idx, adjective_idx, relation_type

    if existing_relation != relation_type:
        print(
            f"Adjective relation already exists between {noun.lower()} and {adjective.lower()} "
            "with a different relation type."
        )
        return False, noun_idx, adjective_idx, existing_relation

    return True, noun_idx, adjective_idx, relation_type


def add_adj_relation(noun: str, adjective: str, relation: str):
    if relation not in adj_relation_list:
        raise ValueError(f"Unknown adjective relation: {relation}")
    relation_type = adj_relation_list.index(relation) + 1
    return _add_adj_relation_entry(noun, adjective, relation_type)


def add_adj_relation_by_type(noun: str, adjective: str, relation_type: int):
    return _add_adj_relation_entry(noun, adjective, relation_type)


def save_adj_relation_data(file_path="adj_relation_data.npz"):
    _ensure_adj_capacity()
    np.savez(
        file_path,
        adj_relation_map=adj_relation_map,
        adjective_list=np.array(adjective_list, dtype=object),
        adj_relation_list=np.array(adj_relation_list, dtype=object),
        lr_per_adjective=lr_per_adjective,
        lr_adj_relation=lr_adj_relation,
    )


def load_adj_relation_data(file_path="adj_relation_data.npz"):
    global adj_relation_map, adjective_list, adj_relation_list, lr_per_adjective, lr_adj_relation

    if not os.path.exists(file_path):
        return False

    data = np.load(file_path, allow_pickle=True)
    adj_relation_map = data["adj_relation_map"].astype(np.int64, copy=False)
    adjective_list = data["adjective_list"].tolist()
    adj_relation_list = data["adj_relation_list"].tolist()
    lr_per_adjective = ensure_vector_capacity(data["lr_per_adjective"], adjective_number)
    lr_adj_relation = ensure_vector_capacity(data["lr_adj_relation"], adj_relation_num)

    return (
        adj_relation_map,
        adjective_list,
        adj_relation_list,
        lr_per_adjective,
        lr_adj_relation,
    )


if __name__ == "__main__":
    save_adj_relation_data()
