import os

import numpy as np

from ._storage_utils import ensure_vector_capacity

noun_number = 500
noun_dim = 50
relation_num = 5

DEFAULT_NOUNS = [
    "apple",
    "banana",
    "fruit",
    "cat",
    "dog",
    "car",
    "tree",
    "house",
    "book",
    "phone",
    "computer",
    "person",
    "city",
    "country",
    "continent",
    "planet",
    "star",
    "galaxy",
    "universe",
    "water",
    "fire",
    "air",
    "earth",
    "light",
    "darkness",
    "time",
    "space",
    "animal",
    "teacher",
    "job",
]

DEFAULT_RELATIONS = [
    "include",
    "belong to",
    "job_relation",
]

relation_map = np.zeros((noun_number, noun_number), dtype=np.int64)
noun_list = list(DEFAULT_NOUNS)
relation_list = list(DEFAULT_RELATIONS)
lr_relation = np.ones(relation_num)
lr_per_embedding = np.ones(noun_number)


def _ensure_lr_relation_capacity() -> None:
    global lr_relation
    lr_relation = ensure_vector_capacity(lr_relation, relation_num)


def _ensure_default_relations() -> None:
    for relation in DEFAULT_RELATIONS:
        if relation not in relation_list:
            if len(relation_list) >= relation_num:
                raise ValueError(
                    f"relation_list is full; cannot register default relation '{relation}'"
                )
            relation_list.append(relation)


def _placeholder_name(index: int) -> str:
    return f"noun_{int(index)}"


def is_defined_noun_index(index: int) -> bool:
    index = int(index)
    return 0 <= index < len(noun_list) and noun_list[index] != _placeholder_name(index)


def bind_noun_to_index(noun: str, index: int) -> int:
    noun = noun.lower()
    index = int(index)
    if index < 0 or index >= noun_number:
        raise ValueError(f"noun index must be in [0, {noun_number - 1}]")

    if noun in noun_list:
        return noun_list.index(noun)

    while len(noun_list) <= index:
        noun_list.append(_placeholder_name(len(noun_list)))

    current_value = noun_list[index]
    if current_value == noun or current_value == _placeholder_name(index):
        noun_list[index] = noun
        return index

    raise ValueError(f"noun slot {index} is already bound to '{current_value}'")


def _ensure_noun(noun: str) -> int:
    noun = noun.lower()
    if noun in noun_list:
        return noun_list.index(noun)
    if len(noun_list) >= noun_number:
        raise ValueError("noun_list is full; cannot register a new noun")
    noun_list.append(noun)
    return len(noun_list) - 1


def _normalize_relation_type(relation_type: int) -> int:
    relation_type = int(relation_type)
    if relation_type < 1 or relation_type > relation_num:
        raise ValueError(f"relation_type must be in [1, {relation_num}]")
    return relation_type


def _add_relation_entry(noun1: str, noun2: str, relation_type: int):
    i_idx = _ensure_noun(noun1)
    j_idx = _ensure_noun(noun2)
    relation_type = _normalize_relation_type(relation_type)
    existing_relation = int(relation_map[i_idx, j_idx])

    if existing_relation == 0:
        relation_map[i_idx, j_idx] = relation_type
        return True, i_idx, j_idx, relation_type

    if existing_relation != relation_type:
        print(
            f"Relation already exists between {noun1.lower()} and {noun2.lower()} "
            "with a different relation type."
        )
        return False, i_idx, j_idx, existing_relation

    return True, i_idx, j_idx, relation_type


def add_relation(noun1, noun2, relation):
    if relation not in relation_list:
        raise ValueError(f"Unknown relation: {relation}")
    relation_type = relation_list.index(relation) + 1
    return _add_relation_entry(noun1, noun2, relation_type)


def add_relation_by_type(noun1, noun2, relation_type):
    return _add_relation_entry(noun1, noun2, relation_type)


def save_relation_data(file_path="relation_data.npz"):
    _ensure_lr_relation_capacity()
    np.savez(
        file_path,
        relation_map=relation_map,
        noun_list=np.array(noun_list, dtype=object),
        relation_list=np.array(relation_list, dtype=object),
        lr_per_embedding=lr_per_embedding,
        lr_relation=lr_relation,
    )


def load_relation_data(file_path="relation_data.npz"):
    global relation_map, noun_list, relation_list, lr_per_embedding, lr_relation

    if not os.path.exists(file_path):
        return False

    data = np.load(file_path, allow_pickle=True)
    relation_map = data["relation_map"].astype(np.int64, copy=False)
    noun_list = data["noun_list"].tolist()
    relation_list = data["relation_list"].tolist()
    _ensure_default_relations()
    lr_per_embedding = data["lr_per_embedding"]
    lr_relation = ensure_vector_capacity(data["lr_relation"], relation_num)

    return relation_map, noun_list, relation_list, lr_per_embedding, lr_relation


if __name__ == "__main__":
    save_relation_data()
