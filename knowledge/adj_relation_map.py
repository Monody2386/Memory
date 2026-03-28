import os

import numpy as np

import knowledge.relation_map as noun_rm

adjective_number = 500
adjective_dim = noun_rm.noun_dim
adj_relation_num = 5

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
]

adj_relation_map = np.zeros((noun_rm.noun_number, adjective_number), dtype=np.int64)
adjective_list = list(DEFAULT_ADJECTIVES)
adj_relation_list = list(DEFAULT_ADJ_RELATIONS)
lr_per_adjective = np.ones(adjective_number)
lr_adj_relation = np.ones(adj_relation_num)


def _ensure_adj_capacity() -> None:
    global lr_per_adjective, lr_adj_relation

    if len(lr_per_adjective) < adjective_number:
        padded = np.ones(adjective_number)
        padded[: len(lr_per_adjective)] = lr_per_adjective
        lr_per_adjective = padded
    elif len(lr_per_adjective) > adjective_number:
        lr_per_adjective = lr_per_adjective[:adjective_number]

    if len(lr_adj_relation) < adj_relation_num:
        padded = np.ones(adj_relation_num)
        padded[: len(lr_adj_relation)] = lr_adj_relation
        lr_adj_relation = padded
    elif len(lr_adj_relation) > adj_relation_num:
        lr_adj_relation = lr_adj_relation[:adj_relation_num]


def add_adj_relation(noun: str, adjective: str, relation: str):
    if relation not in adj_relation_list:
        raise ValueError(f"Unknown adjective relation: {relation}")

    if noun not in noun_rm.noun_list:
        noun_rm.noun_list.append(noun)
    if adjective not in adjective_list:
        adjective_list.append(adjective)

    noun_idx = noun_rm.noun_list.index(noun)
    adjective_idx = adjective_list.index(adjective)
    relation_type = adj_relation_list.index(relation) + 1

    if adj_relation_map[noun_idx, adjective_idx] == 0:
        adj_relation_map[noun_idx, adjective_idx] = relation_type
        return True, noun_idx, adjective_idx, relation_type

    if adj_relation_map[noun_idx, adjective_idx] != relation_type:
        print(
            f"Adjective relation already exists between {noun} and {adjective} "
            "with a different relation type."
        )
        return False, noun_idx, adjective_idx, adj_relation_map[noun_idx, adjective_idx]

    return True, noun_idx, adjective_idx, relation_type


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
    lr_per_adjective = data["lr_per_adjective"]
    lr_adj_relation = data["lr_adj_relation"]
    _ensure_adj_capacity()

    return (
        adj_relation_map,
        adjective_list,
        adj_relation_list,
        lr_per_adjective,
        lr_adj_relation,
    )


if __name__ == "__main__":
    save_adj_relation_data()
