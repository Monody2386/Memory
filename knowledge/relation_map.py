import os

import numpy as np

noun_number = 500
noun_dim = 50

# The model currently allocates five relation transforms. Keep the persisted
# learning-rate array aligned with that capacity even if only a subset of
# relation names is actively used.
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
]

DEFAULT_RELATIONS = [
    "include",
    "belong to",
]

relation_map = np.zeros((noun_number, noun_number), dtype=np.int64)
noun_list = list(DEFAULT_NOUNS)
relation_list = list(DEFAULT_RELATIONS)
lr_relation = np.ones(relation_num)
lr_per_embedding = np.ones(noun_number)


def _ensure_lr_relation_capacity() -> None:
    global lr_relation

    if len(lr_relation) < relation_num:
        padded = np.ones(relation_num)
        padded[: len(lr_relation)] = lr_relation
        lr_relation = padded
    elif len(lr_relation) > relation_num:
        lr_relation = lr_relation[:relation_num]


def add_relation(noun1, noun2, relation):
    if relation not in relation_list:
        raise ValueError(f"Unknown relation: {relation}")

    if noun1 not in noun_list:
        noun_list.append(noun1)
    if noun2 not in noun_list:
        noun_list.append(noun2)

    i_idx = noun_list.index(noun1)
    j_idx = noun_list.index(noun2)
    relation_type = relation_list.index(relation) + 1

    if relation_map[i_idx, j_idx] == 0:
        relation_map[i_idx, j_idx] = relation_type
        return True, i_idx, j_idx, relation_type

    if relation_map[i_idx, j_idx] != relation_type:
        print(
            f"Relation already exists between {noun1} and {noun2} "
            "with a different relation type."
        )
        return False, i_idx, j_idx, relation_map[i_idx, j_idx]

    return True, i_idx, j_idx, relation_type


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
    lr_per_embedding = data["lr_per_embedding"]
    lr_relation = data["lr_relation"]
    _ensure_lr_relation_capacity()

    return relation_map, noun_list, relation_list, lr_per_embedding, lr_relation


if __name__ == "__main__":
    save_relation_data()
