import importlib
import os

import torch

rm = importlib.import_module("knowledge.relation_map")
from .knowledge_map import knowledge_map
from .training import (
    begin_feed_training,
    end_feed_training,
    random_feed,
    run_long_training_and_save,
)

MODEL_PATH = "knowledge_map_one.pt"


def train_via_feed_relations(relations):
    begin_feed_training()

    if not rm.noun_list:
        raise ValueError("noun_list is empty after loading relation_data.npz")
    if not rm.relation_list:
        raise ValueError("relation_list is empty after loading relation_data.npz")

    for noun1, noun2, relation in relations:
        random_feed(noun1, noun2, relation)

    end_feed_training()


def predict_next_word(word: str, relation, top_k: int = 5):
    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError("relation_data.npz not found")

    if word not in rm.noun_list:
        raise ValueError(f"word '{word}' not in noun_list")

    if isinstance(relation, str):
        if relation not in rm.relation_list:
            raise ValueError(f"relation '{relation}' not in relation_list")
        relation_type = rm.relation_list.index(relation) + 1
    elif isinstance(relation, int):
        relation_type = relation
    else:
        raise TypeError("relation should be a relation name or relation_type integer")

    model = knowledge_map(rm.noun_dim, rm.noun_dim)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        i_idx = rm.noun_list.index(word)
        i_tensor = torch.tensor(i_idx, dtype=torch.long)
        x = model.embedding(i_tensor)
        y_pred = model.relations[int(relation_type) - 1](x)
        top_indices, top_scores = model.query_similarity(y_pred, top_k=top_k)

    results = []
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        if 0 <= idx < len(rm.noun_list):
            results.append((rm.noun_list[idx], float(score)))
        else:
            results.append((f"<unk_{idx}>", float(score)))

    return results


if __name__ == "__main__":
    relations = [("apple", "fruit", "include")]
    train_via_feed_relations(relations)
    print(predict_next_word("banana", "include"))
    run_long_training_and_save()
