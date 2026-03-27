import importlib
import os

import torch

rm = importlib.import_module("knowledge.relation_map")
from .knowledge_map import knowledge_map, train_average, train_random

knowledge_map_one = knowledge_map(rm.noun_dim, rm.noun_dim)
MODEL_PATH = "knowledge_map_one.pt"
_FEED_TRAIN_READY = False


def _load_training_state() -> None:
    if os.path.exists(MODEL_PATH):
        knowledge_map_one.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError(
            "relation_data.npz not found. Generate relation data before training."
        )


def begin_feed_training():
    global _FEED_TRAIN_READY
    _load_training_state()
    _FEED_TRAIN_READY = True


def end_feed_training():
    global _FEED_TRAIN_READY
    if not _FEED_TRAIN_READY:
        return
    rm.save_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)
    _FEED_TRAIN_READY = False


def run_long_training_and_save():
    _load_training_state()
    relation_map, _, _, lr_per_embedding, lr_relation = rm.load_relation_data()
    train_average(knowledge_map_one, relation_map, lr_per_embedding, lr_relation)
    rm.save_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)


def run_short_training_and_save(relation_learn, save=True):
    if not _FEED_TRAIN_READY:
        _load_training_state()

    for i_idx, j_idx, relation_type in relation_learn:
        train_random(
            knowledge_map_one,
            i_idx,
            j_idx,
            relation_type,
            rm.lr_per_embedding,
            rm.lr_relation,
        )

    if save:
        rm.save_relation_data()
        torch.save(knowledge_map_one.state_dict(), MODEL_PATH)


def random_feed(noun1, noun2, relation):
    created, i_idx, j_idx, relation_type = rm.add_relation(noun1, noun2, relation)
    relation_type = int(relation_type)

    if created:
        run_short_training_and_save([(i_idx, j_idx, relation_type)], save=False)
        print(noun1, noun2, relation)
    else:
        print(relation_type)
        print(noun1, noun2, rm.relation_list[relation_type - 1])
