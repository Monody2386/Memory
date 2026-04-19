import importlib
import os

import torch

rm = importlib.import_module("knowledge.relation_map")
arm = importlib.import_module("knowledge.adj_relation_map")
from .adj_map import adj_map, train_adj_average, train_adj_random, train_joint_average
from .knowledge_map import knowledge_map, train_average, train_random

knowledge_map_one = knowledge_map(rm.noun_dim, rm.noun_dim)
adj_map_one = adj_map(knowledge_map_one.embedding, rm.noun_dim, rm.noun_dim)
MODEL_PATH = "knowledge_map_one.pt"
ADJ_MODEL_PATH = "adj_map_one.pt"
_FEED_TRAIN_READY = False


def _load_training_state() -> None:
    if os.path.exists(MODEL_PATH):
        knowledge_map_one.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    if os.path.exists(ADJ_MODEL_PATH):
        adj_map_one.load_state_dict(torch.load(ADJ_MODEL_PATH, map_location="cpu"), strict=False)

    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError(
            "relation_data.npz not found. Generate relation data before training."
        )

    arm.load_adj_relation_data()


def _save_training_state() -> None:
    rm.save_relation_data()
    arm.save_adj_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)
    torch.save(adj_map_one.state_dict(), ADJ_MODEL_PATH)


def begin_feed_training():
    global _FEED_TRAIN_READY
    _load_training_state()
    _FEED_TRAIN_READY = True


def end_feed_training():
    global _FEED_TRAIN_READY
    if not _FEED_TRAIN_READY:
        return
    _save_training_state()
    _FEED_TRAIN_READY = False


def run_long_training_and_save():
    if not _FEED_TRAIN_READY:
        _load_training_state()

    relation_map = rm.relation_map
    lr_per_embedding = rm.lr_per_embedding
    lr_relation = rm.lr_relation
    train_average(knowledge_map_one, relation_map, lr_per_embedding, lr_relation)
    rm.save_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)


def run_adj_long_training_and_save():
    if not _FEED_TRAIN_READY:
        _load_training_state()

    if not arm.adjective_list:
        arm.save_adj_relation_data()
        arm.load_adj_relation_data()

    loss = train_adj_average(
        adj_map_one,
        arm.adj_relation_map,
        rm.lr_per_embedding,
        arm.lr_per_adjective,
        arm.lr_adj_relation,
    )
    rm.save_relation_data()
    arm.save_adj_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)
    torch.save(adj_map_one.state_dict(), ADJ_MODEL_PATH)
    return loss


def run_joint_training_and_save():
    _load_training_state()
    relation_map, _, _, lr_per_embedding, lr_relation = rm.load_relation_data()
    adj_relation_map, _, _, lr_per_adjective, lr_adj_relation = arm.load_adj_relation_data()
    if adj_relation_map is False:
        raise FileNotFoundError(
            "adj_relation_data.npz not found. Generate adjective relation data before joint training."
        )

    loss = train_joint_average(
        knowledge_map_one=knowledge_map_one,
        adj_map_one=adj_map_one,
        relation_map=relation_map,
        adj_relation_map=adj_relation_map,
        lr_per_embedding=lr_per_embedding,
        lr_relation=lr_relation,
        lr_per_adjective=lr_per_adjective,
        lr_adj_relation=lr_adj_relation,
    )
    _save_training_state()
    return loss


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


def apply_language_training_samples(samples, save=True):
    if not _FEED_TRAIN_READY:
        _load_training_state()

    results = {"noun_noun": [], "adj_noun": []}
    noun_noun_samples = getattr(samples, "noun_noun_samples", None)
    if noun_noun_samples is None and isinstance(samples, dict):
        noun_noun_samples = samples.get("noun_noun_samples", [])
    adj_noun_samples = getattr(samples, "adj_noun_samples", None)
    if adj_noun_samples is None and isinstance(samples, dict):
        adj_noun_samples = samples.get("adj_noun_samples", [])

    for sample in noun_noun_samples or []:
        created, i_idx, j_idx, relation_type = rm.add_relation_by_type(
            sample.source_noun,
            sample.target_noun,
            sample.relation_type,
        )
        loss = train_random(
            knowledge_map_one,
            i_idx,
            j_idx,
            relation_type,
            rm.lr_per_embedding,
            rm.lr_relation,
        )
        results["noun_noun"].append(
            {
                "created": created,
                "source_idx": i_idx,
                "target_idx": j_idx,
                "relation_type": int(relation_type),
                "loss": float(loss),
            }
        )

    for sample in adj_noun_samples or []:
        created, noun_idx, adjective_idx, relation_type = arm.add_adj_relation_by_type(
            sample.noun,
            sample.adjective,
            sample.relation_type,
        )
        loss = train_adj_random(
            adj_map_one,
            noun_idx,
            adjective_idx,
            relation_type,
            rm.lr_per_embedding,
            arm.lr_per_adjective,
            arm.lr_adj_relation,
        )
        results["adj_noun"].append(
            {
                "created": created,
                "noun_idx": noun_idx,
                "adjective_idx": adjective_idx,
                "relation_type": int(relation_type),
                "loss": float(loss),
            }
        )

    if save:
        _save_training_state()
    return results


def train_sentence_online(
    sentence,
    noun_relation_type=None,
    adjective_relation_types=None,
    infer_missing=False,
    save=True,
):
    grammar = importlib.import_module("grammar_layer")
    samples = grammar.sentence_to_knowledge_samples(
        sentence,
        noun_relation_type=noun_relation_type,
        adjective_relation_types=adjective_relation_types,
        infer_missing=infer_missing,
    )
    results = apply_language_training_samples(samples, save=save)
    return samples, results


def random_feed(noun1, noun2, relation):
    created, i_idx, j_idx, relation_type = rm.add_relation(noun1, noun2, relation)
    relation_type = int(relation_type)

    if created:
        run_short_training_and_save([(i_idx, j_idx, relation_type)], save=False)
        print(noun1, noun2, relation)
    else:
        print(relation_type)
        print(noun1, noun2, rm.relation_list[relation_type - 1])
