"""Interactive adj-noun trainer.

Input format:
    noun relation adjective

Examples:
    apple color red
    apple taste sweet
"""

from __future__ import annotations

import importlib
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

rm = importlib.import_module("knowledge.relation_map")
arm = importlib.import_module("knowledge.adj_relation_map")
from knowledge.training import (
    ADJ_MODEL_PATH,
    MODEL_PATH,
    adj_map_one,
    begin_feed_training,
    knowledge_map_one,
    run_adj_long_training_and_save,
    train_adj_random,
)


STOP_TOKENS = {"stop", "quit", "exit"}
YES_TOKENS = {"y", "yes"}
NO_TOKENS = {"n", "no"}


def _load_existing_state() -> None:
    if rm.load_relation_data() is False:
        rm.save_relation_data()
        rm.load_relation_data()
    if arm.load_adj_relation_data() is False:
        arm.save_adj_relation_data()
        arm.load_adj_relation_data()

    model_path = pathlib.Path(MODEL_PATH)
    if model_path.exists():
        knowledge_map_one.load_state_dict(torch.load(model_path, map_location="cpu"))

    adj_model_path = pathlib.Path(ADJ_MODEL_PATH)
    if adj_model_path.exists():
        adj_map_one.load_state_dict(torch.load(adj_model_path, map_location="cpu"), strict=False)


def _parse_triple(line: str) -> tuple[str, str, str]:
    tokens = line.strip().lower().split()
    if len(tokens) < 3:
        raise ValueError("Please enter: <noun> <adj_relation> <adjective>")

    noun = tokens[0]
    adjective = tokens[-1]
    relation = " ".join(tokens[1:-1])
    return noun, relation, adjective


def _relation_name(relation_type: int) -> str:
    relation_idx = int(relation_type) - 1
    if 0 <= relation_idx < len(arm.adj_relation_list):
        return arm.adj_relation_list[relation_idx]
    return f"adj_relation_type_{int(relation_type)}"


def _ensure_relation_name(relation: str) -> tuple[int, bool]:
    relation = relation.lower()
    if relation in arm.adj_relation_list:
        return arm.adj_relation_list.index(relation) + 1, False
    if len(arm.adj_relation_list) >= arm.adj_relation_num:
        available = ", ".join(arm.adj_relation_list)
        raise ValueError(
            f"adj_relation_list is full; cannot register new relation '{relation}'. "
            f"Available relations: {available}"
        )
    arm.adj_relation_list.append(relation)
    return len(arm.adj_relation_list), True


def _ask_overwrite_relation(noun: str, adjective: str, old_relation_type: int, new_relation: str) -> bool:
    old_relation = _relation_name(old_relation_type)
    prompt = (
        f"Adj relation conflict: {noun} -> {adjective} already uses "
        f"'{old_relation}'. Overwrite with '{new_relation}'? [y/n] "
    )
    while True:
        answer = input(prompt).strip().lower()
        if answer in YES_TOKENS:
            return True
        if answer in NO_TOKENS:
            return False
        print("Please answer y or n.")


def _train_one(noun: str, relation: str, adjective: str) -> dict:
    noun_before = noun in rm.noun_list
    adjective_before = adjective in arm.adjective_list
    noun_idx_before = rm.noun_list.index(noun) if noun_before else None
    adjective_idx_before = arm.adjective_list.index(adjective) if adjective_before else None
    existing_relation = (
        0
        if noun_idx_before is None or adjective_idx_before is None
        else int(arm.adj_relation_map[noun_idx_before, adjective_idx_before])
    )
    known_relation_type = (
        arm.adj_relation_list.index(relation) + 1
        if relation in arm.adj_relation_list
        else None
    )
    overwrite_relation = False

    if existing_relation != 0 and existing_relation != known_relation_type:
        overwrite_relation = _ask_overwrite_relation(noun, adjective, existing_relation, relation)
        if not overwrite_relation:
            noun_idx = int(noun_idx_before)
            adjective_idx = int(adjective_idx_before)
            stored_relation_type = existing_relation
            loss = train_adj_random(
                adj_map_one,
                noun_idx,
                adjective_idx,
                int(stored_relation_type),
                rm.lr_per_embedding,
                arm.lr_per_adjective,
                arm.lr_adj_relation,
            )
            return {
                "noun": noun,
                "relation": _relation_name(stored_relation_type),
                "input_relation": relation,
                "adjective": adjective,
                "noun_idx": noun_idx,
                "adjective_idx": adjective_idx,
                "relation_type": int(stored_relation_type),
                "relation_conflict": True,
                "overwritten": False,
                "loss": float(loss),
            }

    relation_type, relation_created = _ensure_relation_name(relation)
    if overwrite_relation:
        noun_idx = int(noun_idx_before)
        adjective_idx = int(adjective_idx_before)
        arm.adj_relation_map[noun_idx, adjective_idx] = relation_type
        stored_relation_type = relation_type
    else:
        _, noun_idx, adjective_idx, stored_relation_type = arm.add_adj_relation_by_type(
            noun,
            adjective,
            relation_type,
        )

    loss = train_adj_random(
        adj_map_one,
        noun_idx,
        adjective_idx,
        int(stored_relation_type),
        rm.lr_per_embedding,
        arm.lr_per_adjective,
        arm.lr_adj_relation,
    )
    return {
        "noun": noun,
        "relation": relation,
        "adjective": adjective,
        "noun_idx": int(noun_idx),
        "adjective_idx": int(adjective_idx),
        "relation_type": int(stored_relation_type),
        "new_noun": not noun_before,
        "new_adjective": not adjective_before,
        "new_relation_name": relation_created,
        "new_relation": existing_relation == 0,
        "relation_conflict": overwrite_relation,
        "overwritten": overwrite_relation,
        "loss": float(loss),
    }


def main() -> None:
    _load_existing_state()
    begin_feed_training()
    print("Interactive adj-noun trainer")
    print("Format: <noun> <adj_relation> <adjective>")
    print(f"Adj relations: {', '.join(arm.adj_relation_list)}")
    print("Type 'stop' to run train_adj_average and save.")

    trained_count = 0
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            line = "stop"

        if not line:
            continue
        if line.lower() in STOP_TOKENS:
            break

        try:
            noun, relation, adjective = _parse_triple(line)
            result = _train_one(noun, relation, adjective)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        trained_count += 1
        print(result)

    print(f"Running train_adj_average for {trained_count} accepted triples...")
    loss = run_adj_long_training_and_save()
    print({"saved": ["relation_data.npz", "adj_relation_data.npz", MODEL_PATH, ADJ_MODEL_PATH], "average_loss": loss})


if __name__ == "__main__":
    main()
