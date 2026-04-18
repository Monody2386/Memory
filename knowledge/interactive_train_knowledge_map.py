"""Interactive noun-relation-noun trainer for knowledge_map.

Input examples:
    apple include fruit
    teacher belong to job
    robin is_a bird

Type "stop" to run average training and save relation/map data.
"""

from __future__ import annotations

import pathlib
import sys
import importlib

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

rm = importlib.import_module("knowledge.relation_map")
from knowledge.training import (
    MODEL_PATH,
    begin_feed_training,
    knowledge_map_one,
    run_long_training_and_save,
    run_short_training_and_save,
)


STOP_TOKENS = {"stop", "quit", "exit"}
YES_TOKENS = {"y", "yes"}
NO_TOKENS = {"n", "no"}


def _load_existing_state() -> None:
    if rm.load_relation_data() is False:
        rm.save_relation_data()
        rm.load_relation_data()

    model_path = pathlib.Path(MODEL_PATH)
    if model_path.exists():
        knowledge_map_one.load_state_dict(torch.load(model_path, map_location="cpu"))


def _parse_triple(line: str) -> tuple[str, str, str]:
    tokens = line.strip().lower().split()
    if len(tokens) < 3:
        raise ValueError("Please enter: <noun> <relation> <noun>")

    source_noun = tokens[0]
    target_noun = tokens[-1]
    relation_name = " ".join(tokens[1:-1])

    return source_noun, relation_name, target_noun


def _ensure_relation_name(relation_name: str) -> tuple[int, bool]:
    relation_name = relation_name.lower()
    if relation_name in rm.relation_list:
        return rm.relation_list.index(relation_name) + 1, False
    if len(rm.relation_list) >= rm.relation_num:
        available = ", ".join(rm.relation_list)
        raise ValueError(
            f"relation_list is full; cannot register new relation '{relation_name}'. "
            f"Available relations: {available}"
        )
    rm.relation_list.append(relation_name)
    return len(rm.relation_list), True


def _relation_name(relation_type: int) -> str:
    relation_idx = int(relation_type) - 1
    if 0 <= relation_idx < len(rm.relation_list):
        return rm.relation_list[relation_idx]
    return f"relation_type_{int(relation_type)}"


def _ask_overwrite_relation(
    source_noun: str,
    target_noun: str,
    old_relation_type: int,
    new_relation_name: str,
) -> bool:
    old_relation_name = _relation_name(old_relation_type)
    prompt = (
        f"Relation conflict: {source_noun} -> {target_noun} already uses "
        f"'{old_relation_name}'. Overwrite with '{new_relation_name}'? [y/n] "
    )

    while True:
        answer = input(prompt).strip().lower()
        if answer in YES_TOKENS:
            return True
        if answer in NO_TOKENS:
            return False
        print("Please answer y or n.")


def _train_one(source_noun: str, relation_name: str, target_noun: str) -> dict:
    source_before = source_noun in rm.noun_list
    target_before = target_noun in rm.noun_list
    source_idx_before = rm.noun_list.index(source_noun) if source_before else None
    target_idx_before = rm.noun_list.index(target_noun) if target_before else None
    existing_relation = (
        0
        if source_idx_before is None or target_idx_before is None
        else int(rm.relation_map[source_idx_before, target_idx_before])
    )
    existing_relation_name = _relation_name(existing_relation) if existing_relation else None
    known_relation_type = (
        rm.relation_list.index(relation_name) + 1
        if relation_name in rm.relation_list
        else None
    )
    overwrite_relation = False

    if existing_relation != 0 and existing_relation != known_relation_type:
        overwrite_relation = _ask_overwrite_relation(
            source_noun,
            target_noun,
            existing_relation,
            relation_name,
        )
        if not overwrite_relation:
            source_idx = int(source_idx_before)
            target_idx = int(target_idx_before)
            stored_relation_type = existing_relation
            run_short_training_and_save(
                [(source_idx, target_idx, int(stored_relation_type))],
                save=False,
            )
            return {
                "new_source_noun": False,
                "new_target_noun": False,
                "new_relation_name": False,
                "new_relation": False,
                "relation_conflict": True,
                "overwritten": False,
                "kept_relation": existing_relation_name,
                "source_noun": source_noun,
                "relation": existing_relation_name,
                "input_relation": relation_name,
                "target_noun": target_noun,
                "source_idx": source_idx,
                "target_idx": target_idx,
                "relation_type": int(stored_relation_type),
            }

    relation_type, relation_created = _ensure_relation_name(relation_name)

    if overwrite_relation:
        source_idx = int(source_idx_before)
        target_idx = int(target_idx_before)
        rm.relation_map[source_idx, target_idx] = relation_type
        stored_relation_type = relation_type
    else:
        _, source_idx, target_idx, stored_relation_type = rm.add_relation_by_type(
            source_noun,
            target_noun,
            relation_type,
        )
    run_short_training_and_save([(source_idx, target_idx, int(stored_relation_type))], save=False)
    return {
        "new_source_noun": not source_before,
        "new_target_noun": not target_before,
        "new_relation_name": relation_created,
        "new_relation": existing_relation == 0,
        "relation_conflict": overwrite_relation,
        "overwritten": overwrite_relation,
        "previous_relation": existing_relation_name if overwrite_relation else None,
        "source_noun": source_noun,
        "relation": relation_name,
        "target_noun": target_noun,
        "source_idx": int(source_idx),
        "target_idx": int(target_idx),
        "relation_type": int(stored_relation_type),
    }


def main() -> None:
    _load_existing_state()
    begin_feed_training()
    print("Interactive knowledge_map trainer")
    print("Format: <noun> <relation> <noun>")
    print(f"Relations: {', '.join(rm.relation_list)}")
    print("Type 'stop' to run train_average and save.")

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
            source_noun, relation_name, target_noun = _parse_triple(line)
            result = _train_one(source_noun, relation_name, target_noun)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        trained_count += 1
        print(result)

    print(f"Running train_average for {trained_count} accepted triples...")
    run_long_training_and_save()
    print("Saved relation_data.npz and knowledge_map_one.pt")


if __name__ == "__main__":
    main()
