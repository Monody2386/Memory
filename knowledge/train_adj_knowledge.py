"""File-driven training for adj-noun relations.

Training data format:
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

from knowledge.training import (
    ADJ_MODEL_PATH,
    MODEL_PATH,
    adj_map_one,
    begin_feed_training,
    knowledge_map_one,
    run_adj_long_training_and_save,
    train_adj_random,
)

rm = importlib.import_module("knowledge.relation_map")
arm = importlib.import_module("knowledge.adj_relation_map")

DEFAULT_ADJ_TRAINING_DATA_PATH = pathlib.Path(__file__).with_name(
    "adj_knowledge_training_data.txt"
)


def _ensure_saved_data() -> None:
    if rm.load_relation_data() is False:
        rm.save_relation_data()
        rm.load_relation_data()
    if arm.load_adj_relation_data() is False:
        arm.save_adj_relation_data()
        arm.load_adj_relation_data()


def _parse_training_line(line: str) -> tuple[str, str, str] | None:
    line = line.split("#", 1)[0].strip().lower()
    if not line:
        return None

    tokens = line.split()
    if len(tokens) < 3:
        raise ValueError("expected: <noun> <adj_relation> <adjective>")

    noun = tokens[0]
    relation = " ".join(tokens[1:-1])
    adjective = tokens[-1]
    return noun, adjective, relation


def load_adj_training_relations(file_path=DEFAULT_ADJ_TRAINING_DATA_PATH):
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    relations = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            parsed = _parse_training_line(line)
        except ValueError as exc:
            raise ValueError(f"{file_path}:{line_number}: {exc}") from exc

        if parsed is not None:
            relations.append(parsed)

    return relations


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


def train_via_adj_relations(relations, *, conflict_policy="keep", run_average=True):
    _ensure_saved_data()
    begin_feed_training()

    results = []
    for noun, adjective, relation in relations:
        relation_type, relation_created = _ensure_relation_name(relation)
        noun_before = noun in rm.noun_list
        adjective_before = adjective in arm.adjective_list
        noun_idx_before = rm.noun_list.index(noun) if noun_before else None
        adjective_idx_before = arm.adjective_list.index(adjective) if adjective_before else None
        existing_relation = (
            0
            if noun_idx_before is None or adjective_idx_before is None
            else int(arm.adj_relation_map[noun_idx_before, adjective_idx_before])
        )
        conflict = existing_relation != 0 and existing_relation != relation_type

        if conflict:
            if conflict_policy == "error":
                raise ValueError(
                    f"adj relation conflict for {noun} -> {adjective}: "
                    f"existing '{_relation_name(existing_relation)}', new '{relation}'"
                )
            if conflict_policy == "overwrite":
                noun_idx = int(noun_idx_before)
                adjective_idx = int(adjective_idx_before)
                arm.adj_relation_map[noun_idx, adjective_idx] = relation_type
                stored_relation_type = relation_type
                overwritten = True
            elif conflict_policy == "keep":
                noun_idx = int(noun_idx_before)
                adjective_idx = int(adjective_idx_before)
                stored_relation_type = existing_relation
                overwritten = False
            else:
                raise ValueError("conflict_policy must be: keep, overwrite, or error")
        else:
            _, noun_idx, adjective_idx, stored_relation_type = arm.add_adj_relation_by_type(
                noun,
                adjective,
                relation_type,
            )
            overwritten = False

        loss = train_adj_random(
            adj_map_one,
            noun_idx,
            adjective_idx,
            int(stored_relation_type),
            rm.lr_per_embedding,
            arm.lr_per_adjective,
            arm.lr_adj_relation,
        )
        results.append(
            {
                "noun": noun,
                "relation": _relation_name(stored_relation_type),
                "input_relation": relation,
                "adjective": adjective,
                "noun_idx": int(noun_idx),
                "adjective_idx": int(adjective_idx),
                "relation_type": int(stored_relation_type),
                "new_noun": not noun_before,
                "new_adjective": not adjective_before,
                "new_relation_name": relation_created,
                "new_relation": existing_relation == 0,
                "relation_conflict": conflict,
                "overwritten": overwritten,
                "loss": float(loss),
            }
        )

    average_loss = run_adj_long_training_and_save() if run_average else None
    return results, average_loss


def train_adj_from_file(
    file_path=DEFAULT_ADJ_TRAINING_DATA_PATH,
    *,
    conflict_policy="keep",
    run_average=True,
):
    relations = load_adj_training_relations(file_path)
    return train_via_adj_relations(
        relations,
        conflict_policy=conflict_policy,
        run_average=run_average,
    )


if __name__ == "__main__":
    results, average_loss = train_adj_from_file()
    print(f"Trained {len(results)} adj-noun triples from {DEFAULT_ADJ_TRAINING_DATA_PATH}")
    for result in results:
        print(result)
    print({"average_loss": average_loss, "model_path": MODEL_PATH, "adj_model_path": ADJ_MODEL_PATH})
