"""Run joint average training for noun-noun and adj-noun knowledge maps."""

from __future__ import annotations

import importlib

from .training import ADJ_MODEL_PATH, MODEL_PATH, run_joint_training_and_save

rm = importlib.import_module("knowledge.relation_map")
arm = importlib.import_module("knowledge.adj_relation_map")


def _count_active_relations(matrix, relation_count: int) -> int:
    count = 0
    for indices in zip(*matrix.nonzero()):
        relation_type = int(matrix[indices])
        if 1 <= relation_type <= relation_count:
            count += 1
    return count


def run_joint_training() -> dict:
    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError("relation_data.npz not found. Train noun knowledge first.")

    adj_loaded = arm.load_adj_relation_data()
    if adj_loaded is False:
        raise FileNotFoundError("adj_relation_data.npz not found. Train adj knowledge first.")

    noun_relation_count = _count_active_relations(rm.relation_map, rm.relation_num)
    adj_relation_count = _count_active_relations(arm.adj_relation_map, arm.adj_relation_num)
    loss = run_joint_training_and_save()

    return {
        "noun_relation_count": noun_relation_count,
        "adj_relation_count": adj_relation_count,
        "joint_loss": loss,
        "saved": [
            "relation_data.npz",
            "adj_relation_data.npz",
            MODEL_PATH,
            ADJ_MODEL_PATH,
        ],
    }


def main() -> None:
    print(run_joint_training())


if __name__ == "__main__":
    main()
