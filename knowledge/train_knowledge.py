import importlib
import os
import pathlib

import torch

rm = importlib.import_module("knowledge.relation_map")
from .knowledge_map import knowledge_map
from .training import (
    begin_feed_training,
    run_long_training_and_save,
    run_short_training_and_save,
)

MODEL_PATH = "knowledge_map_one.pt"
DEFAULT_TRAINING_DATA_PATH = pathlib.Path(__file__).with_name("knowledge_training_data.txt")


def _ensure_relation_data() -> None:
    if rm.load_relation_data() is False:
        rm.save_relation_data()
        rm.load_relation_data()


def _parse_training_line(line: str) -> tuple[str, str, str] | None:
    line = line.split("#", 1)[0].strip().lower()
    if not line:
        return None

    tokens = line.split()
    if len(tokens) < 3:
        raise ValueError("expected: <noun1> <relation> <noun2>")

    source_noun = tokens[0]
    relation = " ".join(tokens[1:-1])
    target_noun = tokens[-1]
    return source_noun, target_noun, relation


def load_training_relations(file_path=DEFAULT_TRAINING_DATA_PATH):
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


def _ensure_relation_name(relation: str) -> tuple[int, bool]:
    relation = relation.lower()
    if relation in rm.relation_list:
        return rm.relation_list.index(relation) + 1, False
    if len(rm.relation_list) >= rm.relation_num:
        available = ", ".join(rm.relation_list)
        raise ValueError(
            f"relation_list is full; cannot register new relation '{relation}'. "
            f"Available relations: {available}"
        )
    rm.relation_list.append(relation)
    return len(rm.relation_list), True


def _relation_name(relation_type: int) -> str:
    relation_idx = int(relation_type) - 1
    if 0 <= relation_idx < len(rm.relation_list):
        return rm.relation_list[relation_idx]
    return f"relation_type_{int(relation_type)}"


def train_via_feed_relations(relations, *, conflict_policy="keep", run_average=True):
    _ensure_relation_data()
    begin_feed_training()

    if not rm.noun_list:
        raise ValueError("noun_list is empty after loading relation_data.npz")
    if not rm.relation_list:
        raise ValueError("relation_list is empty after loading relation_data.npz")

    results = []
    for noun1, noun2, relation in relations:
        source_before = noun1 in rm.noun_list
        target_before = noun2 in rm.noun_list
        source_idx_before = rm.noun_list.index(noun1) if source_before else None
        target_idx_before = rm.noun_list.index(noun2) if target_before else None
        existing_relation = (
            0
            if source_idx_before is None or target_idx_before is None
            else int(rm.relation_map[source_idx_before, target_idx_before])
        )
        known_relation_type = (
            rm.relation_list.index(relation) + 1
            if relation in rm.relation_list
            else None
        )
        conflict = existing_relation != 0 and existing_relation != known_relation_type
        relation_created = False

        if conflict:
            if conflict_policy == "error":
                raise ValueError(
                    f"relation conflict for {noun1} -> {noun2}: "
                    f"existing '{_relation_name(existing_relation)}', new '{relation}'"
                )
            if conflict_policy == "overwrite":
                relation_type, relation_created = _ensure_relation_name(relation)
                source_idx = int(source_idx_before)
                target_idx = int(target_idx_before)
                rm.relation_map[source_idx, target_idx] = relation_type
                stored_relation_type = relation_type
                overwritten = True
            elif conflict_policy == "keep":
                source_idx = int(source_idx_before)
                target_idx = int(target_idx_before)
                stored_relation_type = existing_relation
                overwritten = False
            else:
                raise ValueError("conflict_policy must be: keep, overwrite, or error")
        else:
            relation_type, relation_created = _ensure_relation_name(relation)
            _, source_idx, target_idx, stored_relation_type = rm.add_relation_by_type(
                noun1,
                noun2,
                relation_type,
            )
            overwritten = False

        run_short_training_and_save([(source_idx, target_idx, int(stored_relation_type))], save=False)
        results.append(
            {
                "source_noun": noun1,
                "relation": _relation_name(stored_relation_type),
                "input_relation": relation,
                "target_noun": noun2,
                "source_idx": int(source_idx),
                "target_idx": int(target_idx),
                "relation_type": int(stored_relation_type),
                "new_source_noun": not source_before,
                "new_target_noun": not target_before,
                "new_relation_name": relation_created,
                "new_relation": existing_relation == 0,
                "relation_conflict": conflict,
                "overwritten": overwritten,
            }
        )

    if run_average:
        run_long_training_and_save()
    return results


def train_from_file(
    file_path=DEFAULT_TRAINING_DATA_PATH,
    *,
    conflict_policy="keep",
    run_average=True,
):
    relations = load_training_relations(file_path)
    return train_via_feed_relations(
        relations,
        conflict_policy=conflict_policy,
        run_average=run_average,
    )


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
    results = train_from_file()
    print(f"Trained {len(results)} triples from {DEFAULT_TRAINING_DATA_PATH}")
    for result in results:
        print(result)
