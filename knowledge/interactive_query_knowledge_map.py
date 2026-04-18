"""Interactive noun-relation query for knowledge_map.

Input examples:
    apple include
    teacher belong to

Type "stop" to exit.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

rm = importlib.import_module("knowledge.relation_map")
from knowledge.training import MODEL_PATH, knowledge_map_one


STOP_TOKENS = {"stop", "quit", "exit"}
DEFAULT_TOP_K = 3


def _load_state() -> None:
    if rm.load_relation_data() is False:
        raise FileNotFoundError("relation_data.npz not found. Train or initialize relation data first.")

    model_path = pathlib.Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train knowledge_map first.")
    knowledge_map_one.load_state_dict(torch.load(model_path, map_location="cpu"))
    knowledge_map_one.eval()


def _parse_query(line: str) -> tuple[str, str]:
    tokens = line.strip().lower().split()
    if len(tokens) < 2:
        raise ValueError("Please enter: <noun> <relation>")

    source_noun = tokens[0]
    relation_name = " ".join(tokens[1:])

    if source_noun not in rm.noun_list:
        raise ValueError(f"Unknown noun '{source_noun}'")
    if relation_name not in rm.relation_list:
        available = ", ".join(rm.relation_list)
        raise ValueError(f"Unknown relation '{relation_name}'. Available relations: {available}")

    return source_noun, relation_name


def query(source_noun: str, relation_name: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    source_idx = rm.noun_list.index(source_noun)
    relation_type = rm.relation_list.index(relation_name) + 1
    relation_idx = relation_type - 1

    with torch.no_grad():
        source_tensor = torch.tensor(source_idx, dtype=torch.long)
        source_embedding = knowledge_map_one.embedding(source_tensor)
        predicted_target = knowledge_map_one.relations[relation_idx](source_embedding)
        top_indices, top_scores = knowledge_map_one.query_similarity(predicted_target, top_k=top_k)

    results = []
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        noun = rm.noun_list[int(idx)] if int(idx) < len(rm.noun_list) else f"noun_{int(idx)}"
        results.append(
            {
                "noun": noun,
                "noun_idx": int(idx),
                "score": float(score),
            }
        )
    return results


def main() -> None:
    _load_state()
    print("Interactive knowledge_map query")
    print("Format: <noun> <relation>")
    print(f"Relations: {', '.join(rm.relation_list)}")
    print("Type 'stop' to exit.")

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
            source_noun, relation_name = _parse_query(line)
            results = query(source_noun, relation_name)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(
            {
                "source_noun": source_noun,
                "relation": relation_name,
                "top1_3": results[:3],
            }
        )


if __name__ == "__main__":
    main()
