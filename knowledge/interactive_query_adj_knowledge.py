"""Interactive adj-noun query.

Input format:
    noun adj_relation

Examples:
    apple color
    apple taste
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
from knowledge.training import ADJ_MODEL_PATH, MODEL_PATH, adj_map_one, knowledge_map_one


STOP_TOKENS = {"stop", "quit", "exit"}
DEFAULT_TOP_K = 3


def _load_state() -> None:
    if rm.load_relation_data() is False:
        raise FileNotFoundError("relation_data.npz not found. Train or initialize relation data first.")
    if arm.load_adj_relation_data() is False:
        raise FileNotFoundError("adj_relation_data.npz not found. Train or initialize adj data first.")

    model_path = pathlib.Path(MODEL_PATH)
    if model_path.exists():
        knowledge_map_one.load_state_dict(torch.load(model_path, map_location="cpu"))

    adj_model_path = pathlib.Path(ADJ_MODEL_PATH)
    if not adj_model_path.exists():
        raise FileNotFoundError(f"{ADJ_MODEL_PATH} not found. Train adj_map first.")

    adj_map_one.load_state_dict(torch.load(adj_model_path, map_location="cpu"), strict=False)
    knowledge_map_one.eval()
    adj_map_one.eval()


def _parse_query(line: str) -> tuple[str, str]:
    tokens = line.strip().lower().split()
    if len(tokens) < 2:
        raise ValueError("Please enter: <noun> <adj_relation>")

    noun = tokens[0]
    relation = " ".join(tokens[1:])

    if noun not in rm.noun_list:
        raise ValueError(f"Unknown noun '{noun}'")
    if relation not in arm.adj_relation_list:
        available = ", ".join(arm.adj_relation_list)
        raise ValueError(f"Unknown adj relation '{relation}'. Available relations: {available}")
    return noun, relation


def query(noun: str, relation: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    noun_idx = rm.noun_list.index(noun)
    relation_idx = arm.adj_relation_list.index(relation)

    with torch.no_grad():
        noun_tensor = torch.tensor(noun_idx, dtype=torch.long)
        noun_embedding = adj_map_one.embedding(noun_tensor)
        predicted_adjective = adj_map_one.relations[relation_idx](noun_embedding)
        top_indices, top_scores = adj_map_one.query_adjective_similarity(
            predicted_adjective,
            top_k=top_k,
        )

    results = []
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        adjective = (
            arm.adjective_list[int(idx)]
            if int(idx) < len(arm.adjective_list)
            else f"adj_{int(idx)}"
        )
        results.append(
            {
                "adjective": adjective,
                "adjective_idx": int(idx),
                "score": float(score),
            }
        )
    return results


def main() -> None:
    _load_state()
    print("Interactive adj-noun query")
    print("Format: <noun> <adj_relation>")
    print(f"Adj relations: {', '.join(arm.adj_relation_list)}")
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
            noun, relation = _parse_query(line)
            results = query(noun, relation)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print({"noun": noun, "relation": relation, "top1_3": results[:3]})


if __name__ == "__main__":
    main()
