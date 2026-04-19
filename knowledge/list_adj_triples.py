"""Print all saved noun-adjective relation triples from adj_map data."""

from __future__ import annotations

import importlib
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

rm = importlib.import_module("knowledge.relation_map")
arm = importlib.import_module("knowledge.adj_relation_map")


def _noun_name(index: int) -> str:
    if 0 <= index < len(rm.noun_list):
        return rm.noun_list[index]
    return f"noun_{index}"


def _adjective_name(index: int) -> str:
    if 0 <= index < len(arm.adjective_list):
        return arm.adjective_list[index]
    return f"adj_{index}"


def _relation_name(relation_type: int) -> str:
    relation_idx = int(relation_type) - 1
    if 0 <= relation_idx < len(arm.adj_relation_list):
        return arm.adj_relation_list[relation_idx]
    return f"adj_relation_type_{int(relation_type)}"


def iter_triples():
    if rm.load_relation_data() is False:
        raise FileNotFoundError("relation_data.npz not found. Train or initialize relation data first.")
    if arm.load_adj_relation_data() is False:
        raise FileNotFoundError("adj_relation_data.npz not found. Train or initialize adj data first.")

    noun_indices, adjective_indices = np.nonzero(arm.adj_relation_map)
    for noun_idx, adjective_idx in zip(noun_indices.tolist(), adjective_indices.tolist()):
        relation_type = int(arm.adj_relation_map[noun_idx, adjective_idx])
        yield {
            "noun": _noun_name(int(noun_idx)),
            "relation": _relation_name(relation_type),
            "adjective": _adjective_name(int(adjective_idx)),
            "noun_idx": int(noun_idx),
            "adjective_idx": int(adjective_idx),
            "relation_type": relation_type,
        }


def _print_adj_relation_list() -> None:
    print(f"Adj relation list ({len(arm.adj_relation_list)}):")
    if not arm.adj_relation_list:
        print("  <empty>")
        return

    for idx, relation in enumerate(arm.adj_relation_list, start=1):
        print(f"  {idx}: {relation}")


def _print_adjective_list() -> None:
    print(f"Adjective list ({len(arm.adjective_list)}):")
    if not arm.adjective_list:
        print("  <empty>")
        return

    for idx, adjective in enumerate(arm.adjective_list):
        print(f"  {idx}: {adjective}")


def _print_noun_list() -> None:
    print(f"Noun list ({len(rm.noun_list)}):")
    if not rm.noun_list:
        print("  <empty>")
        return

    for idx, noun in enumerate(rm.noun_list):
        print(f"  {idx}: {noun}")


def main() -> None:
    triples = list(iter_triples())
    _print_adj_relation_list()
    print()
    _print_adjective_list()
    print()
    _print_noun_list()
    print()

    if not triples:
        print("No noun-adj_relation-adjective triples found.")
        return

    print(f"Found {len(triples)} noun-adj_relation-adjective triples:")
    for triple in triples:
        print(
            f"{triple['noun']} {triple['relation']} {triple['adjective']} "
            f"(noun_idx={triple['noun_idx']}, adjective_idx={triple['adjective_idx']}, "
            f"relation_type={triple['relation_type']})"
        )


if __name__ == "__main__":
    main()
