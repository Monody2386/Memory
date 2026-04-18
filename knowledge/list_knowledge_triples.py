"""Print all saved noun-relation-noun triples from knowledge_map data."""

from __future__ import annotations

import importlib
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

rm = importlib.import_module("knowledge.relation_map")


def _noun_name(index: int) -> str:
    if 0 <= index < len(rm.noun_list):
        return rm.noun_list[index]
    return f"noun_{index}"


def _relation_name(relation_type: int) -> str:
    relation_idx = int(relation_type) - 1
    if 0 <= relation_idx < len(rm.relation_list):
        return rm.relation_list[relation_idx]
    return f"relation_type_{int(relation_type)}"


def iter_triples():
    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError("relation_data.npz not found. Train or initialize relation data first.")

    source_indices, target_indices = np.nonzero(rm.relation_map)
    for source_idx, target_idx in zip(source_indices.tolist(), target_indices.tolist()):
        relation_type = int(rm.relation_map[source_idx, target_idx])
        yield {
            "source_noun": _noun_name(int(source_idx)),
            "relation": _relation_name(relation_type),
            "target_noun": _noun_name(int(target_idx)),
            "source_idx": int(source_idx),
            "target_idx": int(target_idx),
            "relation_type": relation_type,
        }


def _print_relation_list() -> None:
    print(f"Relation list ({len(rm.relation_list)}):")
    if not rm.relation_list:
        print("  <empty>")
        return

    for idx, relation in enumerate(rm.relation_list, start=1):
        print(f"  {idx}: {relation}")


def _print_noun_list() -> None:
    print(f"Noun list ({len(rm.noun_list)}):")
    if not rm.noun_list:
        print("  <empty>")
        return

    for idx, noun in enumerate(rm.noun_list):
        print(f"  {idx}: {noun}")


def main() -> None:
    triples = list(iter_triples())
    _print_relation_list()
    print()
    _print_noun_list()
    print()

    if not triples:
        print("No noun-relation-noun triples found.")
        return

    print(f"Found {len(triples)} noun-relation-noun triples:")
    for triple in triples:
        print(
            f"{triple['source_noun']} {triple['relation']} {triple['target_noun']} "
            f"(source_idx={triple['source_idx']}, target_idx={triple['target_idx']}, "
            f"relation_type={triple['relation_type']})"
        )


if __name__ == "__main__":
    main()
