from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

import torch


def relation_type_to_index(relation_type: int, relation_count: int) -> int:
    rel_idx = int(relation_type) - 1
    if rel_idx < 0 or rel_idx >= relation_count:
        raise ValueError(f"relation_type must be in [1, {relation_count}]")
    return rel_idx


def iter_active_relations(matrix, relation_count: int) -> Iterator[Tuple[int, ...]]:
    for raw_indices in zip(*matrix.nonzero()):
        item_indices = tuple(int(i) for i in raw_indices)
        relation_type = int(matrix[item_indices])
        if 1 <= relation_type <= relation_count:
            yield (*item_indices, relation_type)


def scale_row_gradients(grad: torch.Tensor | None, learning_rates: Sequence[float]) -> None:
    if grad is None:
        return
    lr_tensor = torch.as_tensor(
        learning_rates,
        device=grad.device,
        dtype=grad.dtype,
    ).unsqueeze(1)
    grad.mul_(lr_tensor)


def scale_relation_gradients(relations, learning_rates: Sequence[float]) -> None:
    for rel_idx, relation in enumerate(relations):
        rel_grad = relation.weight.grad
        if rel_grad is not None:
            rel_grad.mul_(float(learning_rates[rel_idx]))


def decay_learning_rates(
    learning_rates,
    indices: Iterable[int],
    decay: float,
    min_lr: float,
) -> None:
    for idx in sorted({int(index) for index in indices}):
        learning_rates[idx] = max(float(learning_rates[idx]) * decay, min_lr)

