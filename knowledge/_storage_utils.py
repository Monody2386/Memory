from __future__ import annotations

import numpy as np


def ensure_vector_capacity(values, target_size: int, fill_value: float = 1.0):
    if len(values) < target_size:
        padded = np.full(target_size, fill_value, dtype=np.asarray(values).dtype)
        padded[: len(values)] = values
        return padded
    if len(values) > target_size:
        return values[:target_size]
    return values
