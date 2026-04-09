from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .reward_types import RewardSample


class RewardMemory:
    def __init__(self, samples: Optional[Sequence[RewardSample]] = None):
        self._samples: List[RewardSample] = list(samples or [])

    def add_sample(self, sample: RewardSample) -> None:
        self._samples.append(sample)

    def add_many(self, samples: Iterable[RewardSample]) -> None:
        self._samples.extend(samples)

    def all_samples(self) -> List[RewardSample]:
        return list(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)
