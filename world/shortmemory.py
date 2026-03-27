from collections import deque

import torch

from knowledge.relation_map import (
    lr_per_embedding,
    noun_dim,
    noun_number,
    relation_map,
    relation_num,
)


def search_relation(relation_map, i, j, relation_type):
    return relation_map[i][j] == relation_type


def update_relation_map(relation_map, i, j, relation_type):
    relation_map[i][j] = relation_type


class ScoredTensorQueue:
    def __init__(self, maxlen=100, device="cpu"):
        self.maxlen = maxlen
        self.device = device
        self.queue = deque()

    def append(self, tensor, score=0.0, noun_type=None, action_type=None):
        tensor = tensor.to(self.device)
        self.queue.append((tensor, score, noun_type, action_type))
        while len(self.queue) > self.maxlen:
            self.queue.popleft()

    def set_maxlen(self, new_maxlen):
        self.maxlen = new_maxlen
        while len(self.queue) > self.maxlen:
            self.queue.popleft()

    def filter_by_score(self, threshold, mode="ge"):
        if mode == "ge":
            self.queue = deque(
                [(t, s, n, a) for t, s, n, a in self.queue if s >= threshold]
            )
        elif mode == "le":
            self.queue = deque(
                [(t, s, n, a) for t, s, n, a in self.queue if s <= threshold]
            )
        else:
            raise ValueError("mode must be 'ge' or 'le'")

    def get_stack(self):
        if len(self.queue) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                [],
                [],
            )

        tensors, scores, noun_types, action_types = zip(*self.queue)
        return (
            torch.stack(tensors),
            torch.tensor(scores, device=self.device),
            list(noun_types),
            list(action_types),
        )

    def get_latest_n(self, n):
        if not self.queue:
            return torch.empty(0, device=self.device)
        latest = list(self.queue)[-n:]
        tensors = [t for t, _, _, _ in latest]
        return torch.stack(tensors)

    def __len__(self):
        return len(self.queue)

    def filter_by_type(self, noun_type=None, action_type=None):
        new_queue = []
        for t, s, n, a in self.queue:
            if noun_type is not None and n != noun_type:
                continue
            if action_type is not None and a != action_type:
                continue
            new_queue.append((t, s, n, a))
        self.queue = deque(new_queue)

    def clear(self):
        self.queue.clear()


short_memory = ScoredTensorQueue(maxlen=50, device="cpu")
