from dataclasses import dataclass

import torch


@dataclass
class Episode:
    time: int
    place: int
    subject: int
    predicate: int
    object: int
    place_emb: torch.Tensor
    subject_emb: torch.Tensor
    object_emb: torch.Tensor


class EpisodeMemory:
    def __init__(self):
        self.episodes = []
        self.index = {}  # token_id -> set(eid)

    def add(self, ep: Episode):
        eid = len(self.episodes)
        self.episodes.append(ep)

        for token in [ep.time, ep.place, ep.subject, ep.predicate, ep.object]:
            self.index.setdefault(int(token), set()).add(eid)

    def lookup(self, token_id: int):
        return [self.episodes[eid] for eid in sorted(self.index.get(int(token_id), []))]
