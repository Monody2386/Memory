from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Episode:
    time: int
    place: int
    subject: int
    predicate: int
    object: int

    place_emb: torch.Tensor     # [d]
    subject_emb: torch.Tensor   # [d]
    object_emb: torch.Tensor    # [d]

class EpisodeMemory:
    def __init__(self):
        self.episodes = []
        self.index = {}  # token_id -> set(eid)

    def add(self, ep: Episode):
        eid = len(self.episodes)

        
        self.episodes.append(ep)

        # 建立倒排索引
        for field in [ep.time, ep.place, ep.subject, ep.predicate, ep.object]:
            for token in field:
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(eid)

