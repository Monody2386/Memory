import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from train_relation_map import noun_number, noun_dim,  relation_map, lr_per_embedding
from relation_map import relation_map, relation_num


def search_relation(relation_map, i, j, relation_type):
    relation = relation_map[i][j]
    if relation == relation_type:
        return True
    return False

def update_relation_map(relation_map, i, j, relation_type):
    relation_map[i][j] = relation_type



class ScoredTensorQueue:
    """
    管理张量队列，每个张量绑定一个数值(score)
    - 自动保持队列长度
    - 可根据 score 条件选择性丢弃张量
    - 返回堆叠后的 tensor 和对应的 scores
    """
    def __init__(self, maxlen=100, device='cpu'):
        self.maxlen = maxlen
        self.device = device
        self.queue = deque()  # 存储 (tensor, score) 元组
    
    def append(self, tensor, score=0.0, noun_type=None, action_type=None):
        tensor = tensor.to(self.device)
        self.queue.append((tensor, score, noun_type, action_type))
        
        while len(self.queue) > self.maxlen:
            self.queue.popleft()
    
    def set_maxlen(self, new_maxlen):
        """
        动态修改队列长度
        """
        self.maxlen = new_maxlen
        while len(self.queue) > self.maxlen:
            self.queue.popleft()
    
    def filter_by_score(self, threshold, mode='ge'):
        """
        根据 score 选择性丢弃
        threshold: 阈值
        mode: 'ge' 保留 score >= threshold
              'le' 保留 score <= threshold
        """
        if mode == 'ge':
            self.queue = deque([(t, s) for t, s in self.queue if s >= threshold])
        elif mode == 'le':
            self.queue = deque([(t, s) for t, s in self.queue if s <= threshold])
        else:
            raise ValueError("mode must be 'ge' or 'le'")
    
    def get_stack(self):
        if len(self.queue) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                [],
                []
            )
        
        tensors, scores, noun_types, action_types = zip(*self.queue)
    
        return (
            torch.stack(tensors),
            torch.tensor(scores, device=self.device),
            list(noun_types),
            list(action_types)
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

short_memory = ScoredTensorQueue(maxlen=50, device='cpu')




