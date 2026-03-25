import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from relation_map import noun_number, noun_dim, relation_num, relation_map, lr_per_embedding

class inf_concat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(inf_concat, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def adj(model, i, j, relation_type):
    i_embedding = model.embedding.weight.data[i].clone().detach().requires_grad_(True)
    j_embedding = model.embedding.weight.data[j]
    relation_embeddings = model.relations[relation_type - 1].weight.data
    j_predict = relation_embeddings @ i_embedding
    loss = F.mse_loss(j_predict, j_embedding)
    loss.backward()
    with torch.no_grad():
        i_embedding -= lr_per_embedding[i] * i_embedding.grad
    i_embedding.grad.zero_()
    return i_embedding

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
    
    def append(self, tensor, score=0.0):
        """
        添加一个张量和对应的 score
        """
        tensor = tensor.to(self.device)
        self.queue.append((tensor, score))
        # 保持队列长度
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
        """
        返回堆叠后的 tensor 和 scores
        """
        if len(self.queue) == 0:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        tensors, scores = zip(*self.queue)
        return torch.stack(tensors), torch.tensor(scores, device=self.device)
    def get_latest_n(self, n):

        if not self.queue:
            return torch.empty(0, device=self.device)
        latest_tensors = [t for t, _ in list(self.queue)[-n:]]
        return torch.stack(latest_tensors)
    
    def __len__(self):
        return len(self.queue)
    
    def clear(self):
        self.queue.clear()

short_memory = ScoredTensorQueue(maxlen=50, device='cpu')



