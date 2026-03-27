import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from knowledge.relation_map import noun_number, noun_dim, relation_map, relation_num, lr_per_embedding
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

