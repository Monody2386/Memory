from relation_map import noun_number, noun_dim, relation_map, noun_list, embedding_list, relation_list, relation_embedding_list, lr_noun, lr_relation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# relation is i to j
def train_relation_map(i, j):
    if relation_map[i][j] != 0:
        return True
    else:
        return False
lr_per_embedding = torch.ones(noun_number)
class knowledge_map(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(knowledge_map, self).__init__()
        self.embedding = nn.Embedding(noun_number, input_dim)
        self.relations = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(5)
        ])

    def forward(self, input_index, target_index, relation_type):
        x = self.embedding(input_index)
        target = self.embedding(target_index)
        if relation_type == 1:
            y = self.relations[0](x)
        elif relation_type == 2:
            y = self.relations[1](x)
        elif relation_type == 3:
            y = self.relations[2](x)
        elif relation_type == 4:
            y = self.relations[3](x)
        elif relation_type == 5:
            y = self.relations[4](x)   
        return y, target

knowledge_map_one = knowledge_map(noun_dim, noun_dim)
#random train
def random_train_random(i, j, relation_type):
    optimizer = torch.optim.Adam(knowledge_map_one.parameters(), lr=0.001)
    for step in range(100):
        optimizer.zero_grad()
        y_pred, y_target = knowledge_map_one(torch.tensor(i), torch.tensor(j), relation_type)
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()
        with torch.no_grad():
            grad = torch.embedding.weight.grad  # (N, d)
            grad *= lr_per_embedding.unsqueeze(1)
        optimizer.step()

#average train
def random_train_average(relation_map):
    optimizer = torch.optim.Adam(knowledge_map_one.parameters(), lr=0.001)
    for i in range(noun_number):
        for j in range(noun_number):
            if relation_map[i][j] != 0:
                y_pred, y_target = knowledge_map_one(torch.tensor(i), torch.tensor(j), relation_map[i][j])
                loss = F.mse_loss(y_pred, y_target)
                loss.backward()
    with torch.no_grad():
        grad = torch.embedding.weight.grad 
        grad *= lr_per_embedding.unsqueeze(1)
    optimizer.step()



