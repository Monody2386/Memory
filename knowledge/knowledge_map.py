import torch
import torch.nn as nn
import torch.nn.functional as F

from ._training_utils import (
    decay_learning_rates,
    iter_active_relations,
    relation_type_to_index,
    scale_relation_gradients,
    scale_row_gradients,
)
from .relation_map import noun_number, relation_num, save_relation_data


class knowledge_map(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(noun_number, input_dim)
        self.relations = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in range(relation_num)]
        )

    def forward(self, input_index, target_index, relation_type):
        rel_idx = relation_type_to_index(relation_type, len(self.relations))
        x = self.embedding(input_index)
        target = self.embedding(target_index)
        y = self.relations[rel_idx](x)
        return y, target

    def query_similarity(self, query_tensor, top_k=5):
        embed_weights = self.embedding.weight
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)

        top_k = min(int(top_k), embed_weights.shape[0])
        similarity = F.cosine_similarity(query_tensor, embed_weights, dim=1)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        return top_indices, top_scores

    def query_relation_by_tensor(self, query_tensor, top_k=1):
        if query_tensor.dim() > 1:
            query_tensor = query_tensor.view(-1)

        weights = torch.stack([rel.weight.view(-1) for rel in self.relations])
        query_tensor = F.normalize(query_tensor, dim=0)
        weights = F.normalize(weights, dim=1)

        top_k = min(int(top_k), weights.shape[0])
        similarity = torch.matmul(weights, query_tensor)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        return top_indices, top_scores


def train_random(knowledge_map_one, i, j, relation_type, lr_per_embedding, lr_relation):
    device = next(knowledge_map_one.parameters()).device
    rel_idx = relation_type_to_index(relation_type, len(knowledge_map_one.relations))

    i_idx = int(i)
    j_idx = int(j)
    i_tensor = torch.tensor(i_idx, dtype=torch.long, device=device)
    j_tensor = torch.tensor(j_idx, dtype=torch.long, device=device)

    lr_i = float(lr_per_embedding[i_idx])
    lr_j = float(lr_per_embedding[j_idx])
    lr_rel = float(lr_relation[rel_idx])

    lr_decay_embedding = 0.95
    lr_decay_relation = 0.95
    min_lr = 1e-6
    loss = None

    for _ in range(100):
        knowledge_map_one.zero_grad(set_to_none=True)
        y_pred, y_target = knowledge_map_one(i_tensor, j_tensor, int(relation_type))
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()

        with torch.no_grad():
            emb_grad = knowledge_map_one.embedding.weight.grad
            if emb_grad is not None:
                knowledge_map_one.embedding.weight[i_idx] -= lr_i * emb_grad[i_idx]
                if j_idx != i_idx:
                    knowledge_map_one.embedding.weight[j_idx] -= lr_j * emb_grad[j_idx]

            rel_grad = knowledge_map_one.relations[rel_idx].weight.grad
            if rel_grad is not None:
                knowledge_map_one.relations[rel_idx].weight -= lr_rel * rel_grad

    decay_learning_rates(lr_per_embedding, [i_idx, j_idx], lr_decay_embedding, min_lr)
    decay_learning_rates(lr_relation, [rel_idx], lr_decay_relation, min_lr)
    return float(loss.item()) if loss is not None else 0.0


def train_average(
    knowledge_map_one,
    relation_map,
    lr_per_embedding,
    lr_relation,
    epochs=100,
    lr_decay_embedding=0.95,
    lr_decay_relation=0.95,
    min_lr=1e-6,
):
    optimizer = torch.optim.Adam(knowledge_map_one.parameters(), lr=0.001)
    involved_nouns = set()
    involved_relation_indices = set()
    relation_entries = list(iter_active_relations(relation_map, len(knowledge_map_one.relations)))

    if not relation_entries:
        return 0.0

    last_loss = None
    device = next(knowledge_map_one.parameters()).device

    for _ in range(int(epochs)):
        optimizer.zero_grad()
        losses = []

        for i, j, relation_type in relation_entries:
            involved_nouns.update((i, j))
            rel_idx = relation_type_to_index(relation_type, len(knowledge_map_one.relations))
            involved_relation_indices.add(rel_idx)

            y_pred, y_target = knowledge_map_one(
                torch.tensor(i, dtype=torch.long, device=device),
                torch.tensor(j, dtype=torch.long, device=device),
                relation_type,
            )
            losses.append(F.mse_loss(y_pred, y_target))

        loss = torch.stack(losses).mean()
        loss.backward()

        with torch.no_grad():
            scale_row_gradients(knowledge_map_one.embedding.weight.grad, lr_per_embedding)
            scale_relation_gradients(knowledge_map_one.relations, lr_relation)

        optimizer.step()
        last_loss = float(loss.item())

    decay_learning_rates(lr_per_embedding, involved_nouns, lr_decay_embedding, min_lr)
    decay_learning_rates(lr_relation, involved_relation_indices, lr_decay_relation, min_lr)
    return last_loss if last_loss is not None else 0.0


def save_all(knowledge_map_one, model_path):
    save_relation_data()
    torch.save(knowledge_map_one.state_dict(), model_path)
