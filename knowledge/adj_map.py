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
from .adj_relation_map import adjective_number, adj_relation_num, save_adj_relation_data


class adj_map(nn.Module):
    def __init__(self, shared_noun_embedding, input_dim, output_dim):
        super().__init__()
        self.embedding = shared_noun_embedding
        self.adjective_embedding = nn.Embedding(adjective_number, input_dim)
        self.relations = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in range(adj_relation_num)]
        )

    def forward(self, noun_index, adjective_index, relation_type):
        rel_idx = relation_type_to_index(relation_type, len(self.relations))
        noun_emb = self.embedding(noun_index)
        adjective_target = self.adjective_embedding(adjective_index)
        adjective_pred = self.relations[rel_idx](noun_emb)
        return adjective_pred, adjective_target

    def query_adjective_similarity(self, query_tensor, top_k=5):
        embed_weights = self.adjective_embedding.weight
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)

        top_k = min(int(top_k), embed_weights.shape[0])
        similarity = F.cosine_similarity(query_tensor, embed_weights, dim=1)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        return top_indices, top_scores


def train_adj_random(
    adj_map_one,
    noun_idx,
    adjective_idx,
    relation_type,
    lr_per_embedding,
    lr_per_adjective,
    lr_adj_relation,
):
    device = next(adj_map_one.parameters()).device
    rel_idx = relation_type_to_index(relation_type, len(adj_map_one.relations))

    noun_idx = int(noun_idx)
    adjective_idx = int(adjective_idx)
    noun_tensor = torch.tensor(noun_idx, dtype=torch.long, device=device)
    adjective_tensor = torch.tensor(adjective_idx, dtype=torch.long, device=device)

    lr_noun = float(lr_per_embedding[noun_idx])
    lr_adj = float(lr_per_adjective[adjective_idx])
    lr_rel = float(lr_adj_relation[rel_idx])

    lr_decay_embedding = 0.95
    lr_decay_relation = 0.95
    min_lr = 1e-6
    loss = None

    for _ in range(100):
        adj_map_one.zero_grad(set_to_none=True)
        y_pred, y_target = adj_map_one(noun_tensor, adjective_tensor, int(relation_type))
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()

        with torch.no_grad():
            noun_grad = adj_map_one.embedding.weight.grad
            if noun_grad is not None:
                adj_map_one.embedding.weight[noun_idx] -= lr_noun * noun_grad[noun_idx]

            adjective_grad = adj_map_one.adjective_embedding.weight.grad
            if adjective_grad is not None:
                adj_map_one.adjective_embedding.weight[adjective_idx] -= lr_adj * adjective_grad[adjective_idx]

            rel_grad = adj_map_one.relations[rel_idx].weight.grad
            if rel_grad is not None:
                adj_map_one.relations[rel_idx].weight -= lr_rel * rel_grad

    decay_learning_rates(lr_per_embedding, [noun_idx], lr_decay_embedding, min_lr)
    decay_learning_rates(lr_per_adjective, [adjective_idx], lr_decay_embedding, min_lr)
    decay_learning_rates(lr_adj_relation, [rel_idx], lr_decay_relation, min_lr)
    return float(loss.item()) if loss is not None else 0.0


def train_adj_average(
    adj_map_one,
    adj_relation_map,
    lr_per_embedding,
    lr_per_adjective,
    lr_adj_relation,
    epochs=100,
    lr_decay_embedding=0.95,
    lr_decay_relation=0.95,
    min_lr=1e-6,
):
    optimizer = torch.optim.Adam(
        list(adj_map_one.embedding.parameters())
        + list(adj_map_one.adjective_embedding.parameters())
        + list(adj_map_one.relations.parameters()),
        lr=0.001,
    )
    involved_nouns = set()
    involved_adjectives = set()
    involved_relation_indices = set()
    relation_entries = list(iter_active_relations(adj_relation_map, len(adj_map_one.relations)))

    if not relation_entries:
        return 0.0

    last_loss = None
    device = next(adj_map_one.parameters()).device

    for _ in range(int(epochs)):
        optimizer.zero_grad()
        losses = []

        for noun_idx, adjective_idx, relation_type in relation_entries:
            involved_nouns.add(noun_idx)
            involved_adjectives.add(adjective_idx)
            rel_idx = relation_type_to_index(relation_type, len(adj_map_one.relations))
            involved_relation_indices.add(rel_idx)

            y_pred, y_target = adj_map_one(
                torch.tensor(noun_idx, dtype=torch.long, device=device),
                torch.tensor(adjective_idx, dtype=torch.long, device=device),
                relation_type,
            )
            losses.append(F.mse_loss(y_pred, y_target))

        loss = torch.stack(losses).mean()
        loss.backward()

        with torch.no_grad():
            scale_row_gradients(adj_map_one.embedding.weight.grad, lr_per_embedding)
            scale_row_gradients(adj_map_one.adjective_embedding.weight.grad, lr_per_adjective)
            scale_relation_gradients(adj_map_one.relations, lr_adj_relation)

        optimizer.step()
        last_loss = float(loss.item())

    decay_learning_rates(lr_per_embedding, involved_nouns, lr_decay_embedding, min_lr)
    decay_learning_rates(lr_per_adjective, involved_adjectives, lr_decay_embedding, min_lr)
    decay_learning_rates(lr_adj_relation, involved_relation_indices, lr_decay_relation, min_lr)
    return last_loss if last_loss is not None else 0.0


def train_joint_average(
    knowledge_map_one,
    adj_map_one,
    relation_map,
    adj_relation_map,
    lr_per_embedding,
    lr_relation,
    lr_per_adjective,
    lr_adj_relation,
    epochs=100,
    lr_decay_embedding=0.95,
    lr_decay_relation=0.95,
    min_lr=1e-6,
):
    optimizer = torch.optim.Adam(
        list(knowledge_map_one.parameters())
        + list(adj_map_one.adjective_embedding.parameters())
        + list(adj_map_one.relations.parameters()),
        lr=0.001,
    )

    involved_nouns = set()
    involved_adjectives = set()
    involved_knowledge_relation_indices = set()
    involved_adj_relation_indices = set()
    noun_relation_entries = list(
        iter_active_relations(relation_map, len(knowledge_map_one.relations))
    )
    adj_relation_entries = list(
        iter_active_relations(adj_relation_map, len(adj_map_one.relations))
    )

    if not noun_relation_entries and not adj_relation_entries:
        return 0.0

    last_loss = None
    device = next(knowledge_map_one.parameters()).device

    for _ in range(int(epochs)):
        optimizer.zero_grad()
        losses = []

        for noun_i, noun_j, relation_type in noun_relation_entries:
            involved_nouns.update((noun_i, noun_j))
            rel_idx = relation_type_to_index(relation_type, len(knowledge_map_one.relations))
            involved_knowledge_relation_indices.add(rel_idx)

            y_pred, y_target = knowledge_map_one(
                torch.tensor(noun_i, dtype=torch.long, device=device),
                torch.tensor(noun_j, dtype=torch.long, device=device),
                relation_type,
            )
            losses.append(F.mse_loss(y_pred, y_target))

        for noun_idx, adjective_idx, relation_type in adj_relation_entries:
            involved_nouns.add(noun_idx)
            involved_adjectives.add(adjective_idx)
            rel_idx = relation_type_to_index(relation_type, len(adj_map_one.relations))
            involved_adj_relation_indices.add(rel_idx)

            y_pred, y_target = adj_map_one(
                torch.tensor(noun_idx, dtype=torch.long, device=device),
                torch.tensor(adjective_idx, dtype=torch.long, device=device),
                relation_type,
            )
            losses.append(F.mse_loss(y_pred, y_target))

        loss = torch.stack(losses).mean()
        loss.backward()

        with torch.no_grad():
            scale_row_gradients(knowledge_map_one.embedding.weight.grad, lr_per_embedding)
            scale_relation_gradients(knowledge_map_one.relations, lr_relation)
            scale_row_gradients(adj_map_one.adjective_embedding.weight.grad, lr_per_adjective)
            scale_relation_gradients(adj_map_one.relations, lr_adj_relation)

        optimizer.step()
        last_loss = float(loss.item())

    decay_learning_rates(lr_per_embedding, involved_nouns, lr_decay_embedding, min_lr)
    decay_learning_rates(lr_per_adjective, involved_adjectives, lr_decay_embedding, min_lr)
    decay_learning_rates(lr_relation, involved_knowledge_relation_indices, lr_decay_relation, min_lr)
    decay_learning_rates(lr_adj_relation, involved_adj_relation_indices, lr_decay_relation, min_lr)
    return last_loss if last_loss is not None else 0.0


def save_adj_all(adj_map_one, model_path):
    save_adj_relation_data()
    torch.save(adj_map_one.state_dict(), model_path)
