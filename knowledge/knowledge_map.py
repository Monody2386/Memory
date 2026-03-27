from .relation_map import noun_number, relation_num, save_relation_data
import torch
import torch.nn as nn
import torch.nn.functional as F


class knowledge_map(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(knowledge_map, self).__init__()
        self.embedding = nn.Embedding(noun_number, input_dim)
        self.relations = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in range(relation_num)]
        )

    def forward(self, input_index, target_index, relation_type):
        rel_idx = int(relation_type) - 1
        if rel_idx < 0 or rel_idx >= len(self.relations):
            raise ValueError(f"relation_type must be in [1, {len(self.relations)}]")

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
    rel_idx = int(relation_type) - 1
    if rel_idx < 0 or rel_idx >= len(knowledge_map_one.relations):
        raise ValueError(f"relation_type={relation_type} exceeds relation capacity")

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

    lr_per_embedding[i_idx] = max(
        float(lr_per_embedding[i_idx]) * lr_decay_embedding, min_lr
    )
    lr_per_embedding[j_idx] = max(
        float(lr_per_embedding[j_idx]) * lr_decay_embedding, min_lr
    )
    lr_relation[rel_idx] = max(float(lr_relation[rel_idx]) * lr_decay_relation, min_lr)

    return loss.item()


def train_average(
    knowledge_map_one,
    relation_map,
    lr_per_embedding,
    lr_relation,
    lr_decay_embedding=0.95,
    lr_decay_relation=0.95,
    min_lr=1e-6,
):
    optimizer = torch.optim.Adam(knowledge_map_one.parameters(), lr=0.001)
    optimizer.zero_grad()
    involved_nouns = set()
    involved_relation_types = set()

    for i in range(relation_map.shape[0]):
        for j in range(relation_map.shape[1]):
            rt = relation_map[i][j]
            if rt == 0:
                continue

            rt_i = int(rt)
            if rt_i < 1 or rt_i > len(knowledge_map_one.relations):
                continue

            involved_nouns.add(i)
            involved_nouns.add(j)
            involved_relation_types.add(rt_i)

            y_pred, y_target = knowledge_map_one(torch.tensor(i), torch.tensor(j), rt_i)
            loss = F.mse_loss(y_pred, y_target)
            loss.backward()

    with torch.no_grad():
        grad = knowledge_map_one.embedding.weight.grad
        if grad is not None:
            lr_t = torch.tensor(
                lr_per_embedding, device=grad.device, dtype=grad.dtype
            ).unsqueeze(1)
            grad *= lr_t

        for rel_idx in range(len(knowledge_map_one.relations)):
            rel_grad = knowledge_map_one.relations[rel_idx].weight.grad
            if rel_grad is not None:
                rel_grad *= float(lr_relation[rel_idx])

    optimizer.step()

    for idx in involved_nouns:
        lr_per_embedding[idx] = max(
            float(lr_per_embedding[idx]) * lr_decay_embedding, min_lr
        )
    for rt_i in involved_relation_types:
        rel_idx = rt_i - 1
        lr_relation[rel_idx] = max(
            float(lr_relation[rel_idx]) * lr_decay_relation, min_lr
        )


def save_all(knowledge_map_one, model_path):
    save_relation_data()
    torch.save(knowledge_map_one.state_dict(), model_path)


