"""Legacy adjective helpers kept as a thin compatibility layer."""

import torch
import torch.nn.functional as F


def adj(model, i, j, relation_type, lr_per_embedding=None):
    rel_idx = int(relation_type) - 1
    if rel_idx < 0 or rel_idx >= len(model.relations):
        raise ValueError(f"relation_type must be in [1, {len(model.relations)}]")

    i_idx = int(i)
    j_idx = int(j)
    if lr_per_embedding is None:
        lr_per_embedding = getattr(model, "lr_per_embedding", None)
    if lr_per_embedding is None:
        raise ValueError("lr_per_embedding must be provided for compatibility adj updates")

    i_embedding = model.embedding.weight.data[i_idx].clone().detach().requires_grad_(True)
    j_embedding = model.embedding.weight.data[j_idx].detach()
    relation_weight = model.relations[rel_idx].weight.data
    j_predict = relation_weight @ i_embedding
    loss = F.mse_loss(j_predict, j_embedding)
    loss.backward()

    with torch.no_grad():
        updated_embedding = i_embedding - float(lr_per_embedding[i_idx]) * i_embedding.grad

    i_embedding.grad.zero_()
    return updated_embedding.detach()
