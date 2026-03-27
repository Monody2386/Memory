from relation_map import noun_number, noun_dim, add_relation, load_relation_data, save_relation_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# relation is i to j

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
        if relation_type == 1: # classify to
            y = self.relations[0](x)
        elif relation_type == 2: # include
            y = self.relations[1](x)
        elif relation_type == 3: # 
            y = self.relations[2](x)
        elif relation_type == 4:
            y = self.relations[3](x)
        elif relation_type == 5:
            y = self.relations[4](x)   
        return y, target
    
    def query_similarity(self, query_tensor, top_k=5):
        """
        输入查询 tensor，计算与 embedding 层所有向量的余弦相似度
        :param query_tensor: 输入查询向量，shape: [input_dim] 或 [1, input_dim]
        :param top_k: 返回最相似的前 k 个结果
        :return: 最相似的索引、相似度值
        """
        # 1. 获取 embedding 层所有词向量 (noun_number, input_dim)
        embed_weights = self.embedding.weight  # shape: [num_embeddings, embedding_dim]

        # 2. 规整查询向量形状 [1, input_dim]
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)  # [input_dim] → [1, input_dim]

        # 3. 计算余弦相似度
        # F.cosine_similarity 会自动广播，结果 shape: [num_embeddings]
        similarity = F.cosine_similarity(query_tensor, embed_weights, dim=1)

        # 4. 取相似度最高的 top-k
        top_scores, top_indices = torch.topk(similarity, k=top_k)

        return top_indices, top_scores
    


    def query_relation_by_tensor(self, query_tensor, top_k=1):
        """
        输入一个 tensor，找到最相似的 relation（基于 linear weight）
        
        :param query_tensor: shape = [input_dim * output_dim] 或 [N]
        :param top_k: 返回最相似的 relation 数量
        :return: relation indices, similarity scores
        """

        # 1. 展平 query
        if query_tensor.dim() > 1:
            query_tensor = query_tensor.view(-1)
        
        # 2. 收集所有 relation 的 weight
        weights = []
        for rel in self.relations:
            w = rel.weight.view(-1)  # flatten
            weights.append(w)

        weights = torch.stack(weights)  # [num_rel, dim]

        # 3. 归一化（很重要）
        query_tensor = F.normalize(query_tensor, dim=0)
        weights = F.normalize(weights, dim=1)

        # 4. 计算相似度
        similarity = torch.matmul(weights, query_tensor)  # [num_rel]

        # 5. 取 top-k
        top_scores, top_indices = torch.topk(similarity, k=top_k)





        return top_indices, top_scores



# 先从文件恢复 relation_map / noun_list / relation_list（如果存在）
# load_relation_data()

# if os.path.exists(MODEL_PATH):
#     knowledge_map_one.load_state_dict(torch.load(MODEL_PATH))

# query = torch.randn(noun_dim)  # shape: [128]

# # 查询相似度（返回前 5 个最相似）
# indices, scores = knowledge_map_one.query_similarity(query, top_k=5)

# print("embedding", indices)
# print("score", scores)
#random train
def train_random(knowledge_map_one, i, j, relation_type, lr_per_embedding, lr_relation):
    """
    对单条 (i -> j, relation_type) 做“按各自学习率”的梯度下降更新：
    - embedding[i] 用 lr_per_embedding[i]
    - embedding[j] 用 lr_per_embedding[j]
    - relations[relation_type-1] 用 lr_relation[relation_type-1]
    """
    device = next(knowledge_map_one.parameters()).device
    rel_idx = int(relation_type) - 1
    if rel_idx < 0 or rel_idx >= len(knowledge_map_one.relations):
        raise ValueError(f"relation_type={relation_type} 对应的 rel_idx={rel_idx} 超出 relations 数量。")

    i_idx = int(i)
    j_idx = int(j)

    # 确保索引是 long，且 tensor 在同一设备
    i_tensor = torch.tensor(i_idx, dtype=torch.long, device=device)
    j_tensor = torch.tensor(j_idx, dtype=torch.long, device=device)

    # 将 lr 转成标量（支持 numpy / torch tensor）
    lr_i = float(lr_per_embedding[i_idx])
    lr_j = float(lr_per_embedding[j_idx])
    lr_rel = float(lr_relation[rel_idx])

    # 训练后对参与的 embedding / relation 学习率做持久衰减
    lr_decay_embedding = 0.95
    lr_decay_relation = 0.95
    min_lr = 1e-6

    for _ in range(100):
        knowledge_map_one.zero_grad(set_to_none=True)
        y_pred, y_target = knowledge_map_one(i_tensor, j_tensor, int(relation_type))
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()

        with torch.no_grad():
            # embedding 的梯度在 embedding.weight.grad 上
            emb_grad = knowledge_map_one.embedding.weight.grad
            if emb_grad is not None:
                knowledge_map_one.embedding.weight[i_idx] -= lr_i * emb_grad[i_idx]
                if j_idx != i_idx:
                    knowledge_map_one.embedding.weight[j_idx] -= lr_j * emb_grad[j_idx]

            rel_grad = knowledge_map_one.relations[rel_idx].weight.grad
            if rel_grad is not None:
                knowledge_map_one.relations[rel_idx].weight -= lr_rel * rel_grad

    # 更新学习率（持久化依赖外层再调用 save_relation_data）
    lr_per_embedding[i_idx] = max(float(lr_per_embedding[i_idx]) * lr_decay_embedding, min_lr)
    lr_per_embedding[j_idx] = max(float(lr_per_embedding[j_idx]) * lr_decay_embedding, min_lr)
    lr_relation[rel_idx] = max(float(lr_relation[rel_idx]) * lr_decay_relation, min_lr)

    return loss.item()

#average train
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
    for i in range(noun_number):
        for j in range(noun_number):
            rt = relation_map[i][j]
            if rt != 0:
                rt_i = int(rt)
                if rt_i < 1 or rt_i > len(knowledge_map_one.relations):
                    continue
                involved_nouns.add(i)
                involved_nouns.add(j)
                involved_relation_types.add(rt_i)

                y_pred, y_target = knowledge_map_one(
                    torch.tensor(i), torch.tensor(j), rt_i
                )
                loss = F.mse_loss(y_pred, y_target)
                loss.backward()
    with torch.no_grad():
        grad = knowledge_map_one.embedding.weight.grad
        if grad is not None:
            lr_t = torch.tensor(lr_per_embedding, device=grad.device, dtype=grad.dtype).unsqueeze(1)
            grad *= lr_t

        # 关系层梯度按对应学习率缩放
        for rel_idx in range(len(knowledge_map_one.relations)):
            rel_grad = knowledge_map_one.relations[rel_idx].weight.grad
            if rel_grad is not None:
                rel_lr_t = float(lr_relation[rel_idx])
                rel_grad *= rel_lr_t
    optimizer.step()

    # 训练完成后持久衰减：仅对参与训练过的 embedding / relation_type 衰减
    for idx in involved_nouns:
        lr_per_embedding[idx] = max(float(lr_per_embedding[idx]) * lr_decay_embedding, min_lr)
    for rt_i in involved_relation_types:
        rel_idx = rt_i - 1
        lr_relation[rel_idx] = max(float(lr_relation[rel_idx]) * lr_decay_relation, min_lr)

def save_all(knowledge_map_one, model_path):
    """训练完成后统一保存：关系数据 + 模型参数。"""
    save_relation_data()
    torch.save(knowledge_map_one.state_dict(), model_path)


# if __name__ == "__main__":
#     add_relation("apple", "fruit", 1)
#     random_train_random(noun_list.index("apple"), noun_list.index("fruit"), 1)
#     save_all()
