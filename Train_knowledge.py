import random
from typing import Optional
import os

import torch

import relation_map as rm
from Feed_relations import begin_feed_training, end_feed_training, random_feed
from train_relation_map import knowledge_map

MODEL_PATH = "knowledge_map_one.pt"


def train_via_feed_relations(num_steps: int = 100, seed: Optional[int] = None):
    """
    feed_relation 多轮训练流程：
    1. 开始训练：载入模型参数 + relation_data（只发生一次）
    2. 多轮 random_feed：反复更新模型参数与学习率（不落盘）
    3. 训练结束：统一保存 relation_data + 模型参数（只落盘一次）
    """
    if seed is not None:
        random.seed(seed)

    begin_feed_training()

    if not rm.noun_list:
        raise ValueError("noun_list 为空：请先准备/落盘 relation_data.npz。")
    if not rm.relation_list:
        raise ValueError("relation_list 为空：请先准备/落盘 relation_data.npz。")

    for _ in range(num_steps):
        noun1 = random.choice(rm.noun_list)
        noun2 = random.choice(rm.noun_list)
        relation = random.choice(rm.relation_list)
        random_feed(noun1, noun2, relation)

    end_feed_training()


def predict_next_word(word: str, relation, top_k: int = 5):
    """
    基于训练好的模型做推断：给定 (word, relation)，预测下一个词的 top_k。

    参数:
    - word: 输入词（必须在 rm.noun_list 里）
    - relation: 关系名（字符串，必须在 rm.relation_list 里；或直接传 relation_type:int）
    - top_k: 返回 top-k 候选
    """
    loaded = rm.load_relation_data()
    # if loaded is False:
    #     raise FileNotFoundError("relation_data.npz 不存在，无法进行推断。")
    # rm.ensure_relation_defaults()

    if word not in rm.noun_list:
        raise ValueError(f"输入词 `{word}` 不在 rm.noun_list 中。")

    if isinstance(relation, str):
        if relation not in rm.relation_list:
            raise ValueError(f"关系 `{relation}` 不在 rm.relation_list 中。")
        relation_type = rm.relation_list.index(relation)
    elif isinstance(relation, int):
        relation_type = relation
    else:
        raise TypeError("relation 需要是字符串（关系名）或整数（relation_type）。")

    if not (1 <= relation_type <= 5):
        raise ValueError("relation_type 需要在 1..5 之间（0 是无关系占位，不可用）。")

    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"  # 这里保持 CPU，避免你环境 CUDA 不一致导致报错

    model = knowledge_map(rm.noun_dim, rm.noun_dim)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到已训练模型文件：{MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        i_idx = rm.noun_list.index(word)
        i_tensor = torch.tensor(i_idx, dtype=torch.long, device=device)
        # 预测目标 embedding：relation_linear(embedding[word])
        x = model.embedding(i_tensor)  # [noun_dim]
        y_pred = model.relations[int(relation_type) - 1](x)  # [noun_dim]

        top_indices, top_scores = model.query_similarity(y_pred, top_k=top_k)

    results = []
    # top_indices 可能落在 [0, noun_number) 之外 rm.noun_list 的长度，所以这里做兜底
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        if 0 <= idx < len(rm.noun_list):
            results.append((rm.noun_list[idx], float(score)))
        else:
            results.append((f"<unk_{idx}>", float(score)))

    return results


if __name__ == "__main__":
    predict_next_word("banana", "include")

