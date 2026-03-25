import os
import torch
import relation_map as rm
from train_relation_map import knowledge_map, train_average, train_random

knowledge_map_one = knowledge_map(rm.noun_dim, rm.noun_dim)
MODEL_PATH = "knowledge_map_one.pt"

# 仅用于“feed_relation 多轮训练”场景：载入一次，训练多轮，最后再统一保存。
_FEED_TRAIN_READY = False


def begin_feed_training():
    """
    开始 feed_relation 驱动的多轮训练：
    - 载入模型参数（如果存在）
    - 载入 relation_map / noun_list / relation_list / 学习率（只做一次）
    """
    global _FEED_TRAIN_READY

    if os.path.exists(MODEL_PATH):
        knowledge_map_one.load_state_dict(torch.load(MODEL_PATH))

    loaded = rm.load_relation_data()
    if loaded is False:
        raise FileNotFoundError("relation_data.npz 不存在，无法训练。请先生成并落盘 relation_data。")

    _FEED_TRAIN_READY = True


def end_feed_training():
    """结束 feed_relation 多轮训练：统一保存 relation_data + 模型参数。"""
    global _FEED_TRAIN_READY
    if not _FEED_TRAIN_READY:
        return
    rm.save_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)
    _FEED_TRAIN_READY = False


def run_long_training_and_save():
    if os.path.exists(MODEL_PATH):
        knowledge_map_one.load_state_dict(torch.load(MODEL_PATH))
    relation_map, noun_list_, relation_list_, lr_per_embedding_, lr_relation_ = rm.load_relation_data()
    # train_average 内部会对参与过的 lr 做持久衰减
    train_average(knowledge_map_one, relation_map, lr_per_embedding_, lr_relation_)
    rm.save_relation_data()
    torch.save(knowledge_map_one.state_dict(), MODEL_PATH)

def run_short_training_and_save(relation_learn, save=True):
    # 如果外部已经 begin_feed_training()，就不要重复载入，避免把训练进度重置。
    if not _FEED_TRAIN_READY:
        if os.path.exists(MODEL_PATH):
            knowledge_map_one.load_state_dict(torch.load(MODEL_PATH))
        loaded = rm.load_relation_data()
        if loaded is False:
            raise FileNotFoundError("relation_data.npz 不存在，无法训练。请先生成并落盘 relation_data。")
    # relation_learn: [(i_idx, j_idx, relation_type), ...]
    for (i_idx, j_idx, relation_type) in relation_learn:
        train_random(knowledge_map_one, i_idx, j_idx, relation_type, rm.lr_per_embedding, rm.lr_relation)
    if save:
        rm.save_relation_data()
        torch.save(knowledge_map_one.state_dict(), MODEL_PATH)

def random_feed(noun1, noun2, relation):
    [bool, i_idx, j_idx, relation_type] = rm.add_relation(noun1, noun2, relation)
    # 关系类型在“已存在但类型不同”分支会从 relation_map 读出 numpy.float64，这里统一转成 int。
    relation_type = int(relation_type)
    if bool:
        run_short_training_and_save([(i_idx, j_idx, relation_type)], save=False)
        print(noun1, noun2, relation)
    else:
        print(relation_type)
        print(noun1, noun2, rm.relation_list[relation_type - 1])
