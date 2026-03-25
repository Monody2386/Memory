noun_number = 500
noun_dim = 50
relation_num = 10
import numpy as np
import os
relation_map = np.zeros((noun_number, noun_number), dtype=np.int64)
noun_list = []
relation_list = []
lr_relation = np.ones(relation_num)
lr_per_embedding = np.ones(noun_number)

noun_list = ["apple", "banana", "fruit", "cat", "dog", "car", "tree", "house", "book", "phone", "computer", "person", "city",
              "country", "continent", "planet", "star", "galaxy", "universe", "water", "fire", "air", "earth", "light", 
              "darkness", "time", "space", "animal",]
relation_list = ["include", "belong to"]
def add_relation(noun1, noun2, relation):
    if noun1 not in noun_list:
        noun_list.append(noun1)
    if noun2 not in noun_list:
        noun_list.append(noun2)
    i_idx = noun_list.index(noun1)
    j_idx = noun_list.index(noun2)
    relation_type = relation_list.index(relation) + 1
    if relation_map[i_idx, j_idx] == 0:
        relation_map[i_idx, j_idx] = relation_type
        return True, i_idx, j_idx, relation_type
    if relation_map[i_idx, j_idx] != relation_type:
        print(f"Relation already exists between {noun1} and {noun2} with different relation type.")
        return False, i_idx, j_idx, relation_map[i_idx, j_idx]
    return True, i_idx, j_idx, relation_type


def save_relation_data(file_path="relation_data.npz"):
    """
    保存关系图数据到 npz 文件。
    包含：relation_map、noun_list、relation_list
    """
    np.savez(
        file_path,
        relation_map=relation_map,
        noun_list=np.array(noun_list, dtype=object),
        relation_list=np.array(relation_list, dtype=object),
        lr_per_embedding=lr_per_embedding,
        lr_relation=lr_relation,
    )


def load_relation_data(file_path="relation_data.npz"):
    """

    """
    global relation_map, noun_list, relation_list, lr_per_embedding, lr_relation

    if not os.path.exists(file_path):
        return False

    data = np.load(file_path, allow_pickle=True)
    # 兼容旧版 relation_data.npz：历史上 relation_map 可能是 float64，这里强制转为 int，避免当作 list 下标报错。
    relation_map = data["relation_map"].astype(np.int64, copy=False)
    noun_list = data["noun_list"].tolist()
    relation_list = data["relation_list"].tolist()
    lr_per_embedding = data["lr_per_embedding"]
    lr_relation = data["lr_relation"]

    return relation_map, noun_list, relation_list, lr_per_embedding, lr_relation




if __name__ == "__main__":
    # 仅在直接运行本文件时才落盘默认数据
    save_relation_data()

