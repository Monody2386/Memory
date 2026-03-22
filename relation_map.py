noun_number = 500
noun_dim = 50
relation_num = 10
import numpy as np
relation_map = np.zeros((noun_number, noun_number))
noun_list = []
embedding_list = []
relation_list = []
relation_embedding_list = []
lr_noun = np.zeros(noun_number)
lr_relation = np.zeros(relation_num)

