import numpy as np
import pandas as pd

train = np.load("/Users/hsienhaochen/Documents/WebSearch&Mining/Project_3/data/train.npy")

name_list = ["item_id", "user_id", "region", "city", "parent_category_name", "category_name", "param_1", "param_2", "param_3", "titie", "description", "price", "item_seq_number", "activation_date", "user_type", "image", "image_top_1", "deal_probability"]

for i in range(len(name_list)):
    print(name_list[i], train[0, i])
