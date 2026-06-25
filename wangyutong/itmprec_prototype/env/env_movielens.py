"""轻量MovieLens环境，不需要transformers库"""
# MovieLens环境，读取数据矩阵，计算用户对物品的偏好和距离

import json
import os
import pickle
import numpy as np


class MovieLensENV:
    """兼容T-PRA SteamENV接口的最小环境"""

    def __init__(self, config, split):
        path = os.path.join(config.env_path, 'movielens/')

        with open(os.path.join(path, 'datamaps.json'), encoding='utf-8') as f:
            datamaps = json.load(f)
        self.item2id = datamaps['item2id_dict']
        self.id2item = datamaps['id2item_dict']

        self.reward_mat = np.load(os.path.join(path, f'movielens_{split}.npy'))
        print(f'reward_mat shape:{self.reward_mat.shape}')

        with open(os.path.join(path, f'{split}_distance_mat.pickle'), 'rb') as f:
            self.distance_mat = pickle.load(f)

        self.env_window_length = getattr(config, 'env_window_length', 4)
        self.threshold = getattr(config, 'env_threshold', 50)

    def get_reward(self, userid, item_list, target_item=None):
        if len(item_list) < 2:
            return 0.0, 0.0, 0.0

        item = item_list[-1]
        last_item = item_list[-2]

        if item in self.item2id and last_item in self.item2id:
            itemid = self.item2id[item]
            lastitemid = self.item2id[last_item]

            user_pre = self.reward_mat[userid, itemid] / 5.0

            base = max(self.distance_mat[itemid])
            step_len = self.distance_mat[itemid, lastitemid] / base if base > 0 else 0.0
            reward_step_len = 1.0 - step_len

            forward_len = 0.0
            if target_item is not None and target_item in self.item2id:
                targetid = self.item2id[target_item]
                base_target = max(self.distance_mat[targetid])
                forward_len = (self.distance_mat[lastitemid, targetid] - self.distance_mat[itemid, targetid]) / base_target if base_target > 0 else 0.0

            return user_pre, reward_step_len, forward_len
        else:
            return 0.0, 0.0, 0.0

    def get_dist(self, item1, item2):
        if item1 in self.item2id and item2 in self.item2id:
            itemid1 = self.item2id[item1]
            itemid2 = self.item2id[item2]
            return self.distance_mat[itemid1, itemid2]
        return 0

    def get_item_list(self):
        return self.item2id.keys()


def get_envs(name, config, split):
    if name == 'movielens':
        return MovieLensENV(config, split)
    raise ValueError(f"MovieLens env only supports 'movielens', got {name}")
