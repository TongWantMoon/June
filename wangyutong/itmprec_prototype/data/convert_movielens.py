# 把MovieLens 100K原始数据转成T-PRA能用的格式，包括json、npy、pickle

import json
import os
import pickle
import random
import numpy as np
from collections import defaultdict


DATA_ROOT = r"E:\SCHOOL!\考核\work\ml-100k\ml-100k"
OUT_ROOT = r"E:\SCHOOL!\考核\T-PRA-main\env\movielens"


def load_ratings(path):
    ratings = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                ratings.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
    return ratings


def load_items(path):
    items = {}
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            title = parts[1]
            items[item_id] = title
    return items


def build_datamaps(items):
    item2id = {}
    id2item = {}
    for idx, (raw_id, title) in enumerate(sorted(items.items()), start=0):
        item2id[title] = idx
        id2item[str(idx)] = title
    return {"item2id_dict": item2id, "id2item_dict": id2item}


def build_reward_matrix(ratings, num_users, num_items, item2id):
    mat = np.zeros((num_users, num_items), dtype=np.float32)
    for user_id, raw_item_id, rating, _ in ratings:
        item_title = items[raw_item_id]
        item_idx = item2id[item_title]
        mat[user_id - 1, item_idx] = rating  # user_id is 1-based
    return mat


def build_distance_matrix(reward_mat):
    # 用余弦相似度计算用户评分向量
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(reward_mat.T)  # item x item
    # 相似度转距离
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def build_sequences(ratings, item2id, all_items, min_len=5):
    user_items = defaultdict(list)
    for user_id, raw_item_id, rating, timestamp in ratings:
        user_items[user_id].append((timestamp, raw_item_id, rating))

    sequences = []
    for user_id in sorted(user_items.keys()):
        sorted_items = sorted(user_items[user_id], key=lambda x: x[0])
        pos_seq = [items[raw_id] for _, raw_id, rating in sorted_items if rating >= 3]
        if len(pos_seq) >= min_len:
            # 从用户没互动过的物品里随机选目标
            interacted = set(items[raw_id] for _, raw_id, _ in sorted_items)
            candidates = [item for item in all_items if item not in interacted]
            target_item = random.choice(candidates) if candidates else "Unknown"
            sequences.append({
                "userid_encoded": user_id - 1,
                "pos_seq_name": pos_seq,
                "target_item": target_item,
            })
    return sequences


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 加载物品数据
    items = load_items(os.path.join(DATA_ROOT, "u.item"))
    print(f"Loaded {len(items)} items")

    # 构建映射表
    datamaps = build_datamaps(items)
    with open(os.path.join(OUT_ROOT, "datamaps.json"), 'w') as f:
        json.dump(datamaps, f)
    print(f"Saved datamaps.json")

    item2id = datamaps["item2id_dict"]
    num_users = 943
    num_items = len(items)
    all_item_titles = list(item2id.keys())

    # 训练集
    train_ratings = load_ratings(os.path.join(DATA_ROOT, "u1.base"))
    train_mat = build_reward_matrix(train_ratings, num_users, num_items, item2id)
    np.save(os.path.join(OUT_ROOT, "movielens_train.npy"), train_mat)
    print(f"Saved movielens_train.npy shape={train_mat.shape}")

    train_dist = build_distance_matrix(train_mat)
    with open(os.path.join(OUT_ROOT, "train_distance_mat.pickle"), 'wb') as f:
        pickle.dump(train_dist, f)
    print(f"Saved train_distance_mat.pickle shape={train_dist.shape}")

    train_seqs = build_sequences(train_ratings, item2id, all_item_titles)
    with open(os.path.join(OUT_ROOT, "new_train_ran.json"), 'w') as f:
        json.dump(train_seqs, f)
    print(f"Saved new_train_ran.json with {len(train_seqs)} users")

    # 测试集
    test_ratings = load_ratings(os.path.join(DATA_ROOT, "u1.test"))
    test_mat = build_reward_matrix(test_ratings, num_users, num_items, item2id)
    np.save(os.path.join(OUT_ROOT, "movielens_test.npy"), test_mat)
    print(f"Saved movielens_test.npy shape={test_mat.shape}")

    test_dist = build_distance_matrix(test_mat)
    with open(os.path.join(OUT_ROOT, "test_distance_mat.pickle"), 'wb') as f:
        pickle.dump(test_dist, f)
    print(f"Saved test_distance_mat.pickle shape={test_dist.shape}")

    test_seqs = build_sequences(test_ratings, item2id, all_item_titles)
    with open(os.path.join(OUT_ROOT, "new_test_ran.json"), 'w') as f:
        json.dump(test_seqs, f)
    print(f"Saved new_test_ran.json with {len(test_seqs)} users")

    print("\nAll MovieLens conversion done!")
