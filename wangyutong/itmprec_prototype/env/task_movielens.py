"""兼容T-PRA BaseTask的MovieLens任务适配器"""
# 任务适配器，把用户历史数据包装成Task格式，让Agent能读取

import json
import os


class MovieLensTask:
    """MovieLens 100K数据的任务包装器"""

    def __init__(self, split='train'):
        # 先尝试从当前文件往上找到 T-PRA-main
        current = os.path.dirname(os.path.abspath(__file__))
        while True:
            candidate = os.path.join(current, 'T-PRA-main', 'env', 'movielens')
            if os.path.exists(candidate):
                data_dir = candidate
                break
            parent = os.path.dirname(current)
            if parent == current:
                # 兜底路径
                data_dir = os.path.join(os.path.dirname(current), 'T-PRA-main', 'env', 'movielens')
                break
            current = parent
        data_dir = os.path.normpath(data_dir)
        print(f"[task_movielens] data_dir={data_dir}")

        with open(os.path.join(data_dir, f'new_{split}_ran.json'), encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        """把用户历史返回成字符串，作为问题给Agent"""
        entry = self.data[idx]
        history = entry.get('pos_seq_name', [])
        history_str = ', '.join(history[:10]) if history else 'No history'
        target = entry.get('target_item', 'Unknown')
        return f"User history: {history_str}. Target item: {target}."

    def __len__(self):
        return len(self.data)

    def get_userid(self, idx):
        return self.data[idx].get('userid_encoded', idx)

    def get_target_item(self, idx):
        return self.data[idx].get('target_item', 'Unknown')

    def get_history_actions(self, idx):
        return self.data[idx].get('pos_seq_name', [])

    def evaluate(self, idx, answer):
        target = self.get_target_item(idx)
        return 1.0 if target in answer else 0.0


def get_task(name, split, suffix):
    if name == 'movielens':
        return MovieLensTask(split)
    raise ValueError(f"MovieLens task only supports 'movielens', got {name}")
