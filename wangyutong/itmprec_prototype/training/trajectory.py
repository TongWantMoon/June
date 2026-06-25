# 轨迹管理，记录对话的每一步，包括思考、动作、观察、奖励、意图

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

@dataclass
class TrajectoryStep:
    '''记录单步的信息'''
# thought: str  #LLM 的思考内容
    action: Dict  #执行的动作
    observation: Dict  #环境反馈
    reward_dict: Dict  #奖励的四个分量
    intent_state: Dict #这一步的意图状态
    critic_value: float = 0.0  #Critic 对这一步的评价


    def to_dict(self):
        return asdict(self)


class TrajectoryBuffer:

    def __init__(self):
        # idx映射到候选轨迹列表，每个轨迹是TrajectoryStep列表
        self.trajectories: Dict[str, List[List[TrajectoryStep]]] = {}

    def add_step(self, idx, step: TrajectoryStep, candidate_idx: int = 0):
        idx_str = str(idx)
        if idx_str not in self.trajectories:
            self.trajectories[idx_str] = []
        if len(self.trajectories[idx_str]) <= candidate_idx:
            self.trajectories[idx_str].append([])
        self.trajectories[idx_str][candidate_idx].append(step)

    def cumulative_reward(self, trajectory: List[TrajectoryStep]) -> float:
        return sum(step.reward_dict.get("total", 0.0) for step in trajectory)

    def to_dict(self):
        return {
            idx: [[step.to_dict() for step in traj] for traj in trajs]
            for idx, trajs in self.trajectories.items()
        }

    def build_dpo_data(self):
        rows = []
        for idx_str, trajectories in self.trajectories.items():
            if len(trajectories) < 2:
                continue
            rewards = [self.cumulative_reward(traj) for traj in trajectories]
            chosen_idx = rewards.index(max(rewards))
            rejected_idx = rewards.index(min(rewards))
            if chosen_idx == rejected_idx:
                continue
            chosen = trajectories[chosen_idx]
            rejected = trajectories[rejected_idx]
            rows.append(
                {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "Prefer the trajectory that better guides user intent while preserving recommendation quality.",
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": json.dumps(self._compact_traj(chosen), ensure_ascii=False),
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": json.dumps(self._compact_traj(rejected), ensure_ascii=False),
                    },
                }
            )
        return rows

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "trajectory_buffer.json"), "w", encoding="utf-8") as fout:
            json.dump(self.to_dict(), fout, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "trajectory_dpo_data.json"), "w", encoding="utf-8") as fout:
            json.dump(self.build_dpo_data(), fout, indent=2, ensure_ascii=False)

    def _compact_traj(self, trajectory: List[TrajectoryStep]):
        return [
            {
                "thought": step.thought,
                "action": step.action,
                "reward": step.reward_dict.get("total", 0.0),
                "intent": step.intent_state.get("summary", ""),
            }
            for step in trajectory
        ]
