from dataclasses import asdict, dataclass

from .intent import cosine
'''奖励计算器'''

# @dataclass
class RewardBreakdown:
    recommendation: float   #推荐质量,用户对这个推荐物品的偏好程度,env_dialogue.py 的 _recommend() 返回的 user_pre
    intention_alignment: float #意图对齐,推荐物品是否符合用户明确表达过的喜好,计算当前推荐物品与 intent_state.liked_attributes 的关键词重叠率
    dialogue_quality: float #主动对话动作的质量（ask 和 clarify 的得分取决于当前 confidence）,env_dialogue.py 的 _ask()/_clarify()/_explain() 返回
    target_progress: float #推荐物品是否让用户更接近目标物品,env_dialogue.py 的 _recommend() 返回的 forward_len
    total: float

    def to_dict(self):
        return asdict(self)


class MultiObjectiveReward:
    def __init__(self, alpha=0.4, beta=0.2, gamma=0.2, delta=0.2):#默认权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute(self, action, observation, previous_intent, current_intent, target_intention):
        recommendation = float(observation.get("recommendation_reward", 0.0))
        dialogue_quality = float(observation.get("dialogue_quality", 0.0))
        intention_alignment = max(0.0, cosine(current_intent.embedding, target_intention.embedding)) #现在的对齐度
        previous_alignment = max(0.0, cosine(previous_intent.embedding, target_intention.embedding)) #上一步的对齐度
        target_progress = intention_alignment - previous_alignment

        total = (
            #总奖励
            self.alpha * recommendation
            + self.beta * intention_alignment
            + self.gamma * dialogue_quality
            + self.delta * target_progress
        )
        return RewardBreakdown(recommendation, intention_alignment, dialogue_quality, target_progress, total)
