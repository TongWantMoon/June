from dataclasses import dataclass
'''配置中心'''

@dataclass
class ITMPRecConfig:
    enable_intent: bool = True #启用意图管理	系统会记录用户喜好
    enable_dialogue_actions: bool = True#是否启用主动对话动作（ask/clarify/explain）
    enable_ipg: bool = True #是否启用IPG牵引式重排，开启会向目标靠近
    enable_trajectory_dpo: bool = True #是否启用轨迹DPO训练，开启会记录完整对话轨迹用于训练，否则只记录单步
    ipg_topk: int = 5 #重排时考虑的近邻数量
    memory_topk: int = 3#记忆检索时找几个最相似用户
    #多目标奖励的四个权重
    alpha_recommendation: float = 0.4  #推荐质量
    beta_intention_alignment: float = 0.2 #意图对齐
    gamma_dialogue_quality: float = 0.2 #对话质量
    delta_target_progress: float = 0.2 #目标进度

    #API参数
    use_api: bool = True
    api_key: str = ""  #我在命令行运行
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.7
    max_tokens: int = 512

    @classmethod
    def from_args(cls, args):
        return cls(
            enable_intent=getattr(args, "enable_intent", False),
            enable_dialogue_actions=getattr(args, "enable_dialogue_actions", False),
            enable_ipg=getattr(args, "enable_ipg", False),
            enable_trajectory_dpo=getattr(args, "enable_trajectory_dpo", False),
            ipg_topk=getattr(args, "ipg_topk", 5),
            memory_topk=getattr(args, "memory_topk", 3),
            alpha_recommendation=getattr(args, "reward_alpha", 0.4),
            beta_intention_alignment=getattr(args, "reward_beta", 0.2),
            gamma_dialogue_quality=getattr(args, "reward_gamma", 0.2),
            delta_target_progress=getattr(args, "reward_delta", 0.2),
            use_api=getattr(args, "use_api", True),
            api_key=getattr(args, "api_key", ""),
            base_url=getattr(args, "base_url", ""),
            model=getattr(args, "model", "gpt-4o-mini"),
            temperature=getattr(args, "temperature", 0.7),
            max_tokens=getattr(args, "max_tokens", 512),
        )
