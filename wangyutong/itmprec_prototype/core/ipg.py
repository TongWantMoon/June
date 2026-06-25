from dataclasses import dataclass
from typing import Iterable, List, Optional

from .intent import HashingTextEmbedder, cosine
'''在 Grounding 之后，对候选物品进行牵引式重排,确保推荐的物品不仅和用户喜好相关，还向目标意图靠近。'''

@dataclass
class IPGScore:
    item: str
    preference_score: float  #候选物品和用户偏好的相似度
    guidance_score: float
    total_score: float


class IPGReranker:
    """输入：grounding 后的种子物品、已排除物品、意图状态、目标意图   输出：重排后的最优物品"""

    def __init__(self, base_env=None, grounding_model=None, embedder=None, topk: int = 5):
        self.base_env = base_env
        self.grounding_model = grounding_model
        self.embedder = embedder or HashingTextEmbedder()
        self.topk = topk

    def rerank_from_seed(self, seed_item: str, exclude_items=None, intent_state=None, target_intention=None):
        candidates = self._candidate_items(seed_item, exclude_items or [])
        if not candidates:
            return seed_item
        scored = self.score_items(candidates, intent_state, target_intention)
        return scored[0].item if scored else seed_item

    def score_items(self, items: Iterable[str], intent_state=None, target_intention=None, userid=None) -> List[IPGScore]:
        current_embedding = getattr(intent_state, "embedding", None)
        target_embedding = getattr(target_intention, "embedding", None)
        current_target = cosine(current_embedding, target_embedding)
        scored = []
        for item in items:
            item_embedding = self.embedder.encode(str(item))
            preference = self._preference_score(item, userid)
            guidance = cosine(item_embedding, target_embedding) - current_target
            total = preference * guidance
            scored.append(IPGScore(item, preference, guidance, total))
        return sorted(scored, key=lambda row: row.total_score, reverse=True)

    def _candidate_items(self, seed_item: str, exclude_items: List[str]) -> List[str]:
        if self.grounding_model is not None:
            try:
                candidates = self.grounding_model.get_topk_near_item([seed_item], self.topk)[0]
                return [item for item in candidates if item not in exclude_items]
            except Exception:
                pass
        if self.base_env is not None:
            try:
                items = list(self.base_env.get_item_list())
                return [item for item in items if item not in exclude_items][: self.topk]
            except Exception:
                pass
        return [seed_item] if seed_item not in exclude_items else []

    def _preference_score(self, item: str, userid=None) -> float:
        if self.base_env is not None and userid is not None:
            try:
                item_list = ["__dummy_previous__", item]
                return float(self.base_env.get_reward(userid, item_list, item)[0])
            except Exception:
                pass
        return 1.0
