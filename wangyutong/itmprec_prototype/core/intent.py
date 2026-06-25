import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional
'''意图抽取'''

DEFAULT_SLOTS = {
    "liked_attributes": [],
    "disliked_attributes": [],
    "accepted_items": [],
    "rejected_items": [],
    "open_questions": [],
}


@dataclass
class TargetIntention:
    '''最终要引导用户去哪里'''
    description: str   #目标意图的文字描述
    keywords: List[str]  #关键词列表
    target_item: str     #目标物品名称
    embedding: List[float] = field(default_factory=list)#目标意图的向量表示

    def to_dict(self):
        return asdict(self)


@dataclass
class IntentState:
    '''当前意图状态'''
    summary: str
    slots: Dict[str, List[str]]
    embedding: List[float]
    confidence: float = 0.0
    turn_id: int = 0

    def to_dict(self):
        return asdict(self)


class HashingTextEmbedder:

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[A-Za-z0-9_']+", (text or "").lower())
        for token in tokens: #分词
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> float:
    if not a or not b:
        return 0.0
    av = list(a)
    bv = list(b)
    size = min(len(av), len(bv))
    if size == 0:
        return 0.0
    dot = sum(av[i] * bv[i] for i in range(size))
    an = math.sqrt(sum(av[i] * av[i] for i in range(size))) or 1.0
    bn = math.sqrt(sum(bv[i] * bv[i] for i in range(size))) or 1.0
    return dot / (an * bn)


class IntentExtractor:
    '''从 LLM 输出和用户反馈中自动提取关键词'''
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingTextEmbedder()

    def build_target_intention(self, target_item: str, question: str = "") -> TargetIntention:
        description = f"Guide the user toward the intention represented by target item: {target_item}."
        keywords = self._keywords(f"{target_item} {question}", limit=12)
        embedding = self.embedder.encode(f"{description} {' '.join(keywords)}")
        return TargetIntention(description, keywords, target_item, embedding)

    def extract_intent(self, dialogue_history, history_items=None, turn_id: int = 0) -> IntentState:
        history_items = history_items or []
        text = self._history_to_text(dialogue_history)
        liked = self._keywords(" ".join(history_items[-10:]) + " " + text, limit=10)
        slots = {key: list(value) for key, value in DEFAULT_SLOTS.items()}
        slots["liked_attributes"] = liked
        summary = self._summarize(liked, history_items)
        return IntentState(summary, slots, self.embedder.encode(summary + " " + text), 0.25, turn_id)

    def update_intent(self, state: IntentState, observation: Dict, action=None) -> IntentState:
        slots = {key: list(state.slots.get(key, [])) for key in DEFAULT_SLOTS}
        action_item = getattr(action, "grounded_item", None) or getattr(action, "argument", "")
        accepted = observation.get("accepted", False)
        rejected = observation.get("rejected", False)
        feedback = observation.get("feedback", "")

        if accepted and action_item:
            self._append_unique(slots["accepted_items"], action_item)
            self._append_unique(slots["liked_attributes"], action_item)
        if rejected and action_item:
            self._append_unique(slots["rejected_items"], action_item)
            self._append_unique(slots["disliked_attributes"], action_item)
        if getattr(action, "type", "") in {"ask", "clarify"}:
            self._append_unique(slots["open_questions"], getattr(action, "argument", ""))

        feedback_keywords = self._keywords(feedback, limit=4)
        for keyword in feedback_keywords:
            self._append_unique(slots["liked_attributes"], keyword)

        confidence_delta = 0.12 if accepted else 0.05 if feedback else -0.03
        confidence = min(1.0, max(0.0, state.confidence + confidence_delta))
        summary = self._summary_from_slots(slots)
        embedding = self.embedder.encode(summary + " " + feedback)
        return IntentState(summary, slots, embedding, confidence, state.turn_id + 1)

    @staticmethod
    def _history_to_text(dialogue_history) -> str:
        if isinstance(dialogue_history, str):
            return dialogue_history
        if isinstance(dialogue_history, list):
            return " ".join(map(str, dialogue_history))
        return str(dialogue_history or "")

    @staticmethod
    def _keywords(text: str, limit: int = 8) -> List[str]:
        stop = {"the", "and", "for", "with", "that", "this", "user", "item", "target", "toward"}
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_']+", (text or "").lower())
        seen = []
        for token in tokens:
            if token not in stop and token not in seen:
                seen.append(token)
            if len(seen) >= limit:
                break
        return seen

    @staticmethod
    def _append_unique(values: List[str], value: str):
        if value and value not in values:
            values.append(value)

    @staticmethod
    def _summarize(liked: List[str], history_items: List[str]) -> str:
        recent = ", ".join(history_items[-3:])
        keywords = ", ".join(liked[:5]) or "uncertain preferences"
        return f"User currently appears interested in {keywords}. Recent items: {recent}."

    @staticmethod
    def _summary_from_slots(slots: Dict[str, List[str]]) -> str:
        liked = ", ".join(slots.get("liked_attributes", [])[:6]) or "unknown"
        disliked = ", ".join(slots.get("disliked_attributes", [])[:4]) or "none"
        accepted = ", ".join(slots.get("accepted_items", [])[-3:]) or "none"
        return f"Liked signals: {liked}. Disliked signals: {disliked}. Accepted items: {accepted}."


class IntentTracker:
    '''每轮对话后更新 IntentState'''
    def __init__(self, extractor: Optional[IntentExtractor] = None):
        self.extractor = extractor or IntentExtractor()
        self.states = {}

    def initialize(self, idx, dialogue_history, history_items=None):
        state = self.extractor.extract_intent(dialogue_history, history_items, turn_id=0)
        self.states[idx] = state
        return state

    def update(self, idx, observation, action=None):
        state = self.states.get(idx)
        if state is None:
            state = self.extractor.extract_intent("", [], turn_id=0)
        state = self.extractor.update_intent(state, observation, action)
        self.states[idx] = state
        return state
