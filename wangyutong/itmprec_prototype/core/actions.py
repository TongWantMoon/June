import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional
'''把 LLM 的文本变成结构化的数据'''
VALID_ACTIONS = {"recommend", "ask", "clarify", "explain"}

'''每个动作被解析后，会变成一个ActionSchema对象'''
@dataclass
class ActionSchema:
    type: str    #动作类型（recommend/ask/clarify/explain）
    argument: str  #动作参数（比如推荐物品名称、问的问题内容）
    raw_text: str  #LLM原始输出的文本
    valid: bool   #是否解析成功
    grounded_item: Optional[str] = None  #经过Grounding后的真实物品

    def to_dict(self):
        return asdict(self)


class ActionParser:
    """用正则表达式从LLM输出中提取动作变成结构化数据"""

    _pattern = re.compile(r"^\s*([A-Za-z_]+)\s*\[(.*)\]\s*$")

    def parse_one(self, text: str) -> ActionSchema:
        raw = "" if text is None else str(text).strip()  # 1. 清理文本
        match = self._pattern.match(raw)  # 2. 尝试匹配
        if not match:
            return self._fallback(raw)  # 3. 匹配失败

        action_type = match.group(1).strip().lower()  # 4. 提取动作名
        argument = self._clean_argument(match.group(2))  # 5. 提取参数
        if action_type not in VALID_ACTIONS or not argument:  # 6. 检查是否有效
            return self._fallback(raw)
        return ActionSchema(action_type, argument, raw, True)  # 7. 返回成功

    def parse_many(self, texts: Iterable[str]) -> List[ActionSchema]:
        return [self.parse_one(text) for text in texts]

    @staticmethod
    def _clean_argument(argument: str) -> str: #去掉首尾引号
        argument = argument.strip()
        if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in {"'", '"'}:
            argument = argument[1:-1]
        return argument.strip()

    @staticmethod
    def _fallback(raw: str) -> ActionSchema:#把无效输出变成继续对话提问
        return ActionSchema(
            type="clarify",
            argument="Please specify your preference.",
            raw_text=raw,
            valid=False,
        )


class ActionExecutor:
    """根据动作类型，决定是否需要Grounding和IPG重排"""

    def __init__(self, grounding_model=None, ipg_reranker=None):
        self.grounding_model = grounding_model
        self.ipg_reranker = ipg_reranker

    def execute(self, action, idx, task, history_items, env_wrapper, intent_state, target_intention):
        grounded_action = self.ground_action(action, history_items, intent_state, target_intention)
        return env_wrapper.step(idx, grounded_action, task, history_items, intent_state, target_intention)

    def ground_action(self, action, history_items, intent_state=None, target_intention=None):
        if action.type not in {"recommend", "explain"}:#只有recommend和explain需要Grounding
            return action

        exclude = list(history_items) # 已经推荐过的物品，避免重复推荐
        grounded = action.argument
        if self.grounding_model is not None:
            try:
                grounded = self.grounding_model.get_top_near_item(action.argument, exclude)#找最相似的物品
            except AttributeError:
                grounded = self.grounding_model.get_topk_near_item([action.argument], len(exclude) + 1)[0][0]
            except Exception:
                grounded = action.argument

        if action.type == "recommend" and self.ipg_reranker is not None:
            grounded = self.ipg_reranker.rerank_from_seed(
                grounded,
                exclude_items=exclude,
                intent_state=intent_state,
                target_intention=target_intention,
            )
        action.grounded_item = grounded
        return action
