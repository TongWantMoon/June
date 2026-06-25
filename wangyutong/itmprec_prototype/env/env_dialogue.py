# 对话环境的封装，把recommend、ask、clarify、explain四种动作映射成奖励

class DialogueEnvWrapper:
    """T-PRA表格环境的自然语言封装层"""

    def __init__(self, base_env):
        self.base_env = base_env

    def step(self, idx, action, task, history_items, intent_state, target_intention):
        if action.type == "recommend":
            return self._recommend(idx, action, task, history_items, target_intention)
        if action.type == "explain":
            return self._explain(action, intent_state)
        if action.type == "ask":
            return self._ask(action, intent_state)
        if action.type == "clarify":
            return self._clarify(action, intent_state)
        return {
            "feedback": "Invalid action. Please use recommend, ask, clarify, or explain.",
            "accepted": False,
            "rejected": True,
            "recommendation_reward": 0.0,
            "dialogue_quality": 0.0,
            "raw_rewards": [0.0, 0.0, 0.0],
        }

    def _recommend(self, idx, action, task, history_items, target_intention):
        item = action.grounded_item or action.argument
        item_list = list(history_items) + [item]
        if len(item_list) < 2:
            item_list = ["__dummy_prev__"] + item_list
        try:
            user_pre, step_len, forward_len = self.base_env.get_reward(
                task.get_userid(idx),
                item_list,
                target_intention.target_item,
            )
        except Exception:
            user_pre, step_len, forward_len = 0.0, 0.0, 0.0
        accepted = user_pre >= 0.5
        rejected = user_pre < 0.2
        feedback = (
            f"The user {'accepts' if accepted else 'does not fully accept'} recommendation {item}. "
            f"preference={user_pre:.2f}, step={step_len:.2f}, progress={forward_len:.2f}."
        )
        return {
            "feedback": feedback,
            "accepted": accepted,
            "rejected": rejected,
            "recommendation_reward": max(0.0, float(user_pre)),
            "dialogue_quality": 0.4 + 0.4 * max(0.0, float(step_len)),
            "target_progress_signal": float(forward_len),
            "raw_rewards": [float(user_pre), float(step_len), float(forward_len)],
            "item": item,
        }

    @staticmethod
    def _ask(action, intent_state):
        return {
            "feedback": f"The user provides a preference clue after being asked: {action.argument}",
            "accepted": False,
            "rejected": False,
            "recommendation_reward": 0.0,
            "dialogue_quality": 0.8 if intent_state.confidence < 0.7 else 0.45,
            "raw_rewards": [0.0, 0.0, 0.0],
        }

    @staticmethod
    def _clarify(action, intent_state):
        return {
            "feedback": f"The user clarifies their current need: {action.argument}",
            "accepted": False,
            "rejected": False,
            "recommendation_reward": 0.0,
            "dialogue_quality": 0.75 if intent_state.confidence < 0.8 else 0.4,
            "raw_rewards": [0.0, 0.0, 0.0],
        }

    @staticmethod
    def _explain(action, intent_state):
        item = action.grounded_item or action.argument
        return {
            "feedback": f"The system explains why {item} matches the current intent: {intent_state.summary}",
            "accepted": False,
            "rejected": False,
            "recommendation_reward": 0.1,
            "dialogue_quality": 0.7,
            "raw_rewards": [0.0, 0.0, 0.0],
            "item": item,
        }
