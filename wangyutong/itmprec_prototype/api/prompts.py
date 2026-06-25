"""给LLM发消息用的prompt模板，兼容OpenAI格式"""
# prompt模板，给LLM发消息时用的格式，包括actor和critic两种


from typing import List, Dict, Any


# 基础字符串prompt（兼容本地模型和T-PRA）
itmp_react_agent_prompt = (
    "You are an intention-aware proactive recommendation agent. "
    "Keep the T-PRA ReAct format, but use explicit user intent and proactive dialogue actions.\n\n"
    "Available actions:\n"
    "(1) recommend[item] - recommend one catalog item.\n"
    "(2) ask[question] - ask a preference elicitation question.\n"
    "(3) clarify[question] - clarify an ambiguous user need.\n"
    "(4) explain[item] - explain why an item can bridge current intent to the target intention.\n\n"
    "Task:\n{question}\n\n"
    "Target Intention:\n{target_intention}\n\n"
    "Current Intent:\n{current_intent}\n\n"
    "Retrieved Memory:\n{memory}\n\n"
    "Previous trajectory:\n{scratchpad}\n\n"
    "Please respond with a Thought and an Action in the format:\n"
    "Thought n: ...\nAction n: ..."
)


def build_actor_messages(
    question: str,
    target_intention: str,
    current_intent: str,
    memory: str,
    scratchpad: str,
) -> List[Dict[str, str]]:
    """构造actor的OpenAI格式消息"""
    system_msg = (
        "You are an intention-aware proactive recommendation agent. "
        "Keep the T-PRA ReAct format, but use explicit user intent and proactive dialogue actions.\n\n"
        "Available actions:\n"
        "(1) recommend[item] - recommend one catalog item.\n"
        "(2) ask[question] - ask a preference elicitation question.\n"
        "(3) clarify[question] - clarify an ambiguous user need.\n"
        "(4) explain[item] - explain why an item can bridge current intent to the target intention.\n\n"
        "Use ask or clarify when intent confidence is low. "
        "Use recommend when you have enough intent evidence. "
        "Use explain when a recommendation may need justification. "
        "Do not repeat recommended items."
    )

    user_msg = (
        f"Task:\n{question}\n\n"
        f"Target Intention:\n{target_intention}\n\n"
        f"Current Intent:\n{current_intent}\n\n"
        f"Retrieved Memory:\n{memory}\n\n"
        f"Previous trajectory:\n{scratchpad}\n\n"
        "Please respond with a Thought and an Action in the format:\n"
        "Thought n: ...\nAction n: ..."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_critic_messages(
    history_list: str,
    current_intent: str,
    target_intention: str,
    recent_actions: str,
) -> List[Dict[str, str]]:
    """构造critic的OpenAI格式消息"""
    system_msg = (
        "You are a critic for proactive recommendation. "
        "Score the state from -5 to 5 by considering user preference, "
        "target-intention progress, dialogue quality, and future potential. "
        "Only reply with one number."
    )

    user_msg = (
        f"History: {history_list}\n"
        f"Current intent: {current_intent}\n"
        f"Target intention: {target_intention}\n"
        f"Recent actions: {recent_actions}\n"
        "Critic: "
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
