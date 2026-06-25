"""DeepSeek API验证，用于ITMPRec Agent。

运行方式：
    set DEEPSEEK_API_KEY=sk-xxx
    python test_deepseek_api.py

或者用命令行传key：
    python test_deepseek_api.py --api_key sk-xxx
"""

import sys, os
TPRA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'T-PRA-main'))
WORK_ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for r in (TPRA_ROOT, WORK_ROOT):
    if r not in sys.path:
        sys.path.insert(0, r)

import argparse
from itmprec_prototype.api.llm_api import DeepSeekInterface
from itmprec_prototype.api.prompts import build_actor_messages, build_critic_messages


def test_deepseek_api(api_key: str):
    """Test DeepSeek API with a simple actor + critic prompt."""
    print("=" * 60)
    print("DeepSeek API Validation for ITMPRec")
    print("=" * 60)

    # 创建接口
    llm = DeepSeekInterface(api_key=api_key, model="deepseek-v4-pro", temperature=0.7, max_tokens=512)
    print(f"[INFO] API base_url: {llm.base_url}")
    print(f"[INFO] Model: {llm.model}")
    print()

    # 测试1：Actor prompt
    print("[TEST 1] Actor prompt (recommendation agent)")
    actor_msgs = build_actor_messages(
        question="Recommend a game for a user who likes RPGs.",
        target_intention='{"target_item": "The Witcher 3", "description": "Open-world RPG"}',
        current_intent='{"summary": "User likes RPG games", "confidence": 0.6}',
        memory="User previously played Skyrim and Dark Souls.",
        scratchpad="Thought 1: The user enjoys challenging RPGs.\nAction 1: recommend[Dark Souls]\nObservation 1: User accepted.",
    )
    print("Prompt messages:")
    for msg in actor_msgs:
        print(f"  {msg['role']}: {msg['content'][:100]}...")
    print()

    print("Calling DeepSeek API...")
    responses = llm.generate(actor_msgs, n=1)
    if not responses or not responses[0]:
        print("[FAIL] API returned empty response. Check API key and connectivity.")
        return False

    print(f"[OK] Response:\n{responses[0]}")
    print()

    # 检查格式
    if "Thought" in responses[0] and "Action" in responses[0]:
        print("[PASS] Response contains Thought and Action.")
    else:
        print("[WARN] Response may not follow ReAct format. Consider prompt tuning.")
    print()

    # 测试2：Critic prompt
    print("[TEST 2] Critic prompt")
    critic_msgs = build_critic_messages(
        history_list="[Skyrim, Dark Souls]",
        current_intent="User likes RPG games",
        target_intention="The Witcher 3",
        recent_actions="[Dark Souls]",
    )
    print("Prompt messages:")
    for msg in critic_msgs:
        print(f"  {msg['role']}: {msg['content'][:100]}...")
    print()

    print("Calling DeepSeek API...")
    critic_responses = llm.generate(critic_msgs, n=1)
    if not critic_responses or not critic_responses[0]:
        print("[FAIL] Critic API returned empty response.")
        return False

    print(f"[OK] Response: {critic_responses[0]}")
    print()

    # 尝试解析为float
    try:
        score = float(critic_responses[0].strip().split()[0])
        print(f"[PASS] Parsed critic score: {score}")
    except ValueError:
        print("[WARN] Could not parse critic score as float. Consider prompt tuning.")
    print()

    print("=" * 60)
    print("[ALL PASSED] DeepSeek API is ready for ITMPRec.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key")
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] No API key provided. Please set DEEPSEEK_API_KEY environment variable or pass --api_key.")
        print("Example: set DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)

    success = test_deepseek_api(args.api_key)
    sys.exit(0 if success else 1)
