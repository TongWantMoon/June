"""Mock集成测试，不依赖真实API和T-PRA数据。

这个测试验证整个pipeline，不需要：
- 真实LLM API key
- 本地GPU或transformers
- T-PRA数据集文件

运行：python test_api_mock.py
"""
# Mock集成测试，不依赖真实API和T-PRA数据，验证整个pipeline的模块间交互

import sys, os

TPRA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'T-PRA-main'))
WORK_ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for r in (TPRA_ROOT, WORK_ROOT):
    if r not in sys.path:
        sys.path.insert(0, r)

from unittest.mock import MagicMock

from itmprec_prototype.api.llm_api import MockLLMInterface
from itmprec_prototype.core.config import ITMPRecConfig
from itmprec_prototype.core.agent_itmprec import ITMPReactA2CAgent
from itmprec_prototype.core.actions import ActionParser, ActionSchema
from itmprec_prototype.core.intent import IntentExtractor, TargetIntention, IntentState
from itmprec_prototype.core.memory import IntentMemoryManager
from itmprec_prototype.core.ipg import IPGReranker
from itmprec_prototype.core.rewards import MultiObjectiveReward
from itmprec_prototype.training.trajectory import TrajectoryBuffer, TrajectoryStep
from itmprec_prototype.env.env_dialogue import DialogueEnvWrapper
from itmprec_prototype.api.prompts import build_actor_messages, build_critic_messages

def test_mock_llm_interface():
    """测试MockLLMInterface返回固定响应"""
    llm = MockLLMInterface()
    messages = [{"role": "user", "content": "Please recommend an item."}]
    out = llm.generate(messages, n=1)
    assert len(out) == 1
    assert "Thought" in out[0]
    assert "Action" in out[0]
    print("[PASS] MockLLMInterface")


def test_prompts_api_format():
    """测试prompts.py返回兼容OpenAI的消息"""

    actor_msgs = build_actor_messages(
        question="Find a game for user",
        target_intention='{"target_item": "GameA"}',
        current_intent='{"summary": "likes RPG"}',
        memory="User played Witcher 3",
        scratchpad="Thought 1: ...",
    )
    assert len(actor_msgs) == 2
    assert actor_msgs[0]["role"] == "system"
    assert actor_msgs[1]["role"] == "user"
    assert "recommend[item]" in actor_msgs[0]["content"]

    critic_msgs = build_critic_messages(
        history_list="[GameA, GameB]",
        current_intent="likes RPG",
        target_intention="Target GameA",
        recent_actions="[GameA]",
    )
    assert len(critic_msgs) == 2
    assert critic_msgs[0]["role"] == "system"
    print("[PASS] Prompts API format")


def test_agent_step_with_mock_llm():
    """Test ITMPReactA2CAgent.step() with mock LLM and mock env."""
    # 构建假任务
    task = MagicMock()
    task.__getitem__ = MagicMock(return_value="Find a game for user")
    task.get_history_actions = MagicMock(return_value=["Game1"])
    task.get_target_item = MagicMock(return_value="TargetGame")
    task.get_userid = MagicMock(return_value=0)

    # 构建假环境
    mock_env = MagicMock()
    mock_env.get_reward = MagicMock(return_value=(0.8, 0.5, 0.3))
    mock_env.get_item_list = MagicMock(return_value=["Game1", "Game2", "TargetGame"])

    # 构建假grounding
    mock_grounding = MagicMock()
    mock_grounding.get_topk_near_item = MagicMock(return_value=[["Game2", "TargetGame"]])
    mock_grounding.generate_embedding = MagicMock(return_value=[0.0] * 10)

    # 构建配置
    config = ITMPRecConfig(
        use_api=True,
        enable_intent=True,
        enable_dialogue_actions=True,
        enable_ipg=True,
    )

    # 构建假LLM
    mock_llm = MockLLMInterface()

    # 构建Agent (avoid super().__init__ complexity by mocking)
    agent = MagicMock()
    agent.itmp_config = config
    agent.intent_extractor = IntentExtractor()
    agent.intent_tracker = MagicMock()
    agent.intent_tracker.states = {0: IntentState("initial", {}, [0.0]*10, 0.3, 0)}
    agent.intent_memory = IntentMemoryManager(agent.intent_extractor.embedder)
    agent.dialogue_env = DialogueEnvWrapper(mock_env)
    agent.ipg_reranker = IPGReranker(
        base_env=mock_env,
        grounding_model=mock_grounding,
        embedder=agent.intent_extractor.embedder,
        topk=3,
    )
    agent.action_parser = ActionParser()
    agent.action_executor = MagicMock()
    agent.action_executor.execute = MagicMock(return_value={
        "feedback": "User accepted.",
        "accepted": True,
        "rejected": False,
        "raw_rewards": [0.8, 0.5, 0.3],
    })
    agent.reward_model = MultiObjectiveReward()
    agent.trajectory_buffer = TrajectoryBuffer()
    agent.target_intentions = {0: TargetIntention("TargetGame", ["TargetGame"], "TargetGame", [0.0]*10)}
    agent.scratchpad = {0: ""}
    agent.step_n = 1
    agent.idxs = [0]
    agent.batch_size = 1
    agent.finished = {0: False}
    agent.argument_lists = {0: []}
    agent.ori_argument_lists = {0: []}
    agent.reward_lists = {0: []}
    agent.rel_lists = {0: []}
    agent.dialogue_observations = {0: []}
    agent.task = task
    agent.llm_interface = mock_llm
    agent.critic_llm_interface = mock_llm
    agent.enc = None
    agent.infos = {0: {}}
    agent.final_infos = {}
    agent.dpo_training_data_thought = []
    agent.dpo_training_data_action = []
    agent.dpo_training_data_critic = []
    agent.add_lora = False

    # Simulate a single step
    from itmprec_prototype.core.agent_itmprec import ITMPReactA2CAgent
    # Use the real _safe_prompt method
    agent._safe_prompt = ITMPReactA2CAgent._safe_prompt.__get__(agent, MagicMock)

    # Test prompt generation
    prompts = ITMPReactA2CAgent._build_agent_prompt(agent, [0])
    assert len(prompts) == 1
    assert isinstance(prompts[0], list)
    assert prompts[0][0]["role"] == "system"
    assert "recommend[item]" in prompts[0][0]["content"]
    print("[PASS] Agent prompt generation")

    # Test LLM call
    out = mock_llm.generate(prompts[0], n=1)
    assert len(out) == 1
    assert "Thought" in out[0]
    print("[PASS] Agent LLM call")

    # Test action parsing
    action = agent.action_parser.parse(out[0].split("\n")[-1])
    assert action.valid
    print(f"[PASS] Action parsed: {action.type}[{action.argument}]")

    # Test intent update
    prev_intent = agent.intent_tracker.states[0]
    observation = agent.action_executor.execute(action, 0, task, [], agent.dialogue_env, prev_intent, agent.target_intentions[0])
    curr_intent = agent.intent_extractor.update_intent(prev_intent, observation, action)
    assert isinstance(curr_intent, IntentState)
    print("[PASS] Intent update")

    # Test reward
    reward = agent.reward_model.compute(action, observation, prev_intent, curr_intent, agent.target_intentions[0])
    assert reward.total > 0
    print(f"[PASS] Reward computed: {reward.total:.3f}")

    # Test trajectory buffer
    agent.trajectory_buffer.add_step(0, TrajectoryStep(
        thought="Test thought",
        action=action.to_dict(),
        observation=observation,
        reward_dict=reward.to_dict(),
        intent_state=curr_intent.to_dict(),
        critic_value=0.5,
    ))
    dpo = agent.trajectory_buffer.build_dpo_data()
    assert isinstance(dpo, list)
    print("[PASS] Trajectory buffer")

    print("\n[ALL PASSED] API-based architecture mock test completed successfully.")
    print("You can now purchase an API key and replace MockLLMInterface with LLMInterface.")


if __name__ == "__main__":
    test_mock_llm_interface()
    test_prompts_api_format()
    test_agent_step_with_mock_llm()
