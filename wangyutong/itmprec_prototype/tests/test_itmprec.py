# 单元测试，覆盖ActionParser、IntentExtractor、IPGReranker等核心模块的基本逻辑

import sys
import os

TPRA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "T-PRA-main")
TPRA_ROOT = os.path.normpath(TPRA_ROOT)
WORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_ROOT = os.path.normpath(WORK_ROOT)

for _root in (TPRA_ROOT, WORK_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import unittest
from unittest.mock import MagicMock, patch

from itmprec_prototype.core.actions import ActionSchema, ActionParser, ActionExecutor
from itmprec_prototype.core.intent import IntentState, TargetIntention, IntentExtractor, IntentTracker
from itmprec_prototype.core.memory import SensoryMemory, ShortTermMemory, LongTermMemory, IntentMemoryManager
from itmprec_prototype.core.ipg import IPGReranker
from itmprec_prototype.core.rewards import MultiObjectiveReward
from itmprec_prototype.training.trajectory import TrajectoryBuffer, TrajectoryStep
from itmprec_prototype.env.env_dialogue import DialogueEnvWrapper
from itmprec_prototype.core.config import ITMPRecConfig


class TestActionParser(unittest.TestCase):
    """ActionParser单元测试"""

    def test_recommend(self):
        parser = ActionParser()
        action = parser.parse("recommend[Call of Duty]")
        self.assertEqual(action.type, "recommend")
        self.assertEqual(action.argument, "Call of Duty")
        self.assertTrue(action.valid)

    def test_ask(self):
        parser = ActionParser()
        action = parser.parse("ask[What genre do you prefer?]")
        self.assertEqual(action.type, "ask")
        self.assertEqual(action.argument, "What genre do you prefer?")
        self.assertTrue(action.valid)

    def test_clarify(self):
        parser = ActionParser()
        action = parser.parse("clarify[Please specify your preference.]")
        self.assertEqual(action.type, "clarify")
        self.assertEqual(action.argument, "Please specify your preference.")
        self.assertTrue(action.valid)

    def test_explain(self):
        parser = ActionParser()
        action = parser.parse("explain[The Witcher 3]")
        self.assertEqual(action.type, "explain")
        self.assertEqual(action.argument, "The Witcher 3")
        self.assertTrue(action.valid)

    def test_invalid_action(self):
        parser = ActionParser()
        action = parser.parse("buy[Call of Duty]")
        self.assertFalse(action.valid)

    def test_quoted_item(self):
        parser = ActionParser()
        action = parser.parse("recommend[\"Half-Life 2\"]")
        self.assertEqual(action.argument, "Half-Life 2")
        self.assertTrue(action.valid)

    def test_unquoted_item(self):
        parser = ActionParser()
        action = parser.parse("recommend[Half-Life 2]")
        self.assertEqual(action.argument, "Half-Life 2")
        self.assertTrue(action.valid)

    def test_parse_many(self):
        parser = ActionParser()
        actions = parser.parse_many(["recommend[A]", "ask[B]?", "clarify[C]"])
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[0].type, "recommend")
        self.assertEqual(actions[1].type, "ask")
        self.assertEqual(actions[2].type, "clarify")


class TestIntentExtractor(unittest.TestCase):
    """Unit tests: intent update after accept/reject/preference."""

    def setUp(self):
        self.extractor = IntentExtractor()

    def test_build_target_intention(self):
        ti = self.extractor.build_target_intention("TargetItem", "question text")
        self.assertEqual(ti.target_item, "TargetItem")
        self.assertIn("TargetItem", ti.description)

    def test_initialize_intent(self):
        tracker = IntentTracker(self.extractor)
        state = tracker.initialize(0, "question", ["item1", "item2"])
        self.assertIn("item1", state.summary)
        self.assertIn("item2", state.summary)

    def test_update_after_accept(self):
        tracker = IntentTracker(self.extractor)
        state = tracker.initialize(0, "question", ["item1"])
        observation = {
            "feedback": "accepted",
            "accepted": True,
            "rejected": False,
            "item": "item2",
        }
        action = ActionSchema(type="recommend", argument="item2", raw_text="recommend[item2]", valid=True, grounded_item="item2")
        new_state = self.extractor.update_intent(state, observation, action)
        self.assertIn("item2", new_state.slots["accepted_items"])
        self.assertGreater(new_state.confidence, state.confidence)

    def test_update_after_reject(self):
        tracker = IntentTracker(self.extractor)
        state = tracker.initialize(0, "question", ["item1"])
        observation = {
            "feedback": "rejected",
            "accepted": False,
            "rejected": True,
            "item": "item2",
        }
        action = ActionSchema(type="recommend", argument="item2", raw_text="recommend[item2]", valid=True, grounded_item="item2")
        new_state = self.extractor.update_intent(state, observation, action)
        self.assertIn("item2", new_state.slots["rejected_items"])
        self.assertLess(new_state.confidence, state.confidence)

    def test_update_after_ask(self):
        tracker = IntentTracker(self.extractor)
        state = tracker.initialize(0, "question", ["item1"])
        observation = {
            "feedback": "The user provides a preference clue after being asked: Do you like FPS?",
            "accepted": False,
            "rejected": False,
        }
        action = ActionSchema(type="ask", argument="Do you like FPS?", raw_text="ask[Do you like FPS?]", valid=True)
        new_state = self.extractor.update_intent(state, observation, action)
        self.assertIn("FPS", new_state.slots["open_questions"])


class TestIPGReranker(unittest.TestCase):
    """Unit tests: IPG score and filtering."""

    def test_ipg_score_increases_when_nearer_target(self):
        mock_env = MagicMock()
        mock_env.get_item_list = MagicMock(return_value=["A", "B", "C"])
        mock_env.get_reward = MagicMock(return_value=(0.6, 0.5, 0.2))

        mock_grounding = MagicMock()
        mock_grounding.get_topk_near_item = MagicMock(return_value=[["A", "B", "C"]])
        mock_grounding.generate_embedding = MagicMock(return_value=[0.0] * 10)

        def embed_side_effect(text):
            vec = [0.0] * 10
            if "C" in str(text):
                vec[0] = 1.0
            elif "B" in str(text):
                vec[0] = 0.5
            elif "A" in str(text):
                vec[0] = 0.1
            return vec

        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(side_effect=embed_side_effect)

        reranker = IPGReranker(
            base_env=mock_env,
            grounding_model=mock_grounding,
            embedder=mock_embedder,
            topk=3,
        )

        target = TargetIntention("target", ["target"], "target", [1.0] + [0.0] * 9)
        current = IntentState("current", {}, [0.0] * 10, 0.5, 0)
        current.slots["rejected_items"] = ["B"]
        history = ["item1", "B"]
        grounded = "C"
        result = reranker.rerank_from_seed(grounded, exclude_items=history, intent_state=current, target_intention=target)

        # B在排除列表里，不应该出现在结果中
        self.assertNotIn("B", result)
        # C应该被选中，因为离目标最近
        self.assertEqual(result, "C")


class TestMultiObjectiveReward(unittest.TestCase):
    """Unit tests: reward composition."""

    def test_reward_composition(self):
        reward_model = MultiObjectiveReward(alpha=0.4, beta=0.2, gamma=0.2, delta=0.2)
        action = ActionSchema(type="recommend", argument="item", raw_text="", valid=True)
        observation = {
            "recommendation_reward": 0.8,
            "dialogue_quality": 0.6,
        }
        prev = IntentState("prev", {}, [0.1, 0.0], 0.5, 0)
        curr = IntentState("curr", {}, [0.5, 0.0], 0.7, 1)
        target = TargetIntention("target", ["target"], "target", [1.0, 0.0])

        reward = reward_model.compute(action, observation, prev, curr, target)
        self.assertAlmostEqual(reward.recommendation, 0.8, places=1)
        self.assertAlmostEqual(reward.dialogue_quality, 0.6, places=1)
        self.assertGreater(reward.total, 0.0)


class TestTrajectoryBuffer(unittest.TestCase):
    """Unit tests: trajectory buffer DPO construction."""

    def test_add_and_save(self):
        buf = TrajectoryBuffer()
        buf.add_step(0, TrajectoryStep(
            thought="t1", action={"type": "recommend"}, observation={"feedback": "ok"},
            reward_dict={"total": 1.0}, intent_state={"summary": "s"}, critic_value=0.5
        ))
        buf.add_step(1, TrajectoryStep(
            thought="t2", action={"type": "ask"}, observation={"feedback": "ok"},
            reward_dict={"total": 0.5}, intent_state={"summary": "s"}, critic_value=0.3
        ))
        dpo = buf.build_dpo_data()
        self.assertEqual(len(dpo), 1)
        self.assertIn("chosen", dpo[0])
        self.assertIn("rejected", dpo[0])


class TestDialogueEnvWrapper(unittest.TestCase):
    """Unit tests: dialogue env wrapper."""

    def test_recommend_boundary(self):
        """Ensure _recommend handles short item_list gracefully."""
        mock_env = MagicMock()
        mock_env.get_reward = MagicMock(return_value=(0.6, 0.5, 0.2))
        wrapper = DialogueEnvWrapper(mock_env)
        action = ActionSchema(type="recommend", argument="item", raw_text="", valid=True, grounded_item="item")
        task = MagicMock()
        task.get_userid = MagicMock(return_value=0)
        target = TargetIntention("target", ["target"], "target", [0.0] * 10)
        result = wrapper._recommend(0, action, task, ["prev_item"], target)
        self.assertIn("feedback", result)
        self.assertTrue(result["accepted"])

    def test_invalid_action(self):
        mock_env = MagicMock()
        wrapper = DialogueEnvWrapper(mock_env)
        action = ActionSchema(type="buy", argument="item", raw_text="", valid=False)
        result = wrapper.step(0, action, None, [], None, None)
        self.assertTrue(result["rejected"])


class TestRegression(unittest.TestCase):
    """Regression: all ITMP flags disabled -> should fall back to T-PRA behavior."""

    def test_config_defaults_to_tpra(self):
        config = ITMPRecConfig.from_args(MagicMock())
        self.assertFalse(config.enable_intent)
        self.assertFalse(config.enable_dialogue_actions)
        self.assertFalse(config.enable_ipg)
        self.assertFalse(config.enable_trajectory_dpo)


if __name__ == "__main__":
    unittest.main()
