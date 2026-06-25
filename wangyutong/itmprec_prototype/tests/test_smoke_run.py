# A阶段冒烟测试，Mock掉所有外部依赖，不花钱、不调用API、不加载数据，验证完整流程

import sys, os
import types
import unittest
import json
from unittest.mock import MagicMock
import tempfile
import shutil

# 确保T-PRA和工作目录在搜索路径里
TPRA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'T-PRA-main'))
WORK_ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for r in (TPRA_ROOT, WORK_ROOT):
    if r not in sys.path:
        sys.path.insert(0, r)


class FakeReactA2CAgent:
    """简单的假Agent，避免MagicMock继承问题。"""
    def __init__(self, task, idxs, args, rec_env, grounding_model,
                 max_steps=30, agent_prompt=None, react_llm=None,
                 reflect_llm=None, critic_llm=None, reflections_memory=None,
                 actor_memory=None, critic_memory=None, critic_model=None,
                 **kwargs):
        self.idxs = idxs
        self.scratchpad = {}
        self.step_n = 1
        self.finished = {}
        self.argument_lists = {}
        self.ori_argument_lists = {}
        self.reward_lists = {}
        self.rel_lists = {}
        self.value_lists = {}
        self.target_dist_lists = {}
        self.infos = {}
        self.final_infos = {}
        self.dpo_training_data_thought = []
        self.dpo_training_data_action = []
        self.dpo_training_data_critic = []
        self.mlp_training_data_critic = []
        self.reflections = []
        self.actor_memory = {}
        self.critic_memory = {}
        self.add_lora = False
        self.enc = None
        self.batch_size = getattr(args, 'batch_size', 1)
        self.thought_num = getattr(args, 'thought_num', 2)
        self.action_num = getattr(args, 'action_num', 2)
        self.reward_func = getattr(args, 'reward_func', 0)
        self.max_steps = max_steps
        self.task = task
        self.critic_model = None
        self.critic_llm = None
        self.reflect_llm = None

    def _build_info(self, idxs):
        for idx in idxs:
            self.infos[idx] = {"task": self.task[idx] if hasattr(self.task, '__getitem__') else "test"}

    def single_run(self, idxs, reset=True):
        for idx in idxs:
            self.scratchpad[idx] = "Thought 1: test\nAction 1: recommend[test_item]"
            self.finished[idx] = True

    def run(self, reset=True, reflect_strategy=None, outfilename=""):
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i: i + self.batch_size]
            self._build_info(temp_idxs)
            self.single_run(temp_idxs, reset)
        if outfilename:
            os.makedirs(outfilename, exist_ok=True)
            with open(os.path.join(outfilename, "trajs_agent.json"), "w") as f:
                json.dump(self.infos, f)

    def is_finished(self):
        return True

    def is_halted(self):
        return False


class TestSmokeRun(unittest.TestCase):
    """Smoke test: verify run_itmprec.py main() runs end-to-end with mocks."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="itmprec_smoke_")
        self.save_dir = os.path.join(self.test_dir, "smoke_test")
        os.makedirs(self.save_dir, exist_ok=True)

        # 清除缓存模块，避免旧import
        for mod_name in list(sys.modules.keys()):
            if "itmprec_prototype" in mod_name or mod_name in [
                "torch", "tasks", "env", "env.env", "mlp", "models", "models.llama",
                "Agents", "Agents.agent_base", "Agents.agent_a2c", "Agents.agent_reflexion",
                "Agents.prompts", "Agents.fewshots", "tiktoken", "langchain", "langchain.prompts",
            ]:
                del sys.modules[mod_name]

        # 向sys.modules注入假模块
        # 1. fake torch
        fake_torch = types.ModuleType("torch")
        fake_torch.load = MagicMock(return_value={})
        fake_torch.FloatTensor = MagicMock
        sys.modules["torch"] = fake_torch

        # 2. fake tasks
        fake_tasks = types.ModuleType("tasks")
        mock_task = MagicMock()
        mock_task.__len__ = MagicMock(return_value=5)
        mock_task.__getitem__ = MagicMock(return_value="Find a game for user")
        mock_task.get_history_actions = MagicMock(return_value=["Game1"])
        mock_task.get_target_item = MagicMock(return_value="TargetGame")
        mock_task.get_userid = MagicMock(return_value=0)
        fake_tasks.get_task = MagicMock(return_value=mock_task)
        sys.modules["tasks"] = fake_tasks

        # 3. fake env
        fake_env_pkg = types.ModuleType("env")
        fake_env = MagicMock()
        fake_env.get_reward = MagicMock(return_value=(0.8, 0.5, 0.3))
        fake_env.get_item_list = MagicMock(return_value=["Game1", "Game2", "TargetGame"])
        fake_env.get_dist = MagicMock(return_value=0.5)
        fake_env.get_max_step = MagicMock(return_value=10)
        fake_env_pkg.get_envs = MagicMock(return_value=fake_env)
        sys.modules["env"] = fake_env_pkg
        fake_env_sub = types.ModuleType("env.env")
        fake_grounding = MagicMock()
        fake_grounding.get_topk_near_item = MagicMock(return_value=[["Game2", "TargetGame"]])
        fake_grounding.generate_embedding = MagicMock(return_value=[0.0] * 10)
        fake_env_sub.Grounding_Model_LLAMA = MagicMock(return_value=fake_grounding)
        sys.modules["env.env"] = fake_env_sub

        # 4. 假mlp
        fake_mlp = types.ModuleType("mlp")
        mock_mlp_instance = MagicMock()
        mock_mlp_instance.return_value = MagicMock(return_value=MagicMock(item=MagicMock(return_value=[0.5])))
        fake_mlp.MLP = MagicMock(return_value=mock_mlp_instance)
        sys.modules["mlp"] = fake_mlp

        # 5. 假models.llama
        fake_models = types.ModuleType("models")
        sys.modules["models"] = fake_models
        fake_llama = types.ModuleType("models.llama")
        mock_llama_instance = MagicMock()
        mock_llama_instance.generate_responses_from_llama = MagicMock(
            return_value=["Thought 1: I should ask.\nAction 1: ask[What genre?]"]
        )
        fake_llama.LlamaInterface = MagicMock(return_value=mock_llama_instance)
        sys.modules["models.llama"] = fake_llama

        # 6. 假agents（用简单类代替MagicMock避免__setattr__递归）ursion)
        fake_agents = types.ModuleType("Agents")
        sys.modules["Agents"] = fake_agents

        fake_agents_base = types.ModuleType("Agents.agent_base")
        fake_agents_base.ReactAgent = FakeReactA2CAgent
        fake_agents_base.parse_action = lambda list_str, thought: ([], [])
        fake_agents_base.format_step = lambda steps: [s.strip() for s in steps]
        fake_agents_base.truncate_scratchpad = lambda sp, tokenizer=None: sp
        sys.modules["Agents.agent_base"] = fake_agents_base

        fake_agents_a2c = types.ModuleType("Agents.agent_a2c")
        fake_agents_a2c.ReactA2CAgent = FakeReactA2CAgent
        fake_agents_a2c.extract_floats_list = lambda output: [0.5]
        fake_agents_a2c.format_step = lambda steps: [s.strip() for s in steps]
        fake_agents_a2c.save_info = lambda infos, outfilename: None
        fake_agents_a2c.NpEncoder = json.JSONEncoder
        sys.modules["Agents.agent_a2c"] = fake_agents_a2c

        fake_agents_reflexion = types.ModuleType("Agents.agent_reflexion")
        fake_agents_reflexion.ReactReflectAgent = FakeReactA2CAgent
        sys.modules["Agents.agent_reflexion"] = fake_agents_reflexion
        fake_agents_prompts = types.ModuleType("Agents.prompts")
        sys.modules["Agents.prompts"] = fake_agents_prompts
        fake_agents_fewshots = types.ModuleType("Agents.fewshots")
        sys.modules["Agents.fewshots"] = fake_agents_fewshots

        # 7. fake tiktoken
        fake_tiktoken = types.ModuleType("tiktoken")
        fake_tiktoken.encoding_for_model = MagicMock(return_value=MagicMock(encode=MagicMock(return_value=[])))
        sys.modules["tiktoken"] = fake_tiktoken

        # 8. fake langchain
        fake_langchain = types.ModuleType("langchain")
        sys.modules["langchain"] = fake_langchain
        fake_langchain_prompts = types.ModuleType("langchain.prompts")
        fake_langchain_prompts.PromptTemplate = MagicMock
        sys.modules["langchain.prompts"] = fake_langchain_prompts

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        for mod_name in list(sys.modules.keys()):
            if mod_name in ["torch", "tasks", "env", "env.env", "mlp", "models", "models.llama",
                           "Agents", "Agents.agent_base", "Agents.agent_a2c", "Agents.agent_reflexion",
                           "Agents.prompts", "Agents.fewshots", "tiktoken", "langchain", "langchain.prompts"]:
                del sys.modules[mod_name]

    def test_smoke_run_itmprec(self):
        """Verify full pipeline runs with injected mocks."""
        # Force fresh import after module injection
        from itmprec_prototype.training import run_itmprec
        import importlib
        importlib.reload(run_itmprec)

        original_argv = sys.argv
        sys.argv = [
            "run_itmprec.py",
            "--save_dir", "smoke_test",
            "--save_dir_base", self.test_dir,
            "--task_start_index", "0",
            "--task_end_index", "2",
            "--batch_size", "1",
            "--Max_Iteration", "3",
            "--enable_intent",
            "--enable_dialogue_actions",
        ]
        try:
            run_itmprec.main()
        finally:
            sys.argv = original_argv

        # Verify output files exist
        expected_files = ["intent_memory.json"]
        for fname in expected_files:
            fpath = os.path.join(self.save_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Expected output file not found: {fname}")

        print(f"[PASS] Smoke test completed. Output files verified in: {self.save_dir}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
