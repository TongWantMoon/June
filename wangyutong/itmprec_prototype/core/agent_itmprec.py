# 核心Agent，继承T-PRA的ReactA2CAgent，加了意图、IPG、记忆、轨迹等模块

import json
import os
from collections import defaultdict

from Agents.agent_a2c import ReactA2CAgent, extract_floats_list, format_step, save_info
from Agents.agent_base import truncate_scratchpad
from Agents.prompts import react_agent_prompt

from .actions import ActionExecutor, ActionParser
from .config import ITMPRecConfig
from ..env.env_dialogue import DialogueEnvWrapper
from .intent import IntentExtractor, IntentTracker
from .ipg import IPGReranker
from .memory import IntentMemoryManager
from ..api.prompts import itmp_react_agent_prompt, build_actor_messages, build_critic_messages
from .rewards import MultiObjectiveReward
from ..training.trajectory import TrajectoryBuffer, TrajectoryStep
from ..api.llm_api import LLMInterface, MockLLMInterface, DeepSeekInterface


class ITMPReactA2CAgent(ReactA2CAgent):
    """继承T-PRA的ReactA2CAgent，不改原文件，只在新类里扩展"""

    def __init__(
        self,
        task,
        idxs,
        args,
        rec_env,
        grounding_model,
        max_steps=30,
        react_llm=None,
        reflect_llm=None,
        critic_llm=None,
        reflections_memory=None,
        actor_memory=None,
        critic_memory=None,
        critic_model=None,
        itmp_config=None,
    ):
        self.itmp_config = itmp_config or ITMPRecConfig.from_args(args)
        self.intent_extractor = IntentExtractor()
        self.intent_tracker = IntentTracker(self.intent_extractor)
        self.intent_memory = IntentMemoryManager(self.intent_extractor.embedder)
        self.dialogue_env = DialogueEnvWrapper(rec_env)
        self.ipg_reranker = IPGReranker(
            base_env=rec_env,
            grounding_model=grounding_model,
            embedder=self.intent_extractor.embedder,
            topk=self.itmp_config.ipg_topk,
        )
        self.action_parser = ActionParser()
        self.action_executor = ActionExecutor(
            grounding_model=grounding_model,
            ipg_reranker=self.ipg_reranker if self.itmp_config.enable_ipg else None,
        )
        self.reward_model = MultiObjectiveReward(
            alpha=self.itmp_config.alpha_recommendation,
            beta=self.itmp_config.beta_intention_alignment,
            gamma=self.itmp_config.gamma_dialogue_quality,
            delta=self.itmp_config.delta_target_progress,
        )
        self.trajectory_buffer = TrajectoryBuffer()
        self.target_intentions = {}

        super().__init__(
            task,
            idxs,
            args,
            rec_env,
            grounding_model,
            max_steps=max_steps,
            agent_prompt=react_agent_prompt,
            react_llm=react_llm,
            reflect_llm=reflect_llm,
            critic_llm=critic_llm,
            reflections_memory=reflections_memory,
            actor_memory=actor_memory,
            critic_memory=critic_memory,
            critic_model=critic_model,
        )
        # 根据配置选择LLM接口
        if self.itmp_config.use_api:
            if react_llm is not None and isinstance(react_llm, (LLMInterface, MockLLMInterface, DeepSeekInterface)):
                self.llm_interface = react_llm
            else:
                # 如果检测到deepseek相关的base_url或model，自动创建DeepSeekInterface
                if "deepseek" in (self.itmp_config.base_url or "") or "deepseek" in (self.itmp_config.model or ""):
                    self.llm_interface = DeepSeekInterface(
                        api_key=self.itmp_config.api_key,
                        model=self.itmp_config.model,
                        temperature=self.itmp_config.temperature,
                        max_tokens=self.itmp_config.max_tokens,
                    )
                else:
                    self.llm_interface = LLMInterface(
                        api_key=self.itmp_config.api_key,
                        base_url=self.itmp_config.base_url,
                        model=self.itmp_config.model,
                        temperature=self.itmp_config.temperature,
                        max_tokens=self.itmp_config.max_tokens,
                    )
            if critic_llm is not None and isinstance(critic_llm, (LLMInterface, MockLLMInterface, DeepSeekInterface)):
                self.critic_llm_interface = critic_llm
            else:
                if "deepseek" in (self.itmp_config.base_url or "") or "deepseek" in (self.itmp_config.model or ""):
                    self.critic_llm_interface = DeepSeekInterface(
                        api_key=self.itmp_config.api_key,
                        model=self.itmp_config.model,
                        temperature=self.itmp_config.temperature,
                        max_tokens=self.itmp_config.max_tokens,
                    )
                else:
                    self.critic_llm_interface = LLMInterface(
                        api_key=self.itmp_config.api_key,
                        base_url=self.itmp_config.base_url,
                        model=self.itmp_config.model,
                        temperature=self.itmp_config.temperature,
                        max_tokens=self.itmp_config.max_tokens,
                    )
        else:
            self.llm_interface = react_llm
            self.critic_llm_interface = critic_llm

        # API模式开启时，覆盖llm引用，让ReactA2CAgent
        # 使用API接口而不是本地Llama模型
        if self.itmp_config.use_api:
            self.react_llm = self.llm_interface
            self.reflect_llm = self.llm_interface
            self.critic_llm = self.critic_llm_interface

        self.dialogue_observations = defaultdict(list)

    def run(self, reset=True, reflect_strategy=None, outfilename=""):
        for i in range(0, len(self.idxs), self.batch_size):
            temp_idxs = self.idxs[i : i + self.batch_size]
            print("EPOCH: ", i)
            print(f"temp_idxs:{temp_idxs}")

            self._ensure_itmp_state(temp_idxs)
            self.single_run(temp_idxs, reset)
            self._build_info(temp_idxs)

            self.final_infos["trajs"] = self.infos
            self.final_infos["reflections"] = self.reflections
            self.final_infos["actor_memory"] = self.actor_memory
            self.final_infos["critic_memory"] = self.critic_memory
            self.final_infos["dpo_training_data_thought"] = self.dpo_training_data_thought
            self.final_infos["dpo_training_data_action"] = self.dpo_training_data_action
            self.final_infos["dpo_training_data_critic"] = self.dpo_training_data_critic
            self.final_infos["mlp_training_data_critic"] = self.mlp_training_data_critic

            save_info(self.final_infos, outfilename)
            self._save_itmp_outputs(outfilename)
            print(f"OUTPUTFILE NAME IS:{outfilename}")

    def step(self, idxs, thought_num=None, action_num=None, adv_gamma=0.5):
        if not (
            self.itmp_config.enable_intent
            or self.itmp_config.enable_dialogue_actions
            or self.itmp_config.enable_ipg
            or self.itmp_config.enable_trajectory_dpo
        ):
            return super().step(idxs, thought_num or self.thought_num, action_num or self.action_num, adv_gamma)

        thought_num = thought_num or self.thought_num
        action_num = action_num or self.action_num
        self._ensure_itmp_state(idxs)

        for idx in idxs:
            self.scratchpad[idx] += f"\nThought {self.step_n}:"

        pre_state_value = self.prompt_critic_llm(idxs)
        scratchpad_backup = self.scratchpad.copy()

        thought_candidates = []
        for _ in range(thought_num):
            thoughts = self._safe_prompt(idxs, fallback_text="I should analyze the user's intent and the target item before making a recommendation.")
            thought_candidates.append(thoughts)

        candidate_rows = []
        for thought_idx, thoughts in enumerate(thought_candidates):
            for _ in range(action_num):
                for pos, idx in enumerate(idxs):
                    self.scratchpad[idx] = (
                        scratchpad_backup[idx] + " " + thoughts[pos] + f"\nAction {self.step_n}:"
                    )
                actions_text = self._safe_prompt(idxs, fallback_text="clarify[Please specify your preference.]")
                actions = self.action_parser.parse_many(actions_text)
                candidate_rows.append((thought_idx, thoughts, actions_text, actions))

        # 评估所有候选并记录轨迹
        candidate_results = []
        for candidate_idx, (thought_idx, thoughts, actions_text, actions) in enumerate(candidate_rows):
            results_per_idx = {}
            for pos, idx in enumerate(idxs):
                action = actions[pos]
                history_items = self.task.get_history_actions(idx) + self.argument_lists[idx]
                previous_intent = self.intent_tracker.states[idx]
                target_intention = self.target_intentions[idx]
                observation = self.action_executor.execute(
                    action,
                    idx,
                    self.task,
                    history_items,
                    self.dialogue_env,
                    previous_intent,
                    target_intention,
                )
                current_intent = self.intent_extractor.update_intent(previous_intent, observation, action)
                reward = self.reward_model.compute(
                    action, observation, previous_intent, current_intent, target_intention
                )
                results_per_idx[idx] = {
                    "thought": thoughts[pos],
                    "action_text": actions_text[pos],
                    "action": action,
                    "observation": observation,
                    "intent": current_intent,
                    "reward": reward,
                }
                self.trajectory_buffer.add_step(
                    idx,
                    TrajectoryStep(
                        thought=thoughts[pos],
                        action=action.to_dict(),
                        observation=observation,
                        reward_dict=reward.to_dict(),
                        intent_state=current_intent.to_dict(),
                        critic_value=float(pre_state_value[pos]) if pos < len(pre_state_value) else 0.0,
                    ),
                    candidate_idx=candidate_idx,
                )
            candidate_results.append(results_per_idx)

        # 为每个idx选择最优候选
        selected = {}
        for pos, idx in enumerate(idxs):
            best = None
            for candidate_idx, results_per_idx in enumerate(candidate_results):
                row = results_per_idx[idx]
                score = row["reward"].total
                if best is None or score > best["score"]:
                    best = {"score": score, **row}
            selected[idx] = best

        for pos, idx in enumerate(idxs):
            row = selected[idx]
            action = row["action"]
            observation = row["observation"]
            reward = row["reward"]
            current_intent = row["intent"]
            target_intention = self.target_intentions[idx]
            grounded_or_arg = action.grounded_item or action.argument

            self.scratchpad[idx] = (
                scratchpad_backup[idx]
                + " "
                + row["thought"]
                + f"\nAction {self.step_n}: "
                + f"{action.type}[{grounded_or_arg}]"
            )
            self.scratchpad[idx] += f"\nObservation {self.step_n}: {observation['feedback']} Final reward={reward.total:.2f}"
            if action.argument != grounded_or_arg:
                self.scratchpad[idx] += f"\nObservation {self.step_n}: [{action.argument}] can not be recommended, instead, {action.type}[{grounded_or_arg}]"

            if action.type == "recommend":
                self.argument_lists[idx].append(grounded_or_arg)
                self.ori_argument_lists[idx].append(action.argument)
                self.rel_lists[idx].append(observation.get("raw_rewards", [0.0, 0.0, 0.0]))
                if grounded_or_arg == target_intention.target_item:
                    self.finished[idx] = True

            self.reward_lists[idx].append(float(reward.total))
            self.dialogue_observations[idx].append(observation)
            self.intent_tracker.states[idx] = current_intent
            self.intent_memory.add_memory(
                f"idx={idx}; action={action.type}[{grounded_or_arg}]; observation={observation['feedback']}; intent={current_intent.summary}",
                kind="short_term",
                idx=idx,
                turn_id=self.step_n,
                metadata={"reward": reward.to_dict()},
            )
            self._append_lightweight_dpo(idx, scratchpad_backup[idx], row, pre_state_value[pos])

        self.step_n += 1
        print(self.step_n)

    def _build_agent_prompt(self, idxs):
        """Build API-compatible messages for each idx."""
        prompts = []
        for idx in idxs:
            self._ensure_itmp_state([idx])
            target_intention = self.target_intentions[idx]
            current_intent = self.intent_tracker.states[idx]
            memory = self.intent_memory.summarize_memory(current_intent.summary, topk=self.itmp_config.memory_topk)
            scratchpad = truncate_scratchpad(self.scratchpad[idx], tokenizer=self.enc)
            messages = build_actor_messages(
                question=self.task[idx],
                target_intention=json.dumps(target_intention.to_dict(), ensure_ascii=False),
                current_intent=json.dumps(current_intent.to_dict(), ensure_ascii=False),
                memory=memory,
                scratchpad=scratchpad,
            )
            prompts.append(messages)
        return prompts

    def _build_critic_prompt(self, idxs, action=None):
        """Build API-compatible messages for critic."""
        prompts = []
        for pos, idx in enumerate(idxs):
            self._ensure_itmp_state([idx])
            temp_list = self.task.get_history_actions(idx) + self.argument_lists[idx]
            if action is not None:
                temp_list += [action[pos]]
            messages = build_critic_messages(
                history_list=str(temp_list[-10:]),
                current_intent=self.intent_tracker.states[idx].summary,
                target_intention=self.target_intentions[idx].description,
                recent_actions=str(self.argument_lists[idx][-5:]),
            )
            prompts.append(messages)
        return prompts

    def prompt_agent(self, idxs):
        """Override to support API messages format."""
        if self.itmp_config.use_api:
            prompts = self._build_agent_prompt(idxs)
            results = []
            for messages in prompts:
                out = self.llm_interface.generate(messages, n=1)
                results.append(format_step(out)[0] if out else "")
            return results
        else:
            # Fallback to T-PRA local model behavior
            return super().prompt_agent(idxs)

    def prompt_critic_llm(self, idxs, action=None):
        if self.itmp_config.use_api:
            prompts = self._build_critic_prompt(idxs, action)
            results = []
            for messages in prompts:
                out = self.critic_llm_interface.generate(messages, n=1)
                results.append(format_step(out)[0] if out else "")
            return extract_floats_list(results)
        else:
            return super().prompt_critic_llm(idxs, action)

    def _ensure_itmp_state(self, idxs):
        for idx in idxs:
            if idx not in self.target_intentions:
                self.target_intentions[idx] = self.intent_extractor.build_target_intention(
                    self.task.get_target_item(idx),
                    self.task[idx],
                )
            if idx not in self.intent_tracker.states:
                state = self.intent_tracker.initialize(idx, self.task[idx], self.task.get_history_actions(idx))
                self.intent_memory.add_memory(
                    f"Initial intent for idx={idx}: {state.summary}",
                    kind="long_term",
                    idx=idx,
                    turn_id=0,
                )

    def _safe_prompt(self, idxs, fallback_text=None):
        for _ in range(5):
            try:
                values = self.prompt_agent(idxs)
                if all(value != "" for value in values):
                    return values
            except Exception as exc:
                print(f"prompt exception: {exc}")
        fallback = fallback_text or "clarify[Please specify your preference.]"
        return [fallback for _ in idxs]

    def _append_lightweight_dpo(self, idx, scratchpad_before, row, pre_value):
        conversation = [
            {"from": "system", "value": "You are an intention-aware proactive recommendation agent."},
            {
                "from": "human",
                "value": truncate_scratchpad("Question: " + self.task[idx] + scratchpad_before, tokenizer=self.enc),
            },
        ]
        chosen = {"from": "gpt", "value": row["thought"]}
        rejected = {"from": "gpt", "value": "I should recommend without considering user intent."}
        if chosen["value"] != rejected["value"]:
            self.dpo_training_data_thought.append(
                {"conversations": conversation, "chosen": chosen, "rejected": rejected}
            )

        action = row["action"]
        grounded_or_arg = action.grounded_item or action.argument
        self.dpo_training_data_action.append(
            {
                "conversations": conversation,
                "chosen": {"from": "gpt", "value": f"{action.type}[{grounded_or_arg}]"},
                "rejected": {"from": "gpt", "value": "recommend[unknown item]"},
            }
        )
        reward_value = row["reward"].total
        if reward_value != pre_value:
            self.dpo_training_data_critic.append(
                {
                    "conversations": conversation,
                    "chosen": {"from": "gpt", "value": f"{reward_value:.2f}"},
                    "rejected": {"from": "gpt", "value": f"{float(pre_value):.2f}"},
                }
            )

    def _build_info(self, idxs):
        super()._build_info(idxs)
        for idx in idxs:
            self.infos[idx]["target_intention"] = self.target_intentions[idx].to_dict()
            self.infos[idx]["intent_state"] = self.intent_tracker.states[idx].to_dict()
            self.infos[idx]["dialogue_observations"] = self.dialogue_observations[idx]

    def _save_itmp_outputs(self, outfilename):
        if not outfilename:
            return
        os.makedirs(outfilename, exist_ok=True)
        self.trajectory_buffer.save(outfilename)
        self.intent_memory.save(os.path.join(outfilename, "intent_memory.json"))
