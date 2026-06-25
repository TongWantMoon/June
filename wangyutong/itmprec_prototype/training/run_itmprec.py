# 入口文件，解析命令行参数，初始化环境、模型、Agent，开始跑实验

import sys
import os

# 确保项目根目录在 sys.path 最前面，让 Python 能正确找到 itmprec_prototype 包
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # wangyutong
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 找到 T-PRA-main 目录
def find_tpra_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(current, "T-PRA-main")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

TPRA_ROOT = find_tpra_root(_project_root)
if TPRA_ROOT is None:
    TPRA_ROOT = os.path.join(os.path.dirname(_project_root), "T-PRA-main")
TPRA_ROOT = os.path.normpath(TPRA_ROOT)
if TPRA_ROOT and TPRA_ROOT not in sys.path:
    sys.path.insert(0, TPRA_ROOT)

import argparse
import random
from datetime import datetime
from functools import partial

import numpy as np


class MockGroundingModel:

    def __init__(self, task):
        self.task = task
        self.embed_dim = 768

    def get_top_near_item(self, item, his_seq):
        if item not in his_seq:
            return item
        if his_seq:
            return his_seq[0]
        return item

    def get_topk_near_item(self, items, k):
        return [item for item in items[:k]]

    def generate_embedding(self, text):
        np.random.seed(hash(text) % 2**31)
        emb = np.random.randn(self.embed_dim).astype(np.float32)
        np.random.seed(None)
        return emb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="itmp_agent_a2c")
    parser.add_argument("--task", type=str, default="steam")
    parser.add_argument("--env", type=str, default="steam")
    parser.add_argument("--task_split", type=str, default="train")
    parser.add_argument("--task_start_index", type=int, default=0)
    parser.add_argument("--task_end_index", type=int, default=100)
    parser.add_argument("--backend", type=str, default="llama")
    parser.add_argument("--env_path", type=str, default="./env")
    parser.add_argument("--modelpath", type=str, default="/data/wangmz/LLaMA-Factory/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--grounding_model_path", type=str, default="/data/wangmz/LLaMA-Factory/meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--factory_path", type=str, default="/data/wangmz/LLaMA-Factory")
    parser.add_argument("--peftpath", type=str, default="./lora/steam_lora_dpo_1")
    parser.add_argument("--add_lora", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--random", default=True, action="store_true")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_dir_base", type=str, default="new_itmp")
    parser.add_argument("--thought_num", type=int, default=2)
    parser.add_argument("--action_num", type=int, default=2)
    parser.add_argument("--reward_func", type=int, default=0)
    parser.add_argument("--input_data_suffix", type=str, default="ran")
    parser.add_argument("--Max_Iteration", type=int, default=15)
    parser.add_argument("--env_threshold", type=float, default=50)
    parser.add_argument("--env_window_length", type=int, default=4)
    parser.add_argument("--input_file_name", default=None)

    parser.add_argument("--enable_intent", action="store_true", default=True)
    parser.add_argument("--enable_dialogue_actions", action="store_true", default=True)
    parser.add_argument("--enable_ipg", action="store_true", default=True)
    parser.add_argument("--enable_trajectory_dpo", action="store_true", default=True)
    parser.add_argument("--ipg_topk", type=int, default=5)
    parser.add_argument("--memory_topk", type=int, default=3)
    parser.add_argument("--reward_alpha", type=float, default=0.4)
    parser.add_argument("--reward_beta", type=float, default=0.2)
    parser.add_argument("--reward_gamma", type=float, default=0.2)
    parser.add_argument("--reward_delta", type=float, default=0.2)

    # API相关参数，默认用DeepSeek
    parser.add_argument("--use_api", action="store_true", default=True, help="用API调用大模型，替代本地模型")
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""), help="API密钥，默认从环境变量DEEPSEEK_API_KEY读取")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1", help="API的基础地址")
    parser.add_argument("--model", type=str, default="deepseek-v4-pro", help="模型名称")
    # 注意：--temperature上面已经定义了，这里直接复用
    parser.add_argument("--max_tokens", type=int, default=512, help="API调用时的最大token数")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_api:
        from itmprec_prototype.env.task_movielens import get_task
    else:
        from tasks import get_task

    from itmprec_prototype.core.agent_itmprec import ITMPReactA2CAgent
    from itmprec_prototype.core.config import ITMPRecConfig

    if args.use_api:
        from itmprec_prototype.env.env_movielens import get_envs, MovieLensENV
        from itmprec_prototype.api.llm_api import DeepSeekInterface
    else:
        import torch
        from env import get_envs
        from env.env import Grounding_Model_LLAMA
        from mlp import MLP
        from models.llama import LlamaInterface

    # 如果env_path没指定，就自动设成T-PRA的env目录
    if not hasattr(args, 'env_path') or args.env_path == './env':
        args.env_path = os.path.join(TPRA_ROOT, 'env')
    task = get_task(args.task, args.task_split, args.input_data_suffix)

    modelname = args.backend
    if args.add_lora:
        pathname = args.peftpath.replace("/", "_")
        modelname += f"_{pathname}"
    print(f"model={modelname}; time={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    if not os.path.exists(args.save_dir_base):
        os.mkdir(args.save_dir_base)
    outfilename = f"{args.save_dir_base}/{args.save_dir}"
    print(outfilename)

    idxs_all = list(range(len(task)))
    if args.random:
        random.Random(233).shuffle(idxs_all)
    idxs = idxs_all[args.task_start_index : args.task_end_index]

    if not args.use_api and args.backend != "llama":
        raise ValueError(f"Model not found: {args.backend}")

    if args.use_api:
        # 纯API模式：跳过本地Llama模型加载
        model = None
    else:
        llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
        model = partial(llama.generate_responses_from_llama, temperature=args.temperature, stop=["\n"])

    envs = get_envs(args.env, args, args.task_split)

    if args.use_api:
        # 纯API模式：用轻量Mock grounding模型
        grounding_model = MockGroundingModel(args.env)
    else:
        grounding_model = Grounding_Model_LLAMA(args.grounding_model_path, args.env)

    if args.use_api:
        mlp_model = None
    else:
        mlp_model = MLP(input_dim=4096, hidden_dims=[1024, 256], output_dim=1)
        if args.add_lora:
            mlp_model_path = "_".join(args.save_dir.split("_")[:2] + ["train"])
            mlp_model.load_state_dict(torch.load(f"{args.save_dir_base}/{mlp_model_path}/mlp_model.pth"))
            mlp_model.eval()
            print(f"MLP loaded {mlp_model_path}/mlp_model.pth")

    agent = ITMPReactA2CAgent(
        task,
        idxs,
        args,
        envs,
        grounding_model,
        max_steps=args.Max_Iteration,
        react_llm=model,
        reflect_llm=model,
        critic_llm=model,
        critic_model=mlp_model,
        itmp_config=ITMPRecConfig.from_args(args),
    )
    print("start")
    agent.run(outfilename=outfilename)
    print(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))


if __name__ == "__main__":
    main()
