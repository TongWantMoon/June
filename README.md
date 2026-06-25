[README.md](https://github.com/user-attachments/files/29331714/README.md)



项目概述

本项目是我在 T-PRA 框架基础上，针对主动推荐场景中的六个关键不足进行系统性改进后的原型实现。T-PRA 是一个优秀的对话式推荐框架，采用 ReAct 推理链、Actor-Critic 强化学习和 DPO 偏好优化。但在研究过程中我发现，T-PRA 在主动对话能力、意图建模、目标导向牵引、记忆管理、奖励设计和 API 适配等方面存在提升空间。因此，我以 T-PRA 为 Backbone，引入 ITMPRec 的意图驱动思想、IPG-Rec 的牵引式重排策略、RecAgent 的三层记忆架构和 PECRS 的两阶段候选生成方法，构建了一个具备主动对话、目标导向牵引、多目标优化的增强型推荐系统原型。所有新增模块独立开发，未修改 T-PRA 原代码，保持了系统的可扩展性和兼容性。


对 T-PRA 的六大改进方向

改进一：被动响应到主动对话。T-PRA 原版的 Agent 是被动响应式的，用户提问后才生成推荐。我引入 intent.py 意图引擎，使系统能够主动发起对话（ask 动作），在用户意图不明确时主动澄清，在多轮交互中逐步引导用户向目标靠拢。

改进二：无意图建模到意图演化追踪。T-PRA 缺乏对用户意图状态的显式建模。我实现了 IntentExtractor 和 IntentTracker，不仅分析当前意图，还追踪意图在多轮对话中的演化过程。confidence 机制确保低置信度时系统选择询问而非猜测。

改进三：单一排序到牵引式重排。T-PRA 的推荐排序只考虑用户偏好。我引入 ipg.py 的 IPGReranker，采用 preference_score 乘以 guidance_score 的联合打分方式，使推荐既满足用户喜好，又向 target_item 方向推进。

改进四：简单历史记录到三层记忆管理。T-PRA 的记忆管理较简单。我参考 RecAgent 架构，设计了 Sensory、ShortTerm、LongTerm 三层记忆，分别处理即时上下文、对话历史和跨对话的长期偏好，并支持 embedding 向量检索。

改进五：单目标奖励到多目标优化。T-PRA 的奖励函数只关注推荐准确性。我设计了四目标加权奖励：推荐准确性 alpha=0.4、对话效率 beta=0.2、目标牵引度 gamma=0.2、用户满意度 delta=0.2。gamma 的设计尤为重要，确保 Agent 不只推用户喜欢的，还要向目标靠拢。

改进六：本地模型依赖到 API 灵活适配。T-PRA 强依赖本地 transformers 模型。我封装了 llm_api.py 统一接口，支持 DeepSeek API 调用和 Mock 降级，降低了硬件门槛，提高了部署灵活性。


核心模块清单

一、配置与入口层

config.py 是配置中心，统一管理所有参数。包括 DeepSeek API 参数（base_url、model、timeout 等）、多目标奖励权重（alpha、beta、gamma、delta）、IPG 重排参数、记忆检索参数等。所有参数集中管理，改一处全局生效。

run_itmprec.py 是主入口脚本。解析命令行参数，初始化环境、Agent、记忆模块，运行训练或评估循环。支持 API 模式（调用 DeepSeek）和 Mock 模式（本地模拟）。包含 MockGroundingModel，替代 T-PRA 原版的本地 transformers 模型，实现 API 模式下的轻量运行。


二、Agent 核心层

agent_itmprec.py 是核心 Agent，继承自 T-PRA 的 ReactA2CAgent。保留原版的 ReAct 推理循环、Critic 评估、DPO 训练接口。新增 use_api 分支，覆盖 llm 引用为 DeepSeekInterface；支持多候选评估；注入意图状态和记忆上下文到 prompt，使 LLM 在推理时具备用户意图和历史的上下文感知。


三、动作与意图层

actions.py 是动作体系。定义四种动作：inform（告知信息）、ask（主动提问）、recommend（直接推荐）、explain（解释原因）。包含 ActionSchema（动作结构定义）、ActionParser（解析 LLM 自然语言输出为结构化动作）、ActionExecutor（执行动作并返回环境反馈）。Parser 做了容错设计，LLM 输出格式不稳定时能兜底返回默认动作。

intent.py 是意图引擎。TargetIntention 定义目标物品；IntentState 表示用户当前意图阶段（如浏览中、比较中、已明确）；IntentExtractor 从对话历史中抽取意图状态（调用 DeepSeek 推理）；IntentTracker 维护意图序列演化。confidence 机制：低置信度时（如用户说随便看看），系统主动 ask 澄清，不盲目推荐。这是从被动响应到主动对话的核心支撑。


四、记忆与重排层

memory.py 是三层记忆管理。Sensory Memory（感知记忆，仅当前和上一轮，列表结构，查询最快）；ShortTerm Memory（短期记忆，本次对话全部历史，JSON 格式，包含 role、content、timestamp、intent_state）；LongTerm Memory（长期记忆，跨对话的用户画像和偏好，embedding 向量检索）。IntentMemoryManager 总控写入分层、读取合并、定期清理。相较 T-PRA 的简单历史拼接，大幅提升了上下文管理的精细度。

ipg.py 是 IPG 牵引式重排器。核心公式：score = preference_score × guidance_score。preference_score 衡量用户对该候选物品的偏好程度（基于类型匹配、历史评分等）；guidance_score 衡量该候选对 target_item 的引导作用（属性相似度）。两者乘积确保推荐既让用户满意、又向目标推进。这是将 IPG-Rec 思想引入 T-PRA 框架的关键模块。


五、环境与奖励层

env_dialogue.py 是对话环境封装。接收 Agent 的 action，模拟用户反馈（B 阶段调用 DeepSeek API 模拟用户角色，A 阶段用 mock），计算奖励，返回新观测和 done 标志。四种 action 分别对应不同的奖励逻辑：recommend 命中 target_item 给高奖励并结束对话；ask 获得有效信息给中等奖励；inform 提供信息给小额奖励；explain 解释合理给小额奖励。相比 T-PRA 的通用环境，针对对话式推荐场景做了特化。

rewards.py 是多目标奖励函数。四个目标加权求和：alpha=0.4（推荐准确性，是否命中 target_item）、beta=0.2（对话效率，轮数越少越好）、gamma=0.2（目标牵引度，中间步骤是否朝 target_item 推进）、delta=0.2（用户满意度，反馈正面与否）。gamma 是关键设计，确保 Agent 不只推用户喜欢的，还要向目标靠拢。这解决了 T-PRA 单目标奖励下系统缺乏目标导向的问题。


六、训练与提示层

trajectory.py 是轨迹 DPO 模块。收集整段对话的轨迹（state、action、reward、next_state 序列），支持成对比较（如轨迹 A 三轮成功 vs 轨迹 B 十轮失败，A 为 win，B 为 lose）。生成 preference pair 喂给 T-PRA 的 DPO 接口，实现 trajectory_level 的策略优化，捕捉长程对话策略。这是将 DPO 从单步优化扩展到整段对话级别的关键改进。

prompts.py 是提示工程管理。双轨制设计：保留 T-PRA 原版字符串模板，兼容本地模型；新增 OpenAI messages 格式转换函数，适配 DeepSeek API。包含 actor_prompt（Agent 推理）、critic_prompt（Critic 评估）、intent_extraction_prompt（意图抽取）、user_simulation_prompt（用户模拟）。Prompt 与代码分离，调 prompt 无需改代码，重启即可生效。这是工程上的可维护性设计。


七、数据集适配层

env_movielens.py 是 MovieLens 100K 环境。轻量级实现，不依赖 transformers。包含用户画像、物品集合、交互逻辑。状态表示为向量化形式，便于 Agent 处理。这是针对 API 模式去 transformers 依赖的适配层。

task_movielens.py 是 MovieLens 任务适配器。将用户-目标对格式转换为 T-PRA 框架能理解的 task 格式（初始状态、目标、评估函数）。

convert_movielens.py 是数据转换脚本。读取 MovieLens 100K 原始文件（u.data、u.item、u.user），处理后生成 datamaps（id 映射）、npy（特征矩阵）、pickle（Python 对象）、json（可读格式）。关键功能：生成 target_item 字段，为每个用户-目标对指定明确的推荐目标。这是实现目标导向推荐的数据基础。


八、API 封装层

llm_api.py 是 API 调用封装。三层结构：LLMInterface（基类，定义统一接口 generate）、DeepSeekInterface（真实调用，base_url=https://api.deepseek.com/v1，模型 deepseek-v4-pro，支持 30 秒超时、3 次重试、错误兜底）、MockLLMInterface（Mock 模式，返回预设结果，用于 A 阶段冒烟测试和 API 不可用时降级）。统一接口设计，换模型只需改配置，无需改业务代码。这是 T-PRA 从本地模型到 API 模式的关键适配层。


九、测试层

test_smoke_run.py 是 A 阶段冒烟测试。使用 FakeReactA2CAgent（解决 MagicMock 继承问题）和 MockLLMInterface，不调用任何 API，验证逻辑链路和数据流是否通。这是三阶段验证的第一关。

test_api_quick.py 是 B 阶段 API 连通测试。验证 DeepSeek API 的响应格式和延迟。

test_deepseek_api.py 是 DeepSeek API 专项验证。测试 Actor prompt 和 Critic prompt 的生成效果，检查 LLM 输出是否符合 ReAct 格式（包含 Thought 和 Action）。

test_itmprec.py 是 ITMPRec 模块单元测试。覆盖 intent、ipg、memory、actions 等核心模块的独立功能验证。

test_api_mock.py 是 API 层的 Mock 测试。验证 LLMInterface 的接口一致性，测试 Mock 和真实模式切换是否正常。


实验验证方法

本项目采用三阶段验证策略，确保每一步都有据可查。

A 阶段：冒烟测试（Mock 模式）。目的：不调用任何 API，验证代码逻辑和数据流。方法：使用 FakeReactA2CAgent 替代真正 Agent（解决 MagicMock 继承导致的 _mock_methods AttributeError），MockLLMInterface 替代 DeepSeek，MockGroundingModel 替代 transformers。结果：所有模块逻辑链路通，接口格式正确。

B 阶段：真实 API 测试。配置：模型 deepseek-v4-pro，base_url=https://api.deepseek.com/v1，timeout=30 秒。运行：test_api_quick.py 测试连通性，run_itmprec.py 跑完整链路。结果：API 响应正常，Agent 能生成 Thought，Action 能解析，环境能返回奖励。

C 阶段：问题修复与迭代。修复了以下问题：
（1）target_item 缺失：convert_movielens.py 未生成 target_item，导致 env_dialogue.py 判断命中时报 KeyError。修复：基于用户历史评分，选择评分高但互动少的物品作为目标。
（2）依赖冲突：T-PRA 强依赖 transformers，API 模式下不需要。修复：用 env_movielens.py 替代 T-PRA 的 env 初始化，避免加载 transformers。
（3）openai 和 tiktoken 缺失：手动 pip 安装，使用阿里云镜像。
（4）Connection error：llm_api.py 添加 timeout=30，避免无限等待。
（5）greenlet 编译失败：改用 langchain-core，避免 SQLAlchemy 依赖。
（6）清华镜像 403：换阿里云镜像。


数据集说明

使用 MovieLens 100K 数据集，包含 10 万条评分记录，覆盖约 943 个用户和 1682 部电影。数据转换流程：运行 convert_movielens.py 读取原始文件，生成 datamaps（用户 id 和物品 id 的映射表）、npy（特征向量矩阵）、pickle（Python 序列化对象，便于快速加载）、json（可读格式，便于调试查看），并为每个用户-目标对生成 target_item（推荐目标）。

target_item 的生成策略：基于用户历史评分，选择评分较高但互动次数较少的物品作为目标。既保证有挑战性，又避免选择用户已看过多次的物品。


环境配置

本项目使用 conda 创建独立虚拟环境，避免与系统 Python 环境冲突。


使用方式

方式一：API 模式（推荐，需要 DeepSeek API Key）。先执行 conda activate itmprec，然后 set DEEPSEEK_API_KEY=你的_api_key，最后执行 python run_itmprec.py --mode api --dataset movielens100k --episodes 10。

方式二：Mock 模式（无需 API，验证逻辑）。先执行 conda activate itmprec，然后执行 python run_itmprec.py --mode mock --dataset movielens100k --episodes 10。

方式三：快速 API 连通测试。先执行 conda activate itmprec，然后 set DEEPSEEK_API_KEY=你的_api_key，最后执行 python test_api_quick.py --api_key 你的_api_key。


运行输出说明

本节说明各脚本运行后的典型输出及其含义，便于理解系统运行状态和排查问题。

test_smoke_run.py（A 阶段冒烟测试）

运行命令：python test_smoke_run.py

典型输出示例：

[INFO] 开始冒烟测试...
[INFO] 环境初始化完成
[INFO] Agent 创建成功
[第1轮] Action: ask, Reward: 0.05, Done: False
[第2轮] Action: ask, Reward: 0.05, Done: False
[第3轮] Action: recommend, Reward: 1.0, Done: True
[结果] 命中目标: True, 对话轮数: 3, 总奖励: 1.1
[INFO] 测试通过

输出含义：

Action: ask 表示系统主动询问用户；Action: recommend 表示系统直接推荐物品。Reward: 0.05 表示 ask 动作获得的小额奖励；Reward: 1.0 表示 recommend 命中目标物品的高额奖励。Done: True 表示对话结束。命中目标: True 表示最终推荐成功；对话轮数: 3 表示共进行 3 轮交互；总奖励: 1.1 表示整段对话的累计奖励。测试通过表示所有模块逻辑链路正常，无报错。

如果出现 KeyError 或 AttributeError，说明数据格式或类继承有问题，需检查 convert_movielens.py 是否正确生成 target_item 以及 FakeReactA2CAgent 是否包含必要属性。

test_api_quick.py（B 阶段 API 连通测试）

运行命令：python test_api_quick.py --api_key 你的_api_key

典型输出示例：

[TEST] API base_url: https://api.deepseek.com/v1
[TEST] 模型: deepseek-v4-pro
[OK] API 响应成功，耗时 1.2 秒
[PASS] 响应包含 Thought 和 Action
[OK] Critic 评分: 0.75
[ALL PASSED] API 已就绪

输出含义：

API 响应成功表示网络连接正常，DeepSeek API 可访问。耗时 1.2 秒表示 API 响应时间，若超过 30 秒会触发 timeout。响应包含 Thought 和 Action 表示 LLM 输出符合 ReAct 格式规范。Critic 评分: 0.75 表示 Critic 模块对当前策略的评估分数，越高越好。API 已就绪表示所有连通性检查通过，可进行完整链路测试。

如果出现 Connection error 或 timeout，说明网络问题或 API 繁忙，可稍后重试或切换至 Mock 模式。

test_deepseek_api.py（DeepSeek API 专项验证）

运行命令：python test_deepseek_api.py --api_key 你的_api_key

典型输出示例：

[TEST 1] Actor prompt（推荐 Agent）
  system: 你是一个对话式推荐系统...
  user: 请推荐一部悬疑电影...
[OK] 响应:
Thought: 用户喜欢悬疑片，我推荐《无间道》。
Action: recommend[无间道]
[PASS] 响应包含 Thought 和 Action

[TEST 2] Critic prompt（评估器）
  system: 你是一个评估器...
  user: 评估以下推荐...
[OK] 响应: 0.85
[PASS] 成功解析为 float 类型: 0.85

[ALL PASSED] DeepSeek API 已准备就绪

输出含义：

Actor prompt 测试 Agent 的推理和动作生成能力。响应包含 Thought 和 Action 表示 LLM 正确理解 ReAct 格式。Critic prompt 测试评估器对推荐质量的打分能力。成功解析为 float 表示 Critic 输出可被代码正确读取为数值。如果出现 响应为空 或 无法解析，说明 prompt 需要调优或 API 返回异常。

run_itmprec.py（主程序运行）

运行命令：python run_itmprec.py --mode mock --dataset movielens100k --episodes 10

典型输出示例：

[INFO] 配置加载完成
[INFO] 环境初始化: MovieLens 100K
[INFO] Agent 初始化完成
[Episode 1/10] 目标物品: 电影A (类型: 悬疑)
[第1轮] Thought: 用户意图不明确，先询问偏好
          Action: ask[您喜欢什么类型？]
          Reward: 0.05
[第2轮] Thought: 用户回答喜欢悬疑片
          Action: recommend[电影A]
          Reward: 1.0
[Episode 1] 结果: 命中=True, 轮数=2, 总奖励=1.05
[Episode 2/10] 目标物品: 电影B (类型: 爱情)
...
[汇总] 总回合: 10, 命中数: 8, 命中率: 0.80, 平均轮数: 3.2

输出含义：

Episode 表示第几轮独立对话实验。目标物品显示当前要推荐的目标及其属性。Thought 是 Agent 的推理过程（ReAct 中的 Reasoning）。Action 是 Agent 执行的具体动作。Reward 是环境返回的奖励值。命中表示是否成功推荐目标物品。汇总统计展示整体性能，命中率是核心指标。

Mock 模式下不调用真实 API，所有 LLM 响应由 MockLLMInterface 预设返回，适合快速验证逻辑和调试。API 模式下调用真实 DeepSeek API，结果更真实但消耗 API 额度。

convert_movielens.py（数据转换）

运行命令：python convert_movielens.py

典型输出示例：

[INFO] 读取 MovieLens 100K 数据
[INFO] 用户数: 943, 电影数: 1682, 评分数: 100000
[INFO] 生成 datamaps...
[INFO] 生成特征矩阵 (npy)...
[INFO] 生成 target_item...
[OK] 数据转换完成
[OK] 输出文件: datamaps.pkl, features.npy, data.json

输出含义：

用户数、电影数、评分数显示原始数据规模。生成 datamaps 建立用户 ID 和物品 ID 的连续映射。生成特征矩阵将文本属性转为数值向量。生成 target_item 为每个用户指定推荐目标。转换完成后即可运行主程序。

如果输出 文件未找到，说明 MovieLens 100K 原始数据（u.data、u.item、u.user）未放置到正确目录，需检查数据路径。


设计原则与工程规范

最小侵入式改造：充分利用 T-PRA 已有能力，只新增或覆盖必要部分，不修改 T-PRA 原代码。所有新模块放在 itmprec_prototype 目录下，与 T-PRA 物理隔离。T-PRA 有更新时可直接同步，不影响我的改造。

配置驱动：config.py 集中管理所有参数，改配置不改代码，降低维护成本。

接口统一：llm_api.py 提供统一 LLM 接口，换模型只需改配置；env_movielens.py 保持与 T-PRA BaseEnv 的接口兼容，可无缝切换。

降级兜底：MockLLMInterface 和 MockGroundingModel 提供 API 不可用时的降级方案，确保系统始终可运行。

分阶段验证：A 阶段（冒烟测试）验证逻辑；B 阶段（API 测试）验证真实链路；C 阶段（修复迭代）解决实际问题。每阶段完成后人工确认，再进入下一阶段。这是工程上的稳健性实践，避免一次性引入过多变量导致问题难以定位。


注意事项

（1）API Key 安全：config.py 中 api_key 默认为空字符串，运行时通过环境变量 DEEPSEEK_API_KEY 或命令行参数传入。切勿将真实 Key 硬编码到代码中。

（2）数据文件不上传：MovieLens 原始数据、生成的 npy/pickle/json 文件体积较大，已加入 .gitignore，不会进入代码仓库。首次使用时需运行 convert_movielens.py 生成本地数据。

（3）学术原型声明：本项目为学术考核原型，核心思想（意图建模、目标导向、多轮主动、LLM 模拟用户）已在原型中验证可行。实现做了工程简化，与 ITMPRec 论文的完整系统存在差距，但改进方向和技术路线与论文一致。


