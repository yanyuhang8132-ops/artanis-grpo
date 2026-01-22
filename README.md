

README

一个面向意图驱动网络配置更新的闭环原型：
Translator（大模型生成配置 JSON） → Verifier（多模态验证器评分） → Reward（结合格式+SLA约束的奖励） → GRPO（基于验证反馈的策略优化） → 推理与满足率评估。

当前版本支持输出两类配置更新（block_type）：

- acl：ACL rule（含 rate_limit Mbps）
- slb：SLB 负载均衡（virtual_ip + real_servers/weight）

---

1. 项目结构（核心目录）

    artanis-grpo-release/
    ├─ src/
    │  ├─ translator/
    │  │  ├─ api.py                      # 统一的生成接口 generate(prompt, n)
    │  │  └─ legacy/...                  # 旧版推理/训练脚本与数据
    │  ├─ verifier/
    │  │  ├─ api.py                      # score(raw_idx, policy_configs, ...)
    │  │  └─ legacy/CPV/...              # CPV legacy verifier（verify_sample）
    │  └─ rl/
    │     ├─ reward_fn.py                # compute_reward：格式+SLA+base_score融合
    │     └─ grpo_reward_wrapper.py      # GRPORewardWrapper：对齐 raw_idx/raw_idxs
    ├─ scripts/
    │  ├─ train_grpo.py                  # dry-run: 仅跑生成→reward→adv→日志
    │  ├─ train_grpo_rl.py               # RL训练版：计算logprob+PPO/GRPO更新
    │  ├─ train_grpo_rl.py               # RL训练版：计算logprob+PPO/GRPO更新
    │  └─ req_satisfation.py             # 最终输出“满足率”统计脚本
    ├─ data/
    │  └─ processed/grpo_json/
    │     ├─ train_30.json
    │     └─ test_30.json
    ├─ checkpoints/
    │  └─ translator/
    │     ├─ sft_lora/...                # SFT LoRA adapter
    │     └─ grpo_lora/checkpoint-200/   # GRPO输出 adapter 权重目录
    └─ result/                           # 推理输出与评估输出

---

2. 核心概念与工作流

2.1 Translator

输入：instruction（包含拓扑、设备配置、当前流、意图）
输出：严格 JSON array，每个元素为一个 block（acl 或 slb），示例：

    [
      {
        "block_type": "acl",
        "device": "X",
        "acl_rules": {
          "rules": [
            {"permit": "1->2", "rate_limit": "60 Mbps", "interface": "Fa0/0"}
          ]
        },
        "type": "added"
      }
    ]

注意：非 JSON例如自然语言解释、Markdown会被 reward_fn 判为 invalid_json，奖励极低。

2.2 Verifier

- src/verifier/api.py::score(raw_idx, policy_configs)：返回 [0,1] 的 base score。
- 运行 CPV 的 verify_sample 评分逻辑。

1. base = verifier.score(raw_idx, ...)
2. valid_json：格式有效性（无效则直接很低）
3. sla：从 prompt 里解析 SLA（如 “minimum bandwidth of 47 Mbps”）
4. limits：从 completion JSON 内抽取 ACL 的 rate_limit（以及扩展的 SLB/ACL字段）
5. quality：根据 limits 是否满足/接近 SLA 得到质量分
6. reward：最终合成（已验证 “10 Mbps vs 100 Mbps” 会拉开差距）

2.4 GRPO

- dry-run（scripts/train_grpo.py）：不更新权重，只验证“生成→打分→adv→日志”链路。
- RL训练（scripts/train_grpo_rl.py）：在 LoRA 参数上做更新（类似 PPO/GRPO + KL penalty）。

---

3. 环境与启动建议

3.1 运行入口必须在项目根目录

    cd ~/USER_YCJ/MyCode/artanis-grpo-release
    export PYTHONPATH=$(pwd)

后续所有 python scripts/xxx.py ... 都在根目录运行。

3.2 Conda / CUDA

为了避免 SSL 握手失败（SSLEOFError）和依赖版本冲突，建议按照以下拆解步骤进行安装：

3.2.1 创建基础环境

Bash

    # 修改 yml 后创建环境
    conda env create -f conda_env_artanis.yml -n artanis
    conda activate artanis

3.2.2 手动安装 PyTorch (针对 CUDA 11.8)

由于官方源在部分网络下不稳定，使用国内镜像源手动安装指定版本的 Torch。务必在安装前确保环境已激活：

Bash

    # 使用清华源或阿里源安装 torch 2.1.1
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      --trusted-host pypi.tuna.tsinghua.edu.cn

3.2.3 安装 Flash-Attention

flash-attn 必须在 Torch 安装完成后进行。为了避免 setup.py 找不到 torch 的问题，需使用 --no-build-isolation 参数：

Bash

    pip install flash-attn==2.7.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

---

4. 配置文件与关键参数

本项目使用 configs/grpo.yaml 统一管理数据路径、模型路径、推理超参与训练超参。配置字段位于 grpo: 节点下，常用字段说明如下。

4.1 数据与索引对齐

- train_file / test_file：训练与测试数据（JSON list）。每条样本至少包含 instruction、output（可为空或用于 teacher completion）。
- raw_idx_field：数据中用于映射验证器场景索引的字段名。默认可设为 null，此时脚本将使用样本在文件中的顺序索引 sample_idx 作为 raw_idx。
  - 约定：验证器侧数据集（CPV augmented_dataset）按 scenario_id 顺序可索引；raw_idx 直接对应 scenario_id 或列表下标。
- completion_field：从数据中读取参考输出字段名（通常为 output）。在 dry-run 中可用于构造生成候选；在 RL 训练中一般由 Translator 实际生成覆盖。

4.2 Translator 模型路径

- translator_model_path：基座模型目录（例如 Meta-Llama-3-8B-Instruct）。
- translator_adapter_path：LoRA adapter 目录（SFT 或 GRPO 的训练产物）。
  - 推荐目录结构（与 HuggingFace/PEFT 保存习惯一致）：
    - checkpoints/translator/grpo_lora/checkpoint-200/
      - adapter_model.safetensors
      - adapter_config.json

4.3 推理与生成超参

- max_new_tokens：最大生成长度（从 516起，显存不足时可降到 64/32）。
- temperature / top_p：采样参数（影响多样性与 JSON 合规率）。
- num_generations：每条样本生成候选数（用于 GRPO 的组内相对优势计算；也用于“低分重试”策略的候选池）。

4.4 训练超参（RL）

- lr：LoRA 可训练参数的学习率。
- beta_kl：KL 惩罚系数（约束策略偏离参考策略）。
- clip_eps：PPO/GRPO 风格裁剪范围。
- grad_accum：梯度累积步数（显存不足时可增大，以时间换显存）。
- save_every：保存间隔（按 step 计）。
- device：cuda 或 cpu。

---

5. 运行流程

本节给出推荐的端到端执行顺序：环境 → dry-run → 推理（含验证重试）→ 满足率评估 →（可选）RL 训练。

5.1 环境检查

在项目根目录执行：

    cd ~/USER_YCJ/MyCode/artanis-grpo-release
    export PYTHONPATH=$(pwd)
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

建议同时验证 Translator 与 Verifier API 可导入：

    python -c "from src.translator.api import generate; from src.verifier.api import score; print('import_ok')"

---

6. Dry-run：验证闭环是否可用

dry-run 目标：不更新模型参数，仅验证 Translator → Reward/Verifier → GRPO wrapper → 日志落盘 全链路可跑通。

    python scripts/train_grpo.py --config configs/grpo.yaml

dry-run 输出目录由配置项 out_dir 决定，典型产物：

- experiments/runs/<run_name>/dryrun_logs.json
- experiments/runs/<run_name>/run_config.json

---

7. 推理与验证：生成、过滤与落盘

7.1 一键推理(验证驱动重试策略)

推理脚本读取 test_30.json，逐条调用 src/translator/api.py 的 generate() 生成 model_output，并写出：

- result/generated_output_30.jsonl：逐行 JSON，便于流式处理
- result/generated_output_30_summary.json：汇总统计（平均时延、均值奖励、合规率等）

输出字段规范：

- id：样本编号（从 1 开始或与 sample_idx 对齐均可，保持一致即可）
- user_input：输入指令（instruction 或其摘要）
- reference：参考输出（数据中的 output 字段）
- model_output：模型生成的 JSON array 字符串
- inference_time：单条推理耗时（秒，float）

    python scripts/infer_translator.py   --input data/processed/grpo_json/test_30.json   --base_model src/translator/legacy/INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct   --adapter checkpoints/translator/grpo_lora/checkpoint-200   --out_dir ./result

为提升输出合规率与满足率，推理阶段可加入验证器反馈回路：

- 对每条样本最多生成 4 次；
- 每次生成后计算验证得分（或 reward）；
- 若分数低于阈值则重新生成；
- 达到阈值则提前停止；否则使用最后一次结果落盘。

这一策略的优点：

- 以少量额外推理开销换取显著更高的合规率与 SLA 通过率；
- 不依赖 RL 训练即可提升基线效果；
- 与后续 GRPO/RL 的目标一致，便于对齐“训练前后”的评估口径。

---

8. 满足率评估

项目提供 scripts/req_satisfation.py 用于对推理输出进行满足率统计与对比分析。其典型输入为：

- result/generated_output_30.jsonl

运行：

    python scripts/req_satisfation.py

典型输出包括：

- SLA/约束满足率统计表（CSV）
- 若干中间构造文件（例如拓扑矩阵、容量矩阵的对比结果）

---

9. GRPO 训练

当显存允许时，可运行 scripts/train_grpo_rl.py 执行 LoRA 参数的 GRPO 风格更新。训练产物为 PEFT LoRA adapter 权重，保存到：

- checkpoints/translator/grpo_lora/checkpoint-<step>/

训练启动：

    python scripts/train_grpo_rl.py --config configs/grpo.yaml

训练保存行为由以下因素共同决定：

- save_every：保存间隔（step）
- translator_adapter_path：输出/加载路径策略

9.1 显存与稳定性建议

若出现 OOM，通常优先调整：

- 降低 max_new_tokens
- 降低 num_generations
- 提高 grad_accum
- 采用 fp16/bf16（取决于 GPU 支持）
- 避免一次性对多 completion 批量做 logprob（可改为逐条/分块计算）

---

10. 输出格式约束（ACL 与 SLB）

10.1 ACL block

必需字段：

- block_type: "acl"
- device
- acl_rules.rules[]，每条 rule 建议包含：
  - permit（字符串）
  - rate_limit（例如 "60 Mbps"）
  - interface（例如 "Fa0/0"）
- type（例如 "added"）

10.2 SLB block

必需字段：

- block_type: "slb"
- device
- slb_config.virtual_ip（例如 "10.0.0.75 / 255.255.255.255")
- slb_config.real_servers[]，每个元素包含：
  - ip
  - weight
- type
