GRPO-Intent-Train/
├── model/
│   ├── base_model/                      # Meta-Llama-3-8B-Instruct 本地目录（或软链接）
│   └── lora_adapter_sft/                # SFT 微调得到的 LoRA adapter（起点）
├── data/
│   ├── train_70.json                    # 每条包含 instruction + raw_idx (+可选 output)
│   └── test_70.json
├── reward/
│   ├── reward_fn.py                     # verifier reward: completion(JSON)+raw_idx -> score(0~1)
│   └── grpo_reward_wrapper.py           # TRL适配器: prompts/completions + raw_idx -> List[float]
├── scripts/
│   └── run_grpo_lora_train.py           # ✅ GRPO训练入口（最终版）
├── inference/
│   └── generate_with_grpo.py            # ✅ GRPO后模型生成入口（最终版）
└── output/
    └── grpo/
        └── exp_70_seed42/
            ├── adapter/                 # ✅ 训练产物（LoRA adapter）
            ├── tokenizer/               # 可选：tokenizer保存
            └── run_config.json          # 可选：记录超参
