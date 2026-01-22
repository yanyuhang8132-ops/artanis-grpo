import os
import json
import argparse
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from trl import GRPOConfig, GRPOTrainer

from reward.grpo_reward_wrapper import GRPORewardWrapper


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_llama3_prompt(instruction: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "你是一个网络配置助手，请根据网络状态和意图生成配置更新。\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def ensure_exists(path: str, name: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def load_dataset_with_prompt(path: str, limit: Optional[int] = None) -> Dataset:
    ds = Dataset.from_json(path)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    needed = {"instruction", "raw_idx"}
    missing = needed - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset {path} missing fields: {missing}. Existing: {ds.column_names}")

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": build_llama3_prompt(ex["instruction"]),
            "raw_idx": int(ex["raw_idx"]),
        }

    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c not in ("instruction", "raw_idx")])
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "raw_idx")])
    return ds


def main():
    parser = argparse.ArgumentParser()

    # model paths
    parser.add_argument("--base_model", type=str,
                        default="INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--init_adapter", type=str,
                        default="INFOCOM26-Intent-Model/output/llama3_network_config_lora/adapter")

    # data
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--eval_json", type=str, default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--eval_limit", type=int, default=200)

    # output
    parser.add_argument("--output_root", type=str, default="output/grpo")
    parser.add_argument("--exp_name", type=str, default=None)

    # training
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    # grpo generation
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.0)  # KL penalty weight

    # logging/save
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # reward debug
    parser.add_argument("--reward_verbose", action="store_true", default=False)
    parser.add_argument("--reward_log_first_n_zero", type=int, default=0)

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    base_model = ensure_exists(args.base_model, "base_model")
    init_adapter = ensure_exists(args.init_adapter, "init_adapter")
    train_json = ensure_exists(args.train_json, "train_json")
    eval_json = None if args.eval_json is None else ensure_exists(args.eval_json, "eval_json")

    if args.exp_name is None:
        tag = os.path.splitext(os.path.basename(train_json))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{tag}_seed{args.seed}_{ts}"

    exp_dir = os.path.join(args.output_root, f"exp_{args.exp_name}")
    adapter_out = os.path.join(exp_dir, "adapter")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    tok_out = os.path.join(exp_dir, "tokenizer")
    os.makedirs(adapter_out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tok_out, exist_ok=True)

    print("=== GRPO Train (Final) ===")
    print(f"base_model    : {base_model}")
    print(f"init_adapter  : {init_adapter}")
    print(f"train_json    : {train_json}")
    print(f"eval_json     : {eval_json}")
    print(f"exp_dir       : {exp_dir}")
    print(f"adapter_out   : {adapter_out}")
    print(f"ckpt_dir      : {ckpt_dir}")
    print("==========================")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(tok_out)

    # policy: base + init_adapter (trainable)
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    policy.config.use_cache = False
    policy = PeftModel.from_pretrained(policy, init_adapter, is_trainable=True)

    # ref: base + init_adapter (frozen)
    ref = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ref.config.use_cache = False
    ref = PeftModel.from_pretrained(ref, init_adapter, is_trainable=False)
    ref.eval()

    train_ds = load_dataset_with_prompt(train_json, limit=args.train_limit)
    eval_ds = None
    if eval_json is not None:
        eval_ds = load_dataset_with_prompt(eval_json, limit=args.eval_limit)

    reward_func = GRPORewardWrapper(
        raw_id_field_candidates=["raw_idx"],
        verbose=args.reward_verbose,
        log_first_n_zero=args.reward_log_first_n_zero,
    )

    cfg = GRPOConfig(
        output_dir=ckpt_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,

        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_new_tokens,
        beta=args.beta,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=policy,
        ref_model=ref,
        args=cfg,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存最终 adapter
    trainer.save_model(adapter_out)
    policy.save_pretrained(adapter_out)

    run_cfg = vars(args).copy()
    run_cfg.update({
        "resolved_base_model": base_model,
        "resolved_init_adapter": init_adapter,
        "exp_dir": exp_dir,
        "adapter_out": adapter_out,
        "ckpt_dir": ckpt_dir,
        "tokenizer_out": tok_out,
        "AUG_DATA_PATH": os.getenv("AUG_DATA_PATH", "data/augmented_dataset_all.json"),
        "cuda": bool(torch.cuda.is_available()),
        "num_gpus": torch.cuda.device_count(),
    })
    with open(os.path.join(exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print("\n[OK] GRPO training done.")
    print(f"Final adapter: {adapter_out}")
    print("NOTE: In inference, load ONLY base_model + this final adapter (do NOT stack init_adapter again).")


if __name__ == "__main__":
    main()
