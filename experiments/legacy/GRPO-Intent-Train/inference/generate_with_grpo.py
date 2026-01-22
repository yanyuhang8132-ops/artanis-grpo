import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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

def extract_first_json(text: str):
    """
    从文本中截取第一个完整 JSON（数组[]或对象{}），返回 JSON 字符串；找不到/不完整返回 None。
    兼容偶发的 ```json fence 或前后解释文字。
    """
    s = text.strip()
    if not s:
        return None

    # 找到第一个 '[' 或 '{'
    start = None
    open_ch = close_ch = None
    for i, ch in enumerate(s):
        if ch == "[":
            start = i
            open_ch, close_ch = "[", "]"
            break
        if ch == "{":
            start = i
            open_ch, close_ch = "{", "}"
            break
    if start is None:
        return None

    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : j + 1].strip()

    return None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--grpo_adapter", type=str, required=True,
                        help="Final GRPO adapter dir")

    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True)

    args = parser.parse_args()

    if args.out_jsonl is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_jsonl = f"output/model_output_grpo_{ts}.jsonl"
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.grpo_adapter, is_trainable=False)
    model.eval()

    ds = Dataset.from_json(args.test_json)
    need = {"instruction", "raw_idx"}
    if not need.issubset(set(ds.column_names)):
        raise ValueError(f"test_json missing fields {need}, got {ds.column_names}")

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            instruction = ex["instruction"]
            raw_idx = int(ex["raw_idx"])

            prompt = build_llama3_prompt(instruction)
            toks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

            out = model.generate(
                **toks,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, 
            )

            full = tokenizer.decode(out[0], skip_special_tokens=True)
            prompt_text = tokenizer.decode(toks["input_ids"][0], skip_special_tokens=True)
            completion_raw = full[len(prompt_text):].strip()
            completion = extract_first_json(completion_raw) or completion_raw


            rec: Dict[str, Any] = {
                "id": i,
                "raw_idx": raw_idx,
                "instruction": instruction,
                "model_output_raw": completion_raw,
                "model_output": completion,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
