from __future__ import annotations

import ast
import json
import os
from typing import Any, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_MODEL_PATH = "checkpoints/translator/base_models/infocom26_base_model/Meta-Llama-3-8B-Instruct"

DEFAULT_ADAPTER_PATH = "checkpoints/translator/sft_lora/adapter"

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[torch.nn.Module] = None
_DEVICE: Optional[torch.device] = None


def build_llama3_prompt(instruction: str) -> str:
    """
    强约束：要求模型输出必须是 JSON array，且只输出 JSON，不要解释/markdown。
    """
    sys = (
        "You are a network configuration assistant.\n"
        "You MUST output ONLY a valid JSON array.\n"
        "Do NOT output any explanation, markdown, code fences, or extra text.\n"
        "Your response MUST start with '[' and end with ']'.\n"
        "If you cannot comply, output an empty JSON array: []\n"
    )
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{sys}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _lazy_load() -> None:
    global _TOKENIZER, _MODEL, _DEVICE
    if _TOKENIZER is not None and _MODEL is not None and _DEVICE is not None:
        return

    model_path = os.getenv("TRANSLATOR_MODEL_PATH", DEFAULT_MODEL_PATH)
    adapter_path = os.getenv("TRANSLATOR_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        dtype = torch.float16
    else:
        _DEVICE = torch.device("cpu")
        dtype = torch.float32

    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,  # 手动 .to(device)
        low_cpu_mem_usage=True,
    ).to(_DEVICE)

    base = PeftModel.from_pretrained(base, adapter_path)
    base.eval()

    _MODEL = base


def _try_parse_json_or_literal(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _extract_first_bracketed_array(text: str) -> Optional[str]:
    """
    抽取第一个 [...]（支持字符串内引号转义）
    """
    if not text:
        return None
    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
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
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_to_json_array(raw: str) -> str:
    """
    - 优先抽取 [...] 子串
    - 尝试 json.loads / ast.literal_eval
    - 最终兜底：返回 []（保证 reward 至少进入 valid_json 分支）
    """
    raw = (raw or "").strip()
    if not raw:
        return "[]"

    cand = _extract_first_bracketed_array(raw) or raw
    obj = _try_parse_json_or_literal(cand)

    if obj is None:
        # 没有任何可解析结构：兜底给 []
        return "[]"

    # 规范成 list
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return "[]"

    return json.dumps(obj, ensure_ascii=False)


@torch.no_grad()
def generate(
    instruction: Union[str, List[str]],
    num_generations: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """
    输入：instruction（sample["instruction"] 纯文本）
    输出：长度为 num_generations 的 completion list，每个尽力保证是“合法 JSON array 字符串”
    """
    _lazy_load()
    assert _TOKENIZER is not None and _MODEL is not None and _DEVICE is not None

    prompts = instruction if isinstance(instruction, list) else [instruction]

    outs: List[str] = []
    for inst in prompts:
        prompt = build_llama3_prompt(inst)
        inputs = _TOKENIZER(prompt, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[-1]

        for _ in range(int(num_generations)):
            gen_ids = _MODEL.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                pad_token_id=_TOKENIZER.eos_token_id,
                eos_token_id=_TOKENIZER.eos_token_id,
            )

            new_tokens = gen_ids[0][in_len:]
            text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
            outs.append(_normalize_to_json_array(text))

    return outs
