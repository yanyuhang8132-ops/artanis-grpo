from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.rl.reward_fn import compute_reward
from src.translator.api import build_llama3_prompt, _normalize_to_json_array  # 复用强约束

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML not installed. Please `pip install pyyaml`") from e
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be dict, got {type(obj)}")
    return obj


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"JSON must be a list, got {type(obj)}")
    out: List[Dict[str, Any]] = []
    for i, x in enumerate(obj):
        if not isinstance(x, dict):
            raise ValueError(f"JSON[{i}] must be dict, got {type(x)}")
        out.append(x)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def group_normalize(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    GRPO: 同一个 prompt 下的 K 个 reward 做组内归一化 advantage。
    A_i = (r_i - mean) / (std + eps)
    """
    if not rewards:
        return []
    r = np.array(rewards, dtype=np.float32)
    mu = float(r.mean())
    sd = float(r.std())
    if sd < eps:
        return [0.0 for _ in rewards]
    a = (r - mu) / (sd + eps)
    return [float(x) for x in a.tolist()]


REPO_ROOT = Path(__file__).resolve().parents[1]


def _logprob_of_completion(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    completion_text: str,
) -> torch.Tensor:
    """
    返回：completion token 的 logprob 求和（标量 tensor）
    关键点：只统计 completion 部分 token
    """
    # 拼接成完整 assistant 输出
    full = prompt_text + completion_text

    enc_full = tokenizer(full, return_tensors="pt")
    enc_prompt = tokenizer(prompt_text, return_tensors="pt")

    input_ids = enc_full["input_ids"].to(device)
    attn = enc_full["attention_mask"].to(device)

    prompt_len = enc_prompt["input_ids"].shape[-1]
    # logits: [1, T, V]
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits

    # next-token logprob
    # shift: token t 的概率来自 logits[t-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    # completion 部分对应的是：labels 的位置 >= (prompt_len-1)
    start = max(prompt_len - 1, 0)
    comp_token_logp = token_logp[:, start:]  # [1, comp_len]
    return comp_token_logp.sum(dim=-1).squeeze(0)  # scalar


@torch.no_grad()
def _generate_k(
    model,
    tokenizer,
    device: torch.device,
    instruction: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, List[str]]:
    prompt = build_llama3_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[-1]

    outs: List[str] = []
    for _ in range(k):
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen_ids[0][in_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outs.append(_normalize_to_json_array(text))
    return prompt, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config)["grpo"]
    set_seed(int(cfg.get("seed", 42)))

    # ===== data =====
    train_path = REPO_ROOT / cfg["train_file"]
    data = load_json_list(train_path)
    if not data:
        raise ValueError(f"Empty dataset: {train_path}")

    # 保证 raw_idx 存在（你方案A）
    for i, s in enumerate(data):
        if "raw_idx" not in s:
            s["raw_idx"] = i

    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== model paths =====
    model_path = cfg["translator_model_path"]   # base
    adapter_path = cfg["translator_adapter_path"]  # lora
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # ===== tokenizer =====
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ===== policy (trainable) =====
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    policy = PeftModel.from_pretrained(base, adapter_path, is_trainable=True).to(device)
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.train()

    # ===== reference (frozen) =====
    base_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    ref = PeftModel.from_pretrained(base_ref, adapter_path, is_trainable=False).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False



    # ===== optim =====
    lr = float(cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)

    # ===== GRPO params =====
    num_generations = int(cfg.get("num_generations", 4))
    max_steps = int(cfg.get("max_steps", 200))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    temperature = float(cfg.get("temperature", 0.7))
    top_p = float(cfg.get("top_p", 0.95))

    clip_eps = float(cfg.get("clip_eps", 0.2))
    # beta_kl = float(cfg.get("beta_kl", 0.02))
    beta_kl = 0.0
    grad_accum = int(cfg.get("grad_accum", 1))
    save_every = int(cfg.get("save_every", 50))

    logs = []
    step = 0
    cursor = 0
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        sample = data[cursor % len(data)]
        cursor += 1

        instruction = sample.get("instruction", "")
        raw_idx = int(sample["raw_idx"])

        # 1) generate K samples
        with torch.no_grad():
            prompt_text, completions = _generate_k(
                policy, tok, device, instruction,
                k=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # 2) rewards
        rewards = [compute_reward(c, raw_idx=raw_idx, prompt=instruction) for c in completions]
        adv = group_normalize(rewards)  # list[float]

        # 3) logp_old (detach)
        with torch.no_grad():
            logp_old = torch.stack([
                _logprob_of_completion(policy, tok, device, prompt_text, c) for c in completions
            ])  # [K]
            logp_ref = torch.stack([
                _logprob_of_completion(ref, tok, device, prompt_text, c) for c in completions
            ])  # [K]

        # 4) policy update (one PPO-style step)
        logp_new = torch.stack([
            _logprob_of_completion(policy, tok, device, prompt_text, c) for c in completions
        ])  # [K]

        A = torch.tensor(adv, device=device, dtype=logp_new.dtype)  # [K]
        ratio = torch.exp(logp_new - logp_old)  # [K]

        unclipped = ratio * A
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
        ppo_obj = torch.min(unclipped, clipped).mean()

        # KL 近似：E[logp_new - logp_ref]
        kl = (logp_new - logp_ref).mean()
        kl = torch.tensor(0.0, device=device, dtype=logp_new.dtype)

        loss = -(ppo_obj) + beta_kl * kl
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        log = {
            "step": step,
            "raw_idx": raw_idx,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_adv": float(np.mean(adv)),
            "loss": float(loss.detach().cpu().item()),
            "ppo_obj": float(ppo_obj.detach().cpu().item()),
            "kl": float(kl.detach().cpu().item()),
            "completion_prefix": (completions[0] or "")[:160],
        }
        logs.append(log)

        if step % 1 == 0:
            print(
                f"[step {step}] raw_idx={raw_idx} "
                f"reward={log['mean_reward']:.4f}±{log['std_reward']:.4f} "
                f"loss={log['loss']:.4f} ppo={log['ppo_obj']:.4f} kl={log['kl']:.4f}"
            )

        if (step + 1) % save_every == 0:
            ckpt_dir = out_dir / f"checkpoint-step{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            with (out_dir / "train_logs.json").open("w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"[save] {ckpt_dir}")

        step += 1

    with (out_dir / "train_logs.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote: {out_dir / 'train_logs.json'}")


if __name__ == "__main__":
    main()
