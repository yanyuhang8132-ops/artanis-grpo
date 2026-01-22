from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.rl.grpo_reward_wrapper import GRPORewardWrapper


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    return data


def make_generations(base_completion, k: int) -> List[str]:
    """
    dry-run: 支持 base_completion 是 str / list / dict
    - str: 原样使用
    - list/dict: json.dumps 成字符串
    """
    if base_completion is None:
        base_completion_str = ""
    elif isinstance(base_completion, str):
        base_completion_str = base_completion.strip()
    else:
        # list/dict/other -> stringify
        try:
            import json
            base_completion_str = json.dumps(base_completion, ensure_ascii=False)
        except Exception:
            base_completion_str = str(base_completion)

    if not base_completion_str:
        base_completion_str = "[{'block_type':'acl','device':'R1','acl_rules':{'rules':[]}}]"

    return [base_completion_str for _ in range(k)]



def group_normalize(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """
    GRPO 的组内归一化 advantage：A_i = (r_i - mean(r)) / (std(r)+eps)
    """
    arr = np.array(rewards, dtype=np.float32)
    mu = float(arr.mean())
    sd = float(arr.std())
    adv = (arr - mu) / (sd + eps)
    return adv.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config).get("grpo", {})
    set_seed(int(cfg.get("seed", 42)))

    train_path = REPO_ROOT / cfg["train_file"]
    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_idx_field = cfg.get("raw_idx_field", "raw_idx")  # 你可以在yaml里设成 null/"" 表示不用
    completion_field = cfg.get("completion_field", "output")
    num_generations = int(cfg.get("num_generations", 4))
    max_steps = int(cfg.get("max_steps", 5))

    data = load_json_list(train_path)

    # Ensure every sample has `raw_idx` for downstream verifier/wrapper compatibility
    for i, s in enumerate(data):
        if "raw_idx" not in s:
            s["raw_idx"] = i

    if not data:
        raise ValueError(f"Empty dataset: {train_path}")

    reward_fn = GRPORewardWrapper(verbose=True, log_first_n_zero=2)

    logs = []
    for step in range(max_steps):
        sample_idx = step % len(data)
        sample = data[sample_idx]

        prompt = sample.get("instruction", "") or ""

        # raw_idx: 优先用字段，否则退化为样本下标
        raw_idx = None
        if isinstance(raw_idx_field, str) and raw_idx_field:
            raw_idx = sample.get(raw_idx_field, None)
        if raw_idx is None:
            raw_idx = sample_idx

        # base_completion: 可能是 list/dict/string，都转成 string（只用于 mock 多样本）
        base_completion = sample.get(completion_field, "")
        if isinstance(base_completion, (list, dict)):
            try:
                base_completion = json.dumps(base_completion, ensure_ascii=False)
            except Exception:
                base_completion = str(base_completion)
        else:
            base_completion = str(base_completion)

        from src.translator.api import generate; completions = generate(prompt, num_generations=num_generations)

        # wrapper 期望：prompts/completions/raw_idxs 对齐长度
        prompts = [prompt] * len(completions)
        raw_idxs = [int(raw_idx)] * len(completions)

        rewards = reward_fn(prompts, completions, raw_idxs=raw_idxs)
        adv = group_normalize(rewards)

        log = {
            "step": int(step),
            "sample_idx": int(sample_idx),
            "raw_idx": int(raw_idx),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "rewards": rewards,
            "advantages": adv,
            "completion_prefix": (completions[0] or "")[:160],
        }
        logs.append(log)

        print(
            f"[step {step}] sample_idx={sample_idx} raw_idx={raw_idx} "
            f"mean_reward={log['mean_reward']:.4f} std={log['std_reward']:.4f}"
        )

    with (out_dir / "dryrun_logs.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote: {out_dir / 'dryrun_logs.json'}")


if __name__ == "__main__":
    main()
