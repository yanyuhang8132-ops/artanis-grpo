from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# verifier
from src.verifier.api import score as verifier_score


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got: {type(data)}")
    return data


def _as_text_reference(ref: Any) -> str:
    if isinstance(ref, (dict, list)):
        return json.dumps(ref, ensure_ascii=False)
    if ref is None:
        return ""
    return str(ref)


def _trim_and_extract_first_json_array(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    start = s.find("[")
    if start < 0:
        return s

    depth = 0
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end > start:
        cand = s[start : end + 1].strip()
        return cand
    return s


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _build_policy_configs_from_completion(obj: Any) -> Dict[str, Any]:
    """
    completion JSON array -> policy_configs dict for verifier:
      {
        "acl_rules": { ... }   # 合并所有 acl block 的 rules
        "lb_rules":  { ... }   # 这里用 slb_config 简单塞进去（你现有 key 是 lb_rules）
      }
    """
    acl_all: List[Dict[str, Any]] = []
    slb_all: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return {"acl_rules": {"rules": []}, "lb_rules": {}}

    for block in obj:
        if not isinstance(block, dict):
            continue
        bt = str(block.get("block_type", "")).lower()

        if bt == "acl":
            acl = block.get("acl_rules", {})
            if isinstance(acl, dict):
                rules = acl.get("rules", [])
                if isinstance(rules, list):
                    for r in rules:
                        if isinstance(r, dict):
                            acl_all.append(r)

        if bt in ("slb", "lb"):
            slb = block.get("slb_config", None)
            if isinstance(slb, dict):
                # 你可以按需更严格：只取 real_servers/virtual_ip
                slb_all.append(slb)

    policy = {
        "acl_rules": {"rules": acl_all},
        "lb_rules": {"slb_configs": slb_all},
    }
    return policy


def build_model(base_model_path: str, adapter_path: str, device: str) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_cuda = ("cuda" in device) and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    if not use_cuda:
        model = model.to("cpu")
    return tok, model


@torch.no_grad()
def generate_one(
    tok,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tok(prompt, return_tensors="pt", padding=False)
    dev = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    out = tok.decode(gen[0], skip_special_tokens=True)
    if out.startswith(prompt):
        out = out[len(prompt) :]
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/grpo_json/test_30.json")
    ap.add_argument("--base_model", default="src/translator/legacy/INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/translator/grpo_lora/checkpoint-200")
    ap.add_argument("--out_dir", default="result")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--min_score", type=float, default=0.05, help="verifier score threshold to accept")
    ap.add_argument("--max_tries", type=int, default=4, help="max generations per sample")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "generated_output_30.jsonl"
    summary_path = out_dir / "generated_output_30_summary.json"

    tok, model = build_model(args.base_model, args.adapter, args.device)

    data = load_json_list(in_path)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    records: List[Dict[str, Any]] = []
    times: List[float] = []
    v_scores: List[float] = []
    accepted_cnt = 0
    total_tries = 0

    t0_all = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as fw:
        for idx, sample in enumerate(data, start=1):
            raw_idx = int(sample.get("raw_idx", idx - 1))

            user_input = sample.get("instruction", "") or ""
            reference = _as_text_reference(sample.get("output", ""))

            chosen_output = ""
            chosen_vscore = 0.0
            chosen_time = 0.0
            chosen_try = args.max_tries
            last_error = ""

            for t in range(1, args.max_tries + 1):
                total_tries += 1
                t0 = time.perf_counter()
                raw_out = generate_one(
                    tok,
                    model,
                    user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                t1 = time.perf_counter()
                infer_time = round(t1 - t0, 3)

                model_output = _trim_and_extract_first_json_array(raw_out)
                parsed = _safe_json_loads(model_output)

                if parsed is None:
                    v = 0.0
                    last_error = "invalid_json"
                else:
                    policy_configs = _build_policy_configs_from_completion(parsed)
                    try:
                        v = float(verifier_score(raw_idx, policy_configs=policy_configs, debug=False))
                        last_error = ""
                    except Exception as e:
                        v = 0.0
                        last_error = f"verifier_error:{type(e).__name__}"

                # update best (keep last-by-default, but also keep score/time)
                chosen_output = model_output
                chosen_vscore = v
                chosen_time = infer_time
                chosen_try = t

                # accept early
                if v >= float(args.min_score):
                    accepted_cnt += 1
                    break

            rec = {
                "id": idx,
                "raw_idx": raw_idx,
                "user_input": user_input,
                "reference": reference,
                "model_output": chosen_output,
                "inference_time": chosen_time,
                "verifier_score": round(float(chosen_vscore), 6),
                "tries": int(chosen_try),
                "last_error": last_error,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            times.append(float(chosen_time))
            v_scores.append(float(chosen_vscore))

    total_time = time.perf_counter() - t0_all
    times_sorted = sorted(times)
    v_sorted = sorted(v_scores)

    def pct(arr: List[float], p: float) -> float:
        if not arr:
            return 0.0
        k = int(round((len(arr) - 1) * p))
        return float(arr[k])

    summary = {
        "note": "verifier-driven regenerate: per-sample up to max_tries; accept if verifier_score >= min_score, else keep last try",
        "input": str(in_path),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "count": len(records),
        "min_score": float(args.min_score),
        "max_tries": int(args.max_tries),
        "accepted_rate": accepted_cnt / max(1, len(records)),
        "avg_tries": total_tries / max(1, len(records)),
        "time_sec_total": round(total_time, 3),
        "time_sec_avg": round(sum(times) / max(1, len(times)), 3),
        "time_sec_p50": round(pct(times_sorted, 0.50), 3),
        "time_sec_p90": round(pct(times_sorted, 0.90), 3),
        "verifier_score_avg": round(sum(v_scores) / max(1, len(v_scores)), 6),
        "verifier_score_p50": round(pct(v_sorted, 0.50), 6),
        "verifier_score_p90": round(pct(v_sorted, 0.90), 6),
        "outputs": {
            "jsonl": str(jsonl_path),
            "summary": str(summary_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {jsonl_path}")
    print(f"[ok] wrote: {summary_path}")
    print(f"[summary] accepted_rate={summary['accepted_rate']:.3f} avg_tries={summary['avg_tries']:.2f} "
          f"v_avg={summary['verifier_score_avg']:.4f} time_avg={summary['time_sec_avg']:.3f}s")


if __name__ == "__main__":
    main()
