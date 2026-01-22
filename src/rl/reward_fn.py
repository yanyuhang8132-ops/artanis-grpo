# src/rl/reward_fn.py
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.verifier.api import score as verifier_score


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _safe_json_loads(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_sla_mbps_from_prompt(prompt: Optional[str]) -> Optional[float]:
    if not prompt:
        return None
    # 兼容：minimum bandwidth of 47 Mbps / min_bw: 47 Mbps / SLA ... 47 Mbps
    patterns = [
        r"minimum bandwidth of\s*([0-9]+(?:\.[0-9]+)?)\s*Mbps",
        r"min_bw:\s*([0-9]+(?:\.[0-9]+)?)\s*Mbps",
        r"SLA[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*Mbps",
    ]
    for p in patterns:
        m = re.search(p, prompt, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _parse_mbps(s: Any) -> Optional[float]:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_completion_limits_mbps(obj: Any) -> List[float]:
    """
    从 completion JSON 里抽取可能表示带宽/限速的数值（Mbps）。
    支持：
      - ACL: acl_rules.rules[].rate_limit
      - SLB: 只读取“明确语义是带宽”的字段（避免把 weight 当 Mbps）
    """
    limits: List[float] = []

    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        return limits

    def _extract_acl_limits_from_acl_rules(acl_rules: Any) -> None:
        if not isinstance(acl_rules, dict):
            return
        rules = acl_rules.get("rules", [])
        if not isinstance(rules, list):
            return
        for r in rules:
            if not isinstance(r, dict):
                continue
            mbps = _parse_mbps(r.get("rate_limit"))
            if mbps is not None:
                limits.append(mbps)

    def _extract_slb_limits_from_slb_config(slb_cfg: Any) -> None:
        """
        SLB 里默认不认为 weight 是带宽，所以严格只看带宽语义字段。
        """
        if not isinstance(slb_cfg, dict):
            return

        # 仅抓“明显是带宽”的字段名（你后续要扩展也在这里加）
        bw_keys = [
            "rate_limit",
            "bandwidth",
            "min_bw",
            "sla_bw",
            "bw",
            "limit_mbps",
            "min_bandwidth",
            "max_bandwidth",
            "capacity",
        ]

        for k in bw_keys:
            if k in slb_cfg:
                mbps = _parse_mbps(slb_cfg.get(k))
                if mbps is not None:
                    limits.append(mbps)

        # 有些人会把 QoS 信息塞到 slb_config.qos / slb_config.qos_policy 里
        for qos_key in ["qos", "qos_policy", "policy", "sla"]:
            q = slb_cfg.get(qos_key)
            if isinstance(q, dict):
                for k in bw_keys:
                    if k in q:
                        mbps = _parse_mbps(q.get(k))
                        if mbps is not None:
                            limits.append(mbps)

        # 注意：real_servers[].weight 是“负载分配权重”，不是 Mbps；这里明确不解析

    for block in obj:
        if not isinstance(block, dict):
            continue

        bt = str(block.get("block_type", "")).lower()

        if bt == "acl":
            _extract_acl_limits_from_acl_rules(block.get("acl_rules", {}))

        elif bt == "slb":
            # 有些 slb block 可能同时带 acl_rules（你贴的例子里 acl_rules: None）
            _extract_acl_limits_from_acl_rules(block.get("acl_rules"))
            _extract_slb_limits_from_slb_config(block.get("slb_config", {}))

    return limits



@lru_cache(maxsize=4096)
def _cached_base_score(raw_idx: int) -> float:
    # 这里用你当前已经跑通的 verifier 输出（不依赖 completion）
    return float(verifier_score(int(raw_idx), None, debug=False))


def compute_reward(
    completion: str,
    raw_idx: int,
    prompt: Optional[str] = None,
    debug: bool = False,
) -> float:
    """
    completion-aware reward:
      - base: verifier_score(raw_idx, None)  (场景难度/风险基线，0~1)
      - quality: completion 是否满足 prompt 中 SLA（启发式）
    """
    base = _clip01(_cached_base_score(int(raw_idx)))

    # 1) completion 必须是合法 JSON array（或 dict）
    parsed = _safe_json_loads(completion)
    valid = 1.0 if (isinstance(parsed, list) or isinstance(parsed, dict)) else 0.0
    if valid <= 0.0:
        if debug:
            print(f"[reward] invalid_json raw_idx={raw_idx} base={base:.4f}")
        # 不合法直接给一个很低的分（仍保留 base 的一点点信息）
        return _clip01(0.05 * base)

    # 2) SLA matching
    sla = _extract_sla_mbps_from_prompt(prompt)
    limits = _extract_completion_limits_mbps(parsed)

    # 默认：没有 SLA 就只看合法性（先跑通训练）
    if sla is None:
        quality = 0.6 * valid
    else:
        if not limits:
            # 写了 JSON 但没有任何 rate_limit：通常不可能满足 SLA
            quality = 0.1 * valid
        else:
            best = max(limits)
            # 如果 best >= SLA 给高分；否则按比例给分（但不要太高）
            if best >= sla:
                quality = 1.0
            else:
                quality = max(0.0, best / max(1e-6, sla)) * 0.6  # 上限 0.6

    # 3) 合成：让 reward 真正受 completion 影响（权重大一些）
    # 你现在 base 在 0.39 左右，所以这里让 quality 主导，不然差异太小
    r = _clip01(0.2 * base + 0.8 * quality)

    if debug:
        print(
            f"[reward] raw_idx={raw_idx} base={base:.4f} valid={valid:.1f} "
            f"sla={sla} limits={limits[:5]} quality={quality:.4f} reward={r:.4f}"
        )
    return r
