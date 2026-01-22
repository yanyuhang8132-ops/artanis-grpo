from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from CPV.training.dataloader import encode_traffic_flows, encode_ec_networks, parse_if_string
from inference.Inference2Sample import parse_model_output_configs


# -------------------------
# Load full augmented data
# -------------------------
AUG_PATH = os.getenv("AUG_DATA_PATH", "data/augmented_dataset_all.json")
with open(AUG_PATH, "r", encoding="utf-8") as f:
    FULL_DATA = json.load(f)


# -------------------------
# JSON extraction helpers
# -------------------------
def _find_json_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Find span of the first complete JSON array/object in text.
    Very lightweight bracket matching (good enough for "pure JSON" + occasional trailing text).
    """
    s = text.strip()
    if not s:
        return None

    start = None
    open_ch = None
    close_ch = None

    for i, ch in enumerate(s):
        if ch == "[":
            start, open_ch, close_ch = i, "[", "]"
            break
        if ch == "{":
            start, open_ch, close_ch = i, "{", "}"
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
                return (start, j + 1)

    return None


def extract_first_json(text: str) -> Optional[str]:
    span = _find_json_span(text)
    if span is None:
        return None
    a, b = span
    return text.strip()[a:b].strip()


# -------------------------
# completion -> policy_configs
# -------------------------
def completion_to_policy_configs(completion_str: str) -> Optional[Dict[str, Any]]:
    """
    Input:
      - completion_str: model output (expected to be pure JSON array, but may contain trailing text)
    Output:
      - {"lb_rules": {...}, "acl_rules": {...}}  or None
    """
    json_str = extract_first_json(completion_str)
    if json_str is None:
        return None

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # already structured
    if isinstance(obj, dict) and ("lb_rules" in obj or "acl_rules" in obj):
        return {
            "lb_rules": obj.get("lb_rules", {}) or {},
            "acl_rules": obj.get("acl_rules", {}) or {},
        }

    # standard: block list
    if isinstance(obj, list):
        try:
            policy = parse_model_output_configs(json_str)  # JSON -> {lb_rules, acl_rules}
        except Exception:
            return None

        if not isinstance(policy, dict):
            return None

        return {
            "lb_rules": policy.get("lb_rules", {}) or {},
            "acl_rules": policy.get("acl_rules", {}) or {},
        }

    return None


# -------------------------
# Cache static context per raw_idx
# -------------------------
@lru_cache(maxsize=4096)
def build_static_context(raw_idx: int) -> Dict[str, Any]:
    raw = FULL_DATA[int(raw_idx)]
    N_max = len(parse_if_string(raw["graph_features"]["adjacency"]))

    # graph
    adj = torch.tensor(parse_if_string(raw["graph_features"]["adjacency"]), dtype=torch.float)
    in_degree = torch.tensor(parse_if_string(raw["graph_features"]["in_degree"]), dtype=torch.float)
    out_degree = torch.tensor(parse_if_string(raw["graph_features"]["out_degree"]), dtype=torch.float)
    betweenness = torch.tensor(parse_if_string(raw["graph_features"]["betweenness"]), dtype=torch.float)
    ec_counts = torch.tensor(parse_if_string(raw["graph_features"]["ec_counts"]), dtype=torch.float)
    ec_networks = encode_ec_networks(raw["graph_features"]["ec_networks"], N_max)

    # flows
    flow_features, traffic_pairs = encode_traffic_flows(
        raw["non_graph_features"]["traffic_flows"], num_links=100
    )

    # labels
    isolated = torch.tensor(parse_if_string(raw["labels"]["isolated"]), dtype=torch.float)
    loop = torch.tensor(parse_if_string(raw["labels"]["loop"]), dtype=torch.float)
    blackhole = torch.tensor(parse_if_string(raw["labels"]["blackhole"]), dtype=torch.float)
    reachability = torch.tensor(parse_if_string(raw["labels"]["reachability"]), dtype=torch.float)
    link_overload_pairs = torch.tensor(parse_if_string(raw["labels"]["link_overload_pairs"]), dtype=torch.float)
    sla_labels = torch.tensor(parse_if_string(raw["labels"]["sla_labels"]), dtype=torch.float)
    mask = torch.tensor(parse_if_string(raw["mask"]), dtype=torch.bool)

    # padding (match verifier expectations)
    isolated = F.pad(isolated, (0, N_max - isolated.shape[0]), value=0)
    loop = F.pad(loop, (0, N_max - loop.shape[0]), value=0)
    blackhole = F.pad(blackhole, (0, N_max - blackhole.shape[0]), value=0)
    reachability = F.pad(
        reachability,
        (0, N_max - reachability.shape[0], 0, N_max - reachability.shape[1]),
        value=0,
    )
    link_overload_pairs = F.pad(
        link_overload_pairs,
        (0, N_max - link_overload_pairs.shape[0], 0, N_max - link_overload_pairs.shape[1]),
        value=0,
    )

    return {
        "graph": {
            "adj": adj,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "betweenness": betweenness,
            "ec_counts": ec_counts,
            "ec_networks": ec_networks,
        },
        "non_graph": {
            "traffic_flows": flow_features,
            "traffic_pairs": traffic_pairs,
        },
        "labels": {
            "isolated": isolated,
            "loop": loop,
            "blackhole": blackhole,
            "reachability": reachability,
            "link_overload_pairs": link_overload_pairs,
            "sla_labels": sla_labels,
        },
        "mask": mask,
    }


def build_sample(raw_idx: int, policy_configs: Dict[str, Any]) -> Dict[str, Any]:
    ctx = build_static_context(int(raw_idx))
    return {
        "graph": ctx["graph"],
        "non_graph": {
            "traffic_flows": ctx["non_graph"]["traffic_flows"],
            "traffic_pairs": ctx["non_graph"]["traffic_pairs"],
            "generated_config": policy_configs, 
        },
        "labels": ctx["labels"],
        "mask": ctx["mask"],
    }


def compute_reward(completion_str: str, raw_idx: int) -> float:
    """
    Reward in [0,1]. Invalid output -> 0.
    """
    policy = completion_to_policy_configs(completion_str)
    if policy is None:
        return 0.0

    sample = build_sample(raw_idx=raw_idx, policy_configs=policy)

    from CPV.verifier import verify_sample
    score = verify_sample(sample)  # expected 0~1

    try:
        s = float(score)
    except Exception:
        return 0.0

    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s
