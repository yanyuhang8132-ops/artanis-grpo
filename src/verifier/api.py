# src/verifier/api.py
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# -----------------------------
# Paths / Config
# -----------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../artanis-grpo-release
_DEFAULT_CFG_PATH = _REPO_ROOT / "configs" / "verifier.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def get_cfg() -> Dict[str, Any]:
    """
    verifier.yaml is optional. If missing keys, we fall back to sane defaults.
    Environment overrides:
      - VERIFIER_CONFIG: path to yaml
      - VERIFIER_CKPT:   checkpoint path
      - VERIFIER_DATA:   dataset json path (augmented_dataset_all.json)
      - VERIFIER_DEVICE: cpu/cuda:0...
      - CPV_DIR:         path to CPV folder (default: <repo_root>/CPV)
    """
    cfg_path = Path(os.getenv("VERIFIER_CONFIG", str(_DEFAULT_CFG_PATH)))
    raw = _load_yaml(cfg_path)

    v = raw.get("verifier", raw) if isinstance(raw, dict) else {}
    if not isinstance(v, dict):
        v = {}

    # resolve CPV directory (the one that contains dataloader.py and models/)
    cpv_dir = Path(os.getenv("CPV_DIR", str(_REPO_ROOT / "src" / "verifier" / "legacy" / "CPV"))).resolve()


    # ckpt path
    ckpt_env = os.getenv("VERIFIER_CKPT")
    ckpt_default = _first_existing(
        [
            _REPO_ROOT / "checkpoints" / "verifier" / "multimodal_multitask_model.pth",
            _REPO_ROOT / "checkpoints" / "verifier" / "multimodal_multitask_model_14.pth",
            cpv_dir / "saved_models" / "multimodal_multitask_model.pth",
        ]
    )
    ckpt_path = Path(ckpt_env).resolve() if ckpt_env else (ckpt_default or (_REPO_ROOT / "checkpoints/verifier/multimodal_multitask_model.pth"))

    # data json path (match validate.py / dataloader.py convention)
    data_env = os.getenv("VERIFIER_DATA")
    data_default = _first_existing(
        [
            _REPO_ROOT / "data" / "processed" / "augmented_dataset_all.json",
            _REPO_ROOT / "data" / "raw" / "augmented_dataset_all.json",
            cpv_dir / "data" / "augmented_dataset_all.json",
            _REPO_ROOT / "src" / "verifier" / "legacy" / "CPV" / "data" / "augmented_dataset_all.json",
        ]
    )
    data_json = Path(data_env).resolve() if data_env else (data_default or (_REPO_ROOT / "data/processed/augmented_dataset_all.json"))

    device = os.getenv("VERIFIER_DEVICE", str(v.get("device", "cpu")))

    return {
        "cpv_dir": str(cpv_dir),
        "ckpt_path": str(ckpt_path),
        "data_json": str(data_json),
        "device": device,
        # these defaults match your validate.py/dataloader choices
        "ec_vocab_size": int(v.get("ec_vocab_size", 10000)),
        "ec_embed_dim": int(v.get("ec_embed_dim", 8)),
        "hidden_dim": int(v.get("hidden_dim", 16)),
        "num_heads": int(v.get("num_heads", 4)),
    }


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# -----------------------------
# Load CPV (validate.py-style) modules
# -----------------------------
@lru_cache(maxsize=1)
def _ensure_cpv_importable() -> None:
    """
    Make legacy CPV importable as package 'CPV'.
    We add: <repo_root>/src/verifier/legacy  to sys.path,
    because it contains folder 'CPV/'.
    """
    legacy_root = (_REPO_ROOT / "src" / "verifier" / "legacy").resolve()
    legacy_root_str = str(legacy_root)

    if not legacy_root.exists():
        raise FileNotFoundError(f"legacy root not found: {legacy_root}")

    if legacy_root_str in sys.path:
        sys.path.remove(legacy_root_str)
    sys.path.insert(0, legacy_root_str)

    # sanity check (optional)
    try:
        import CPV  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(f"Failed to import CPV from {legacy_root}: {e}") from e



@lru_cache(maxsize=1)
def _get_dataset():
    _ensure_cpv_importable()
    cfg = get_cfg()

    from CPV.training.dataloader import NetworkDataset  # type: ignore

    json_path = cfg["data_json"]
    ds = NetworkDataset(json_path)
    return ds



@lru_cache(maxsize=1)
def _get_model():
    _ensure_cpv_importable()
    cfg = get_cfg()

    from CPV.models.multimodal_model import MultimodalMultitaskModel  # type: ignore

    ds = _get_dataset()
    sample0 = ds[0]
    N_max = int(ds.N_max)
    flow_dim = int(sample0["non_graph"]["traffic_flows"].shape[1])

    model = MultimodalMultitaskModel(
        N_max=N_max,
        flow_input_dim=flow_dim,
        ec_vocab_size=int(cfg["ec_vocab_size"]),
        ec_embed_dim=int(cfg["ec_embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        # num_heads 可能不同版本没有，保守不传
    )

    sd = torch.load(cfg["ckpt_path"], map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    dev = torch.device(cfg["device"])
    model.to(dev)
    return model

def build_sample(raw_idx: int, policy_configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return a single sample in the SAME tensor format as CPV dataloader provides.
    NOTE: policy_configs is accepted for future extension, but NOT injected into tensors,
    because CPV model/verifier expects all non_graph items to be Tensors.
    """
    ds = _get_dataset()
    if raw_idx < 0 or raw_idx >= len(ds):
        raise IndexError(f"raw_idx out of range: {raw_idx} / len={len(ds)}")

    sample = ds[int(raw_idx)]
    # Keep it clean: graph/non_graph/labels/mask are all tensors.
    return sample


def _risk_from_outputs(outputs: Dict[str, torch.Tensor], mask: torch.Tensor) -> float:
    """
    Convert multitask outputs into a single 'risk' scalar in [0,1]:
    risk = average(sigmoid(logits)) over valid nodes/pairs/flows.
    """
    probs = []
    m = mask.bool()
    for _, out in outputs.items():
        if not isinstance(out, torch.Tensor):
            continue

        # remove batch dim if present
        if out.dim() >= 1 and out.shape[0] == 1:
            out = out.squeeze(0)

        p = torch.sigmoid(out.float())

        if p.dim() == 1 and p.shape[0] == m.shape[0]:
            probs.append(p[m].mean())
        elif p.dim() == 2 and p.shape[0] == m.shape[0] and p.shape[1] == m.shape[0]:
            sub = p[m][:, m]
            probs.append(sub.mean())
        else:
            # flow-level or other shapes
            probs.append(p.mean())

    if not probs:
        return 1.0  # if nothing, treat as risky
    risk = torch.stack(probs).mean().item()
    return float(_clip01(float(risk)))


def score(raw_idx: int, policy_configs: Optional[Dict[str, Any]] = None, debug: bool = False) -> float:
    """
    Return a scalar score in [0,1] for a given raw_idx (+ optional generated config).
    This implementation intentionally matches validate.py pipeline:
      - use CPV dataloader sample
      - load multimodal_multitask_model.pth strictly
      - forward -> multitask probabilities
      - score = 1 - risk
    """
    cfg = get_cfg()
    dev = torch.device(cfg["device"])

    sample = build_sample(int(raw_idx), policy_configs=policy_configs)

    model = _get_model()

    graph = {k: v.to(dev) for k, v in sample["graph"].items()}
    non_graph = {k: v.to(dev) for k, v in sample["non_graph"].items()}
    mask = sample["mask"].to(dev)

    with torch.no_grad():
        outputs = model(graph, non_graph)

    risk = _risk_from_outputs(outputs, mask)
    sc = _clip01(1.0 - risk)

    if debug:
        print(
            f"[verifier] raw_idx={raw_idx} ckpt={cfg['ckpt_path']} data={cfg['data_json']} "
            f"device={cfg['device']} risk={risk:.6f} score={sc:.6f}"
        )
    return sc
