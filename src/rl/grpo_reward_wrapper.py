from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.rl.reward_fn import compute_reward


class GRPORewardWrapper:
    def __init__(
        self,
        raw_id_field_candidates: Optional[List[str]] = None,
        verbose: bool = False,
        log_first_n_zero: int = 0,
    ):
        self.raw_id_field_candidates = raw_id_field_candidates or ["raw_idx", "raw_idxs", "raw_ids", "raw_id"]
        self.verbose = verbose
        self.log_first_n_zero = log_first_n_zero


    def _get_raw_ids(self, kwargs: Dict[str, Any]) -> Optional[List[int]]:
        for k in self.raw_id_field_candidates:
            if k in kwargs:
                v = kwargs[k]
                if isinstance(v, list):
                    return [int(x) for x in v]
        return None

    def _expand_raw_ids(self, raw_ids: List[int], n_prompts: int, n_completions: int) -> List[int]:
        if len(raw_ids) == n_completions:
            return raw_ids

        if len(raw_ids) == n_prompts:
            if n_prompts == 0:
                raise ValueError("n_prompts==0 but got per-prompt raw_ids?")
            if n_completions % n_prompts != 0:
                raise ValueError(
                    f"Cannot infer num_generations: n_completions={n_completions} not divisible by n_prompts={n_prompts}"
                )
            g = n_completions // n_prompts
            out: List[int] = []
            for rid in raw_ids:
                out.extend([rid] * g)
            return out

        raise ValueError(
            f"raw_idx length mismatch: raw_ids={len(raw_ids)} prompts={n_prompts} completions={n_completions}"
        )

    def _expand_prompts(self, prompts: List[str], n_completions: int) -> List[str]:
        n_p = len(prompts)
        if n_p == 0:
            return [""] * n_completions
        if n_p == n_completions:
            return prompts
        if n_completions % n_p != 0:
            return [prompts[i % n_p] for i in range(n_completions)]
        g = n_completions // n_p
        out: List[str] = []
        for p in prompts:
            out.extend([p] * g)
        return out

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        n_p = len(prompts)
        n_c = len(completions)

        raw_ids = self._get_raw_ids(kwargs)
        if raw_ids is None:
            raise KeyError(
                f"Missing raw_idx in kwargs. Tried {self.raw_id_field_candidates}. Available keys: {list(kwargs.keys())}"
            )

        raw_ids = self._expand_raw_ids(raw_ids, n_p, n_c)
        exp_prompts = self._expand_prompts(prompts, n_c)

        rewards: List[float] = []
        zero_cnt = 0

        for i, (c, rid, p) in enumerate(zip(completions, raw_ids, exp_prompts)):
            r = compute_reward(c, raw_idx=rid, prompt=p, debug=False)
            rewards.append(float(r))

            if r <= 0.0:
                zero_cnt += 1
                if self.verbose and self.log_first_n_zero and zero_cnt <= self.log_first_n_zero:
                    prefix = (c.strip()[:180] + "...") if len(c.strip()) > 180 else c.strip()
                    print(f"[reward=0] idx={i} raw_idx={rid} completion_prefix={prefix}")

        if self.verbose:
            avg = sum(rewards) / max(1, len(rewards))
            print(f"[GRPORewardWrapper] completions={n_c} avg_reward={avg:.4f} zero={zero_cnt}/{n_c}")

        return rewards
