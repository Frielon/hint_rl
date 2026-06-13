# Copyright 2026
#
# Reward manager for Hint Penalized RL (HPRL).
#
# Its single job beyond the stock naive manager is to surface the per-rollout
# **applied-hints state** (recorded by hint_tool.HintTool) into ``extra_info``
# so the reward function (hint_reward.compute_score) can apply the hint penalty.
#
# Where the state comes from:
#   * Inline agent-reward-loop path: the agent loop hands each sample's
#     ``extra_fields`` to the reward worker as ``non_tensor_batch["tool_extra_fields"]``.
#   * Colocate path: ``extra_fields`` are flattened to top-level non-tensor keys,
#     so ``applied_hints`` arrives as ``non_tensor_batch["applied_hints"]``.
# We merge from BOTH so the penalty is computed correctly regardless of path.
#
# Loaded via (legacy flags, mapped by migrate_legacy_reward_impl):
#   reward_model.reward_loop_source=importlib
#   reward_model.reward_loop_module_path=<this file>
#   reward_model.reward_loop_class_name=HintRewardManager

from __future__ import annotations

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def _to_py(v):
    """Normalize numpy 0-d object arrays / scalars to plain Python objects."""
    if v is None:
        return None
    if hasattr(v, "item") and getattr(v, "shape", None) == ():
        return v.item()
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:  # noqa: BLE001
            return v
    return v


class HintRewardManager(RewardManagerBase):
    """Naive-style reward manager that merges applied-hints state into extra_info."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # multi-sequence outputs: score only the last sequence
        data_item = data[0]
        ntb = data_item.non_tensor_batch

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = ntb["data_source"]
        ground_truth = ntb["reward_model"]["ground_truth"]
        extra_info = ntb.get("extra_info", {}) or {}

        # --- merge tool / per-rollout state into extra_info ---------------
        tool_extra_fields = ntb.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            tool_extra_fields = _to_py(tool_extra_fields)
            if isinstance(tool_extra_fields, dict):
                extra_info.update(tool_extra_fields.items())

        # colocate path: applied_hints flattened to a top-level non-tensor key.
        if "applied_hints" not in extra_info and "applied_hints" in ntb:
            extra_info["applied_hints"] = _to_py(ntb.get("applied_hints"))

        # Same for the per-turn reasoning lengths feeding the effort-shaping term.
        if "turn_lens" not in extra_info and "turn_lens" in ntb:
            extra_info["turn_lens"] = _to_py(ntb.get("turn_lens"))

        # Same for the failed-hint-call count (feeds the hint-call bonus, which
        # treats a selector-failed call as a genuine emission, and the hprl/* logs).
        if "hint_call_failed" not in extra_info and "hint_call_failed" in ntb:
            extra_info["hint_call_failed"] = _to_py(ntb.get("hint_call_failed"))

        # Same for the over-budget protocol-violation flag (drives the reward's
        # floor-score short-circuit in hint_reward.compute_score).
        if "hint_budget_exceeded" not in extra_info and "hint_budget_exceeded" in ntb:
            extra_info["hint_budget_exceeded"] = _to_py(ntb.get("hint_budget_exceeded"))

        # Same for the per-rollout selector latency (total seconds / #calls) and the
        # box-then-call counters; passed through so hint_budget_callback can log
        # hprl/hint_select_time_* and hprl/hint_call_with_box_*.
        for k in ("hint_select_time", "hint_select_calls", "hint_calls_total", "hint_calls_with_box"):
            if k not in extra_info and k in ntb:
                extra_info[k] = _to_py(ntb.get(k))

        extra_info["num_turns"] = ntb.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = ntb.get("reward_scores", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
