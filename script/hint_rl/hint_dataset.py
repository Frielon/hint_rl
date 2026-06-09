# Copyright 2026
#
# HintBudgetDataset -- the injection side of the HPRL dynamic-budget ratchet.
#
# This is a thin RLHFDataset subclass wired via ``data.custom_cls.path/name``. It
# changes NOTHING unless the HPRL ratchet is on (``data.hprl.enable=true``): when
# off, ``__getitem__`` is a pure passthrough to RLHFDataset, so the same parquet
# and the same code path run an ordinary (static-budget) HPRL job.
#
# When on, for each sampled row it:
#   1. Looks up the problem's CURRENT budget B_q from the shared budget-state
#      JSON (written by the trainer-side ratchet, hint_budget_callback). The file
#      is re-read only when its mtime changes, so this is cheap per item and safe
#      across the StatefulDataLoader worker subprocesses (they share the file).
#      If the problem isn't in the table yet (first epoch / never ratcheted), it
#      falls back to the per-row baked budget from the parquet.
#   2. Re-renders the system prompt for B_q (using the SAME template the data
#      prep used, hint_prompt.render_system) so the policy is told its real,
#      current budget.
#   3. Overwrites the budget the HintTool enforces
#      (tools_kwargs.request_hint.create_kwargs.budget) and the copy under
#      extra_info -- the latter is what the trainer-side ratchet reads back as
#      the "budget these rollouts actually ran under".
#
# Budget state path / knobs come from ``config.hprl`` (== ``data.hprl`` in the
# top-level config, since verl passes the data config to the dataset).

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.utils.dataset.rl_dataset import RLHFDataset

from budget_manager import load_budget_table
from hint_prompt import DEFAULT_BASE_SYSTEM, render_system, render_user

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_create_budget(tools_kwargs: Any, budget: int, tool_name: str = "request_hint") -> None:
    """Set ``tools_kwargs[tool_name].create_kwargs.budget = budget`` in place.

    No-op if the structure isn't the expected nested dict (defensive: a row may
    legitimately carry no tools_kwargs).
    """
    if not isinstance(tools_kwargs, dict):
        return
    tool = tools_kwargs.get(tool_name)
    if not isinstance(tool, dict):
        return
    ck = tool.get("create_kwargs")
    if isinstance(ck, dict):
        ck["budget"] = int(budget)


def _get_create_budget(tools_kwargs: Any, default: int, tool_name: str = "request_hint") -> int:
    if isinstance(tools_kwargs, dict):
        tool = tools_kwargs.get(tool_name)
        if isinstance(tool, dict):
            ck = tool.get("create_kwargs")
            if isinstance(ck, dict) and ck.get("budget") is not None:
                return int(ck["budget"])
    return int(default)


class HintBudgetDataset(RLHFDataset):
    """RLHFDataset that injects the per-problem ratcheted budget B_q at sample time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hprl = self.config.get("hprl", {}) or {}
        self.hprl_enable = bool(hprl.get("enable", False))
        self.hprl_state_path = hprl.get("budget_state_path", None)
        self.hprl_default_budget = int(hprl.get("default_budget", 8))
        self.hprl_tool_name = hprl.get("tool_name", "request_hint")
        # mtime-cached view of {problem_id: B_q}
        self._budget_table: dict[str, int] = {}
        self._budget_mtime: Optional[float] = None
        if self.hprl_enable:
            logger.warning(
                "HintBudgetDataset: dynamic budget ENABLED (state=%s, default_budget=%d)",
                self.hprl_state_path,
                self.hprl_default_budget,
            )

    # ------------------------------------------------------------------ #
    # budget-state reader (mtime-cached)
    # ------------------------------------------------------------------ #
    def _maybe_reload_budgets(self) -> None:
        path = self.hprl_state_path
        if not path or not os.path.exists(path):
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime != self._budget_mtime:
            table = load_budget_table(path)
            if table or self._budget_mtime is None:
                self._budget_table = table
            self._budget_mtime = mtime

    def _budget_for(self, problem_id: Optional[str], baked_budget: int) -> int:
        """Current B_q for a problem; per-row baked budget if not yet ratcheted."""
        if problem_id is None:
            return int(baked_budget)
        self._maybe_reload_budgets()
        return int(self._budget_table.get(str(problem_id), baked_budget))

    # ------------------------------------------------------------------ #
    # injection
    # ------------------------------------------------------------------ #
    def __getitem__(self, item):
        row = super().__getitem__(item)
        if not self.hprl_enable:
            return row  # ratchet off -> identical to RLHFDataset

        extra_info = row.get("extra_info") or {}
        # Only HPRL/tool rows carry hprl_system_base (set by prepare_hint_data).
        # Non-tool rows -- notably the unaided single-turn validation set -- are
        # left exactly as-is so the budget sentence is never injected into them.
        if "hprl_system_base" not in extra_info:
            return row

        problem_id = extra_info.get("problem_id")
        tools_kwargs = row.get("tools_kwargs") or {}

        baked = _get_create_budget(tools_kwargs, self.hprl_default_budget, self.hprl_tool_name)
        budget = self._budget_for(problem_id, baked)

        # 1) re-render the system prompt to advertise the current budget.
        base_system = extra_info.get("hprl_system_base", DEFAULT_BASE_SYSTEM)
        messages = row.get("raw_prompt")
        if isinstance(messages, list) and messages:
            new_system = {"role": "system", "content": render_system(base_system, budget)}
            if messages[0].get("role") == "system":
                messages[0] = new_system
            else:
                messages.insert(0, new_system)
            # 1b) re-render the user message's trailing budget reminder for B_q.
            # Mirrors the system re-render: hprl_user_base is the budget-free,
            # CoT-stripped user text baked by prepare_hint_data; re-append the
            # current budget so the reminder tracks the ratchet (and vanishes at
            # budget 0, when the tool is disabled). Last user turn only.
            user_base = extra_info.get("hprl_user_base")
            if user_base is not None:
                for j in range(len(messages) - 1, -1, -1):
                    if messages[j].get("role") == "user":
                        messages[j] = {"role": "user",
                                       "content": render_user(user_base, budget)}
                        break
        else:
            logger.warning("HintBudgetDataset: row %s has no raw_prompt to re-render", item)

        # 2) overwrite the enforced budget (tool) and the extra_info copy (the
        #    "budget these rollouts ran under" that the ratchet reads back).
        _set_create_budget(tools_kwargs, budget, self.hprl_tool_name)
        ei_tools = extra_info.get("tools_kwargs")
        if ei_tools is not tools_kwargs:  # keep the two copies consistent
            _set_create_budget(ei_tools, budget, self.hprl_tool_name)

        row["tools_kwargs"] = tools_kwargs
        row["extra_info"] = extra_info
        return row
