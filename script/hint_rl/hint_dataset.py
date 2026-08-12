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
from typing import Optional

from verl.utils.dataset.rl_dataset import RLHFDataset

from budget_manager import get_create_budget, load_budget_table, set_create_budget
from hint_prompt import DEFAULT_BASE_SYSTEM, rerender_messages_for_budget

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_create_kwarg(tools_kwargs, key, value, tool_name: str) -> None:
    """Set one key inside tools_kwargs[tool_name]["create_kwargs"] (no-op on any
    missing/odd level -- same tolerant shape-walk as budget_manager's helpers)."""
    if not isinstance(tools_kwargs, dict):
        return
    tool = tools_kwargs.get(tool_name)
    if not isinstance(tool, dict):
        return
    ck = tool.get("create_kwargs")
    if isinstance(ck, dict):
        ck[key] = value


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
        # REFERENCE-PREFIX restart delivery (restart_agent_loop, env-gated like
        # the other HPRL_RESTART_* knobs): the agent loop reads the per-problem
        # reference solutions from create_kwargs, so when the mode is on, copy
        # the parquet's extra_info["hint_reference"] into
        # tools_kwargs.request_hint.create_kwargs at sample time (auto-hint rows
        # only). Off (default) -> rows are byte-identical to before.
        self.hprl_ref_prefix = str(
            os.environ.get("HPRL_RESTART_REFERENCE_PREFIX", "false")
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if self.hprl_enable:
            logger.warning(
                "HintBudgetDataset: dynamic budget ENABLED (state=%s, default_budget=%d, "
                "reference_prefix_plumbing=%s)",
                self.hprl_state_path,
                self.hprl_default_budget,
                self.hprl_ref_prefix,
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
        # Auto-hint (push-hint) rows carry hprl_auto_hint and a HINT-AGNOSTIC prompt
        # (the policy is never told about hints). Update ONLY the ratcheted budget
        # the loop enforces (tools_kwargs.request_hint.create_kwargs.budget) -- never
        # the prompt -- then return. Checked BEFORE hprl_system_base (auto-hint rows
        # deliberately do not set it, so no budget sentence is ever rendered).
        if extra_info.get("hprl_auto_hint"):
            return self._update_auto_hint_budget(row, extra_info)

        # Only HPRL/tool rows carry hprl_system_base (set by prepare_hint_data).
        # Non-tool rows -- notably the unaided single-turn validation set -- are
        # left exactly as-is so the budget sentence is never injected into them.
        if "hprl_system_base" not in extra_info:
            return row

        problem_id = extra_info.get("problem_id")
        tools_kwargs = row.get("tools_kwargs") or {}

        baked = get_create_budget(tools_kwargs, self.hprl_default_budget, self.hprl_tool_name)
        budget = self._budget_for(problem_id, baked)

        # 1) re-render the system + trailing user budget reminder for B_q. Shared
        #    with the k-pack probe expansion (hint_prompt.rerender_messages_for_budget)
        #    so a probe pack at B-j is byte-identical to a dataset render at B-j.
        #    hprl_user_base is the budget-free, CoT-stripped user text baked by
        #    prepare_hint_data; the reminder tracks the ratchet (and vanishes at
        #    budget 0, when the tool is disabled).
        base_system = extra_info.get("hprl_system_base", DEFAULT_BASE_SYSTEM)
        messages = row.get("raw_prompt")
        if isinstance(messages, list) and messages:
            user_base = extra_info.get("hprl_user_base")
            row["raw_prompt"] = rerender_messages_for_budget(messages, base_system, user_base, budget)
        else:
            logger.warning("HintBudgetDataset: row %s has no raw_prompt to re-render", item)

        # 2) overwrite the enforced budget (tool) and the extra_info copy (the
        #    "budget these rollouts ran under" that the ratchet reads back).
        set_create_budget(tools_kwargs, budget, self.hprl_tool_name)
        ei_tools = extra_info.get("tools_kwargs")
        if ei_tools is not tools_kwargs:  # keep the two copies consistent
            set_create_budget(ei_tools, budget, self.hprl_tool_name)

        row["tools_kwargs"] = tools_kwargs
        row["extra_info"] = extra_info
        return row

    def _update_auto_hint_budget(self, row, extra_info):
        """Auto-hint rows: inject the ratcheted B_q into tools_kwargs ONLY.

        Unlike the <hint_call/> path, the auto-hint prompt is hint-agnostic (no tool
        instruction, no budget reminder) -- the LOOP, not the policy, consumes the
        budget -- so the prompt is left untouched. Only the budget the loop enforces
        (tools_kwargs.request_hint.create_kwargs.budget) and its extra_info copy (read
        back by the ratchet as "the budget these rollouts ran under") are updated.
        """
        problem_id = extra_info.get("problem_id")
        tools_kwargs = row.get("tools_kwargs") or {}
        baked = get_create_budget(tools_kwargs, self.hprl_default_budget, self.hprl_tool_name)
        budget = self._budget_for(problem_id, baked)
        set_create_budget(tools_kwargs, budget, self.hprl_tool_name)
        ei_tools = extra_info.get("tools_kwargs")
        if ei_tools is not tools_kwargs:  # keep the two copies consistent
            set_create_budget(ei_tools, budget, self.hprl_tool_name)
        # reference-prefix restart: hand the loop this problem's per-substep
        # reference solutions (the hint_reference JSON string; parsed loop-side).
        # Absent from the parquet -> nothing set; the loop warns + falls back.
        if self.hprl_ref_prefix:
            ref = extra_info.get("hint_reference")
            if ref is not None:
                _set_create_kwarg(tools_kwargs, "hint_reference", ref, self.hprl_tool_name)
                if ei_tools is not tools_kwargs:
                    _set_create_kwarg(ei_tools, "hint_reference", ref, self.hprl_tool_name)
        row["tools_kwargs"] = tools_kwargs
        row["extra_info"] = extra_info
        return row
