# Copyright 2026
#
# StepHint x verl fully-async policy: the rollouter/task-runner subclasses.
#
# STEPHINT training = plain fully-async GRPO (stock FullyAsyncTrainer, stock
# update path -- run_grpo_qwen3_4b_instruct_2507_npu_async.sh geometry) with ONE
# rollout-side twist: each dataset problem carries a reference solution split
# into M <= 4 segments (extra_info.segments, a JSON list of {segment_id, title,
# content}), and of the N = rollout.n rollouts in a GRPO group,
#
#   * rollouts j = 0 .. r*(M-1)-1   (r = data.stephint.rollouts_per_segment,
#     default 2) are PREFIXED: rollout j continues from segments 1..(j//r + 1)
#     -- i.e. r rollouts per prefix depth i, for i = 1..M-1 (the full solution
#     is never given: the last segment is always the model's to produce);
#   * the remaining N - r*(M-1) >= r rollouts generate from scratch.
#
# All N rollouts stay in ONE GRPO group (the rollouter stamps a single uid per
# prompt-group after generation, exactly as stock), so the easier prefixed
# rollouts raise the group's success rate and give the from-scratch rollouts a
# non-degenerate baseline on problems the policy cannot yet solve unaided. The
# prefix tokens are loss-masked by the agent loop (stephint_agent_loop) -- only
# model-generated tokens train.
#
# SANITY CHECK N >= r*M (the user-facing "N >= 2M" at r=2): enforced per group
# here (fail loud) and against the dataset's max M in the run script before
# submit. It guarantees at least r from-scratch rollouts per group.
#
# Plumbing (all inherited from the HPRL async port, hprl_fully_async.py):
#
#   * verl's FullyAsyncRollouter streams ONE prompt-group at a time:
#     _feed_samples -> prepare_single_generation_data (repeats the sample
#     rollout.n times, stamps agent_name="single_turn_agent") ->
#     _process_single_sample_streaming -> manager.generate_sequences_single.
#     That manager call is the single seam where the repeated group exists
#     BEFORE generation -- the subclass below wraps it to (a) stamp the per-row
#     stephint_prefix columns + agent_name="stephint_agent" and (b) restore the
#     dataset non-tensors (extra_info / reward_model / ...) onto the agent-loop
#     output, which stock REPLACES the batch with (found on the HPRL staleness
#     grid 2026-07-23; without the restore those columns never reach the
#     trainer's rollout dumps).
#   * Validation never routes through generate_sequences_single (it uses the
#     batch generate_sequences), so val sets need no segments and roll out
#     plain single-turn -- no gating required.
#   * Ray forbids subclassing an @ray.remote class; the undecorated body is
#     reached via <ActorClass>.__ray_metadata__.modified_class and
#     re-decorated (same single-module concession as hprl_fully_async.py).
#
# No verl core edits; wired by main_stephint_async.py.

from __future__ import annotations

import json
import logging
import os

import numpy as np
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The undecorated class bodies behind the @ray.remote wrappers (see header).
_FULLY_ASYNC_RUNNER_BODY = FullyAsyncTaskRunner.__ray_metadata__.modified_class
_FULLY_ASYNC_ROLLOUTER_BODY = FullyAsyncRollouter.__ray_metadata__.modified_class

# Segments are joined into a prefix with a paragraph break, and the prefix ends
# with one too, so the model's continuation starts on a fresh paragraph. Titles
# are omitted -- the prefix reads as the solution prose itself.
_SEGMENT_JOINER = "\n\n"


def parse_segments(extra_info) -> list[str]:
    """Return the ordered segment contents of one problem's reference solution.

    ``extra_info.segments`` is a JSON string (or already-decoded list) of
    {segment_id, title, content} dicts. Raises ValueError on any malformed row:
    a stephint TRAIN_FILE without segments is a launch misconfiguration, and a
    silent fallback would quietly train plain GRPO.
    """
    if not isinstance(extra_info, dict):
        raise ValueError(f"stephint: extra_info is {type(extra_info).__name__}, expected dict")
    segments = extra_info.get("segments")
    if isinstance(segments, bytes):
        segments = segments.decode("utf-8")
    if isinstance(segments, str):
        segments = json.loads(segments)
    if isinstance(segments, np.ndarray):
        segments = segments.tolist()
    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "stephint: extra_info.segments missing/empty -- is TRAIN_FILE the "
            "*-stephint.parquet build?"
        )
    if not all(isinstance(s, dict) for s in segments):
        raise ValueError("stephint: extra_info.segments items are not dicts")
    segments = sorted(segments, key=lambda s: s.get("segment_id", 0))
    return [str(s.get("content") or "") for s in segments]


def stamp_stephint_prefixes(prompts, *, rollouts_per_segment: int) -> int:
    """Stamp the per-rollout prefix columns onto one repeated prompt-group.

    ``prompts`` is ONE dataset sample repeated N = rollout.n times
    (prepare_single_generation_data), so row 0's extra_info speaks for the
    whole group. Adds, per row:

      * stephint_prefix          -- segments 1..i joined ("" = from scratch)
      * stephint_prefix_segments -- i (0 = from scratch)
      * agent_name               -- "stephint_agent" (overwrites the
                                    "single_turn_agent" stamp so the group
                                    routes through StepHintAgentLoop)

    Returns M (the group's segment count). Raises on N < rollouts_per_segment*M
    -- the per-group side of the run script's dataset-wide sanity check.
    """
    n = len(prompts)
    extra = prompts.non_tensor_batch.get("extra_info")
    if extra is None:
        raise ValueError("stephint: batch has no extra_info non-tensor (dataset column missing)")
    contents = parse_segments(extra[0])
    num_segments = len(contents)

    if n < rollouts_per_segment * num_segments:
        raise ValueError(
            f"stephint sanity check failed: rollout.n={n} < rollouts_per_segment"
            f"({rollouts_per_segment}) * M({num_segments}) for problem "
            f"{extra[0].get('problem_id', '<unknown>')} -- raise rollout.n or "
            f"rebuild the dataset with fewer segments."
        )

    num_prefixed = rollouts_per_segment * (num_segments - 1)
    prefixes, seg_counts = [], []
    for j in range(n):
        if j < num_prefixed:
            depth = j // rollouts_per_segment + 1  # segments 1..depth
            prefixes.append(_SEGMENT_JOINER.join(contents[:depth]) + _SEGMENT_JOINER)
            seg_counts.append(depth)
        else:
            prefixes.append("")
            seg_counts.append(0)

    prompts.non_tensor_batch["stephint_prefix"] = np.array(prefixes, dtype=object)
    prompts.non_tensor_batch["stephint_prefix_segments"] = np.array(seg_counts, dtype=object)
    prompts.non_tensor_batch["agent_name"] = np.array(["stephint_agent"] * n, dtype=object)
    return num_segments


def restore_dataset_non_tensors(src, out):
    """Copy every non-tensor column of ``src`` (the pre-generation sample batch,
    which still carries the dataset fields) that is MISSING from ``out`` (the
    agent-loop output) into ``out``. Returns (out, n_restored).

    Verbatim from hprl_fully_async.py (kept local so script/stephint_baseline
    stays importable on its own): row alignment is trivially safe because
    ``src`` is one sample repeated rollout.n times; the stephint_prefix columns
    stamped above are per-row and length-matched, so they restore too and land
    in the trainer's rollout dumps.
    """
    if src is None or out is None:
        return out, 0
    n = len(out)
    restored = 0
    for key, val in src.non_tensor_batch.items():
        if key in out.non_tensor_batch:
            continue
        if len(val) != n:
            logger.warning(
                "[STEPHINT ASYNC] not restoring non-tensor %r: %d rows vs batch %d", key, len(val), n
            )
            continue
        out.non_tensor_batch[key] = val
        restored += 1
    return out, restored


class _StepHintFullyAsyncRollouterImpl(_FULLY_ASYNC_ROLLOUTER_BODY):
    """FullyAsyncRollouter that stamps the stephint prefix assignment onto every
    training prompt-group right before generation, and restores the dataset
    non-tensors onto the agent-loop output before the sample is queued.

    The wrap lives on the manager's ``generate_sequences_single`` -- the single
    seam between the repeated dataset batch and ``rollout_sample.full_batch =
    ret`` -- instead of copying the whole ``_process_single_sample_streaming``
    body, so upstream edits to the streaming loop keep working (same seam
    choice as HPRLFullyAsyncRollouter).
    """

    async def _init_async_rollout_manager(self):
        await super()._init_async_rollout_manager()

        stephint_cfg = OmegaConf.select(self.config, "data.stephint") or {}
        if not stephint_cfg.get("enable", False):
            print(
                "[STEPHINT ASYNC] data.stephint.enable is false -- rollouter running "
                "STOCK (no solution prefixes will be injected)."
            )
            return
        rollouts_per_segment = int(stephint_cfg.get("rollouts_per_segment", 2))

        mgr = self.async_rollout_manager
        inner = mgr.generate_sequences_single
        logged = False

        async def _stamp_generate_restore(prompts):
            nonlocal logged
            num_segments = stamp_stephint_prefixes(
                prompts, rollouts_per_segment=rollouts_per_segment
            )
            if not logged:
                logged = True
                counts = prompts.non_tensor_batch["stephint_prefix_segments"].tolist()
                print(
                    f"[STEPHINT ASYNC] prefix stamping active (r={rollouts_per_segment}): "
                    f"first group has M={num_segments} segments, per-rollout prefix "
                    f"depths {counts} (0 = from scratch; logged once per rollouter)"
                )
            out = await inner(prompts)
            out, _ = restore_dataset_non_tensors(prompts, out)
            return out

        mgr.generate_sequences_single = _stamp_generate_restore


StepHintFullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(
    _StepHintFullyAsyncRollouterImpl
)


class StepHintFullyAsyncTaskRunnerImpl(_FULLY_ASYNC_RUNNER_BODY):
    """FullyAsyncTaskRunner that builds StepHintFullyAsyncRollouter instead.

    _create_rollouter is a verbatim copy of the base body with the class
    swapped (the base hardcodes the module-global FullyAsyncRollouter, so
    there is no narrower hook). The trainer, MessageQueue and the whole run()
    orchestration are inherited stock -- the GRPO update is untouched.
    """

    def _create_rollouter(self, config) -> None:
        print("[STEPHINT ASYNC MAIN] Starting create rollouter (StepHintFullyAsyncRollouter)...")
        rollouter = StepHintFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[STEPHINT ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[STEPHINT ASYNC MAIN] StepHintFullyAsyncRollouter created and initialized successfully")
