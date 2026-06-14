# Copyright 2026
#
# k-pack probe-row construction -- the verl-free half of the k-pack expansion.
#
# ``render_variant_rows`` turns a block of "variant" rows (each a shallow copy of a
# source row, as produced by DataProto.select_idxs -- so every slot initially ALIASES
# the source row's mutable objects) into genuine probe packs: it re-renders the prompt
# at the probe budget, sets the tool budget, and stamps a fresh uid + unique index.
#
# It is split out of HPRLRayPPOTrainer so the subtle "deepcopy-before-mutate so the
# source rows stay untouched" logic can be unit-tested WITHOUT importing verl (the
# trainer pulls in Ray/verl). See test_kpack_expansion.py. The surrounding DataProto
# select_idxs/concat plumbing stays in the trainer; this operates purely on the
# resulting dict-of-(numpy object array)s.

from __future__ import annotations

import copy
import uuid

from budget_manager import set_create_budget
from hint_prompt import DEFAULT_BASE_SYSTEM, rerender_messages_for_budget


def _identity(x):
    return x


def render_variant_rows(nt, var_budget, *, tool_name: str = "request_hint", to_py=_identity) -> None:
    """In place: render select_idxs'd ``variant`` rows into probe packs at ``var_budget``.

    Args:
        nt: the variants' non_tensor dict, key -> numpy object array. Each slot starts
            ALIASING its source row's objects (select_idxs shares references for object
            arrays), so every per-row object is DEEP-COPIED before mutation -- the source
            rows (kept as the normal pack) are never touched.
        var_budget: one probe budget per variant row (``len == array length``).
        tool_name: the hint tool key under tools_kwargs.
        to_py: optional normalizer for object-array elements (the trainer passes its
            ``_to_py``; tests pass identity).

    Per row this sets: the re-rendered ``raw_prompt`` (system + trailing user reminder at
    the probe budget, byte-identical to a dataset render); the probe budget on both the
    top-level ``tools_kwargs`` and the ``extra_info.tools_kwargs`` copy; an ``hprl_probe_budget``
    marker; a fresh ``uid`` (so the probe pack is its own GRPO group); and a unique negative
    ``index`` (real extra_info.index are >= 0 -- keeps the rollout-trace per-sample counter
    from merging a probe pack with its source or a sibling probe).
    """
    v_extra = nt["extra_info"]
    v_raw = nt.get("raw_prompt")
    v_tk = nt.get("tools_kwargs")
    v_uid = nt["uid"]
    v_index = nt.get("index")
    for vj, b in enumerate(var_budget):
        b = int(b)
        ei = copy.deepcopy(to_py(v_extra[vj]) or {})
        base_system = ei.get("hprl_system_base", DEFAULT_BASE_SYSTEM)
        user_base = ei.get("hprl_user_base")
        if v_raw is not None:
            v_raw[vj] = rerender_messages_for_budget(to_py(v_raw[vj]) or [], base_system, user_base, b)
        # set the probe budget on the tool config (both copies the dataset keeps consistent).
        set_create_budget(ei.get("tools_kwargs"), b, tool_name)
        ei["hprl_probe_budget"] = b  # marker (logging / debugging only)
        v_extra[vj] = ei
        if v_tk is not None:
            tk = copy.deepcopy(to_py(v_tk[vj]) or {})
            set_create_budget(tk, b, tool_name)
            v_tk[vj] = tk
        v_uid[vj] = str(uuid.uuid4())  # fresh uid -> probe pack is its own GRPO group
        if v_index is not None:
            v_index[vj] = -(vj + 1)  # unique, distinct from all real (>= 0) sample indices
