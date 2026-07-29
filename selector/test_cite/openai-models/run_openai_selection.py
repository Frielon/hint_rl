#!/usr/bin/env python3
"""Test a CLOSED OpenAI model (gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini,
...) on the Template F hint-selection + citation task, scored against the same
Codex reference labels as ``gpt_oss_eval/`` (``run_20260617_224209`` by default).

Design: this is a THIN wrapper around the battle-tested gpt-oss driver
``gpt_oss_eval/run_gpt_oss_selection.py``. We reuse it wholesale so the eval is
apples-to-apples with the open-weight run:

  * the SAME editable Template F prompt (``gpt_oss_eval/prompt_template_F.py``),
  * the SAME hint-pool pruning (drop ``X.0`` step-guidance hints + ``type``),
  * the SAME LaTeX-tolerant ``<output>`` parser,
  * the SAME selection scoring (``agreement_hint_id`` / ``agreement_major_step`` /
    ``majority_agrees_hint_id``) and citation summary (verbatim-quote tiers,
    folded into ``_summary.json``).

The ONLY thing we change is the model-call layer: we monkeypatch the driver's
``one_sample`` with one that speaks the real OpenAI API (``api.openai.com``) and
adapts per-model quirks:

  * reasoning models (``gpt-5*`` / ``o1*`` / ``o3*`` / ``o4*``) use
    ``max_completion_tokens`` (not ``max_tokens``), reject ``temperature`` /
    ``top_p`` overrides (only the default is allowed), and take an optional
    ``reasoning_effort``;
  * standard chat models (``gpt-4.1-*`` / ``gpt-4o-*``) use the usual
    ``temperature`` / ``top_p`` / ``max_tokens``.

Output layout mirrors the label run (``step{N}/<problem_id>/<request_id>.json``)
plus a top-level ``_summary.json``, identical to the gpt-oss run so
``compare_runs.py`` and ``gpt_oss_eval/viewer.py`` can read either.

One model per invocation; ``run_openai_eval.sh`` loops the model list.

Examples:
    # single model, full corpus, n=8:
    OPENAI_API_KEY=sk-... python run_openai_selection.py --model gpt-4o-mini -n 8
    # smoke test: 3 rows/step:
    python run_openai_selection.py --model gpt-4.1-mini --limit 3
    # reasoning model, low effort:
    python run_openai_selection.py --model gpt-5-nano --reasoning-effort low
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

HERE = Path(__file__).resolve().parent                 # .../test_cite/openai-models
TEST_CITE = HERE.parent                                # .../test_cite
GPT_OSS = TEST_CITE / "gpt_oss_eval"                   # reuse its driver + prompt + parser
SELECTOR = TEST_CITE.parent                            # .../hint_rl/selector

# gpt_oss_eval first so its LOCAL prompt_template_F / check_citations win, matching
# how run_gpt_oss_selection.py resolves them.
sys.path.insert(0, str(GPT_OSS))
sys.path.insert(0, str(SELECTOR))

# Reuse the whole gpt-oss driver: prompt building, pruning, scoring, citation.
import run_gpt_oss_selection as drv                     # noqa: E402
# OpenAI-aware model-call layer (shared with the multi-round driver).
from openai_sampler import (  # noqa: E402
    is_reasoning_model, openai_one_sample, set_reasoning_effort,
)
import pricing                                           # noqa: E402  (token cost)

# swap the model-call layer; everything else in drv is reused unchanged.
drv.one_sample = openai_one_sample

# --------------------------------------------------------------------------- #
# LOCAL editable prompt: if openai-models/prompt_template_F.py exists it DRIVES
# the run (with the default --prompt-source local) -- edit THAT file to revise
# the prompt. Loaded explicitly by path and overrides the reused driver's prompt,
# so it wins regardless of import ordering; absent, drv's shared copy is used.
# --------------------------------------------------------------------------- #
import importlib.util                                    # noqa: E402

LOCAL_PROMPT = HERE / "prompt_template_F.py"


def _load_local_prompt() -> bool:
    if not LOCAL_PROMPT.is_file():
        return False
    spec = importlib.util.spec_from_file_location("prompt_template_F_local", LOCAL_PROMPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    drv.prompt_template_F = mod
    drv.build_prompt = mod.build_prompt
    return True


USING_LOCAL_PROMPT = _load_local_prompt()


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def snapshot_scripts(out_dir: Path) -> None:
    """Copy every script this run depends on into out_dir/_scripts/ so a result
    dir is fully self-describing: this OpenAI wrapper + runner, the reused
    gpt-oss driver + Template F prompt + citation scorer + shared parser."""
    dst = out_dir / "_scripts"
    dst.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),                       # this wrapper
        HERE / "run_openai_eval.sh",                    # the entry runner
        HERE / "compare_runs.py",                       # the comparison-table tool
        HERE / "openai_sampler.py",                     # shared OpenAI model-call layer
        GPT_OSS / "run_gpt_oss_selection.py",           # reused driver
        Path(drv.prompt_template_F.__file__),           # the prompt actually used (local if present)
        GPT_OSS / "check_citations.py",                 # citation scorer
        SELECTOR / "run_hint_selection_model.py",       # shared parser + OpenAI client
    ]
    manifest: list[str] = []
    for src in sources:
        try:
            (dst / src.name).write_text(src.read_text())
            manifest.append(f"{src.name}  <-  {src}")
        except Exception as exc:  # noqa: BLE001
            manifest.append(f"{src.name}  <-  {src}  (FAILED: {exc})")
    (dst / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="OpenAI model id, e.g. gpt-5-nano / gpt-4.1-mini / "
                         "gpt-4.1-nano / gpt-4o-mini")
    ap.add_argument("--label-run", default=os.environ.get("LABEL_RUN", "run_20260617_224209"),
                    help="labeled run dir or bare run_<ts> under results/debug|runs")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <here>/results/<label_run>__<model>__<ts>)")
    ap.add_argument("--steps", default=os.environ.get("STEPS") or None,
                    help="comma list of steps to score (default: all present)")
    ap.add_argument("--limit", type=lambda s: int(s) if s else None,
                    default=(int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None),
                    help="cap rows per step (smoke test). Default: all 100/step.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-query rows already saved in out-dir (default: resume)")
    ap.add_argument("--prompt-source", choices=["local", "stored"],
                    default=os.environ.get("PROMPT_SOURCE", "local"),
                    help="local: rebuild from gpt_oss_eval/prompt_template_F.py "
                         "(default, same prompt as the gpt-oss run); stored: reuse "
                         "the exact prompt Codex saw (old schema, un-pruned hints)")
    # endpoint / model
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("-n", "--n", type=int, default=int(os.environ.get("N_SAMPLES", "8")),
                    help="samples per row (default 8; bump to 16 to match the gpt-oss run)")
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.3")),
                    help="ignored for reasoning models (gpt-5*/o*)")
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")),
                    help="ignored for reasoning models")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "16384")),
                    help="max_tokens (chat) / max_completion_tokens (reasoning, "
                         "covers hidden CoT -- keep generous)")
    ap.add_argument("--reasoning-effort", default=os.environ.get("REASONING_EFFORT", "low"),
                    help="reasoning models only: minimal|low|medium|high (default low)")
    # concurrency -- modest by default to stay under OpenAI rate limits
    ap.add_argument("--row-workers", type=int, default=int(os.environ.get("ROW_WORKERS", "8")),
                    help="rows processed concurrently")
    ap.add_argument("--sample-workers", type=int, default=int(os.environ.get("SAMPLE_WORKERS", "4")),
                    help="samples drawn concurrently within a row")
    ap.add_argument("--save-raw", action="store_true", default=True,
                    help="store each sample's raw completion (default on)")
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("FATAL: no API key. `source env.sh` or set OPENAI_API_KEY.")

    # honor --reasoning-effort (only used for reasoning models by the sampler).
    set_reasoning_effort(args.reasoning_effort)

    label_dir = drv.resolve_label_dir(args.label_run)
    steps = {int(s) for s in args.steps.split(",")} if args.steps else None

    if args.out_dir:
        args.out_dir = Path(args.out_dir).resolve()
    else:
        tag = args.model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = HERE / "results" / f"{label_dir.name}__{tag}__{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records_in = drv.discover_records(label_dir, steps, args.limit)
    reasoning = is_reasoning_model(args.model)
    print(f"label_dir = {label_dir}")
    print(f"out_dir   = {args.out_dir}")
    print(f"endpoint  = {args.base_url}   model={args.model}"
          f"{'  [reasoning: effort=' + str(args.reasoning_effort) + ']' if reasoning else ''}")
    print(f"prompt    = {args.prompt_source}  ({drv.prompt_template_F.__file__})  "
          f"{'(LOCAL editable copy)' if USING_LOCAL_PROMPT else '(shared gpt_oss_eval copy)'}")
    print(f"rows      = {len(records_in)}  n={args.n} "
          f"{'temp=(default) top_p=(default)' if reasoning else f'temp={args.temperature} top_p={args.top_p}'} "
          f"max_tokens={args.max_tokens} row_workers={args.row_workers}")
    if not records_in:
        print("no rows to score; exiting.")
        return 0

    snapshot_scripts(args.out_dir)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "api_key": "***redacted***",
        "is_reasoning_model": reasoning,
        "label_dir": str(label_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    client = drv.make_client(args.base_url, args.api_key)
    out_records: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.row_workers) as ex:
        futs = {ex.submit(drv.process_record, rec, client, args): rec for rec in records_in}
        bar = tqdm(as_completed(futs), total=len(futs), unit="row",
                   desc=f"openai:{args.model}", dynamic_ncols=True, smoothing=0.1)
        for fut in bar:
            rec = futs[fut]
            res = fut.result()
            if res is None:
                continue
            op = drv.out_path_for(args.out_dir, rec)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
            out_records.append(res)
            scored = [r for r in out_records if r.get("n_parsed", 0) > 0]
            if scored:
                bar.set_postfix(
                    agree=f"{sum(r['agreement_hint_id'] for r in scored)/len(scored):.2f}",
                    refresh=False)
        bar.close()

    summary = drv.corpus_summary(out_records)
    rid_to_trace = {str(rec.get("request_id")): drv.trace_for(rec) for rec in records_in}
    summary["citation"] = drv.corpus_citation(out_records, rid_to_trace)
    summary["model"] = args.model
    summary["usage"] = pricing.usage_and_cost(out_records, args.model)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ndone in {time.time()-t0:.0f}s  ->  {args.out_dir}")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("by_step", "citation")}, indent=2))
    u = summary["usage"]
    print(f"\ntokens: {u['prompt_tokens']:,} in + {u['completion_tokens']:,} out "
          f"= {u['total_tokens']:,}   est cost: "
          f"{('$' + format(u['est_cost_usd'], '.4f')) if u['est_cost_usd'] is not None else 'n/a (no price)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
