#!/usr/bin/env python3
"""Test a CLOSED OpenAI model (gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini,
...) on the MULTI-ROUND hint-selection + citation task, using the exact same
prompt and benchmark as ``gpt_oss_eval/multi-cite-gpt-eval/`` and reporting the
same ``_summary.json`` stats.

Design: a THIN wrapper around ``multi-cite-gpt-eval/run_multi_selection.py``. We
reuse it wholesale so the eval is apples-to-apples with the open-weight run:

  * the SAME multi-round Template F prompt (``prompt_template_multiF.py``, with
    per-hint ``status: completed|pending``),
  * the SAME static benchmark (``multi-cite-gpt-eval/benchmark.jsonl``, 266 rows),
  * the SAME parser and the SAME scoring/summary:
      - ``mean_agreement_strict`` / ``mean_agreement_merged`` (x.0==x.1),
      - ``majority_strict_rate`` / ``majority_merged_rate`` / ``mean_self_consistency``,
      - ``selected_completed_rate`` (picks an already-completed hint; should be ~0),
      - ``newly_completed_recall`` (re-recognizes achieved-but-unmarked pending hints),
      - ``citation`` (verbatim-quote tiers), ``by_step`` breakdown.

The ONLY thing swapped is the model-call layer (``mdrv.one_sample`` →
``openai_one_sample`` from ``openai_sampler``), which speaks the real OpenAI API
and adapts per-model quirks (reasoning models: ``max_completion_tokens`` /
``reasoning_effort``, no ``temperature`` / ``top_p``).

One model per invocation; ``run_openai_multi_eval.sh`` loops the model list.

Examples:
    OPENAI_API_KEY=sk-... python run_openai_multi_selection.py --model gpt-4o-mini -n 8
    python run_openai_multi_selection.py --model gpt-5-nano --reasoning-effort low --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

HERE = Path(__file__).resolve().parent                 # .../test_cite/openai-models
TEST_CITE = HERE.parent                                # .../test_cite
GPT_OSS = TEST_CITE / "gpt_oss_eval"
MULTI = GPT_OSS / "multi-cite-gpt-eval"                 # reuse its driver + prompt + benchmark
SELECTOR = TEST_CITE.parent                            # .../hint_rl/selector

# MULTI first so its LOCAL prompt_template_multiF wins, matching run_multi_selection.
for p in (str(SELECTOR), str(GPT_OSS), str(MULTI)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the whole multi-round driver: prompt build, scoring, citation, summary.
import run_multi_selection as mdrv                       # noqa: E402
from openai_sampler import (  # noqa: E402
    is_reasoning_model, openai_one_sample, set_reasoning_effort,
)
import pricing                                           # noqa: E402  (token cost)

# swap the model-call layer; everything else in mdrv is reused unchanged.
mdrv.one_sample = openai_one_sample

DEFAULT_BENCHMARK = MULTI / "benchmark.jsonl"

# --------------------------------------------------------------------------- #
# LOCAL editable prompt: if openai-models/prompt_template_multiF.py exists it
# DRIVES the run -- edit THAT file to revise the prompt. We load it explicitly by
# path (unique module name) and override the reused driver's prompt functions, so
# it wins regardless of sys.path ordering and never silently falls back to the
# shared gpt_oss_eval copy. If the local file is absent, mdrv's shared copy is
# used unchanged.
# --------------------------------------------------------------------------- #
import importlib.util                                    # noqa: E402

# PROMPT_FILE=<path> points the run at a different prompt module with the same
# build_prompt/render_hints_with_status API (e.g. prompt_template_multiF_progress.py
# for the per-completed-hint `progress` variant). Default: the local editable copy.
LOCAL_PROMPT = Path(os.environ.get("PROMPT_FILE") or (HERE / "prompt_template_multiF.py"))


def _load_local_prompt() -> bool:
    if not LOCAL_PROMPT.is_file():
        return False
    spec = importlib.util.spec_from_file_location("prompt_template_multiF_local", LOCAL_PROMPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mdrv.PT = mod
    mdrv.build_prompt = mod.build_prompt
    mdrv.render_hints_with_status = mod.render_hints_with_status
    return True


USING_LOCAL_PROMPT = _load_local_prompt()

# --------------------------------------------------------------------------- #
# Optional parser override: PARSER_FILE=<path to a module exporting parse_output
# + hint_id_of> swaps the reused driver's parser (default: the one it imports
# from run_hint_selection_model). Point it at script/hint_rl/utils.py to score
# with the training-side REVISED parser (lenient escape repair, JSON-array ->
# object coercion, regex hard-fallback, control-char/surrogate scrub).
# --------------------------------------------------------------------------- #
PARSER_FILE = os.environ.get("PARSER_FILE") or ""


def _load_parser_override() -> bool:
    if not PARSER_FILE:
        return False
    p = Path(PARSER_FILE).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("selector_parser_override", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mdrv.parse_output = mod.parse_output
    mdrv.hint_id_of = mod.hint_id_of
    return True


USING_PARSER_OVERRIDE = _load_parser_override()

# Lone UTF-16 surrogates (a malformed \uXXXX escape in the API's JSON survives
# json decoding as an unpaired surrogate codepoint) crash the utf-8 row-file
# write -- killed the 2026-07-29 gpt-5.4-nano run at row 38. Drop them from
# every string in the record before serializing (the parser override scrubs
# only the delivered selection fields, not the raw dump).
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _scrub_surrogates(o):
    if isinstance(o, str):
        return _SURROGATE_RE.sub("", o)
    if isinstance(o, list):
        return [_scrub_surrogates(x) for x in o]
    if isinstance(o, dict):
        return {k: _scrub_surrogates(v) for k, v in o.items()}
    return o


# --------------------------------------------------------------------------- #
def snapshot_scripts(out_dir: Path) -> None:
    """Copy every script this run depends on into out_dir/_scripts/ so a result
    dir is fully self-describing."""
    dst = out_dir / "_scripts"
    dst.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__).resolve(),
        HERE / "run_openai_multi_eval.sh",
        HERE / "compare_multi_runs.py",
        HERE / "openai_sampler.py",
        MULTI / "run_multi_selection.py",
        Path(mdrv.PT.__file__),                          # the prompt actually used (local if present)
        MULTI / "build_benchmark.py",
        GPT_OSS / "check_citations.py",
        SELECTOR / "run_hint_selection_model.py",
    ]
    if USING_PARSER_OVERRIDE:
        sources.append(Path(PARSER_FILE).expanduser().resolve())  # the parser actually used
    manifest: list[str] = []
    for src in sources:
        try:
            (dst / src.name).write_text(src.read_text())
            manifest.append(f"{src.name}  <-  {src}")
        except Exception as exc:  # noqa: BLE001
            manifest.append(f"{src.name}  <-  {src}  (FAILED: {exc})")
    (dst / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="OpenAI model id, e.g. gpt-5-nano / gpt-4.1-mini / "
                         "gpt-4.1-nano / gpt-4o-mini")
    ap.add_argument("--benchmark", default=os.environ.get("BENCHMARK", str(DEFAULT_BENCHMARK)),
                    help="multi-round benchmark.jsonl (default: the gpt-oss one, "
                         "so runs are apples-to-apples)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <here>/multi_results/multi__<model>__<ts>)")
    ap.add_argument("--steps", default=os.environ.get("STEPS") or None)
    ap.add_argument("--limit", type=lambda s: int(s) if s else None,
                    default=(int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None),
                    help="cap rows per step (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-query rows already saved in out-dir (default: resume)")
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
                    help="max_tokens (chat) / max_completion_tokens (reasoning)")
    ap.add_argument("--reasoning-effort", default=os.environ.get("REASONING_EFFORT", "low"),
                    help="reasoning models only: minimal|low|medium|high (default low)")
    ap.add_argument("--row-workers", type=int, default=int(os.environ.get("ROW_WORKERS", "8")))
    ap.add_argument("--sample-workers", type=int, default=int(os.environ.get("SAMPLE_WORKERS", "4")))
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no model calls")
    ap.add_argument("--save-raw", action="store_true", default=True)
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.api_key:
        raise SystemExit("FATAL: no API key. `source env.sh` or set OPENAI_API_KEY.")
    set_reasoning_effort(args.reasoning_effort)

    steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    rows = mdrv.load_benchmark(Path(args.benchmark), steps, args.limit)

    if args.out_dir:
        args.out_dir = Path(args.out_dir).resolve()
    else:
        tag = args.model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = HERE / "multi_results" / f"multi__{tag}__{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reasoning = is_reasoning_model(args.model)
    print(f"benchmark = {args.benchmark}  ({len(rows)} rows)")
    print(f"out_dir   = {args.out_dir}")
    print(f"endpoint  = {args.base_url}   model={args.model}"
          f"{'  [reasoning: effort=' + str(args.reasoning_effort) + ']' if reasoning else ''}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"prompt    = {mdrv.PT.__file__}  "
          f"{'(LOCAL editable copy)' if USING_LOCAL_PROMPT else '(shared gpt_oss_eval copy)'}")
    print(f"parser    = {PARSER_FILE + '  (OVERRIDE)' if USING_PARSER_OVERRIDE else 'run_hint_selection_model.parse_output (driver default)'}")
    print(f"rows      = {len(rows)}  n={args.n} "
          f"{'temp=(default) top_p=(default)' if reasoning else f'temp={args.temperature} top_p={args.top_p}'} "
          f"max_tokens={args.max_tokens} row_workers={args.row_workers}")
    if not rows:
        print("no rows; exiting.")
        return 0

    snapshot_scripts(args.out_dir)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "api_key": "***redacted***",
        "is_reasoning_model": reasoning,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    client = None if args.dry_run else mdrv.make_client(args.base_url, args.api_key)
    out_records: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=(1 if args.dry_run else args.row_workers)) as ex:
        futs = {ex.submit(mdrv.process_row, row, client, args): row for row in rows}
        bar = tqdm(as_completed(futs), total=len(futs), unit="row",
                   desc=f"openai-multi:{args.model}", dynamic_ncols=True, smoothing=0.1)
        for fut in bar:
            res = fut.result()
            if res is None:
                continue
            res = _scrub_surrogates(res)
            op = mdrv.out_path_for(args.out_dir, futs[fut])
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
            out_records.append(res)
            scored = [r for r in out_records if r.get("n_parsed", 0) > 0]
            if scored:
                bar.set_postfix(
                    agree=f"{sum(r['agreement_merged'] for r in scored)/len(scored):.2f}",
                    refresh=False)
        bar.close()

    if args.dry_run:
        plens = [r.get("_prompt_len", 0) for r in out_records]
        print(f"\n[dry-run] built {len(out_records)} prompts; "
              f"mean prompt len {sum(plens)//len(plens) if plens else 0} chars. No model calls.")
        return 0

    summary = mdrv.corpus_summary(out_records)
    summary["model"] = args.model
    summary["usage"] = pricing.usage_and_cost(out_records, args.model)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ndone in {time.time()-t0:.0f}s  ->  {args.out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_step"}, indent=2))
    u = summary["usage"]
    print(f"\ntokens: {u['prompt_tokens']:,} in + {u['completion_tokens']:,} out "
          f"= {u['total_tokens']:,}   est cost: "
          f"{('$' + format(u['est_cost_usd'], '.4f')) if u['est_cost_usd'] is not None else 'n/a (no price)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
