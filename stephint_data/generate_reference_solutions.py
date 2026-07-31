#!/usr/bin/env python3
"""Generate segmented reference solutions for the single-turn DAPO parquet.

The script extracts the last user message from each row's ``prompt`` field,
asks Codex for one to four sequential solution sub-steps, constructs the full
reference solution from those sub-steps, and saves each record in its own
resumable result subdirectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "Dolci-RL-Zero-Math-7B_dapo_formatted-single-turn.parquet"
DEFAULT_RESULT_DIR = ROOT / "result"
DEFAULT_SCHEMA = ROOT / "reference_solution_response_schema.json"
DEFAULT_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.5")
DEFAULT_REASONING_EFFORT = os.environ.get("CODEX_REASONING_EFFORT", "high")
DEFAULT_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT_SECONDS", "600"))
RESULT_RESPONSE_RE = re.compile(
    r"\s*<result>\s*(.*?)\s*</result>\s*", re.DOTALL | re.IGNORECASE
)


class GenerationError(RuntimeError):
    """A Codex invocation failed or returned an invalid solution."""


@dataclass(frozen=True)
class ProblemRecord:
    row_index: int
    problem_id: str
    problem: str
    ground_truth: str
    source_metadata: dict[str, Any]


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values and nested arrays to JSON-safe objects."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    value = json_safe(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return value


def extract_problem(prompt_value: Any) -> str:
    """Extract the final user-message content from a parquet prompt value."""
    prompt = json_safe(prompt_value)
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if not stripped:
            raise ValueError("prompt is empty")
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        prompt = decoded

    if isinstance(prompt, dict):
        prompt = prompt.get("messages", prompt.get("prompt", prompt))

    if not isinstance(prompt, list):
        raise ValueError(f"unsupported prompt type: {type(prompt).__name__}")

    fallback_contents: list[str] = []
    user_contents: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        fallback_contents.append(text)
        if str(message.get("role", "")).lower() == "user":
            user_contents.append(text)

    if user_contents:
        return user_contents[-1]
    if fallback_contents:
        return fallback_contents[-1]
    raise ValueError("prompt contains no non-empty message content")


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Reading parquet requires pandas and pyarrow. Install "
            "requirements-reference-solutions.txt or run with the project's "
            "'inference' conda environment."
        ) from exc
    return pd


def load_problem_records(
    input_path: Path, *, start: int = 0, limit: Optional[int] = None
) -> list[ProblemRecord]:
    """Load and normalize a selected range of records from the parquet."""
    pd = _load_pandas()
    required = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    frame = pd.read_parquet(input_path, columns=required)

    if start < 0 or start > len(frame):
        raise SystemExit(f"--start {start} is out of range for {len(frame)} rows")
    if limit is not None and limit < 0:
        raise SystemExit("--limit must be non-negative")
    stop = len(frame) if limit is None else min(len(frame), start + limit)

    records: list[ProblemRecord] = []
    for row_index in range(start, stop):
        row = frame.iloc[row_index]
        reward_model = as_mapping(row["reward_model"], "reward_model")
        extra_info = as_mapping(row["extra_info"], "extra_info")
        ground_truth = reward_model.get("ground_truth")
        if ground_truth is None or not str(ground_truth).strip():
            raise ValueError(f"row {row_index} has no reward_model.ground_truth")
        problem_id = str(extra_info.get("problem_id") or f"row-{row_index}")
        records.append(
            ProblemRecord(
                row_index=row_index,
                problem_id=problem_id,
                problem=extract_problem(row["prompt"]),
                ground_truth=str(ground_truth).strip(),
                source_metadata={
                    "data_source": json_safe(row["data_source"]),
                    "ability": json_safe(row["ability"]),
                    "extra_info": extra_info,
                },
            )
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_generation_prompt(
    record: ProblemRecord, response_schema: Mapping[str, Any]
) -> str:
    """Build the per-problem data-generation prompt sent to Codex."""
    schema_text = json.dumps(response_schema, ensure_ascii=False, indent=2)
    return f"""Create a rigorous reference solution for the math problem below.

The dataset's verified target answer is:
{record.ground_truth}

Return the solution as 1 to 4 sequential segments. Each segment must be a
genuine sub-step of the complete solution, not a summary or a hint. Together,
the segments must contain the full derivation from the givens to the final
answer, in the order a reader should follow it.

Requirements:
- Derive and check the target answer; do not merely assert it.
- Use no more than 4 segments, even for a long solution.
- Keep distinct logical phases in distinct segments, without overlap.
- Put all mathematical reasoning needed for that phase in `content`.
- Do not add nested step labels inside `content`.
- The final segment must explicitly conclude with the answer.
- Set `final_answer` to the answer only, as an integer string.
- Answer directly without calling tools or inspecting files.

Your entire response must consist of exactly one `<result>` block. Inside the
block, write exactly one JSON object conforming to the schema below:

<result>
{{"solution_segments": [...], "final_answer": "..."}}
</result>

Do not use Markdown fences. Do not put commentary before or after the tags.

JSON schema:
{schema_text}

<problem>
{record.problem}
</problem>"""


def parse_wrapped_result(response_text: str) -> dict[str, Any]:
    """Extract and parse the sole JSON object wrapped in ``<result>`` tags."""
    match = RESULT_RESPONSE_RE.fullmatch(response_text)
    if match is None:
        raise GenerationError(
            "Codex response must be exactly one <result>...</result> block"
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise GenerationError(f"invalid JSON inside <result>: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationError("JSON inside <result> must be an object")
    return payload


def canonical_integer(value: Any) -> Optional[str]:
    """Normalize an integer-like answer, including simple TeX wrappers."""
    text = str(value).strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    boxed = re.fullmatch(r"\\boxed\s*\{\s*([^{}]+)\s*\}", text)
    if boxed:
        text = boxed.group(1).strip()
    text = text.replace(",", "")
    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    return str(int(text))


def validate_model_payload(
    payload: Any, expected_answer: str
) -> tuple[list[dict[str, Any]], str]:
    """Validate and normalize the structured response from Codex."""
    if not isinstance(payload, dict):
        raise GenerationError("Codex response is not a JSON object")
    if set(payload) != {"solution_segments", "final_answer"}:
        raise GenerationError(
            "Codex response must contain only solution_segments and final_answer"
        )

    segments = payload["solution_segments"]
    if not isinstance(segments, list) or not 1 <= len(segments) <= 4:
        raise GenerationError("solution_segments must contain 1 to 4 items")

    normalized: list[dict[str, Any]] = []
    for expected_id, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise GenerationError(f"segment {expected_id} is not an object")
        if set(segment) != {"segment_id", "title", "content"}:
            raise GenerationError(
                f"segment {expected_id} must contain segment_id, title, and content"
            )
        if segment["segment_id"] != expected_id:
            raise GenerationError(
                f"expected segment_id {expected_id}, got {segment['segment_id']!r}"
            )
        title = str(segment["title"]).strip()
        content = str(segment["content"]).strip()
        if not title or not content:
            raise GenerationError(f"segment {expected_id} has an empty title or content")
        normalized.append(
            {"segment_id": expected_id, "title": title, "content": content}
        )

    generated_answer = str(payload["final_answer"]).strip()
    expected_canonical = canonical_integer(expected_answer)
    generated_canonical = canonical_integer(generated_answer)
    if expected_canonical is None:
        raise GenerationError(f"expected answer is not an integer: {expected_answer!r}")
    if generated_canonical != expected_canonical:
        raise GenerationError(
            f"answer mismatch: expected {expected_answer!r}, got {generated_answer!r}"
        )
    return normalized, generated_answer


def assemble_reference_solution(segments: Iterable[Mapping[str, Any]]) -> str:
    """Construct the canonical full solution from its sequential sub-steps."""
    return "\n\n".join(
        f"Step {segment['segment_id']}: {segment['title']}\n{segment['content']}"
        for segment in segments
    )


def problem_subdir_name(record: ProblemRecord) -> str:
    """Return a filesystem-safe, row-unique directory name for a problem."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", record.problem_id).strip("._-")
    safe_id = safe_id[:120] or "problem"
    return f"{record.row_index:06d}_{safe_id}"


def problem_result_path(result_dir: Path, record: ProblemRecord) -> Path:
    return result_dir / problem_subdir_name(record) / "result.json"


def problem_error_path(result_dir: Path, record: ProblemRecord) -> Path:
    return result_dir / problem_subdir_name(record) / "error.json"


class CodexClient:
    """Small tagged-JSON response client around ``codex exec``."""

    def __init__(
        self,
        *,
        codex_bin: str,
        model: Optional[str],
        reasoning_effort: Optional[str],
        timeout: int,
        retries: int,
        workdir: Path,
        response_schema: Mapping[str, Any],
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.retries = retries
        self.workdir = workdir
        self.response_schema = response_schema

    def _command(self, output_path: Path) -> list[str]:
        command = [
            self.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "-C",
            str(self.workdir),
            "--output-last-message",
            str(output_path),
        ]
        if self.model:
            command.extend(["--model", self.model])
        if self.reasoning_effort:
            command.extend(
                [
                    "--config",
                    f"model_reasoning_effort='{self.reasoning_effort}'",
                ]
            )
        command.append("-")
        return command

    def generate(
        self, record: ProblemRecord
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        prompt = build_generation_prompt(record, self.response_schema)
        failures: list[str] = []
        started = time.monotonic()

        for attempt in range(1, self.retries + 2):
            with tempfile.TemporaryDirectory(
                prefix=f"reference_solution_{record.row_index}_"
            ) as temporary_dir:
                output_path = Path(temporary_dir) / "last_message.txt"
                retry_note = ""
                if failures:
                    retry_note = (
                        "\n\nA prior attempt was invalid. Correct this issue in the "
                        f"new response: {failures[-1]}"
                    )
                try:
                    completed = subprocess.run(
                        self._command(output_path),
                        input=prompt + retry_note,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                        cwd=self.workdir,
                    )
                    if completed.returncode != 0:
                        raise GenerationError(
                            f"Codex exited {completed.returncode}: "
                            f"{completed.stderr[-1500:].strip()}"
                        )
                    if not output_path.exists():
                        raise GenerationError(
                            "Codex did not write --output-last-message"
                        )
                    payload = parse_wrapped_result(
                        output_path.read_text(encoding="utf-8")
                    )
                    segments, final_answer = validate_model_payload(
                        payload, record.ground_truth
                    )
                    return segments, final_answer, {
                        "client": "codex_cli",
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "response_format": "result_wrapped_json",
                        "timeout_seconds": self.timeout,
                        "attempts": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                except (
                    GenerationError,
                    json.JSONDecodeError,
                    OSError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

            if attempt <= self.retries:
                time.sleep(min(2 ** (attempt - 1), 30))

        raise GenerationError("; ".join(failures))


def make_result(
    record: ProblemRecord,
    *,
    segments: list[dict[str, Any]],
    generated_final_answer: str,
    generation_metadata: dict[str, Any],
    input_path: Path,
    input_sha256: str,
) -> dict[str, Any]:
    return {
        "row_index": record.row_index,
        "problem_id": record.problem_id,
        "problem": record.problem,
        "ground_truth": record.ground_truth,
        "reference_solution": assemble_reference_solution(segments),
        "solution_segments": segments,
        "generated_final_answer": generated_final_answer,
        "source_metadata": record.source_metadata,
        "input_file": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "generation_metadata": generation_metadata,
    }


def load_completed_rows(
    records: Iterable[ProblemRecord], result_dir: Path, input_sha256: str
) -> set[int]:
    """Find valid per-problem result files for this exact input parquet."""
    completed: set[int] = set()
    for record in records:
        path = problem_result_path(result_dir, record)
        if not path.exists():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"warning: ignoring malformed result file {path}", file=sys.stderr)
            continue
        if not isinstance(result, dict):
            print(f"warning: ignoring non-object result file {path}", file=sys.stderr)
            continue
        if result.get("input_sha256") != input_sha256:
            raise SystemExit(
                f"{path} belongs to a different input parquet. Choose another "
                "--result-dir or use --overwrite."
            )
        if (
            result.get("row_index") == record.row_index
            and result.get("reference_solution")
        ):
            completed.add(record.row_index)
    return completed


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one problem's JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def error_record(
    record: ProblemRecord, exc: Exception, input_path: Path, input_sha256: str
) -> dict[str, Any]:
    return {
        "row_index": record.row_index,
        "problem_id": record.problem_id,
        "problem": record.problem,
        "ground_truth": record.ground_truth,
        "input_file": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Root directory containing one subdirectory per problem",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="JSON schema embedded in the Codex prompt",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort", default=DEFAULT_REASONING_EFFORT
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected rows instead of resuming them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract and preview up to three selected problems",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    result_dir = args.result_dir.resolve()
    schema_path = args.schema.resolve()

    if not input_path.is_file():
        raise SystemExit(f"input parquet does not exist: {input_path}")
    if not schema_path.is_file():
        raise SystemExit(f"response schema does not exist: {schema_path}")
    try:
        response_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"cannot read response schema {schema_path}: {exc}") from exc
    if not isinstance(response_schema, dict):
        raise SystemExit(f"response schema must be a JSON object: {schema_path}")
    if result_dir == input_path:
        raise SystemExit("--result-dir must not be the input parquet")
    if result_dir.exists() and not result_dir.is_dir():
        raise SystemExit(f"--result-dir exists but is not a directory: {result_dir}")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    if not args.dry_run and shutil.which(args.codex_bin) is None:
        raise SystemExit(f"Codex executable was not found: {args.codex_bin!r}")

    records = load_problem_records(input_path, start=args.start, limit=args.limit)
    input_digest = sha256_file(input_path)

    if args.dry_run:
        preview = {
            "input": str(input_path),
            "input_sha256": input_digest,
            "result_dir": str(result_dir),
            "selected_rows": len(records),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "problems": [
                {
                    "row_index": record.row_index,
                    "problem_id": record.problem_id,
                    "ground_truth": record.ground_truth,
                    "problem": record.problem,
                    "codex_prompt": build_generation_prompt(
                        record, response_schema
                    ),
                }
                for record in records[:3]
            ],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    result_dir.mkdir(parents=True, exist_ok=True)
    completed = set() if args.overwrite else load_completed_rows(
        records, result_dir, input_digest
    )
    pending = [record for record in records if record.row_index not in completed]
    skipped = len(records) - len(pending)

    if args.overwrite:
        for record in pending:
            existing_result = problem_result_path(result_dir, record)
            if existing_result.exists():
                existing_result.unlink()

    client = CodexClient(
        codex_bin=args.codex_bin,
        model=args.model or None,
        reasoning_effort=args.reasoning_effort or None,
        timeout=args.timeout,
        retries=args.retries,
        workdir=ROOT,
        response_schema=response_schema,
    )

    succeeded = 0
    failed = 0

    def process(record: ProblemRecord) -> dict[str, Any]:
        segments, final_answer, metadata = client.generate(record)
        return make_result(
            record,
            segments=segments,
            generated_final_answer=final_answer,
            generation_metadata=metadata,
            input_path=input_path,
            input_sha256=input_digest,
        )

    with tqdm(
        total=len(pending), desc="reference solutions", unit="problem"
    ) as progress:
        if args.max_workers == 1:
            for record in pending:
                try:
                    write_json_atomic(
                        problem_result_path(result_dir, record), process(record)
                    )
                    old_error = problem_error_path(result_dir, record)
                    if old_error.exists():
                        old_error.unlink()
                    succeeded += 1
                except Exception as exc:
                    write_json_atomic(
                        problem_error_path(result_dir, record),
                        error_record(record, exc, input_path, input_digest),
                    )
                    failed += 1
                progress.update(1)
                progress.set_postfix(ok=succeeded, failed=failed, refresh=False)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(process, record): record for record in pending
                }
                for future in as_completed(futures):
                    record = futures[future]
                    try:
                        write_json_atomic(
                            problem_result_path(result_dir, record), future.result()
                        )
                        old_error = problem_error_path(result_dir, record)
                        if old_error.exists():
                            old_error.unlink()
                        succeeded += 1
                    except Exception as exc:
                        write_json_atomic(
                            problem_error_path(result_dir, record),
                            error_record(record, exc, input_path, input_digest),
                        )
                        failed += 1
                    progress.update(1)
                    progress.set_postfix(ok=succeeded, failed=failed, refresh=False)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "result_dir": str(result_dir),
                "selected": len(records),
                "skipped_completed": skipped,
                "succeeded": succeeded,
                "failed": failed,
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
