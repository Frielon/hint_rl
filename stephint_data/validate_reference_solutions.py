#!/usr/bin/env python3
"""Validate per-problem reference-solution files against the source parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from generate_reference_solutions import (
    DEFAULT_INPUT,
    DEFAULT_RESULT_DIR,
    assemble_reference_solution,
    canonical_integer,
    load_problem_records,
    problem_result_path,
    sha256_file,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail when selected parquet rows have no solution yet",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum individual validation errors to print",
    )
    return parser.parse_args(argv)


def validate_result(result: Any, expected, input_digest: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["record is not a JSON object"]
    if result.get("input_sha256") != input_digest:
        errors.append("input_sha256 does not match the parquet")
    if result.get("problem_id") != expected.problem_id:
        errors.append("problem_id does not match the parquet row")
    if result.get("problem") != expected.problem:
        errors.append("extracted problem does not match the parquet row")
    if str(result.get("ground_truth")) != expected.ground_truth:
        errors.append("ground_truth does not match the parquet row")

    segments = result.get("solution_segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 4:
        errors.append("solution_segments must contain 1 to 4 items")
        return errors

    for expected_id, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            errors.append(f"segment {expected_id} is not an object")
            continue
        if segment.get("segment_id") != expected_id:
            errors.append(f"segment {expected_id} has a non-sequential segment_id")
        if not str(segment.get("title", "")).strip():
            errors.append(f"segment {expected_id} has an empty title")
        if not str(segment.get("content", "")).strip():
            errors.append(f"segment {expected_id} has empty content")

    try:
        assembled = assemble_reference_solution(segments)
    except (KeyError, TypeError) as exc:
        errors.append(f"cannot assemble reference solution: {exc}")
    else:
        if result.get("reference_solution") != assembled:
            errors.append("reference_solution is not the canonical segment join")

    generated = canonical_integer(result.get("generated_final_answer"))
    expected_answer = canonical_integer(expected.ground_truth)
    if generated != expected_answer:
        errors.append(
            "generated_final_answer does not match the integer ground truth"
        )
    return errors


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    result_dir = args.result_dir.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input parquet does not exist: {input_path}")
    if not result_dir.is_dir():
        raise SystemExit(f"result directory does not exist: {result_dir}")
    if args.max_errors < 0:
        raise SystemExit("--max-errors must be non-negative")

    expected_records = load_problem_records(
        input_path, start=args.start, limit=args.limit
    )
    expected_by_row = {record.row_index: record for record in expected_records}
    input_digest = sha256_file(input_path)

    seen: set[int] = set()
    valid = 0
    errors: list[str] = []
    for row_index, expected in expected_by_row.items():
        result_path = problem_result_path(result_dir, expected)
        if not result_path.is_file():
            continue
        seen.add(row_index)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{result_path}: invalid JSON: {exc}")
            continue
        row_errors = validate_result(result, expected, input_digest)
        if row_errors:
            errors.extend(
                f"{result_path}, row {row_index}: {message}"
                for message in row_errors
            )
        else:
            valid += 1

    missing = sorted(set(expected_by_row) - seen)
    summary = {
        "input": str(input_path),
        "result_dir": str(result_dir),
        "selected_rows": len(expected_by_row),
        "records_seen": len(seen),
        "valid_records": valid,
        "invalid_records_or_files": len(errors),
        "missing_records": len(missing),
        "allow_partial": args.allow_partial,
    }
    print(json.dumps(summary, indent=2))
    for message in errors[: args.max_errors]:
        print(f"ERROR: {message}")
    if len(errors) > args.max_errors:
        print(f"... {len(errors) - args.max_errors} more errors")
    if missing and not args.allow_partial:
        preview = ", ".join(str(index) for index in missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        print(f"ERROR: missing row indices: {preview}{suffix}")

    return 1 if errors or (missing and not args.allow_partial) else 0


if __name__ == "__main__":
    raise SystemExit(main())
