"""Build a verl/dapo-formatted parquet from EleutherAI/hendrycks_math.

Keeps only problems with level >= 3 and matches the exact schema of
dapo_17k.parquet:
    data_source : str
    prompt      : list<struct<role:string, content:string>>
    ability     : str
    reward_model: struct<style:string, ground_truth:string>
    extra_info  : struct<split:string, index:int64, problem_id:string>
"""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math problem given by the user, "
    "reasoning step by step, and put your final answer within \\boxed{}."
)
USER_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."

DATA_SOURCE = "math_hendrycks"
ABILITY = "math"


def last_boxed_only_string(s: str):
    """Return the last \\boxed{...} (or \\fbox{...}) substring, brace-balanced."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = None
    right = None
    while i < len(s):
        c = s[i]
        if c == "{":
            depth = 1 if depth is None else depth + 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return s[idx : right + 1]


def remove_boxed(b: str):
    """Strip the \\boxed{...} wrapper, returning the inner content."""
    for prefix in ("\\boxed{", "\\fbox{"):
        if b.startswith(prefix):
            assert b.endswith("}")
            return b[len(prefix) : -1]
    return b


def level_of(level_str: str):
    # level_str like "Level 3"; some entries may be "Level ?" -> return None
    try:
        return int(level_str.strip().split()[-1])
    except (ValueError, IndexError, AttributeError):
        return None


def build(split: str, out_path: str, min_level: int = 3):
    rows = []
    skipped_no_box = 0
    skipped_level = 0
    skipped_empty = 0
    for cfg in CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg)[split]
        for ex in ds:
            lvl = level_of(ex["level"])
            if lvl is None or lvl < min_level:
                skipped_level += 1
                continue
            boxed = last_boxed_only_string(ex["solution"])
            if boxed is None:
                skipped_no_box += 1
                continue
            answer = remove_boxed(boxed).strip()
            if not answer:
                skipped_empty += 1
                continue
            problem = ex["problem"].strip()
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": problem + USER_SUFFIX},
                    ],
                    "ability": ABILITY,
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": split,
                        "index": -1,  # filled below
                        "problem_id": f"hendrycks_math-{cfg}-{split}",
                    },
                    "_type": ex["type"],
                    "_level": lvl,
                }
            )

    # assign global index + finalize problem_id
    for i, r in enumerate(rows):
        r["extra_info"]["index"] = i
        r["extra_info"]["problem_id"] = (
            f"hendrycks_math-{r['_type']}-L{r['_level']}-{split}-{i}"
        )
        del r["_type"]
        del r["_level"]

    schema = pa.schema(
        [
            pa.field("data_source", pa.large_string()),
            pa.field(
                "prompt",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                        ]
                    )
                ),
            ),
            pa.field("ability", pa.large_string()),
            pa.field(
                "reward_model",
                pa.struct(
                    [
                        pa.field("style", pa.string()),
                        pa.field("ground_truth", pa.string()),
                    ]
                ),
            ),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("split", pa.string()),
                        pa.field("index", pa.int64()),
                        pa.field("problem_id", pa.string()),
                    ]
                ),
            ),
        ]
    )

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path)
    print(f"[{split}] wrote {len(rows)} rows -> {out_path}")
    print(f"[{split}] skipped (level<{min_level}): {skipped_level}, "
          f"skipped (no boxed): {skipped_no_box}, "
          f"skipped (empty answer): {skipped_empty}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default="hendrycks_math_level3plus.parquet")
    ap.add_argument("--min-level", type=int, default=3)
    args = ap.parse_args()
    build(args.split, args.out, args.min_level)
