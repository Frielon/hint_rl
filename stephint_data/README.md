# Segmented reference-solution generation

`generate_reference_solutions.py` reads the nested conversation in each
parquet `prompt`, selects the last `user` message as the problem, and invokes
`codex exec` once per row. The prompt requires Codex to return exactly one
`<result>...</result>` block whose contents are a JSON object containing one
to four sequential solution sub-steps. The script extracts and parses that
JSON, validates the generated integer answer against
`reward_model.ground_truth`, and constructs `reference_solution` by joining
the validated segments.

The required Codex response looks like:

```text
<result>
{"solution_segments": [...], "final_answer": "34"}
</result>
```

Responses with missing tags, extra text outside the tags, invalid JSON, more
than four segments, or a mismatched answer are retried.

The output identity is `(input_sha256, row_index)`, because `problem_id` is not
unique in this dataset.

Each problem is stored separately:

```text
result/
├── 000000_DAPO-Math-17k-Processed_filtered-request-127-44/
│   └── result.json
├── 000001_DAPO-Math-17k-Processed_filtered-request-127-45/
│   └── result.json
└── ...
```

The zero-padded parquet row index makes every directory unique even when
multiple rows have the same `problem_id`. A failed problem has `error.json` in
its subdirectory instead; that file is removed after a successful retry.

## Run

Authenticate Codex first if needed:

```bash
codex login
```

Check prompt extraction without making API calls:

```bash
./run_reference_solution_generation.sh --limit 2 --dry-run
```

Generate a small batch:

```bash
./run_reference_solution_generation.sh --limit 10 --max-workers 2
```

Generate or resume the full dataset:

```bash
./run_reference_solution_generation.sh
```

The default concurrency is 10 Codex calls. Override it with
`--max-workers N` when needed.

Successful records are written to `result/<row-index>_<problem-id>/result.json`.
Failures go to `error.json` in the corresponding problem directory. On a later
run, successful rows for the same parquet fingerprint are skipped and failed
rows are retried. Use `--overwrite` only when intentionally regenerating the
selected rows.

The defaults follow the referenced pipeline: model `gpt-5.5`, reasoning effort
`high`, a 600-second timeout, and two retries. They can be changed with flags,
for example:

```bash
./run_reference_solution_generation.sh \
  --model gpt-5.5 \
  --reasoning-effort high \
  --timeout 900 \
  --retries 3
```

Use `PYTHON_BIN=/path/to/python` to override the launcher's Python. A suitable
environment needs the packages in `requirements-reference-solutions.txt`.

## Validate

After a full run:

```bash
/shared_home/xutao.ma/miniconda3/envs/inference/bin/python \
  validate_reference_solutions.py --result-dir result
```

For an in-progress or deliberately partial output:

```bash
/shared_home/xutao.ma/miniconda3/envs/inference/bin/python \
  validate_reference_solutions.py --result-dir result --allow-partial
```
