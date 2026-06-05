#!/usr/bin/env python3
"""Launch a Codex CLI session with a prompt and print the response."""
import subprocess

# ---- parameters ----
PROMPT = "Explain what a GRPO loss does in one paragraph."
MODEL = "gpt-5.5"
REASONING_EFFORT = "high"
CODEX_BIN = "codex"
TIMEOUT_SECONDS = 600
SANDBOX = "read-only"  # or "workspace-write"
# --------------------

cmd = [CODEX_BIN, "exec", "--skip-git-repo-check", "--sandbox", SANDBOX]
if MODEL:
    cmd += ["--model", MODEL]
if REASONING_EFFORT:
    cmd += ["--config", f"model_reasoning_effort='{REASONING_EFFORT}'"]
cmd.append("-")

subprocess.run(cmd, input=PROMPT, text=True, timeout=TIMEOUT_SECONDS)
