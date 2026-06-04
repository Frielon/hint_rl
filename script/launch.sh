#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TRAIN_SCRIPT=run_drgrpo_qwen2.5_7b_npu.sh \
    bash "${SCRIPT_DIR}/ray_cluster_launch.sh"
