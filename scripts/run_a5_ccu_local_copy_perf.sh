#!/usr/bin/env bash
# No SO/wheel install: compile the existing isolated CCU kernel on demand.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "${repo}/scripts/run_a5_ccu_local_copy_probe.sh" \
    --benchmark --profile --copy-mode direct --sizes 4194304 \
    --warmup 100 --iters 20 "$@"
