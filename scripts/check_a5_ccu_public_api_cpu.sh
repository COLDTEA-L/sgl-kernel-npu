#!/usr/bin/env bash
set -Eeuo pipefail
A5_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
A5_OUT=${1:-/home/l00934901/profiling}
mkdir -p "$A5_OUT"
A5_CHECK=$(mktemp -d "$A5_OUT/a5-public-cpu.XXXXXX")
mkdir -p "$A5_CHECK/tmp"
export TMPDIR="$A5_CHECK/tmp"
cd "$A5_REPO"
echo "CPU_RESULTS=$A5_CHECK"
for A5_TEST in eid bidirectional fp8 alltoallv peer_plan local_remote channel_sync heuristic pipeline thread_bridge dfx test_runtime; do
  python3 -m unittest discover -s tests/python/deepep \
    -p "test_a5_ccu_${A5_TEST}_cpu.py" \
    > "$A5_CHECK/$A5_TEST.log" 2>&1 || {
    cat "$A5_CHECK/$A5_TEST.log"; exit 1;
  }
  echo "CPU_TEST_PASS $A5_TEST"
  grep -E '^Ran [0-9]+ tests? in |^OK($| )' "$A5_CHECK/$A5_TEST.log"
done
echo "PUBLIC_CCU_CPU_SUITE_PASS results=$A5_CHECK"
