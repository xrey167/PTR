#!/bin/bash
# round 6 queue: runs `pilot.py OUT jobs...` after the pilot process named by $1 (a file name fragment) has finished.
# usage: lane.sh WAIT_FRAGMENT OUT ARM LR SEEDS...   (one arm; D 2048, d 48)
PY=/tmp/claude-0/-home-user-PTR/0cdd792f-914f-548b-b171-d14a7a81ba5b/scratchpad/c001-venv/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
wait_for=$1; out=$2; arm=$3; lr=$4; shift 4
if [ -n "$wait_for" ]; then while pgrep -f "pilot.py calib6/$wait_for" >/dev/null; do sleep 5; done; fi
jobs=""; for s in "$@"; do jobs="$jobs $s:2048:$arm:$lr:48"; done
$PY pilot.py calib6/$out.jsonl $jobs > calib6/$out.log 2>&1
