#!/bin/bash
# round 6: candidate single-knob weakenings of all_to_all (3 rounds, d 48, lr 0.006), seeds 1011-1020, pilot seeds only
# usage: run_candidates.sh NAME 'SPEC_JSON' [armd]
PY=/tmp/claude-0/-home-user-PTR/0cdd792f-914f-548b-b171-d14a7a81ba5b/scratchpad/c001-venv/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
name=$1; spec=$2; armd=$3
jobs=""
for s in 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020; do jobs="$jobs $s:2048:$name:0.006:48"; done
extra=("spec:$name=$spec")
if [ -n "$armd" ]; then extra+=("armd:$name=$armd"); fi
$PY pilot.py calib6/cand-$name.jsonl "${extra[@]}" $jobs > calib6/cand-$name.log 2>&1
