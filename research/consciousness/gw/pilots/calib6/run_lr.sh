#!/bin/bash
# round 6: candidate weakening by learning rate: all_to_all (3 rounds, d 48) at lr 0.001, seeds 1011-1020
PY=/tmp/claude-0/-home-user-PTR/0cdd792f-914f-548b-b171-d14a7a81ba5b/scratchpad/c001-venv/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
jobs=""
for s in 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020; do jobs="$jobs $s:2048:weak_lr001:0.001:48"; done
$PY pilot.py calib6/cand-weak_lr001.jsonl 'spec:weak_lr001={"scheme":"all_to_all"}' $jobs > calib6/cand-weak_lr001.log 2>&1
