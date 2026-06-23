#!/usr/bin/env bash
# Phiên train nhanh: 400 ep vs Minimax d=1 → 800 ep self-play.
# Ghi log ra models/fast_train.log để theo dõi.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=models/fast_train.log

echo ">>> PHASE 1: vs Minimax d=1, 400 ep" | tee "$LOG"
$PY -u train.py --board-size 15 --mode minimax --opponent-depth 1 \
    --episodes 400 --eval-every 200 --eval-games 10 --log-every 50 --no-progress >> "$LOG" 2>&1
echo "=== PHASE1 DONE ===" >> "$LOG"

echo ">>> PHASE 2: self-play, 800 ep" | tee -a "$LOG"
$PY -u train.py --board-size 15 --mode selfplay \
    --episodes 800 --eval-every 200 --eval-games 10 --log-every 50 --no-progress >> "$LOG" 2>&1
echo "=== PHASE2 DONE ===" >> "$LOG"
echo ">>> ALL DONE" >> "$LOG"
