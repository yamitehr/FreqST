#!/usr/bin/env bash
# Run the whole FreqST prototype end to end.
set -e
cd "$(dirname "$0")"
source venv/bin/activate
echo "=== Stage 1: synthetic sanity check ==="
python -m experiments.stage1
echo "=== Stage 1b: real-video qualitative check ==="
python -m experiments.stage1b
echo "=== Stage 3: learnability (two regimes) ==="
python -m experiments.stage3
echo "=== Stage 3 verification: multi-seed + control ==="
python -m experiments.stage3_verify
echo "Done. See results/ and results/REPORT.md"
