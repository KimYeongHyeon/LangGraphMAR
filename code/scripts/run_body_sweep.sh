#!/bin/bash
# Run threshold sweep for body anatomy
# Usage: ./scripts/run_body_sweep.sh

cd "$(dirname "$0")/.."

# Create logs directory
mkdir -p logs

echo "Starting threshold sweep for BODY anatomy..."
echo "Output: results_threshold_sweep/body/"
echo "Log: logs/body_sweep.log"

nohup python scripts/run_threshold_sweep.py \
    --anatomy body \
    --min_threshold 0.03 \
    --max_threshold 0.40 \
    --step 0.01 \
    --output_dir results_threshold_sweep \
    > logs/body_sweep.log 2>&1 &

echo "Process started in background. PID: $!"
echo "Monitor with: tail -f logs/body_sweep.log"
