#!/bin/bash
# Run threshold sweep for head anatomy
# Usage: ./scripts/run_head_sweep.sh

cd "$(dirname "$0")/.."

# Create logs directory
mkdir -p logs

echo "Starting threshold sweep for HEAD anatomy..."
echo "Output: results_threshold_sweep/head/"
echo "Log: logs/head_sweep.log"

nohup python scripts/run_threshold_sweep.py \
    --anatomy head \
    --min_threshold 0.03 \
    --max_threshold 0.40 \
    --step 0.01 \
    --output_dir results_threshold_sweep \
    > logs/head_sweep.log 2>&1 &

echo "Process started in background. PID: $!"
echo "Monitor with: tail -f logs/head_sweep.log"
