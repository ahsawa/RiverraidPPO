#!/bin/bash

source ../.venv/bin/activate

LOG_DIR="logs"

for dir in "$LOG_DIR"/*/
do
    dir=${dir%/}
    echo "Evaluating: $dir"
    python mainevaluate.py --log_dir "$dir" &
done

wait