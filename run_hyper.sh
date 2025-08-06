#!/bin/bash

source ../.venv/bin/activate


for lr in "1e-4" "2.5e-4" "5e-4" 
do
    for clip in 0.1 0.15 0.20 
    do
        clear
        
        before=$(ls -td logs/*/ | head -n 1)
        
        python mainstable.py --lr $lr --clip $clip

        after=$(ls -td logs/*/ | head -n 1)       
    
        if [ "$after" != "$before" ]; then
            echo "New log_dir detected: $after"
            python mainevaluate.py --log_dir "$after"
        else
            echo "ERROR: no new log_dir detected"
        fi
    done
done



