#!/bin/bash

source ../.venv/bin/activate


for sn in "scenario" "scenario2"
do
    for cr in 0 1
    do
        for rs in "clip" "soft"
        do 
            clear
            
            before=$(ls -td logs/*/ | head -n 1)
            
            python mainstable.py --reward_select $rs --curriculum_select $cr --scenario $sn

            after=$(ls -td logs/*/ | head -n 1)       
        
            if [ "$after" != "$before" ]; then
                echo "New log_dir detected: $after"
                python mainevaluate.py --log_dir "$after"
            else
                echo "ERROR: no new log_dir detected"
            fi
        done
    done
done



