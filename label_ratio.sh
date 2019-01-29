#!/bin/bash

for i in 1 2 3 4 5
do
    for ratio in seq 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python test_ind.py --label_ratio $ratio --folder_num $i
    done
done
