#!/bin/bash

for i in 1 2 3 4 5
do
    for ratio in 1 2 3 4 5 6 7 8 9 10
    do
        python test_ind.py --label_ratio $ratio --folder_num $i
    done
done
