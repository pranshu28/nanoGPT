#!/bin/bash


for init_opt in 'adabeta' #'sgdm' #
do
    for init_lr in 0.001 #0.1 0.01 0.001 0.0001
    do
        for init_steps in 100 #10 50 500 1000
        do
            sbatch job.sh $init_opt $init_lr $init_steps 4
            # bash job.sh $init_opt $init_lr $init_steps 1
        done
    done
done
sbatch job.sh '' 0.001 0 4
