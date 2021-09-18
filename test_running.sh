#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"



b=256
logs='log_midx_pop_test'
w=0.01
lr=0.001
# for w in 0.01 0.001 0.0001
for w in 0.01
do
    # for lr in 0.01 0.001 0.0001
    for lr in 0.001
    do
        # for s in 0 1 2 3 4
        for s in 4
        do
            python run_multi_items.py -data 'amazoni' -e 50 --num_workers 8 -lr ${lr} -b $b --log_path ${logs} --sampler $s --weight_decay $w -s 2000

            python run_multi_items.py -data 'amazoni' -e 50 --num_workers 8 -lr ${lr} -b $b --log_path ${logs} --sampler $s --weight_decay $w -s 20

        done
    done
done