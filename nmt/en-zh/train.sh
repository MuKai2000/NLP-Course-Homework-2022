#! /bin/bash

# Path
root=~
pwd=$(pwd)

dataset_dir=/home/koukq0907/nmt/en-zh/niutrans-smt-sample
data_dir=$dataset_dir/data
model_dir=$dataset_dir/model

# Task
src=en
tgt=zh

tag=baseline


CUDA_VISIBLE_DEVICES=0 nohup fairseq-train ${data_dir}/data-bin --arch transformer \
	--source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 5 --num-workers 8 \
	--save-dir ${model_dir}/${tag}/checkpoints &