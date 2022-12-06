#! /bin/bash

# Path
root=~/NLP-Course-Homework-2022
pwd=$(pwd)

dataset_dir=$root/nmt/en-zh/niutrans-smt-sample
data_dir=$dataset_dir/data
model_dir=$dataset_dir/model
config_dir=$root/nmt/en-zh/config

# Task
src=en
tgt=zh

tag=baseline_nlplab

CUDA_VISIBLE_DEVICES=0 nohup fairseq-train ${data_dir}/data-bin \
	--source-lang ${src} --target-lang ${tgt}  \
    --arch transformer \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --num-workers 8 \
    --keep-last-epochs 3 \
	--save-dir ${model_dir}/${tag}/checkpoints \
    --tensorboard-logdir ${model_dir}/${tag}/tensorboard &