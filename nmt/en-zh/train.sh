#! /bin/bash

# Path
root=~/NLP-Course-Homework-2022
pwd=$(pwd)

# dataset=niutrans-smt-sample
dataset=UNv2009

dataset_dir=$root/nmt/en-zh/$dataset
data_dir=$dataset_dir/data
model_dir=$dataset_dir/model
config_dir=$root/nmt/en-zh/config

# Task
src=en
tgt=zh

tag=baseline_un_layer6_256_4_1024

if [ ! -d "${model_dir}/${tag}" ]; then
    mkdir "${model_dir}/${tag}"
fi

cp ${pwd}/train.sh ${model_dir}/${tag}
cp ${root}/fairseq/fairseq/models/transformer.py ${model_dir}/${tag}

CUDA_VISIBLE_DEVICES=0 nohup fairseq-train ${data_dir}/data-bin \
	--source-lang ${src} --target-lang ${tgt}  \
    --arch transformer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --attention-dropout 0.1 --activation-dropout 0.1 \
    --activation-fn relu \
    --encoder-layers 6 \
    --encoder-embed-dim 256 --encoder-attention-heads 4 --encoder-ffn-embed-dim 1024 \
    --decoder-layers 6 \
    --decoder-embed-dim 256 --decoder-attention-heads 4 --decoder-ffn-embed-dim 2048 \
    --share-decoder-input-output-embed \
    --max-epoch 25 --max-update 200000 \
    --max-tokens 2048 --num-workers 8 --update-freq 2 \
    --no-progress-bar --log-interval 100 --seed 1 --report-accuracy\
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 5 \
	--save-dir ${model_dir}/${tag}/checkpoints \
    --tensorboard-logdir ${model_dir}/${tag}/tensorboard &