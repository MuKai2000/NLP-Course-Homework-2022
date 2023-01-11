#! /bin/bash

# Path
root=~/NLP-Course-Homework-2022
pwd=$(pwd)

# dataset=niutrans-smt-sample
dataset=News-Commentary-small

# model set
layers=4
dim=256
heads=4
hidden=1024

tag=baseline_${dataset}_layer${layers}_${dim}_${heads}_${hidden}

dataset_dir=$root/nmt/en-zh/$dataset
data_dir=$dataset_dir/data
model_dir=$dataset_dir/model
config_dir=$root/nmt/en-zh/config

# Task
src=en
tgt=zh

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
    --warmup-init-lr 1e-7 --warmup-updates 5000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --activation-fn relu \
    --encoder-layers ${layers} \
    --encoder-embed-dim ${dim} --encoder-attention-heads ${heads} --encoder-ffn-embed-dim ${hidden} \
    --decoder-layers ${layers} \
    --decoder-embed-dim ${dim} --decoder-attention-heads ${heads} --decoder-ffn-embed-dim ${hidden} \
    --share-decoder-input-output-embed \
    --max-epoch 20 --max-update 200000 \
    --max-tokens 1024 --num-workers 8 --update-freq 4 \
    --no-progress-bar --log-interval 100 --seed 1 --report-accuracy\
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 5 \
	--save-dir ${model_dir}/${tag}/checkpoints \
    --tensorboard-logdir ${model_dir}/${tag}/tensorboard >${model_dir}/${tag}/train.log 2>&1 &

# --warmup-init-lr 1e-7 --warmup-updates 5000 \
# --share-decoder-input-output-embed \