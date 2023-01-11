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

tag=baseline_${dataset}_layer${layers}_${dim}_${heads}_${hidden}_prenorm

dataset_dir=$root/nmt/en-zh/$dataset
data_dir=$dataset_dir/data
model_dir=$dataset_dir/model

# Task
src=en
tgt=zh

CUDA_VISIBLE_DEVICES=0 fairseq-generate ${data_dir}/data-bin \
    --path ${model_dir}/$tag/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --remove-bpe \
    --scoring bleu \
    --results-path ${model_dir}/$tag