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
task=translation

tag=baseline_nlplab

# cp ${pwd}/train-python.sh ${model_dir}
# cp ${config_dir}/train.yaml ${model_dir}
# cp ${root}/fairseq/fairseq/models/transformer/transformer_base.py ${model_dir}

cmd="python3 -u ${root}/fairseq/fairseq_cli/train.py 
    ${data_dir}/data-bin
    --arch transformer
    --train-config ${config_dir}/train.yaml
    --task ${task}
    --skip-invalid-size-inputs-valid-test
    --save-dir ${model_dir}
    --tensorboard-logdir ${model_dir}"

echo -e "\033[34mRun command: \n${cmd} \033[0m"
eval $cmd