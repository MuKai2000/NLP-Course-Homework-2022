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
    --encoder-attention-heads 8
    --task ${task}
    --max-tokens 4096
    --skip-invalid-size-inputs-valid-test
    --save-dir ${model_dir}
    --tensorboard-logdir ${model_dir}"

echo -e "\033[34mRun command: \n${cmd} \033[0m"
eval $cmd


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
if [ ! -d "${model_dir}/${tag}" ]; then
    mkdir "${model_dir}/${tag}"
fi

# cp ${pwd}/train.sh ${model_dir}/${tag}
# cp ${root}/fairseq/fairseq/models/transformer/transformer_base.py ${model_dir}/${tag}

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
    --encoder-embed-dim 128 --encoder-attention-heads 8 --encoder-ffn-embed-dim 2048 \
    --decoder-layers 6 \
    --decoder-embed-dim 512 --decoder-attention-heads 8 --decoder-ffn-embed-dim 2048 \
    --share-decoder-input-output-embed \
    --max-epoch 5 --max-update 200000 \
    --max-tokens 4096 --num-workers 8 \
    --no-progress-bar --log-interval 100 --seed 1 --report-accuracy\
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 3 \
	--save-dir ${model_dir}/${tag}/checkpoints \
    --tensorboard-logdir ${model_dir}/${tag}/tensorboard &