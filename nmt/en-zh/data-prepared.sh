#! /bin/bash

# arg
reload=0
dataset=niutrans-smt-sample # or wmt

# Path
root=~/NLP-Course-Homework-2022
pwd=$(pwd)
dataset_dir=$root/nmt/en-zh/$dataset
if [ ! -d "$dataset_dir" ]; then
    mkdir "$dataset_dir"
fi

data_dir=$dataset_dir/data
if [ ! -d "$data_dir" ]; then
    mkdir "$data_dir"
fi
model_dir=$dataset_dir/model
if [ ! -d "$model_dir" ]; then
    mkdir "$model_dir"
fi


# Tools
tools=$root/tools
scripts=$tools/mosesdecoder/scripts

NORM_PUNC=$scripts/tokenizer/normalize-punctuation.perl
TOKENIZER=$scripts/tokenizer/tokenizer.perl
TRAIN_TC=$scripts/recaser/train-truecaser.perl
TC=$scripts/recaser/truecase.perl
BPEROOT=$tools/subword-nmt/subword_nmt
CLEAN=${scripts}/training/clean-corpus-n.perl
SPLIT=$root/nmt/split.py

# Task
src=en
tgt=zh


if [ "$dataset" = niutrans-smt-sample ]; then
    # NiuTrans.SMT Sample dataset
    if [ $reload -eq 1 ]; then
    curl https:///raw.githubusercontent.com/NiuTrans/NiuTrans.SMT/master/sample-data/sample.tar.gz >> $data_dir/sample.tar.gz
    fi
    # raw
    tar -zxf $data_dir/sample.tar.gz sample-submission-version/Raw-data/chinese.raw.txt
    tar -zxf $data_dir/sample.tar.gz sample-submission-version/Raw-data/english.raw.txt
    mv $pwd/sample-submission-version/Raw-data/*.raw.txt $data_dir/
    mv $data_dir/chinese.raw.txt $data_dir/raw.zh 
    mv $data_dir/english.raw.txt $data_dir/raw.en 
    rm -r $pwd/sample-submission-version
fi

# norm
perl ${NORM_PUNC} -l en < ${data_dir}/raw.en > ${data_dir}/norm.en
perl ${NORM_PUNC} -l zh < ${data_dir}/raw.zh > ${data_dir}/norm.zh

# jieba
python -m jieba -d " " ${data_dir}/norm.zh > ${data_dir}/norm.seg.zh

# token
${TOKENIZER} -l en < ${data_dir}/norm.en > ${data_dir}/norm.tok.en
${TOKENIZER} -l zh < ${data_dir}/norm.seg.zh > ${data_dir}/norm.seg.tok.zh

# truecase
${TRAIN_TC} --model ${model_dir}/truecase-model.en --corpus ${data_dir}/norm.tok.en
${TC} --model ${model_dir}/truecase-model.en < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.true.en

# bpe
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.tok.true.en  -s 32000 -o ${model_dir}/bpecode.en --write-vocabulary ${model_dir}/voc.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/norm.tok.true.en > ${data_dir}/norm.tok.true.bpe.en

python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.seg.tok.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/norm.seg.tok.zh > ${data_dir}/norm.seg.tok.bpe.zh

# clean
mv ${data_dir}/norm.seg.tok.bpe.zh ${data_dir}/toclean.zh
mv ${data_dir}/norm.tok.true.bpe.en ${data_dir}/toclean.en 
${CLEAN} ${data_dir}/toclean zh en ${data_dir}/clean 1 256

# split
python ${SPLIT} ${src} ${data_dir}/clean.en \
                ${tgt} ${data_dir}/clean.zh ${data_dir}/

# dic & binary
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test \
    --destdir ${data_dir}/data-bin