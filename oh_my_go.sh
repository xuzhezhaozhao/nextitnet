#! /usr/bin/env bash

set -e
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

remove_model_dir=1
if [[ ${remove_model_dir} == '1' ]]; then
    rm -rf ${model_dir}.bak
    if [[ -d ${model_dir} ]]; then
        mv ${model_dir} ${model_dir}.bak
    fi
fi

python main.py \
  --model_dir=${model_dir} \
  --export_model_dir=${export_model_dir} \
  --do_train=true \
  --do_eval=true \
  --train_data_path='./test/mtrain.txt' \
  --eval_data_path='./test/meval.txt' \
  --batch_size=8 \
  --eval_batch_size=32 \
  --epoch=10 \
  --min_count=3 \
  --max_seq_lengh=200 \
  --embedding_dim=100 \
  --dilations='1,2,4' \
  --kernel_size=3 \
  --num_sampled=100 \
  --num_gpu=0 \
  --learning_rate=0.02 \
  --warmup_proportion=0.1 \
  --save_summary_steps=10 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_step_count_steps=10
