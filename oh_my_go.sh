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
  --train_data_path='./test/utrain.txt' \
  --eval_data_path='./test/ueval.txt' \
  --batch_size=2 \
  --eval_batch_size=32 \
  --epoch=5 \
  --min_count=1 \
  --max_seq_lengh=5 \
  --embedding_dim=100 \
  --dilations='1,2' \
  --kernel_size=3 \
  --num_sampled=100 \
  --num_gpu=0 \
  --learning_rate=0.01 \
  --warmup_proportion=0.1 \
  --save_summary_steps=100 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_step_count_steps=100
