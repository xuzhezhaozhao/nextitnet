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
  --do_export=true \
  --train_data_path='./test/train.txt' \
  --eval_data_path='./test/eval.txt' \
  --batch_size=8 \
  --eval_batch_size=32 \
  --epoch=10 \
  --min_count=5 \
  --max_seq_length=200 \
  --embedding_dim=100 \
  --dilations='1,2' \
  --kernel_size=3 \
  --num_sampled=20 \
  --num_gpu=0 \
  --recall_k=20 \
  --num_parallel_calls=1 \
  --learning_rate=0.025 \
  --warmup_proportion=0.3 \
  --save_summary_steps=100 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_step_count_steps=100
