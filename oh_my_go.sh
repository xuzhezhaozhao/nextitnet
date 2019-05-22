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

mkdir -p ${model_dir}
python main.py \
  --model_dir=${model_dir} \
  --export_model_dir=${export_model_dir} \
  --do_train=true \
  --do_eval=true \
  --do_export=true \
  --train_data_path='./data/mtrain.txt' \
  --eval_data_path='./data/meval.txt' \
  --batch_size=32 \
  --eval_batch_size=32 \
  --epoch=1 \
  --min_count=50 \
  --max_seq_length=50 \
  --embedding_dim=100 \
  --dilations='1,2' \
  --kernel_size=3 \
  --num_sampled=10 \
  --num_gpu=0 \
  --recall_k=20 \
  --num_parallel_calls=1 \
  --learning_rate=0.025 \
  --warmup_proportion=0.3 \
  --save_summary_steps=100 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_step_count_steps=100
