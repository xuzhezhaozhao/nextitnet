#! /usr/bin/env bash

set -e
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

model_dir=`pwd`/model_dir
export_model_dir=`pwd`/export_model_dir

python main.py \
  --model_dir=${model_dir} \
  --export_model_dir=${export_model_dir} \
  --do_train=true \
  --train_data_path='./test/kd.txt' \
  --batch_size=4 \
  --epoch=5 \
  --min_count=5 \
  --max_seq_lengh=50
  --embedding_dim=100 \
  --dilations='1,2,4' \
  --kernel_size=3 \
  --num_sampled=10 \
  --num_gpu=0 \
  --learning_rate=2e-2 \
  --warmup_proportion=0.1 \
  --save_summary_steps=10 \
  --save_checkpoints_steps=1000 \
  --keep_checkpoint_max=3 \
  --log_step_count_steps=1
