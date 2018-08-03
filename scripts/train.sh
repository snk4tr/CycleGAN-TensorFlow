#!/usr/bin/env bash

pip install tqdm click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python main.py \
--to_train 1 \
--log_dir /data/avatars/homiak_exp/8 \
--config_filename ./configs/photo2avatar.json \
--skip True