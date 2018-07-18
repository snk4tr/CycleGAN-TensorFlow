#!/usr/bin/env bash

pip install tqdm click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python main.py --to_train 0 --log_dir ./output/10 --config_filename ./configs/photo2avatar.json \
--skip True --epoch "10, 20, 10"