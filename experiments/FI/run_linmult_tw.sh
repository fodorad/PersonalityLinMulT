#!/bin/bash

for trait_index in {0..4}; do
    python personalitylinmult/train/train_app.py \
        --output_dir results/FI \
        --experiment_name OOWFR_LinMulT_TW \
        --db_config_path config/FI/dataloader_OOWFR.yaml \
        --model_config_path config/FI/model_OOWFR_LinMulT_TW.yaml \
        --train_config_path config/FI/train.yaml \
        --target_id $trait_index
done