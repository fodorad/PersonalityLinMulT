#!/bin/bash

db=FI
feature=OOWFR
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_app.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml"
