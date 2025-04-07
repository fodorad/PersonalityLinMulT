#!/bin/bash

db=MOSI
feature=OOWFR
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml" \
    --gpu_id 0

db=MOSEI
feature=OOWFR
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml" \
    --gpu_id 0

db=MOSI
feature=OOB
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml" \
    --gpu_id 0

db=MOSEI
feature=OOB
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml"

db=MOSI
feature=WFR
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml"

db=MOSEI
feature=WFR
model=LinMulT
echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

python personalitylinmult/train/train_sentiment.py \
    --output_dir "results/${db}" \
    --experiment_name "${feature}_${model}" \
    --db_config_path "config/${db}/dataloader_${feature}.yaml" \
    --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
    --train_config_path "config/${db}/train.yaml"