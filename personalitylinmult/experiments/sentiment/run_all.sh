#!/bin/bash

dbs=("MOSEI" "MOSI")
features=("OOWFR" "OOB" "WFR")
models=("LinMulT" "MulT")

for db in "${dbs[@]}"; do
  for feature in "${features[@]}"; do
    for model in "${models[@]}"; do
      echo "Running experiment with db=${db}, feature=${feature} and model=${model}"

      python personalitylinmult/train/train_sentiment.py \
        --output_dir "results/${db}" \
        --experiment_name "${feature}_${model}" \
        --db_config_path "config/${db}/dataloader_${feature}.yaml" \
        --model_config_path "config/${db}/model_${feature}_${model}.yaml" \
        --train_config_path "config/${db}/train.yaml"
    done
  done
done

echo "All combinations have been processed!"