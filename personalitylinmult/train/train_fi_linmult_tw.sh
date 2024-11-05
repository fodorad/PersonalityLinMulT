for trait_index in {0..4}; do
    python personalitylinmult/train/train_fi.py --db_config_path config/fi_config.yaml --model_config_path config/fi_OOWFR_LinMulT_tw.yaml --train_config_path config/fi_train_config.yaml --target_id $trait_index --batch_size 32 --output_dir training_results_fi_linmult_tw
done