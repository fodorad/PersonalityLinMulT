import os
import json
import argparse
import yaml
from pprint import pprint
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.functional import mean_absolute_error, r2_score
from exordium.utils.loss import bell_l2_l1_loss, ecl1
from linmult import LinMulT
from personalitylinmult.train.fi import OOWFRDataModule
from personalitylinmult.train.lr_scheduler import WarmupScheduler, CosineAnnealingWarmupScheduler
from personalitylinmult.train.visualization import plot_fi_metrics, plot_fi_histograms
from personalitylinmult.train.callbacks import TimeTrackingCallback


class ModelWrapper(L.LightningModule):
    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')
        self.model = model

        # Set loss function
        loss_fn = self.config.get('loss_fn', 'bell_l2_l1_loss')
        if loss_fn == 'bell_l2_l1_loss':
            self.criterion = bell_l2_l1_loss # bell_l2_l1_loss, nn.L1Loss(), ecl1
        elif loss_fn == 'ecl1':
            self.criterion = ecl1
        else:
            self.criterion = nn.L1Loss()

        # Lists to store epoch-wise metrics
        self.train_losses = []
        self.train_metrics = []

        self.valid_losses = []
        self.valid_preds = []
        self.valid_targets = []
        self.valid_metrics = []
        self.valid_r2 = []

        self.test_preds = []
        self.test_targets = []

        # Set optimizer and scheduler hyperparams
        self.base_lr = self.config.get('base_lr', 1e-3)
        self.total_steps = int(self.config.get('n_epochs', 20) * self.config.get('train_size', 6000) / self.config.get('batch_size', 16)) # n_epochs * train_size / batch_size
        self.warmup_steps = int(self.total_steps * config.get('warmup_steps', 0.05))

        # Set up the optimizer
        self.optimizer_config = self.config.get('optimizer', 'radam')
        self.lr_scheduler_config = self.config.get('lr_scheduler', 'cosine_warmup')

        if self.optimizer_config == 'radam':
            self.optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.hparams.get('weight_decay', 1e-5),
                decoupled_weight_decay=self.hparams.get('decoupled_weight_decay', True)
            )
        elif self.optimizer_config == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.base_lr,
                weight_decay=self.hparams.get('weight_decay', 1e-5)
            )
        else: # Default to Adam
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.base_lr)

        # Set up the scheduler
        if self.lr_scheduler_config == 'cosine_warmup':
            self.scheduler = CosineAnnealingWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.warmup_steps,
                max_lr=self.base_lr,
                total_steps=self.total_steps
            )
        elif self.lr_scheduler_config == 'warmup':
            self.scheduler = WarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.warmup_steps,
                base_lr=self.base_lr
            )
        else: # Default to ReduceLROnPlateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('plateau_factor', 0.1),
                patience=self.config.get('plateau_patience', 5),
                min_lr=1e-6
            )

    def forward(self, x):
        return self.model(x)[0] # LinMulT

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        train_metric = 1 - mean_absolute_error(preds, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_1-mae', train_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['train_loss']
        avg_metric = self.trainer.logged_metrics['train_1-mae']
        
        self.train_losses.append(avg_loss.item())
        self.train_metrics.append(avg_metric.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.valid_preds.append(preds)
        self.valid_targets.append(y)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all collected predictions and targets
        preds = torch.cat(self.valid_preds, dim=0)
        targets = torch.cat(self.valid_targets, dim=0)

        # Calculate 1-MAE and R^2 on the entire validation set
        r2 = r2_score(preds, targets)
        valid_metric = 1 - mean_absolute_error(preds, targets)
        self.log('valid_r2', r2, prog_bar=True, logger=True)
        self.log('valid_1-mae', valid_metric, prog_bar=True, logger=True)

        # Store for plotting
        avg_loss = self.trainer.logged_metrics['val_loss']
        self.valid_losses.append(avg_loss.item())
        self.valid_metrics.append(valid_metric.item())
        self.valid_r2.append(r2.item())
        plot_fi_metrics(self, Path(self.trainer.log_dir).parents[1])
        
        # Clear the collected predictions and targets
        self.valid_preds = []
        self.valid_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        
        self.test_preds.append(preds)
        self.test_targets.append(y)

    def on_test_epoch_end(self):
        # Concatenate all collected predictions and targets
        preds = torch.cat(self.test_preds, dim=0)
        targets = torch.cat(self.test_targets, dim=0)

        # Calculate 1-MAE and R^2 on the entire test set
        r2 = r2_score(preds, targets)
        test_metric = 1 - mean_absolute_error(preds, targets)
        self.log('test_r2', r2, prog_bar=True, logger=True)
        self.log('test_1-mae', test_metric, prog_bar=True, logger=True)

        # Save the test metrics (OCEAN-wise, average 1-MAE, and R^2)
        save_test_metrics_to_csv(self.config, preds, targets, self.trainer.log_dir)

        # Save pred and gt histograms
        np_gt = targets.cpu().detach().numpy()
        np_preds = preds.cpu().detach().numpy()
        plot_fi_histograms(self.config, np_gt, np_preds, Path(self.trainer.log_dir).parents[1])

        # Clear the collected predictions and targets
        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):
        return self.optimizer

    def lr_scheduler_step(self, scheduler, metric):
        """
        Custom step logic for the learning rate scheduler.
        This method will be called automatically by Lightning after each training step.
        """
        # Manually step the custom scheduler
        self.scheduler.step()


def save_test_metrics_to_csv(config, test_preds, test_targets, output_dir):
    """Save the final test metrics to a CSV file (OCEAN-wise, average, and R²)."""
    
    # Assuming OCEAN means we have 5 traits to predict (columns)
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    if 'target_id' in config:
        traits = [traits[config['target_id']]]
    
    # Ensure the predictions and targets have the correct shape
    assert test_preds.shape[1] == len(traits), f"Expected {len(traits)} traits, got {test_preds.shape[1]}."
    
    # Calculate 1 - MAE for each trait
    trait_metrics = {trait: 1 - mean_absolute_error(test_preds[:, i], test_targets[:, i]) for i, trait in enumerate(traits)}
    
    # Calculate the average 1 - MAE across all traits
    average_metric = torch.mean(torch.tensor(list(trait_metrics.values()))).item()
    
    # Calculate R^2 for the entire set
    r2 = r2_score(test_preds, test_targets).item()

    # Prepare the row for the CSV file
    metrics_data = {trait: trait_metrics[trait].item() for trait in traits}
    metrics_data |= {
        "Average_1-MAE": average_metric,
        "R2": r2
    }

    # Convert to DataFrame for saving
    df = pd.DataFrame([metrics_data])
    csv_file = os.path.join(output_dir, 'test_metrics.csv')
    df.to_csv(csv_file, index=False)
    print(f"Test metrics saved to {csv_file}")


def load_config(config_path):
    print(f'Loading config file from {config_path}...')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train_model(config: dict):
    torch.set_float32_matmul_precision(config.get('float32_matmul_precision', 'medium')) # medium, high, highest
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.get('output_dir', 'training_results')) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the DataModule
    n_epochs = config.get('n_epochs', 20)
    batch_size = config.get('batch_size', 16)
    data_module = OOWFRDataModule(config=config)

    # Initialize the model
    model = LinMulT(config=config)
    lightning_model = ModelWrapper(model, config=config)

    # Define the callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_1-mae',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_1-mae',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    checkpoint_last_callback = ModelCheckpoint(
        dirpath=Path(output_dir) / 'checkpoint',
        save_last=True,
    )
    time_tracker = TimeTrackingCallback(output_dir)

    callbacks = [checkpoint_callback, checkpoint_last_callback, time_tracker]

    if config.get('early_stopping', False):

        early_stopping = EarlyStopping(
            monitor='valid_1-mae',
            patience=7,
            mode='max',
            verbose=True
        )

        callbacks.append(early_stopping)

    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")

    # Define the trainer
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [0]),
        max_epochs=n_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=csv_logger,
        num_sanity_val_steps=0,
        #limit_train_batches=0.05,
        #limit_val_batches=0.05,
        #limit_test_batches=0.05,
        #fast_dev_run=True,
        gradient_clip_val=1.0
    )

    # Train the model
    if 'test_only' not in config:
        if 'ckpt_path' in config:
            trainer.fit(lightning_model, datamodule=data_module, ckpt_path=config['ckpt_path'])
        else:
            trainer.fit(lightning_model, datamodule=data_module)


    # Load model checkpoint
    print("Evaluating on the test set...")
    if 'ckpt_path' in config:
        checkpoint_path = config['ckpt_path']
    else:
        checkpoint_path = checkpoint_callback.best_model_path

    print(f"Loading model from: {checkpoint_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path)

    # Visualization
    plot_fi_metrics(lightning_model, output_dir)

    # Test subset
    test_results = trainer.test(lightning_model, datamodule=data_module)


def parse_additional_args(unknown_args):
    additional_args = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--")  # remove leading '--'
        value = unknown_args[i + 1]

        # Try to cast to integer or float if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string if it's not a number

        additional_args[key] = value
    
    if 'gpu_id' in additional_args:
        additional_args['devices'] = [additional_args.get('gpu_id', 0)]

    return additional_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for FI dataset and LinMulT OOWFR")
    parser.add_argument("--db_config_path", type=str, default='config/fi_config.yaml', help="path to the dataset config file")
    parser.add_argument("--model_config_path", type=str, default='config/fi_OOWFR_LinMulT.yaml', help="path to the model config file")
    parser.add_argument("--train_config_path", type=str, default='config/fi_train_config.yaml', help="path to the train config file")
    args, unknown = parser.parse_known_args()

    config = {}
    config |= load_config(args.db_config_path)
    config |= load_config(args.model_config_path)
    config |= load_config(args.train_config_path)

    additional_args = parse_additional_args(unknown)
    config.update(additional_args)
    pprint(config)

    train_model(config)