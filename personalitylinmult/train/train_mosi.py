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
from torchmetrics.functional import mean_absolute_error
from exordium.utils.loss import bell_l2_l1_loss
from linmult import LinMulT
from personalitylinmult.train.mosi import OOWFRDataModule
from personalitylinmult.train.lr_scheduler import WarmupScheduler, CosineAnnealingWarmupScheduler
from personalitylinmult.train.visualization import plot_sentiment_metrics
from personalitylinmult.train.metrics import calculate_sentiment_metrics
from personalitylinmult.train.callbacks import TimeTrackingCallback


class ModelWrapper(L.LightningModule):
    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')
        self.model = model

        # Set loss function
        loss_fn = config.get('loss_fn', 'bell_l2_l1_loss')
        if loss_fn == 'bell_l2_l1_loss':
            self.criterion = bell_l2_l1_loss
        else:
            self.criterion = nn.L1Loss()

        # Lists to store epoch-wise metrics
        self.train_losses = []
        self.train_mae = []

        self.valid_losses = []
        self.valid_preds = []
        self.valid_targets = []
        self.valid_mae = []
        self.valid_corr = []
        self.valid_acc_2 = []
        self.valid_acc_7 = []
        self.valid_f1_7 = []
        self.valid_f1_2 = []

        self.test_preds = []
        self.test_targets = []
        self.best_test_metric = None

        # Set optimizer and scheduler hyperparams
        self.base_lr = config.get('base_lr', 1e-3)
        self.total_steps = config.get('total_steps', 7500) # n_epochs * train_size / batch_size
        self.warmup_steps = int(self.total_steps * config.get('warmup_steps', 0.05))

        # Set up the optimizer
        self.optimizer_config = config.get('optimizer', 'radam')
        self.lr_scheduler_config = config.get('lr_scheduler', 'cosine_warmup')

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
                factor=config.get('plateau_factor', 0.1),
                patience=config.get('plateau_patience', 5),
                min_lr=1e-6
            )

    def forward(self, x):
        return self.model(x)[0] # LinMulT

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(torch.squeeze(preds, dim=1), y)
        train_metric = mean_absolute_error(torch.squeeze(preds, dim=1), y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('train_mae', train_metric, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['train_loss']
        avg_metric = self.trainer.logged_metrics['train_mae']
        
        self.train_losses.append(avg_loss.item())
        self.train_mae.append(avg_metric.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(torch.squeeze(preds, dim=1), y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.valid_preds.append(preds)
        self.valid_targets.append(y)
        return loss

    def on_validation_epoch_end(self):
        # Concatenate all collected predictions and targets
        preds = torch.cat(self.valid_preds, dim=0)
        targets = torch.cat(self.valid_targets, dim=0)

        preds = torch.clamp(preds, min=-3, max=3) # Clip predictions between -3 and 3
        preds_np = preds.cpu().detach().numpy().flatten()  # Shape (N,)
        targets_np = targets.cpu().detach().numpy().flatten()  # Shape (N,)
        results = calculate_sentiment_metrics(preds_np, targets_np)

        self.log('valid_acc_2', results['acc_2'], prog_bar=True, logger=True)
        self.log('valid_acc_7', results['acc_7'], prog_bar=True, logger=True)
        self.log('valid_f1_7', results['f1_7'], prog_bar=True, logger=True)
        self.log('valid_f1_2', results['f1_2'], prog_bar=True, logger=True)
        self.log('valid_mae', results['mae'], prog_bar=True, logger=True)
        self.log('valid_corr', results['corr'], prog_bar=True, logger=True)

        # Store for plotting
        avg_loss = self.trainer.logged_metrics['val_loss']
        self.valid_losses.append(avg_loss.item())
        self.valid_acc_2.append(results['acc_2'])
        self.valid_acc_7.append(results['acc_7'])
        self.valid_f1_7.append(results['f1_7'])
        self.valid_f1_2.append(results['f1_2'])
        self.valid_mae.append(results['mae'])
        self.valid_corr.append(results['corr'])
        plot_dir = Path(self.trainer.log_dir).parents[1] if self.trainer.log_dir is not None else 'test'
        plot_sentiment_metrics(self, plot_dir)
        
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

        # Calculate MAE on the entire test set
        test_metric = mean_absolute_error(torch.squeeze(preds, dim=1), targets)
        self.log('test_mae', test_metric, prog_bar=True, logger=True)

        best_test_metric = self.best_test_metric if self.best_test_metric is not None else 'mae'

        csv_dir = Path(self.trainer.log_dir) if self.trainer.log_dir is not None else 'test'
        save_test_metrics_to_csv(preds, targets, Path(csv_dir) / best_test_metric)

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


def save_test_metrics_to_csv(test_preds, test_targets, output_dir):
    """Save the final test metrics to a CSV file (OCEAN-wise, average, and R²)."""

    test_preds = torch.clamp(test_preds, min=-3, max=3) # Clip predictions between -3 and 3
    preds_np = test_preds.cpu().numpy().flatten()  # Shape (N,)
    targets_np = test_targets.cpu().numpy().flatten()  # Shape (N,)
    results = calculate_sentiment_metrics(preds_np, targets_np)
    results_pd_dict = {k: [v] for k, v in results.items()}

    csv_file = Path(output_dir) / 'test_metrics.csv'
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results_pd_dict)
    results_df.to_csv(str(csv_file), index=False)
    print(f"Test metrics saved to {str(csv_file)}")


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
    # data_module.setup()
    # total_steps = n_epochs * len(data_module.train_dataloader())
    total_steps = n_epochs * config.get('train_size', 1284) / batch_size

    # Initialize the model
    model = LinMulT(config=config)
    lightning_model = ModelWrapper(model, config=config)

    # Define the callbacks and logger
    checkpoint_f1_2_callback = ModelCheckpoint(
        monitor='valid_f1_2',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_f1_2',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    checkpoint_f1_7_callback = ModelCheckpoint(
        monitor='valid_f1_7',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_f1_7',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    checkpoint_acc_2_callback = ModelCheckpoint(
        monitor='valid_acc_2',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_acc_2',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    checkpoint_acc_7_callback = ModelCheckpoint(
        monitor='valid_acc_7',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_acc_7',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    checkpoint_mae_callback = ModelCheckpoint(
        monitor='valid_mae',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_mae',
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )
    checkpoint_corr_callback = ModelCheckpoint(
        monitor='valid_corr',
        dirpath=Path(output_dir) / 'checkpoint',
        filename='best_val_corr',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    time_tracker = TimeTrackingCallback(output_dir)

    callbacks = [
        checkpoint_f1_2_callback,
        checkpoint_f1_7_callback,
        checkpoint_acc_2_callback,
        checkpoint_acc_7_callback,
        checkpoint_mae_callback,
        checkpoint_corr_callback,
    ] #, time_tracker]

    if config.get('early_stopping', False):

        early_stopping = EarlyStopping(
            monitor='valid_mae',
            patience=10,
            mode='min',
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
        #limit_train_batches=0.1,
        #limit_val_batches=0.1,
        #limit_test_batches=0.1,
        #fast_dev_run=True,
        gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(lightning_model, datamodule=data_module)

    # Visualization
    plot_sentiment_metrics(lightning_model, output_dir)

    # Test the model on the test set
    print("Evaluating on the test set...")
    
    best_model_path = checkpoint_mae_callback.best_model_path
    print(f"Loading best mae model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'mae'
    test_results = trainer.test(lightning_model, datamodule=data_module)

    best_model_path = checkpoint_acc_2_callback.best_model_path
    print(f"Loading best acc 2 model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'acc_2'
    test_results = trainer.test(lightning_model, datamodule=data_module)

    best_model_path = checkpoint_acc_7_callback.best_model_path
    print(f"Loading best acc 7 model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'acc_7'
    test_results = trainer.test(lightning_model, datamodule=data_module)

    best_model_path = checkpoint_f1_2_callback.best_model_path
    print(f"Loading best f1 2 model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'f1_2'
    test_results = trainer.test(lightning_model, datamodule=data_module)

    best_model_path = checkpoint_f1_7_callback.best_model_path
    print(f"Loading best f1 7 model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'f1_7'
    test_results = trainer.test(lightning_model, datamodule=data_module)

    best_model_path = checkpoint_corr_callback.best_model_path
    print(f"Loading best corr model from: {best_model_path}")
    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=best_model_path, total_steps=total_steps)
    lightning_model.best_test_metric = 'corr'
    test_results = trainer.test(lightning_model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for MOSI dataset and LinMulT OOWFR")
    parser.add_argument("--db_config_path", type=str, default='config/mosi_config.yaml', help="path to the dataset config file")
    parser.add_argument("--model_config_path", type=str, default='config/mosi_OOWFR_LinMulT.yaml', help="path to the model config file")
    parser.add_argument("--train_config_path", type=str, default='config/mosi_train_config.yaml', help="path to the train config file")
    args = parser.parse_args()

    config = {}
    config |= load_config(args.db_config_path)
    config |= load_config(args.model_config_path)
    config |= load_config(args.train_config_path)
    pprint(config)

    train_model(config)