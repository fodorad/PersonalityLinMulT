from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import lightning as L
from lightning.fabric import seed_everything
from torchmetrics.functional import mean_absolute_error, r2_score
from exordium.utils.loss import bell_l2_l1_loss
from linmult import LinMulT
from personalitylinmult.train.parser import argparser
from personalitylinmult.train.fi import FiDataModule
from personalitylinmult.train.history import History
from personalitylinmult.train.metrics import calculate_app_metrics
from personalitylinmult.train.callbacks import TimeTrackingCallback


class ModelWrapper(L.LightningModule):

    def __init__(self, model, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters('config')

        self.model = model

        loss_fn = config['tasks']['app']['loss_fn']
        if loss_fn == 'bell_l2_l1_loss':
            self.criterion = bell_l2_l1_loss
        else:
            self.criterion = torch.nn.L1Loss()

        self.metrics = config['tasks']['app']['metrics']

        self.train_preds = []
        self.train_targets = []

        self.valid_preds = []
        self.valid_targets = []

        self.test_preds = []
        self.test_targets = []

        self.log_dir = Path(config['experiment_dir'])
        self.history = History(self.log_dir)


    def forward(self, x, mask=None):
        preds_heads = self.model(x, mask) # LinMulT
        return preds_heads[0] # (B, 5) app


    def training_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F), ...]
        mask = [batch[feature_name + '_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        y_true = batch['app']
        y_pred = self(x, mask)
        loss = self.criterion(y_pred, y_true) # preds: (B,5); y: (B,5) or preds: (B,1); y: (B,1)
        racc = 1 - mean_absolute_error(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('train_racc', racc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.train_preds.append(y_pred) # [(B,5),...] or [(B,1),...]
        self.train_targets.append(y_true) # [(B,5),...] or [(B,1),...]
        return loss


    def on_train_epoch_end(self):
        # training_step -> validation_step -> on_valid_epoch_end -> on_train_epoch_end
        preds = torch.cat(self.train_preds, dim=0) # (N,5) or (N,1)
        targets = torch.cat(self.train_targets, dim=0) # (N,5) or (N,1)

        preds = torch.clamp(preds, min=0, max=1) # Clip predictions between 0 and 1

        preds_np = preds.cpu().detach().numpy() # Shape (N,5) or (N,1)
        targets_np = targets.cpu().detach().numpy() # Shape (N,5) or (N,1)

        metrics = calculate_app_metrics(preds_np, targets_np, self.config)
        for metric_name, metric_value in metrics.items():
            self.history.update(phase="train", task='app', metric=metric_name, value=metric_value, epoch=self.current_epoch)

            if metric_name in self.metrics:
                self.log(f'train_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)
                self.history.plot('app', metric_name, 'racc')

        avg_loss = self.trainer.logged_metrics['train_loss']
        self.history.update(phase="train", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('train_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)
        self.history.plot('all', 'avg_loss')
        self.history.save()

        self.train_preds = []
        self.train_targets = []


    def validation_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F), ...]
        mask = [batch[feature_name + '_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        y_true = batch['app']
        y_pred = self(x, mask)
        loss = self.criterion(y_pred, y_true) # preds: (B,5); y: (B,5) or preds: (B,1); y: (B,1)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.valid_preds.append(y_pred) # [(B,5),...] or [(B,1),...]
        self.valid_targets.append(y_true) # [(B,5),...] or [(B,1),...]
        return loss


    def on_validation_epoch_end(self):
        preds = torch.cat(self.valid_preds, dim=0) # (N,5) or (N,1)
        targets = torch.cat(self.valid_targets, dim=0) # (N,5) or (N,1)

        preds = torch.clamp(preds, min=0, max=1) # Clip predictions between 0 and 1

        preds_np = preds.cpu().detach().numpy() # Shape (N,5) or (N,1)
        targets_np = targets.cpu().detach().numpy() # Shape (N,5) or (N,1)

        metrics = calculate_app_metrics(preds_np, targets_np, self.config)
        for metric_name, metric_value in metrics.items():
            self.history.update(phase="valid", task='app', metric=metric_name, value=metric_value, epoch=self.current_epoch)

            if metric_name in self.metrics:
                self.log(f'valid_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)

        avg_loss = self.trainer.logged_metrics['valid_loss']
        self.history.update(phase="valid", task="all", metric="avg_loss", value=avg_loss.item(), epoch=self.current_epoch)
        self.log('valid_loss', avg_loss.item(), prog_bar=True, logger=True, on_epoch=True)

        self.valid_preds = []
        self.valid_targets = []


    def test_step(self, batch, batch_idx):
        x = [batch[feature_name] for feature_name in self.config['feature_list']] # list of shapes [(B,T,F), ...]
        mask = [batch[feature_name + '_mask'] for feature_name in self.config['feature_list']] # list of shapes [(B,T), ...]
        y_true = batch['app']
        y_pred = self(x, mask)

        self.test_preds.append(y_pred) # [(B,5),...] or [(B,1),...]
        self.test_targets.append(y_true) # [(B,5),...] or [(B,1),...]


    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds, dim=0) # # (N,5) or (N,1)
        targets = torch.cat(self.test_targets, dim=0) # # (N,5) or (N,1)

        preds = torch.clamp(preds, min=0, max=1) # Clip predictions between 0 and 1

        preds_np = preds.cpu().detach().numpy() # Shape (N,5) or (N,1)
        targets_np = targets.cpu().detach().numpy() # Shape (N,5) or (N,1)

        metrics = calculate_app_metrics(preds_np, targets_np, self.config, output_path=self.log_dir / 'metrics_test.csv')
        for metric_name, metric_value in metrics.items():
            self.history.update(phase="test", task='app', metric=metric_name, value=metric_value, epoch=self.current_epoch)

            if metric_name in self.metrics:
                self.log(f'test_{metric_name}', metric_value, prog_bar=True, logger=True, on_epoch=True)

        self.history.save_test()
        
        if 'trait_id' in self.config:
            assert preds_np.shape[1] == 1 and targets_np.shape[1] == 1
            df = pd.DataFrame({'Prediction': preds_np[:,0], 'GroundTruth': targets_np[:,0]})
            pred_file = self.log_dir / f'test_predictions_{self.config["trait_id"]}.csv'
            df.to_csv(pred_file, index=False)
            print(f"Test predictions saved to {str(pred_file)}")
        else:
            assert preds_np.shape[1] == 5 and targets_np.shape[1] == 5
            for trait_id in range(5):
                df = pd.DataFrame({'Prediction': preds_np[:,trait_id], 'GroundTruth': targets_np[:,trait_id]})
                pred_file = self.log_dir / f'test_predictions_{trait_id}.csv'
                df.to_csv(pred_file, index=False)
                print(f"Test predictions saved to {str(pred_file)}")

        self.test_preds = []
        self.test_targets = []


    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        base_lr = float(optimizer_config.get('base_lr', 1e-3))
        weight_decay = float(optimizer_config.get('weight_decay', 0))

        # Configure optimizer
        if optimizer_name == 'radam':
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=optimizer_config.get('decoupled_weight_decay', False)
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr)

        # Configure learning rate scheduler
        lr_scheduler_config = self.config.get('lr_scheduler', {})
        lr_scheduler_name = lr_scheduler_config.get('name', 'ReduceLROnPlateau')
        warmup_steps = int(self.trainer.estimated_stepping_batches * config.get('warmup_ratio', 0.))

        # Configure warmup scheduler
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(lr_scheduler_config.get('start_factor', 0.1)),
                total_iters=warmup_steps
            )
        else:
            warmup_scheduler = None

        if lr_scheduler_name == 'ReduceLROnPlateau':
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(lr_scheduler_config.get('factor', 0.1)),
                patience=int(lr_scheduler_config.get('patience', 5)),
                min_lr=1e-8
            )

            # Return for ReduceLROnPlateau (no SequentialLR needed here)
            if warmup_steps > 0:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.SequentialLR(
                            optimizer,
                            schedulers=[warmup_scheduler, main_scheduler],
                            milestones=[warmup_steps]
                        ),
                        "monitor": lr_scheduler_config.get('monitor', 'valid_loss')
                    }
                }
            else:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": main_scheduler,
                        "monitor": lr_scheduler_config.get('monitor', 'valid_loss')
                    }
                }

        if lr_scheduler_name == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr, total_steps=self.trainer.estimated_stepping_batches
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
            }

        # Main scheduler
        if lr_scheduler_name == 'CosineAnnealingLR':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['n_epochs'])
        else:
            raise ValueError(f'Given lr scheduler is not supported: {lr_scheduler_name}')

        # Combine warmup and main scheduler using SequentialLR
        if warmup_steps > 0:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            lr_scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def train_model(config: dict):
    seed_everything(config.get('seed', 42))
    torch.set_float32_matmul_precision(config.get('float32_matmul_precision', 'medium')) # medium, high, highest
    experiment_name = config.get('experiment_name', datetime.now().strftime("%Y%m%d-%H%M%S"))
    if 'target_id' in config:
        experiment_dir = Path(config.get('output_dir', 'results')) / experiment_name / "OCEAN"[config['target_id']]
    else:
        experiment_dir = Path(config.get('output_dir', 'results')) / experiment_name
    if experiment_dir.exists() and 'test_only' not in config:
        raise ValueError(f'Experiment with {experiment_name} already exists. Skip.')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config['experiment_dir'] = str(experiment_dir)

    # Initialize the DataModule
    data_module = FiDataModule(config=config)

    # Initialize the model
    model = LinMulT(config=config)
    lightning_model = ModelWrapper(model, config=config)

    # Define the callbacks and logger
    callbacks = []
    for checkpoint_config in config['checkpoints']:
        callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=experiment_dir / 'checkpoint',
            filename=f"{checkpoint_config['name']}",
            monitor=checkpoint_config['monitor'],
            mode=checkpoint_config['mode'],
            save_top_k=1,
            verbose=True,
            save_weights_only=True
        )
        callbacks.append(callback)

    config_es = config.get('early_stopping', False)
    if config_es:
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor=config_es['monitor'], # 'valid_loss'
            patience=config_es['patience'], # 10
            mode=config_es['mode'], # 'min'
            verbose=True
        )
        callbacks.append(early_stopping)

    time_tracker = TimeTrackingCallback(experiment_dir)
    callbacks.append(time_tracker)

    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=str(experiment_dir), name="csv_logs")

    # Define the trainer
    trainer = L.Trainer(
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [0]),
        max_epochs=config.get('n_epochs', 20),
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
        if 'cp_path' in config:
            trainer.fit(lightning_model, datamodule=data_module, ckpt_path=config['cp_path'])
        else:
            trainer.fit(lightning_model, datamodule=data_module)

    # Test the model on the test set
    print("Evaluating on the test set...")
    if 'cp_path' in config:
        checkpoint_path = config['cp_path']
        print(f"Loading model from: {checkpoint_path}")
    else:
        checkpoint_path = callbacks[1].best_model_path # checkpoint_valid_racc
        print(f"Loading best model from: {checkpoint_path}")

    lightning_model = ModelWrapper.load_from_checkpoint(model=model, checkpoint_path=checkpoint_path, map_location=torch.device(f'cuda:{config.get("gpu_id", 0)}'))
    test_results = trainer.test(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    config: dict = argparser()
    train_model(config)