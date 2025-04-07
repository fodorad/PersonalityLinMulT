import json
from pathlib import Path
from datetime import datetime
import lightning as L


class TimeTrackingCallback(L.Callback):

    def __init__(self, output_dir, device='cuda'):
        self.device = device
        self.output_total_path = Path(output_dir) / 'time.json'
        self.output_epoch_path = Path(output_dir) / 'time_epoch.json'
        self.train_epoch_times = []
        self.validation_epoch_times = []
        self.train_start_time = None
        self.validation_start_time = None
        self.total_validation_time = None
        self.total_training_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start_time = datetime.now()

    def on_train_epoch_end(self, trainer, pl_module):
        train_epoch_time = (datetime.now() - self.train_start_time).total_seconds()
        self.train_epoch_times.append(train_epoch_time)
        print(f"Epoch {trainer.current_epoch + 1} training time: {train_epoch_time:.2f} seconds")

        with open(self.output_epoch_path, 'w') as json_file:
            json.dump({
                'train_epoch_times_sec': self.train_epoch_times,
                'validation_epoch_times_sec': self.validation_epoch_times
            }, json_file, indent=4)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_start_time = datetime.now()

    def on_validation_epoch_end(self, trainer, pl_module):
        validation_epoch_time = (datetime.now() - self.validation_start_time).total_seconds()
        self.validation_epoch_times.append(validation_epoch_time)
        print(f"Epoch {trainer.current_epoch + 1} validation time: {validation_epoch_time:.2f} seconds")

    def on_train_end(self, trainer, pl_module):
        total_training_time = sum(self.train_epoch_times) # seconds
        total_validation_time = sum(self.validation_epoch_times) # seconds
        print(f"Total Training Time: {total_training_time:.2f} seconds | {total_training_time/3600:.2f} hours")
        print(f"Total Validation Time: {total_validation_time:.2f} seconds | {total_validation_time/3600:.2f} hours")
        self.total_validation_time = total_validation_time
        self.total_training_time = total_training_time

        with open(self.output_total_path, 'w') as json_file:
            json.dump({
                'train_epoch_times_sec': self.train_epoch_times,
                'validation_epoch_times_sec': self.validation_epoch_times,
                'total_training_time_sec': self.total_training_time,
                'total_validation_time_sec': self.total_validation_time,
            }, json_file, indent=4)