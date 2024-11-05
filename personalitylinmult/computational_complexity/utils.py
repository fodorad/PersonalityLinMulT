import pickle
from pathlib import Path
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from linmult import LinMulT
from personalitylinmult.train.fi import OOWFRDataset


def load_mult():
    model_config = 'config/experiment_computational_complexity/OOWFR_MulT.yaml'
    model = LinMulT(model_config)
    return model


def load_linmult():
    model_config = 'config/experiment_computational_complexity/OOWFR_LinMulT.yaml'
    model = LinMulT(model_config)
    return model


def load_sample(db_root: str = 'data/db_processed/fi', features: str = 'oowfr', batch_size: int = 64, time_multiplier: float = 1.0):
    tmp_path = Path(db_root) / 'tmp'
    tmp_path.mkdir(parents=True, exist_ok=True)

    batch_path = tmp_path / f'fi_{features}_batch.pkl'
    if batch_path.exists():
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)

    else:
        print('Load OOWFR data...')
        if features == 'oowfr':
            ds = OOWFRDataset('test')
            dl = DataLoader(ds, batch_size=1024, shuffle=False)
            batch, _ = next(iter(dl))
            batch = [inp.cpu().detach().numpy() for inp in batch]
        else:
            batch = [
                np.random.rand(1024,1500,25),
                np.random.rand(1024,450,35),
                np.random.rand(1024,80,768)
            ]

        with open(batch_path, 'wb') as f:
            pickle.dump(batch, f)

    modified_tensors = []
    for tensor in batch:
        orig_batch_size, orig_time_dim, feature_dim = tensor.shape
        sliced_tensor = tensor[:batch_size, :, :]
        new_time_dim = int(orig_time_dim * time_multiplier)

        if new_time_dim < orig_time_dim:
            sliced_tensor = sliced_tensor[:, :new_time_dim, :]
        else:
            pad_length = new_time_dim - orig_time_dim
            padding = np.zeros((batch_size, pad_length, feature_dim))
            sliced_tensor = np.concatenate((sliced_tensor, padding), axis=1)

        modified_tensors.append(sliced_tensor)

    return modified_tensors


if __name__ == "__main__":
    sample_short = load_sample(batch_size=4, time_multiplier=0.5)
    sample_long = load_sample(batch_size=8, time_multiplier=1.5)
    print('short sample shape:', [elem.shape for elem in sample_short])
    print('long sample shape:', [elem.shape for elem in sample_long])