import pickle
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
from exordium.utils.padding import pad_or_crop_time_dim
from exordium.utils.normalize import get_mean_std, standardization


class EgemapsDataset(Dataset):

    def __init__(self, subset: str, db_root: str):
        self.subset = subset
        self.db_root = Path(db_root)
        self.samples = self._load_samples(self.db_root / f'mosei_{subset}_oowfr.pkl')

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        video_ids = list(records.keys())
        samples = []
        for video_id in video_ids:
            clip_ids = list(records[video_id].keys())
            for clip_id in clip_ids:
                samples.append(records[video_id][clip_id]['egemaps_lld'])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lld = self.samples[idx]
        return lld


class AuDataset(Dataset):

    def __init__(self, subset: str, db_root: str):
        self.subset = subset
        self.db_root = Path(db_root)
        self.samples = self._load_samples(self.db_root / f'mosei_{subset}_oowfr.pkl')

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        video_ids = list(records.keys())
        samples = []
        for video_id in video_ids:
            clip_ids = list(records[video_id].keys())
            for clip_id in clip_ids:
                samples.append(records[video_id][clip_id]['au'])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        au = self.samples[idx]
        return au


class TensorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        print('Tensor shape:', self.samples.shape)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx], 0


def calculate_standardization(db_root: Path):

    egemaps_path = Path(db_root) / 'standardization' / 'egemaps.npz'
    if not egemaps_path.exists():
        ds_egemaps = EgemapsDataset('train', Path(db_root) / 'oowfr')
        samples_egemaps = np.vstack(ds_egemaps.samples) # (N, F)
        egemaps_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(DataLoader(TensorDataset(samples_egemaps), batch_size=100, shuffle=False), ndim=2)
        np.savez(str(egemaps_path), mean=mean, std=std)

    au_path = Path(db_root) / 'standardization' / 'au.npz'
    if not au_path.exists():
        ds_au = AuDataset('train', Path(db_root) / 'oowfr')
        samples_au = np.vstack(ds_au.samples) # (N, F)
        au_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(DataLoader(TensorDataset(samples_au), batch_size=100, shuffle=False), ndim=2)
        np.savez(str(au_path), mean=mean, std=std)


class OOWFRDataset(Dataset):

    def __init__(self, subset: str, config: dict | None = None):
        self.config = config if config is not None else {}
        self.subset = subset
        self.db_root = Path(self.config.get('db_root', 'data/db_processed/mosei'))
        self.samples = self._load_samples(self.db_root / 'oowfr' / f'mosei_{subset}_oowfr.pkl')
        self.standardization_params = self._load_standardization_params()

    def _load_standardization_params(self):
        d = {}
        data = np.load(f'{self.db_root}/standardization/egemaps.npz')
        d['egemaps'] = {}
        d['egemaps']['mean'] = torch.FloatTensor(data['mean'])
        d['egemaps']['std'] = torch.FloatTensor(data['std'])
        data = np.load(f'{self.db_root}/standardization/au.npz')
        d['au'] = {}
        d['au']['mean'] = torch.FloatTensor(data['mean'])
        d['au']['std'] = torch.FloatTensor(data['std'])
        return d

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        video_ids = list(records.keys())
        samples = []
        for video_id in video_ids:
            clip_ids = list(records[video_id].keys())
            for clip_id in clip_ids:
                samples.append(records[video_id][clip_id])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = self.samples[idx]
        x_lld = standardization(sample_dict['egemaps_lld'], mean=self.standardization_params['egemaps']['mean'], std=self.standardization_params['egemaps']['std'])
        x_au = standardization(sample_dict['au'], mean=self.standardization_params['au']['mean'], std=self.standardization_params['au']['std'])
        x_lld = pad_or_crop_time_dim(x_lld, 1000)
        x_au = pad_or_crop_time_dim(x_au, 300)
        x_wav2vec = pad_or_crop_time_dim(sample_dict['wav2vec2'], 500)
        x_fabnet = pad_or_crop_time_dim(sample_dict['fabnet'], 300)
        x_roberta = pad_or_crop_time_dim(sample_dict['roberta'], 80)
        x = [x_lld, x_au, x_wav2vec, x_fabnet, x_roberta]
        y = sample_dict['sentiment']
        return x, y


class OOWFRDataModule(L.LightningDataModule):

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}
        self.batch_size = self.config.get('batch_size', 16)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            print('Load OOWFR train data...')
            self.dataset_train = OOWFRDataset('train', self.config)
            print('Load OOWFR valid data...')
            self.dataset_valid = OOWFRDataset('valid', self.config)

        if stage == "test":
            print('Load OOWFR test data...')
            self.dataset_test = OOWFRDataset('test', self.config)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.config.get('num_workers', 3))

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.config.get('num_workers', 3))
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.config.get('num_workers', 3))


if __name__ == "__main__":
    calculate_standardization('data/db_processed/mosei') # calculate and save standardization params

    # Try datamodule
    data_module = OOWFRDataModule()
    data_module.setup('fit')
    data_module.setup('test')

    [print(f'{ds.subset} length:', len(ds)) for ds in [data_module.dataset_train, data_module.dataset_valid, data_module.dataset_test]]

    exit()
    dl = data_module.train_dataloader()
    for x, y in tqdm(dl, total=len(dl)):        
        print('x means:', [torch.mean(seq, 1) for seq in x])
        print('x stds:', [torch.std(seq, 1) for seq in x])
        print('x shapes:', [seq.shape for seq in x])
        print('y shape:', y.shape)
        exit()