import pickle
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import numpy as np
from exordium.utils.padding import pad_or_crop_time_dim
from exordium.utils.normalize import get_mean_std, standardization


def load_gt(db_root: str):
    gt_dict = {}
    for subset in ['train', 'valid', 'test']:
        with open(Path(db_root) / f'hand_features' / f'fi_{subset}_lld_au_bert.pkl', 'rb') as f:
            data = pickle.load(f)
            gt_dict[subset] = {k: y for k, x, y in data}
    return gt_dict


class EgemapsDataset(Dataset):

    def __init__(self, subset: str, db_root: str):
        self.subset = subset
        self.db_root = Path(db_root)
        self.names, self.samples, self.gt = self._load_samples(self.db_root / f'fi_{subset}_lld_au_bert.pkl')

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        names = [k for k, _, _ in records]
        x_dict = {k: x for k, x, _ in records}
        y_dict = {k: y for k, _, y in records}
        return names, x_dict, y_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.names[idx]
        x = self.samples[sample_id]
        lld, au, bert = x
        x = pad_or_crop_time_dim(lld, 1500)
        y = self.gt[sample_id]
        return x, y


class AuDataset(Dataset):

    def __init__(self, subset: str, db_root: str):
        self.subset = subset
        self.db_root = Path(db_root)
        self.names, self.samples, self.gt = self._load_samples(self.db_root / f'fi_{subset}_lld_au_bert.pkl')

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        names = [k for k, _, _ in records]
        x_dict = {k: x for k, x, _ in records}
        y_dict = {k: y for k, _, y in records}
        return names, x_dict, y_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.names[idx]
        x = self.samples[sample_id]
        lld, au, bert = x
        x = pad_or_crop_time_dim(au, 450)
        y = self.gt[sample_id]
        return x, y


def calculate_standardization(db_root: Path):
    egemaps_path = Path(db_root) / 'standardization' / 'egemaps.npz'
    if not egemaps_path.exists():
        mean, std = get_mean_std(DataLoader(EgemapsDataset('train', Path(db_root) / 'hand_features'), batch_size=100, shuffle=False), ndim=3)
        np.savez(str(egemaps_path), mean=mean, std=std)

    au_path = Path(db_root) / 'standardization' / 'au.npz'
    if not au_path.exists():
        mean, std = get_mean_std(DataLoader(AuDataset('train', Path(db_root) / 'hand_features'), batch_size=100, shuffle=False), ndim=3)
        np.savez(str(au_path), mean=mean, std=std)


class OOWFRDataset(Dataset):

    def __init__(self, subset: str, config: dict | None = None):
        self.config = config if config is not None else {}
        self.subset = subset
        self.db_root = Path(self.config.get('db_root', 'data/db_processed/fi'))
        self.samples = self._load_samples(self.db_root / 'oowfr' / f'fi_{subset}_oowfr.pkl')
        self.sample_ids = list(self.samples.keys())
        self.target_id = self.config.get('target_id', None)
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
            samples = pickle.load(f)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dict = self.samples[sample_id]
        x_lld = standardization(sample_dict['lld'], mean=self.standardization_params['egemaps']['mean'], std=self.standardization_params['egemaps']['std'])
        x_au = standardization(sample_dict['au'], mean=self.standardization_params['au']['mean'], std=self.standardization_params['au']['std'])
        x_lld = pad_or_crop_time_dim(x_lld, 1500)
        x_au = pad_or_crop_time_dim(x_au, 450)
        x_wav2vec = pad_or_crop_time_dim(sample_dict['wav2vec'], 1500)
        x_fabnet = pad_or_crop_time_dim(sample_dict['fabnet'], 450)
        x_roberta = pad_or_crop_time_dim(sample_dict['roberta'], 80)
        x = [x_lld, x_au, x_wav2vec, x_fabnet, x_roberta]
        y = sample_dict['ocean']
        if self.target_id is not None: y = np.expand_dims(y[self.target_id], -1) # () -> (1,)
        return x, y


class OOWFRDataModule(L.LightningDataModule):

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}
        self.batch_size = config.get('batch_size', 16)

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
    calculate_standardization('data/db_processed/fi') # calculate and save standardization params

    # Try datamodule
    data_module = OOWFRDataModule({'target_id': 1})
    data_module.setup()
    dl = data_module.train_dataloader()
    for x, y in tqdm(dl, total=len(dl)):        
        print('x means:', [torch.mean(seq, 1) for seq in x])
        print('x stds:', [torch.std(seq, 1) for seq in x])
        print('x shapes:', [seq.shape for seq in x])
        print('y shape:', y.shape)
        exit()