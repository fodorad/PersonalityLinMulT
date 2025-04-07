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
        self.samples = self._load_samples(self.db_root / 'cache' / f'mosei_{subset}.pkl')

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
        self.samples = self._load_samples(self.db_root / 'cache' / f'mosei_{subset}.pkl')

    def _load_samples(self, sample_path: Path):
        with open(sample_path, 'rb') as f:
            records = pickle.load(f)
        video_ids = list(records.keys())
        samples = []
        for video_id in video_ids:
            clip_ids = list(records[video_id].keys())
            for clip_id in clip_ids:
                samples.append(records[video_id][clip_id]['opengraphau'])
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
        ds_egemaps = EgemapsDataset('train', Path(db_root))
        samples_egemaps = np.vstack(ds_egemaps.samples) # (N, F)
        egemaps_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(DataLoader(TensorDataset(samples_egemaps), batch_size=100, shuffle=False), ndim=2)
        np.savez(str(egemaps_path), mean=mean, std=std)

    au_path = Path(db_root) / 'standardization' / 'opengraphau.npz'
    if not au_path.exists():
        ds_au = AuDataset('train', Path(db_root))
        samples_au = np.vstack(ds_au.samples) # (N, F)
        au_path.parent.mkdir(parents=True, exist_ok=True)
        mean, std = get_mean_std(DataLoader(TensorDataset(samples_au), batch_size=100, shuffle=False), ndim=2)
        np.savez(str(au_path), mean=mean, std=std)


class MoseiDataset(Dataset):

    def __init__(self, subset: str, config: dict | None = None):
        self.config = config if config is not None else {}
        self.subset = subset
        self.db_root = Path(self.config.get('db_root', 'data/db_processed/mosei'))
        self.samples = self._load_samples(self.db_root / 'cache' / f'mosei_{subset}.pkl')
        self.standardization_params = self._load_standardization_params()

    def _load_standardization_params(self):
        d = {}
        data = np.load(f'{self.db_root}/standardization/egemaps.npz')
        d['egemaps'] = {}
        d['egemaps']['mean'] = torch.FloatTensor(data['mean'])
        d['egemaps']['std'] = torch.FloatTensor(data['std'])
        data = np.load(f'{self.db_root}/standardization/opengraphau.npz')
        d['opengraphau'] = {}
        d['opengraphau']['mean'] = torch.FloatTensor(data['mean'])
        d['opengraphau']['std'] = torch.FloatTensor(data['std'])
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
        x_lld = standardization(torch.FloatTensor(sample_dict['egemaps_lld']), mean=self.standardization_params['egemaps']['mean'], std=self.standardization_params['egemaps']['std'])
        x_au = standardization(torch.FloatTensor(sample_dict['opengraphau']), mean=self.standardization_params['opengraphau']['mean'], std=self.standardization_params['opengraphau']['std'])
        egemaps_lld, egemaps_lld_mask = pad_or_crop_time_dim(x_lld, 1000)
        opengraphau, opengraphau_mask = pad_or_crop_time_dim(x_au, 300)
        wav2vec2, wav2vec2_mask = pad_or_crop_time_dim(torch.FloatTensor(sample_dict['wav2vec2']), 500)
        fabnet, fabnet_mask = pad_or_crop_time_dim(torch.FloatTensor(sample_dict['fabnet']), 300)
        roberta, roberta_mask = pad_or_crop_time_dim(torch.FloatTensor(sample_dict['roberta']), 80)
        bert, bert_mask = pad_or_crop_time_dim(torch.FloatTensor(sample_dict['bert']), 80)
        y = sample_dict['sentiment']
        return {
            'egemaps_lld': egemaps_lld,
            'egemaps_lld_mask': egemaps_lld_mask,
            'opengraphau': opengraphau,
            'opengraphau_mask': opengraphau_mask,
            'wav2vec2': wav2vec2,
            'wav2vec2_mask': wav2vec2_mask,
            'fabnet': fabnet,
            'fabnet_mask': fabnet_mask,
            'roberta': roberta,
            'roberta_mask': roberta_mask,
            'bert': bert,
            'bert_mask': bert_mask,
            'sentiment': y
        }


class MoseiDataModule(L.LightningDataModule):

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config if config is not None else {}
        self.batch_size = self.config.get('batch_size', 16)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            print('[MOSEI] Load train data...')
            self.dataset_train = MoseiDataset('train', self.config)
            print('[MOSEI] Load valid data...')
            self.dataset_valid = MoseiDataset('valid', self.config)

        if stage == "test":
            print('[MOSEI] Load test data...')
            self.dataset_test = MoseiDataset('test', self.config)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.config.get('num_workers', 3))

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.config.get('num_workers', 3))
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.config.get('num_workers', 3))


if __name__ == "__main__":
    calculate_standardization('data/db_processed/mosei') # calculate and save standardization params