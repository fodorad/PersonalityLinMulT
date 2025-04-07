from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle


DB = Path("data/db/FI")
DB_PROCESSED = Path("data/db_processed/fi")

GT_NAME = {
    'test': 'annotation_test',
    'valid': 'annotation_validation',
    'train': 'annotation_training',
}


def load_gt(subset: str):
    ocean_name = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    with open(DB / 'gt' / f'{GT_NAME[subset]}.pkl', 'rb') as f:
        gt_dict = pickle.load(f, encoding='latin1')

    video_ids = [elem for elem in gt_dict['openness'].keys()]

    gt = {}
    for video_id in video_ids:
        ocean = []

        for trait in ocean_name:
            trait_value = gt_dict[trait][video_id] # neuroticism is already converted to emotional stability
            ocean.append(trait_value)

        gt[video_id[:-4]] = np.array(ocean) # cut .mp4 from the end of filenames

    return gt


if __name__ == "__main__":

    with open(DB_PROCESSED / 'text' / 'fi_roberta.pkl', 'rb') as f:
        roberta_dict = pickle.load(f)

    with open(DB_PROCESSED / 'text' / 'fi_bert.pkl', 'rb') as f:
        bert_dict = pickle.load(f)

    for subset in ['train', 'valid', 'test']:

        gt = load_gt(subset)

        output_path = DB_PROCESSED / 'cache' / f'fi_{subset}.pkl'
        if output_path.exists(): continue

        samples = {}
        for video_id, ocean in tqdm(gt.items(), total=len(gt), desc=f'{subset}'):

            try:
                egemaps_lld = np.load(DB_PROCESSED / 'egemaps_lld' / f'{video_id}.npy') # (T, F)
                wav2vec2 = np.load(DB_PROCESSED / 'wav2vec2' / f'{video_id}.npy') # (T, F)
                with open(DB_PROCESSED / 'fabnet' / f'{video_id}.pkl', 'rb') as f:
                    _, fabnet = pickle.load(f) # (T, F)
                with open(DB_PROCESSED / 'opengraphau' / f'{video_id}.pkl', 'rb') as f:
                    _, opengraphau = pickle.load(f) # (T, F)
                roberta = roberta_dict[video_id] # (T, F)
                bert = bert_dict[video_id] # (T, F)
            except Exception as e:
                print(f'Invalid sample: {video_id} | {e}')
                continue

            sample = [egemaps_lld, opengraphau, wav2vec2, fabnet, roberta, bert]
            samples[video_id] = {
                'egemaps_lld': egemaps_lld,
                'opengraphau': opengraphau,
                'wav2vec2': wav2vec2,
                'fabnet': fabnet,
                'roberta': roberta,
                'bert': bert,
                'ocean': ocean
            }

            #print(f'{video_id}/{clip_id}', float(row['label']), egemaps_lld.shape, wav2vec2.shape, au.shape, fabnet.shape, roberta.shape)
            assert all([elem.ndim == 2 and elem.shape[0] != 0 for elem in sample])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)