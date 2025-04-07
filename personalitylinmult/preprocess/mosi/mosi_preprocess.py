from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle


DB = Path("data/db_processed/mosi")
LABEL_PATH = DB / 'mosi_label.csv'


if __name__ == "__main__":

    df = pd.read_csv(LABEL_PATH)

    with open(DB / 'roberta' / 'mosi_roberta.pkl', 'rb') as f:
        roberta_dict = pickle.load(f)
    
    with open(DB / 'bert' / 'mosi_bert.pkl', 'rb') as f:
        bert_dict = pickle.load(f)

    for subset in ['train', 'valid', 'test']:

        df_subset = df[df['mode'] == subset]

        samples = {}
        for index, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f'{subset}'):
            if row['mode'] != subset: continue

            video_id = row['video_id']
            clip_id = row['clip_id']

            try:
                egemaps_lld = np.load(DB / 'egemaps_lld' / video_id / f'{clip_id}.npy') # (T, F)
                wav2vec2 = np.load(DB / 'wav2vec2' / video_id / f'{clip_id}.npy') # (T, F)
                with open(DB / 'fabnet' / video_id / f'{clip_id}.pkl', 'rb') as f:
                    _, fabnet = pickle.load(f) # (T, F)
                with open(DB / 'opengraphau' / video_id / f'{clip_id}.pkl', 'rb') as f:
                    _, opengraphau = pickle.load(f) # (T, F)
                roberta = roberta_dict[video_id][clip_id] # (T, F)
                bert = bert_dict[video_id][clip_id] # (T, F)
            except Exception as e:
                print(f'Invalid sample: {video_id}/{clip_id} | {e}')
                continue

            if video_id not in samples:
                samples[video_id] = {}

            sample = [egemaps_lld, opengraphau, wav2vec2, fabnet, roberta, bert]
            samples[video_id][clip_id] = {
                'egemaps_lld': egemaps_lld,
                'opengraphau': opengraphau,
                'wav2vec2': wav2vec2,
                'fabnet': fabnet,
                'roberta': roberta,
                'bert': bert,
                'sentiment': float(row['label'])
            }

            assert all([elem.ndim == 2 and elem.shape[0] != 0 for elem in sample])

        output_path = DB / 'cache' / f'mosi_{subset}.pkl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)