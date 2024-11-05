from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from exordium.video.openface import read_openface_au


DB = Path("data/db_processed/mosi")
LABEL_PATH = DB / 'mosi_label.csv'


if __name__ == "__main__":

    df = pd.read_csv(LABEL_PATH)

    with open(DB / 'roberta' / 'mosi_roberta.pkl', 'rb') as f:
        roberta_dict = pickle.load(f)

    for subset in ['train', 'valid', 'test']:

        df_subset = df[df['mode'] == subset]

        samples = {}
        for index, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f'{subset}'):
            if row['mode'] != subset: continue

            video_id = row['video_id']
            clip_id = row['clip_id']

            # collect calculated features
            try:
                egemaps_lld = np.load(DB / 'egemaps_lld' / video_id / f'{clip_id}.npy') # (T, F)
                wav2vec2 = np.load(DB / 'wav2vec2' / video_id / f'{clip_id}.npy') # (T, F)
                _, au, _ = read_openface_au(DB / 'openface' / video_id / str(clip_id) / f'{clip_id}.csv') # (T, F)
                with open(DB / 'fabnet' / video_id / f'{clip_id}.pkl', 'rb') as f:
                    _, fabnet = pickle.load(f) # (T, F)
                roberta = roberta_dict[video_id][clip_id].numpy() # (T, F)
            except:
                print(f'Invalid sample: {video_id}/{clip_id}')
                continue

            if video_id not in samples:
                samples[video_id] = {}

            sample = [egemaps_lld, au, wav2vec2, fabnet, roberta]
            samples[video_id][clip_id] = {
                'egemaps_lld': egemaps_lld,
                'au': au,
                'wav2vec2': wav2vec2,
                'fabnet': fabnet,
                'roberta': roberta,
                'sentiment': float(row['label'])
            } # oowfr

            #print(f'{video_id}/{clip_id}', float(row['label']), egemaps_lld.shape, wav2vec2.shape, au.shape, fabnet.shape, roberta.shape)
            assert all([elem.ndim == 2 and elem.shape[0] != 0 for elem in sample])

        output_path = DB / 'oowfr' / f'mosi_{subset}_oowfr.pkl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        
