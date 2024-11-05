from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from exordium.text.roberta import RobertaWrapper


DB = Path("data/db_processed/mosi")
DB_TEXT = DB / "roberta"
LABEL_PATH = DB / 'mosi_label.csv'


if __name__ == "__main__":
    df = pd.read_csv(LABEL_PATH)

    roberta = RobertaWrapper()
    DB_TEXT.mkdir(parents=True, exist_ok=True)

    d = {}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text'].lower()
        feature = roberta(text)
        feature = feature.squeeze(0)
        video_id = row['video_id']
        clip_id = row['clip_id']
        
        if video_id not in d:
            d[video_id] = {}

        d[video_id][clip_id] = feature.numpy()
    
    with open(DB_TEXT / 'mosi_roberta.pkl', 'wb') as f:
        pickle.dump(d, f)