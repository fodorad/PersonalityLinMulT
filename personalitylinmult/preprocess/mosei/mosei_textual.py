from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from exordium.text.roberta import RobertaWrapper
from exordium.text.bert import BertWrapper


DB = Path("data/db_processed/mosei")
DB_TEXT_ROBERTA = DB / "roberta"
DB_TEXT_BERT = DB / "bert"
LABEL_PATH = DB / 'mosei_label.csv'


if __name__ == "__main__":
    df = pd.read_csv(LABEL_PATH)

    roberta = RobertaWrapper()

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
    
    DB_TEXT_ROBERTA.mkdir(parents=True, exist_ok=True)
    with open(DB_TEXT_ROBERTA / 'mosei_roberta.pkl', 'wb') as f:
        pickle.dump(d, f)

    bert = BertWrapper()

    d = {}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text'].lower()
        feature = bert(text)
        feature = feature.squeeze(0)
        video_id = row['video_id']
        clip_id = row['clip_id']
        
        if video_id not in d:
            d[video_id] = {}

        d[video_id][clip_id] = feature.numpy()
    
    DB_TEXT_BERT.mkdir(parents=True, exist_ok=True)
    with open(DB_TEXT_BERT / 'mosei_bert.pkl', 'wb') as f:
        pickle.dump(d, f)