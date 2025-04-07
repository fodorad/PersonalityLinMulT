from pathlib import Path
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
from prettytable import PrettyTable


DB = Path("data/db/FI")

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


def prior(y_true: np.ndarray, y_pred: np.ndarray):
    # Calculate trait-wise R_acc and R^2 for OCEAN traits
    racc_values = []
    r2_values = []

    for trait_ind in range(5):
        mae = mean_absolute_error(gt[:, trait_ind], preds[:, trait_ind])
        racc = 1 - mae
        r2 = r2_score(gt[:, trait_ind], preds[:, trait_ind])
        racc_values.append(round(racc,3))
        r2_values.append(r2)

    mean_racc = np.mean(racc_values)
    mean_r2 = np.mean(r2_values)

    table = PrettyTable()
    table.field_names = ["Method", "O", "C", "E", "A", "N", "Mean R_acc", "Mean R^2"]
    table.add_row(["Prior"] + racc_values + [round(mean_racc,3), round(mean_r2,3)])

    print(table)


if __name__ == "__main__":
    gt = np.array(list(load_gt('test').values())) # (2000,5)
    mean_preds = np.mean(gt, axis=0) # (5,)
    preds = np.tile(mean_preds, (2000, 1)) # (2000,5)
    prior(gt, preds)