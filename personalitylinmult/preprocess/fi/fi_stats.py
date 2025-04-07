import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pandas as pd
from pathlib import Path
import pickle as pkl
import numpy as np


video_directory = 'data/db_processed/fi/videos'
output_directory = Path('data/db_processed/fi')
traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
gt_path_train = 'data/db/FI/gt/annotation_training.pkl'
gt_path_valid = 'data/db/FI/gt/annotation_validation.pkl'
gt_path_test = 'data/db/FI/gt/annotation_test.pkl' 


def open_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    gt = np.stack([list(data[trait].values()) for trait in traits], axis=1)
    ids = list(data['openness'].keys())
    return gt, ids


def get_video_lengths(directory):
    """
    Calculate the length of all videos in a given directory.
    
    Args:
        directory (str): Path to the directory containing video files
    
    Returns:
        pandas.DataFrame: A DataFrame with video filenames and their lengths in seconds
    """
    video_lengths = []
    filenames = os.listdir(directory)

    for filename in tqdm(filenames, total=len(filenames)):

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            filepath = os.path.join(directory, filename)
            try:
                clip = VideoFileClip(filepath)

                duration = clip.duration # seconds

                clip.close()

                video_lengths.append({
                    'filename': filename, 
                    'duration_seconds': duration
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(video_lengths)

    return df


def save_video_lens(video_directory, path):
    video_lengths_df = get_video_lengths(video_directory)
    video_lengths_df.to_csv(path, index=False)
    print(video_lengths_df.describe())
    print(f"Total number of videos processed: {len(video_lengths_df)}")


def load_video_lens(path):
    gt_train, ids_train = open_pkl(gt_path_train)
    gt_valid, ids_valid = open_pkl(gt_path_valid)
    gt_test, ids_test = open_pkl(gt_path_test)

    try:
        df = pd.read_csv(path)

        if 'duration_seconds' in df.columns:
            total_duration = df['duration_seconds'].sum()
            
            print(f"Total duration of all videos: {int(total_duration)} seconds")
            print(f"Total duration of all videos: {total_duration/3600:.1f} hours")

            for subset, ids in zip(['train', 'valid', 'test'], [ids_train, ids_valid, ids_test]):
                df_subset = df[df['filename'].isin(ids)]
                total_duration_subset = df_subset['duration_seconds'].sum()

                print(f"Number of videos in {subset}:", len(ids))
                print(f"Total duration of {subset} videos: {int(total_duration_subset)} seconds")
                print(f"Total duration of {subset} videos: {total_duration_subset/3600:.1f} hours")
        else:
            print("The 'duration_seconds' column is missing in the CSV file.")

    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    load_video_lens(str(output_directory / 'video_lengths.csv'))