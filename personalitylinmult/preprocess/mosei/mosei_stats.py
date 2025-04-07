import pandas as pd

file_path = "data/db_processed/mosei/mosei_label.csv"
df = pd.read_csv(file_path)

unique_speakers = df['video_id'].nunique()
mode_counts = df['mode'].value_counts()
train_count = mode_counts.get('train', 0)
valid_count = mode_counts.get('valid', 0)
test_count = mode_counts.get('test', 0)

print(f"Number of unique speakers (video_id): {unique_speakers}")
print(f"Number of train samples: {train_count}")
print(f"Number of valid samples: {valid_count}")
print(f"Number of test samples: {test_count}")

train_speakers = set(df[df['mode'] == 'train']['video_id'])
valid_speakers = set(df[df['mode'] == 'valid']['video_id'])
test_speakers = set(df[df['mode'] == 'test']['video_id'])

train_valid_overlap = train_speakers.intersection(valid_speakers)
train_test_overlap = train_speakers.intersection(test_speakers)
valid_test_overlap = valid_speakers.intersection(test_speakers)

print("Speakers in Train set:", len(train_speakers))
print("Speakers in Valid set:", len(valid_speakers))
print("Speakers in Test set:", len(test_speakers))
print("\nOverlap between Train and Valid sets:", train_valid_overlap)
print("Overlap between Train and Test sets:", train_test_overlap)
print("Overlap between Valid and Test sets:", valid_test_overlap)

has_overlap = bool(train_valid_overlap or train_test_overlap or valid_test_overlap)
print("\nAre there any speaker overlaps?", has_overlap)