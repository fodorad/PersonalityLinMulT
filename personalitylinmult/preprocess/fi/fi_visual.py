import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper


DB = Path("data/db_processed/fi")
DB_VIDEOS = DB / "videos"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess FI visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=10000, help='participant id slice end')
    args = parser.parse_args()

    print(f"Using GPU ID: {args.gpu_id}")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)

    video_paths = sorted(list(DB_VIDEOS.glob("**/*.mp4")))[args.start:args.end]
    print('Number of videos:', len(video_paths))

    for v, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc='Videos'):
        start_time = time.time()
        video_name = video_path.parent.name
        video_id = video_path.stem

        if (DB / 'tracker' / video_name / f'{video_id}.vdet').exists(): continue

        try:
            print("video:", video_path)
            videodetections = face_detector.detect_video(video_path, output_path=DB / 'tracker' / video_name / f'{video_id}.vdet')
            track = IouTracker(max_lost=30).label(videodetections).merge().select_topk_biggest_bb_tracks(top_k=2).select_topk_long_tracks(top_k=2).get_center_track()
            print("detected track length:", len(track))

            ids, embeddings = fabnet_extractor.track_to_feature(track, batch_size=30, output_path=DB / 'fabnet' / video_name / f'{video_id}.pkl')
            print('fabnet embeddings:', embeddings.shape)

            ids, au = au_extractor.track_to_feature(track, batch_size=30, output_path=DB / 'opengraphau' / video_name / f'{video_id}.pkl')
            print('au:', au.shape)

        except Exception as e:
            with open(DB / "fi_skip_video.txt", "a") as f:
                f.write(f'{video_name} | {video_id} | {e}\n')

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")