import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from exordium.video.facedetector import RetinaFaceDetector
from exordium.video.tracker import IouTracker
from exordium.video.fabnet import FabNetWrapper
from exordium.video.opengraphau import OpenGraphAuWrapper
from exordium.video.openface import extract_openface_singularity


DB = Path("data/db_processed/mosi")
DB_RAW_VIDEOS = DB / "Raw"
DB_VIDEOS = DB / "Converted"


def reencode_video(raw_video_path):
    converted_video_path = DB_VIDEOS / raw_video_path.parent.name / raw_video_path.name
    converted_video_path.parent.mkdir(parents=True, exist_ok=True)
    if converted_video_path.exists(): return
    os.system(f'ffmpeg -y -i {raw_video_path} -r 30 {converted_video_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess MOSI visual features.")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=2200, help='participant id slice end')
    args = parser.parse_args()

    print(f"Using GPU ID: {args.gpu_id}")
    face_detector = RetinaFaceDetector(gpu_id=args.gpu_id, batch_size=10)
    fabnet_extractor = FabNetWrapper(gpu_id=args.gpu_id)
    au_extractor = OpenGraphAuWrapper(gpu_id=args.gpu_id)

    #for raw_video_path in tqdm(sorted(list(DB_RAW_VIDEOS.glob("**/*.mp4")))):
    #    reencode_video(raw_video_path)

    video_paths = sorted(list(DB_VIDEOS.glob("**/*.mp4")))[args.start:args.end]
    print('Number of videos:', len(video_paths))

    for v, video_path in tqdm(enumerate(video_paths), total=len(video_paths), desc='Videos'):
        start_time = time.time()
        video_name = video_path.parent.name
        video_id = video_path.stem

        try:
            print("video:", video_path)
            videodetections = face_detector.detect_video(video_path, output_path=DB / 'tracker' / video_name / f'{video_id}.vdet')
            track = IouTracker(max_lost=30).label(videodetections).merge().get_center_track()
            print("detected track length:", len(track))

            ids, embeddings = fabnet_extractor.track_to_feature(track, batch_size=30, output_path=DB / 'fabnet' / video_name / f'{video_id}.pkl')
            print('fabnet embeddings:', embeddings.shape)

            ids, au = au_extractor.track_to_feature(track, batch_size=30, output_path=DB / 'opengraphau' / video_name / f'{video_id}.pkl')
            print('au:', au.shape)

            if not (DB / 'openface' / video_name / video_id / f'{video_id}.csv').exists():
                extract_openface_singularity(
                    video_path,
                    DB / 'openface' / video_name / video_id,
                    str(Path('tools/openface_runner/openface_latest.sif').resolve()), 
                    'tools/openface_runner/tools',
                    singularity_args='--bind /nas:/nas')

        except Exception as e:
            with open(DB / "mosi_skip_video.txt", "a") as f:
                f.write(f'{video_name},{video_id},{e}\n')

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")