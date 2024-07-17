import os
import glob

from tqdm import tqdm
import torch


def video_to_images(video_path: str, output_dir: str, fps: int = None):
    """Convert video to images using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    if fps is not None:
        os.system(f"ffmpeg -loglevel error -v error -stats -i '{video_path}' -vf fps={fps} '{output_dir}/%04d.png'")
    else:
        os.system(f"ffmpeg -loglevel error -v error -stats -i '{video_path}' '{output_dir}/%04d.png'")


def soundactions_to_images(soundactions_dir: str, output_dir: str, fps: int = None):
    """Convert soundactions dataset to images."""
    videos = glob.glob(os.path.join(soundactions_dir, "*.mp4"))
    print(f"=> Found {len(videos)} videos in {soundactions_dir}.")
    for video_path in tqdm(videos):
        video_name = os.path.basename(video_path).split(".")[0]
        output_path = os.path.join(output_dir, video_name)
        video_to_images(video_path, output_path, fps)


if __name__ == "__main__":
    soundactions_dir = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/Datasets/SoundActions/video-HD"
    output_dir = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/Datasets/SoundActions/video-HD-frames"
    soundactions_to_images(soundactions_dir, output_dir)
