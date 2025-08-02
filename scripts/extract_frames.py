import os
import cv2
import argparse
from pathlib import Path


def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 30):
    """
    Extract frames from a video file at specified intervals and save them as images.

    :param video_path: Path to the input video file.
    :param output_dir: Directory to save the extracted frames.
    :param interval: Number of frames to skip between saved frames.
    """
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, image)
        success, image = cap.read()
        frame_count += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for frames")
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    args = parser.parse_args()

    extract_frames_from_video(args.video_path, args.output_dir, args.interval)


if __name__ == "__main__":
    main()
