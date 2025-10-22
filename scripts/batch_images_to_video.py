# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
from pathlib import Path
import argparse
from typing import List, Tuple


def convert_images_to_video(image_dir: Path, output_path: Path, fps: int = 30, format: str = 'jpg') -> bool:
    """
    Convert a directory of images to a video file.
    
    Args:
        image_dir: Directory containing image files
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
        format: Image format to process ('png' or 'jpg')
    
    Returns:
        True if successful, False otherwise
    """
    # Get sorted list of image files
    if format.lower() == 'jpg':
        image_files = sorted(list(image_dir.glob('*.jpg')))
    else:
        image_files = sorted(list(image_dir.glob('*.png')))
    
    if not image_files:
        print(f"  Warning: No {format.upper()} files found in {image_dir}")
        return False

    try:
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"  Error: Could not read first image {image_files[0]}")
            return False
            
        height, width = first_img.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"  Error: Could not open video writer for {output_path}")
            return False

        # Process each frame
        for image_file in image_files:
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"  Warning: Could not read {image_file}, skipping")
                continue
            out.write(frame)

        # Release video writer
        out.release()
        print(f"  ✓ Video saved to {output_path} ({len(image_files)} frames)")
        return True
        
    except Exception as e:
        print(f"  Error: Failed to create video - {str(e)}")
        return False


def process_replay_folder(replay_folder: Path, output_dir: Path, fps: int = 30, 
                          depth_format: str = 'png', rgb_format: str = 'jpg') -> List[Tuple[str, bool]]:
    """
    Process a single replay folder to generate videos for depth and rgb images.
    
    Args:
        replay_folder: Path to the replay folder
        output_dir: Directory where output videos will be saved
        fps: Frames per second for output videos
        depth_format: Image format for depth images
        rgb_format: Image format for RGB images
    
    Returns:
        List of tuples containing (video_name, success_status)
    """
    results = []
    folder_name = replay_folder.name
    
    # Define the image directories to process with their respective formats
    image_dirs = [
        ('state/depth/robot.front_camera.left.depth_image', f'{folder_name}_depth.mp4', depth_format),
        ('state/rgb/robot.front_camera.left.rgb_image', f'{folder_name}_rgb.mp4', rgb_format),
    ]
    
    for relative_path, output_filename, img_format in image_dirs:
        image_dir = replay_folder / relative_path
        
        if not image_dir.exists():
            print(f"  Warning: Directory not found: {image_dir}")
            results.append((output_filename, False))
            continue
        
        output_path = output_dir / output_filename
        print(f"  Processing: {relative_path}")
        success = convert_images_to_video(image_dir, output_path, fps, img_format)
        results.append((output_filename, success))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert image sequences to videos from replay folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 batch_images_to_video.py ~/MobilityGenData/replays
  python3 batch_images_to_video.py ~/MobilityGenData/replays --output-dir /path/to/videos
  python3 batch_images_to_video.py ~/MobilityGenData/replays --fps 60
  python3 batch_images_to_video.py ~/MobilityGenData/replays --depth-format png --rgb-format jpg
        """
    )
    parser.add_argument('replays_dir', type=str, 
                       help='Base directory containing replay folders')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory where output videos will be saved (default: replays_dir/videos)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frames per second for output videos (default: 30)')
    parser.add_argument('--depth-format', type=str, default='png', choices=['png', 'jpg'],
                       help='Image format for depth images (default: png)')
    parser.add_argument('--rgb-format', type=str, default='jpg', choices=['png', 'jpg'],
                       help='Image format for RGB images (default: jpg)')
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    
    if not replays_dir.exists():
        raise ValueError(f"Replays directory not found: {replays_dir}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = replays_dir / 'videos'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Depth format: {args.depth_format.upper()}, RGB format: {args.rgb_format.upper()}\n")
    
    # Get all subdirectories in the replays directory
    replay_folders = sorted([d for d in replays_dir.iterdir() if d.is_dir() and d.name != 'videos'])
    
    if not replay_folders:
        print(f"No subdirectories found in {replays_dir}")
        return
    
    print(f"Found {len(replay_folders)} replay folder(s) to process\n")
    
    # Process each replay folder
    all_results = []
    for i, replay_folder in enumerate(replay_folders, 1):
        print(f"[{i}/{len(replay_folders)}] Processing: {replay_folder.name}")
        results = process_replay_folder(replay_folder, output_dir, args.fps, args.depth_format, args.rgb_format)
        all_results.extend(results)
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, success in all_results if success)
    total = len(all_results)
    print(f"Successfully created: {successful}/{total} videos")
    print(f"Output directory: {output_dir}")
    
    if successful < total:
        print("\nFailed conversions:")
        for video_name, success in all_results:
            if not success:
                print(f"  - {video_name}")


if __name__ == "__main__":
    main()

