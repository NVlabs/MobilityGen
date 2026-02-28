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

import argparse
import os
import json
from collections import OrderedDict
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import PIL

MAX_PATH_LENGTH = 200
MAX_DEPTH_VALUE = 100_000

# Get the paths of subdirectories within a directory.
def get_subdirectories(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])

# Get the paths of files within a directory.
def get_files_in_dir(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file()])

# Read an image file into a numpy array.
def load_image(file_path: Path) -> np.ndarray:
    file_extension = str(file_path).split(".")[-1]
    if file_extension not in ["png", "jpg", "jpeg"]:
        raise Exception(f"Unsupported file format: {file_extension}")
    return np.array(PIL.Image.open(file_path))

# Reshape an array from (n, 2) to (new_length, 2) and fill the extra space with zeros.
def pad_array(
    arr       : np.ndarray,
    new_length: int
) -> np.ndarray:
    n = arr.shape[0]
    if n > new_length:
        raise ValueError(f"Input array has more than {new_length} rows.")
    padding = ((0, new_length - n), (0, 0))  # Pad rows only, not columns
    return np.pad(arr, padding, mode='constant', constant_values=0)

# Returns true if feature is a segmentation info, false otherwise.
def is_feature_segmentation_info(state_name: str) -> bool:
    return state_name.split(".")[-1] == "segmentation_info"

# Returns true if feature is an RGB image, false otherwise.
def is_feature_rgb_image(state_name: str) -> bool:
    return state_name.split(".")[-1] == "rgb_image"

# Returns true if feature is a segmentation image, false otherwise.
def is_feature_segmentation_image(state_name: str) -> bool:
    return state_name.split(".")[-1] == "segmentation_image"

# Returns true if feature is a segmentation image, false otherwise.
def is_feature_instance_id_segmentation_image(state_name: str) -> bool:
    return state_name.split(".")[-1] == "instance_id_segmentation_image"

# Returns true if feature is a depth image, false otherwise.
def is_feature_depth_image(state_name: str) -> bool:
    return state_name.split(".")[-1] == "depth_image"

# Returns true if feature is a normal image, false otherwise.
def is_feature_normal_image(state_name: str) -> bool:
    return state_name.split(".")[-1] == "normals_image"

# Converts an idToLabel dictionary into a multi-hot encoding of the labels.
def idToLabel_to_bool_array(
    old_id_to_label: dict,
    label_to_new_id: dict
) -> np.ndarray:
    result = np.zeros(len(label_to_new_id), dtype=bool)
    for value in old_id_to_label.values():
        id = label_to_new_id[value["class"]]
        result[id] = True
    return np.array(result)

# Maps the id within the segmentation image to the new global id.
def remap_segmentation_image(
    image          : np.ndarray,
    label_to_new_id: dict,
    old_id_to_label: dict
) -> np.ndarray:
    max_id = 0
    for old_id, label in old_id_to_label.items():
        assert int(old_id) >= 0
        max_id = max(max_id, int(old_id))
        if label["class"] == "UNLABELLED":
            old_id_of_unlabeled = int(old_id)

    # TODO: Try the map method instead and compare performance.
    mapping_array = np.zeros(max_id + 1, dtype=np.uint8)
    for old_id, label in old_id_to_label.items():
        new_id = label_to_new_id[label["class"]]
        mapping_array[int(old_id)] = new_id

    # There are some incosistencies in the dataset, where some pixels indicate a certain
    # label but that label wasn't present in the old_id_to_label dictionary. We change all
    # those erroenous pixels to the unlabeled class
    if image.max() > max_id:
        image[image > max_id] = old_id_of_unlabeled
    return mapping_array[image]

# Traverse all frames and find out all possible labels, and give each an id.
def get_segmentation_label_lookup(dataset_paths: list[Path]) -> dict:
    labels = set()
    for dataset_dir in dataset_paths:
        for common_state_file in get_files_in_dir(dataset_dir / "state" / "common"):
            common_state = np.load(common_state_file, allow_pickle=True).item()
            for state_name, state_value in common_state.items():
                if is_feature_segmentation_info(state_name):
                    for val in state_value["idToLabels"].values():
                        labels.add(val["class"])

    label_to_id_lookup = OrderedDict()
    for id, label in enumerate(sorted(list(labels))):
        label_to_id_lookup[label] = id
    return label_to_id_lookup

# Load sample files to obtain necessary metadata for each feature.
def get_feature_info(
    dataset_path            : str,
    segmentation_label_to_id: dict
) -> tuple[dict, dict, dict]:
    feature_files_list = {} # Maps feature name to list of files under that feature.
    features = {}

    state_category_dirs = get_subdirectories(dataset_path / "state")
    # For each state, load a sample file to obtain metadata.
    for state_category_dir in state_category_dirs:
        if state_category_dir.name == "common":
            feature_files_list["common"] = get_files_in_dir(state_category_dir)
            sample_common_file = np.load(feature_files_list["common"][0], allow_pickle=True).item()
            for key, val in sample_common_file.items():
                if isinstance(val, np.ndarray):
                    if key == "target_path": # Temporary solution to deal with inconsistent shape of target_path.
                        features["target_path_length"] = {
                            "dtype" : "uint32",
                            "shape" : np.array([0], dtype=np.uint32).shape,
                            "names" : None
                        }
                        val = pad_array(val, MAX_PATH_LENGTH)
                    features[key] = {
                        "dtype" : str(val.dtype),
                        "shape" : val.shape,
                        "names" : None
                    }
                if is_feature_segmentation_info(key): # Feature containing the segmentation info.
                    segmentation_info_array = idToLabel_to_bool_array(val["idToLabels"], segmentation_label_to_id)
                    features[key] = {
                        "dtype" : str(segmentation_info_array.dtype),
                        "shape" : segmentation_info_array.shape,
                        "names" : list(segmentation_label_to_id.keys())
                    }
        else: # If it's not a common state, it is an image state.
            image_state_dirs = get_subdirectories(state_category_dir)
            assert len(image_state_dirs) > 0
            for image_state_dir in image_state_dirs:
                image_state_name = image_state_dir.name
                if is_feature_normal_image(image_state_name) or \
                   is_feature_instance_id_segmentation_image(image_state_name): # don't need thoes two states
                    continue

                feature_files = get_files_in_dir(image_state_dir)
                img = load_image(feature_files[0])
                height, width = img.shape[0], img.shape[1]
                features[image_state_name] = {
                    "dtype" : "image",
                    "shape" : (height, width, 3),
                    "names" : None
                }
                feature_files_list[image_state_name] = feature_files

    extra_info = {}
    extra_info["frame_count"] = len(feature_files_list["common"])

    return features, feature_files_list, extra_info

# Write data frames into the episode.
def write_frames(
    dataset                 : LeRobotDataset,
    feature_files_list      : dict,
    segmentation_label_to_id: dict,
    frame_count             : int,
    task                    : str
) -> None:
    for i in range(frame_count):
        frame = {}
        current_frame_idToLabel = None

        common_state = np.load(feature_files_list["common"][i], allow_pickle=True).item()
        for state_name, state_value in common_state.items():
            if state_name == "target_path": # Temporary solution to deal with inconsistent shape of target_path.
                frame["target_path_length"] = np.array([state_value.shape[0]], dtype=np.uint32)
                state_value = pad_array(state_value, MAX_PATH_LENGTH)
            if isinstance(state_value, np.ndarray):
                frame[state_name] = state_value
            if is_feature_segmentation_info(state_name):
                current_frame_idToLabel = state_value["idToLabels"]
                frame[state_name] = idToLabel_to_bool_array(current_frame_idToLabel, segmentation_label_to_id)

        # Parse the states other than `common`, which are the image states.
        for feature, file_list in feature_files_list.items():
            if feature == "common":
                continue

            img = load_image(file_list[i])
            if is_feature_rgb_image(feature):
                assert img.dtype == np.uint8
                frame[feature] = img
            elif is_feature_depth_image(feature):
                assert img.dtype == np.uint16
                img = img / MAX_DEPTH_VALUE
                img = np.stack([img] * 3, axis=-1) # reshape from (h,w) to (h,w,3)
                frame[feature] = img
            elif is_feature_segmentation_image(feature):
                assert img.dtype == np.uint16
                img = remap_segmentation_image(
                    img,
                    segmentation_label_to_id,
                    current_frame_idToLabel)
                img = np.stack([img] * 3, axis=-1) # reshape from (h,w) to (h,w,3)
                frame[feature] = img

        dataset.add_frame(frame, task)

# Convert a collection of MobilityGen datasets into a single LeRobot dataset.
def convert_to_lerobot_dataset(
    dataset_paths: list[str],
    output_path  : str,
    fps          : int,
    num_processes: int
) -> None:
    num_episodes = len(dataset_paths)
    print(f"Number of episodes: {num_episodes}")

    sample_dataset_path = Path(dataset_paths[0])
    config = json.load(open(sample_dataset_path / "config.json", "r"))

    segmentation_label_to_id = get_segmentation_label_lookup(dataset_paths)

    features, _, _ = get_feature_info(sample_dataset_path, segmentation_label_to_id)

    dataset = LeRobotDataset.create(
        repo_id = "",
        fps = fps,
        root = output_path,
        robot_type = config["robot_type"],
        features = features,
        use_videos = False,
        image_writer_processes = num_processes,
        image_writer_threads = 1,
    )

    for i, dataset_path in enumerate(dataset_paths):
        print(f"Processing {i + 1}/{num_episodes} - {dataset_path.name}")
        _, feature_files_list, extra_info = get_feature_info(dataset_path, segmentation_label_to_id)

        dataset.clear_episode_buffer()
        write_frames(dataset, feature_files_list, segmentation_label_to_id, extra_info["frame_count"], config["scenario_type"])
        dataset.save_episode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to the source MobilityGen dataset(s)")
    parser.add_argument("--output", type=str, default=None, help="Output path for the converted LeRobot dataset")
    parser.add_argument('--batch', action=argparse.BooleanOptionalAction, help="Whether the input directory contains more than one dataset")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the source dataset")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes for the image writer")
    args = parser.parse_args()
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)

    if args.batch:
        convert_to_lerobot_dataset(
            dataset_paths = get_subdirectories(Path(args.input)),
            output_path = args.output,
            fps = args.fps,
            num_processes = args.num_processes
        )
    else:
        convert_to_lerobot_dataset(
            dataset_paths = [Path(args.input)],
            output_path = args.output,
            fps = args.fps,
            num_processes = args.num_processes
        )
