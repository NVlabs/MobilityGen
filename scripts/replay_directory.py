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

"""

This script is responsible for replaying and rendering
a directory of recordings.

"""

import os
import subprocess
import glob
import argparse

if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ['MOBILITY_GEN_DATA']
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--rgb_enabled", type=bool, default=True)
    parser.add_argument("--segmentation_enabled", type=bool, default=True)
    parser.add_argument("--depth_enabled", type=bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=bool, default=False)
    parser.add_argument("--normals_enabled", type=bool, default=False)
    parser.add_argument("--render_rt_subframes", type=int, default=1)
    parser.add_argument("--render_interval", type=int, default=1)
    args = parser.parse_args()

    if args.input is None:
        args.input = os.path.join(DATA_DIR, "recordings")

    if args.output is None:
        args.output = os.path.join(DATA_DIR, "replays")

    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)

    recording_paths = glob.glob(os.path.join(args.input, "*"))

    for recording_path in recording_paths:

        name = os.path.basename(recording_path)

        output_path = os.path.join(args.output, name)

        input_steps = len(glob.glob(os.path.join(recording_path, "state", "common", "*.npy")))
        output_steps = len(glob.glob(os.path.join(output_path, "state", "common", "*.npy")))

        if input_steps == output_steps:
            print(f"Skipping {name} as it is already replayed")
        else:
            print(f"Replaying {name}")

            subprocess.call([
                "./app/python.sh",
                "scripts/replay_implementation.py",
                "--ext-folder", "exts",
                "--enable", "omni.ext.mobility_gen",
                "--enable", "isaacsim.asset.gen.omap",
                "--input_path", recording_path,
                "--output_path", output_path,
                "--render_interval", str(args.render_interval),
                "--render_rt_subframes", str(args.render_rt_subframes),
                "--rgb_enabled", str(args.rgb_enabled),
                "--segmentation_enabled", str(args.segmentation_enabled),
                "--instance_id_segmentation_enabled", str(args.instance_id_segmentation_enabled),
                "--normals_enabled", str(args.normals_enabled),
                "--depth_enabled", str(args.depth_enabled)
            ])