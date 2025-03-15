#!/usr/bin/env python3

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

"""Script for generating augmented scenes for an environment."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Generate augmented scenes for an environment")
    parser.add_argument("--env_url", type=str, required=True,
                       help="URL of the environment USD file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save generated scenes")
    parser.add_argument("--num_scenes", type=int, default=1,
                       help="Number of scenes to generate")
    parser.add_argument("--num_obstacles", type=int, default=20,
                       help="Number of obstacles per scene")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create command for generate_augmented_scenes_implemetation.py
    cmd = [
        "./app/python.sh",
        "scripts/generate_augmented_scenes_implemetation.py",
        "--ext-folder", "exts",
        "--enable", "omni.ext.mobility_gen",
        "--enable", "isaacsim.asset.gen.omap",
        f"--env_url={args.env_url}",
        f"--output_dir={args.output_dir}",
        f"--num_scenes={args.num_scenes}",
        f"--num_obstacles={args.num_obstacles}"
    ]

    # Run the command
    print(f"\nProcessing environment...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
