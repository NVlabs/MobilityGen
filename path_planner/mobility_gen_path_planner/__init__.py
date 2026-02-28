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

from typing import Tuple, List
import numpy as np
import random
from dataclasses import dataclass
import mobility_gen_path_planner._mobility_gen_path_planner_C as _C


@dataclass
class GeneratePathsOutput:
    visited: np.ndarray
    distance_to_start: np.ndarray
    prev_i: np.ndarray
    prev_j: np.ndarray

    def unroll_path(self, end: Tuple[int, int]) -> np.ndarray:
        end = np.array([end[0], end[1]], dtype=np.int64)
        path = _C.unroll_path(end, self.prev_i, self.prev_j)
        return np.array(path)
    
    def get_valid_end_points(self):
        return np.where(self.visited != 0)
    
    def sample_random_end_point(self) -> Tuple[int, int]:
        i, j = self.get_valid_end_points()
        index = random.randint(0, len(i) - 1)
        return (int(i[index]), int(j[index]))

    def sample_random_path(self) -> np.ndarray:
        end = self.sample_random_end_point()
        return self.unroll_path(end)
    

def generate_paths(start: Tuple[int, int], freespace: np.ndarray) -> GeneratePathsOutput:

    start = np.array([start[0], start[1]], dtype=np.int64)
    freespace = freespace.astype(np.uint8)
    visited = np.zeros(freespace.shape, dtype=np.uint8)
    distance_to_start = np.zeros(freespace.shape, dtype=np.float64)
    prev_i = -np.ones((freespace.shape), dtype=np.int64)
    prev_j = -np.ones((freespace.shape), dtype=np.int64)

    _C.generate_paths(
        start,
        freespace,
        visited,
        distance_to_start,
        prev_i,
        prev_j
    )

    return GeneratePathsOutput(
        visited=visited,
        distance_to_start=distance_to_start,
        prev_i=prev_i,
        prev_j=prev_j
    )

def compress_path(path: np.ndarray, eps=1e-3):
    pref = path[1:-1]
    pnext = path[2:]
    pprev = path[:-2]

    vnext = pnext - pref
    vprev = pref - pprev

    keepmask = np.ones((path.shape[0],), dtype=bool)  # keep beginning / end by default
    keepmask[1:-1] = np.sum((vnext - vprev)**2, axis=-1) > eps


    return path[keepmask], keepmask