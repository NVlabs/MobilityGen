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

import os

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import open_stage
import isaacsim.core.api.objects as objects


from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.utils.global_utils import new_world, set_viewport_camera, get_stage
from omni.ext.mobility_gen.scenarios import Scenario, SCENARIOS
from omni.ext.mobility_gen.robots import ROBOTS
from omni.ext.mobility_gen.reader import Reader



def load_scenario(path: str) -> Scenario:
    reader = Reader(path)
    config = reader.read_config()
    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    open_stage(os.path.join(path, "stage.usd"))
    
    stage = get_stage()

    robot_prim_path = "/World/robot"

    if stage.GetPrimAtPath(path).IsValid():
        prim_utils.delete_prim(robot_prim_path)
        
    objects.GroundPlane("/World/ground_plane", visible=False)
    new_world(physics_dt=robot_type.physics_dt)
    occupancy_map = reader.read_occupancy_map()
    robot = robot_type.build(robot_prim_path)
    chase_camera_path = robot.build_chase_camera()
    set_viewport_camera(chase_camera_path)
    robot_type = ROBOTS.get(config.robot_type)
    occupancy_map = OccupancyMap.from_ros_yaml(
        ros_yaml_path=os.path.join(path, "occupancy_map", "map.yaml")
    )
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario

