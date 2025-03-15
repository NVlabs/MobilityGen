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

"""Script for generating augmented scenes with obstacles.

This script provides functionality to:
1. Load a base environment (NRE or default)
2. Generate an occupancy map of the navigable space
3. Add random obstacles in valid locations
4. Save both the augmented scenes and their occupancy maps
"""

# Python imports
import argparse
import asyncio
import cv2
import numpy as np
import os
import random
import sys
import yaml

# Add script directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize simulation
from isaacsim import SimulationApp
simulation_app = SimulationApp(launch_config={"headless": True})

# Omniverse imports
import omni.kit.app
import omni.usd
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.utils.stage import is_stage_loading

# Isaac Sim imports
from isaacsim.core.utils.rotations import euler_angles_to_quat

# USD imports
from pxr import (
    Gf,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics
)

# Extension imports
from omni.ext.mobility_gen.occupancy_map import (
    OccupancyMap,
    OccupancyMapDataValue,
    ROS_FREESPACE_THRESH_DEFAULT,
    ROS_OCCUPIED_THRESH_DEFAULT
)
from omni.ext.mobility_gen.utils.global_utils import get_world
from omni.ext.mobility_gen.utils.occupancy_map_utils import occupancy_map_generate_from_prim_async

# Add obstacle assets here
OBSTACLE_ASSET_URLS = [
    "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/Blocks/nvidia_cube.usd",
    "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/KLT_Bin/small_KLT.usd",
    "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/Mounts/thor_table.usd",
    "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/Pallet/o3dyn_pallet.usd",
    "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/Pallet/pallet.usd",
]

def get_random_obstacle_url():
    """Get a random obstacle URL."""
    return random.choice(OBSTACLE_ASSET_URLS)


def get_mesh_bounding_box(stage: Usd.Stage, prim_path: str) -> dict[str, Gf.Vec3d]:
    """Get axis-aligned bounding box of a USD mesh.

    Args:
        stage: USD stage containing the mesh
        prim_path: Path to the mesh prim

    Returns:
        Dictionary with min/max points and size of bounding box
    """
    mesh_prim = stage.GetPrimAtPath(prim_path)
    if not mesh_prim:
        return None

    bounds = UsdGeom.Mesh(mesh_prim).ComputeWorldBound(0.0, "default")
    box = bounds.ComputeAlignedBox()
    min_point = box.GetMin()
    max_point = box.GetMax()

    return {
        'min': min_point,
        'max': max_point,
        'size': max_point - min_point
    }


def process_occupancy_map(occupancy_map: OccupancyMap, nre_bbox: dict) -> OccupancyMap:
    """Process occupancy map by masking regions outside NRE bbox.

    Args:
        occupancy_map: Original occupancy map
        nre_bbox: Bounding box of NRE mesh in world coordinates

    Returns:
        Processed occupancy map with masked regions
    """
    processed_data = np.array(occupancy_map.ros_image(), dtype=np.uint8)

    if nre_bbox is not None:
        min_point = np.array([[nre_bbox['min'][0], nre_bbox['min'][1]]])
        max_point = np.array([[nre_bbox['max'][0], nre_bbox['max'][1]]])
        min_px = occupancy_map.world_to_pixel_numpy(min_point)[0]
        max_px = occupancy_map.world_to_pixel_numpy(max_point)[0]

        mask = np.zeros_like(processed_data, dtype=bool)
        x_min = int(min(min_px[0], max_px[0]))
        x_max = int(max(min_px[0], max_px[0]))
        y_min = int(min(min_px[1], max_px[1]))
        y_max = int(max(min_px[1], max_px[1]))

        height, width = processed_data.shape
        x_min = max(0, min(x_min, width-1))
        x_max = max(0, min(x_max, width-1))
        y_min = max(0, min(y_min, height-1))
        y_max = max(0, min(y_max, height-1))

        mask[y_min:y_max+1, x_min:x_max+1] = True
        masked_data = np.zeros_like(processed_data)
        masked_data[mask] = processed_data[mask]
        processed_data = masked_data

    occupancy_data = np.zeros_like(processed_data, dtype=np.uint8)
    occupancy_data[processed_data == 0] = OccupancyMapDataValue.OCCUPIED
    occupancy_data[processed_data == 255] = OccupancyMapDataValue.FREESPACE
    occupancy_data[processed_data == 127] = OccupancyMapDataValue.UNKNOWN

    return OccupancyMap(
        data=occupancy_data,
        resolution=occupancy_map.resolution,
        origin=occupancy_map.origin
    )


def set_collision_preset(prim, enable_collision):
    """
    Apply or remove collision settings on a USD prim.

    Args:
        prim (Usd.Prim): The prim to modify collision settings for
        enable_collision (bool): Whether to enable (True) or disable (False) collision

    Note:
        Creates or modifies the physics:collisionEnabled attribute on the prim.
    """
    stage = prim.GetStage()
    prim_path = prim.GetPath()

    collision_api = UsdPhysics.CollisionAPI.Get(stage, prim_path)
    if not collision_api:
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)

    collision_enabled_attr = prim.GetAttribute("physics:collisionEnabled")
    if enable_collision:
        if not collision_enabled_attr or not collision_enabled_attr.Get():
            collision_api.CreateCollisionEnabledAttr(True)


def generate_random_valid_points_in_hull(hull_points, required_num_points, occupancy_map):
    """
    Generate random valid points within a convex hull that are also navigable in the occupancy map.

    Args:
        hull_points (np.ndarray): Array of hull vertices in shape (N, 2) format
        required_num_points (int): Number of valid points to generate
        occupancy_map (np.ndarray): Binary occupancy map where >0 represents navigable space

    Returns:
        np.ndarray: Array of generated points in shape (M, 2) format, where M <= required_num_points

    Note:
        May return fewer points than requested if valid locations cannot be found
        within max_attempts iterations.
    """
    # Ensure hull_points is the right shape
    if len(hull_points.shape) == 3:
        hull_points = hull_points.reshape(-1, 2)

    # Get bounding box of the convex hull
    x_min, y_min = np.min(hull_points, axis=0)
    x_max, y_max = np.max(hull_points, axis=0)

    points = []
    max_attempts = required_num_points * 100  # Prevent infinite loop
    attempts = 0

    while len(points) < required_num_points and attempts < max_attempts:
        # Generate random point within the bounding box
        x = int(np.random.uniform(x_min, x_max))
        y = int(np.random.uniform(y_min, y_max))

        # Ensure point is within image bounds
        if y >= occupancy_map.shape[0] or x >= occupancy_map.shape[1]:
            attempts += 1
            continue

        point = (x, y)

        # Check if the point is inside the convex hull and is navigable
        if cv2.pointPolygonTest(hull_points.astype(np.float32), point, False) >= 0 and occupancy_map[y, x] > 0:
            points.append(point)
            occupancy_map[y, x] = 0

        attempts += 1

    if len(points) < required_num_points:
        print(f"Warning: Could only generate {len(points)} valid points out of {required_num_points} requested")

    return np.array(points)


def detect_env_type(stage: Usd.Stage) -> str:
    """Detect if the environment is an NRE or default environment.

    Args:
        stage: USD stage containing the environment

    Returns:
        "nre" if nerf prim is found, "default" otherwise
    """
    # Get default prim or fallback to /World
    default_prim = stage.GetDefaultPrim()
    root_path = str(default_prim.GetPath())
    root_prim = stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(root_prim):
        if "nerf" in prim.GetPath().pathString and prim.IsA(UsdGeom.Mesh):
            print(f"Detected NRE environment (root: {root_path})")
            return "nre"
    print(f"Detected default environment (root: {root_path})")
    return "default"


class GenerateAugmentedScene:
    """Class for generating augmented scenes with random obstacles.

    This class handles:
    1. Loading a base environment (NRE or default)
    2. Generating an occupancy map of the navigable space
    3. Adding random obstacles in valid locations
    4. Saving both the augmented scenes and their occupancy maps

    Args:
        config: Configuration dictionary containing:
            - env_url: Path to base environment USD file
            - add_obstacles: Whether to add random obstacles
            - ground_plane_args: Ground plane configuration (z_position, is_visible)
            - scene_generation: Scene generation parameters
            - obstacles_config: Obstacle placement configuration
        out_dir: Directory to save generated scenes and maps

    Note:
        Ground plane is automatically added for NRE environments.
        Environment type (NRE/default) is automatically detected.
    """

    def __init__(self, config: dict, out_dir: str) -> None:
        self.CONFIG = config
        self._out_dir = out_dir
        self._env_url = config['env_url']
        self._env_type = "default"  # Will be updated during load_env
        self._stage = None
        self._nre_mesh = None
        self._obstacles = []

        self.add_obstacles = config['add_obstacles']
        self.ground_plane_args = config.get('ground_plane_args', {})

        if self.add_obstacles:
            self._obstacle_sizes = self._precalculate_obstacle_sizes()

    def _find_nre_mesh_prim(self) -> Usd.Prim:
        """Find NRE mesh prim under root hierarchy.

        Returns:
            NRE mesh prim if found, None otherwise
        """
        root_prim = self._stage.GetPrimAtPath(self._get_scene_root_prim())
        for prim in Usd.PrimRange(root_prim):
            if "nerf" in prim.GetPath().pathString and prim.IsA(UsdGeom.Mesh):
                print(f"Found NRE mesh prim: {prim.GetPath()}")
                return prim
        print("No NRE mesh found.")
        return None

    def _set_nre_mesh_collision(self) -> None:
        """Set collision properties for NRE mesh."""
        print("Setting NRE mesh collision...")
        nre_prim = self._find_nre_mesh_prim()
        if nre_prim:
            self._nre_mesh = nre_prim.GetPath()
            set_collision_preset(nre_prim, True)  # Always enable collision
        else:
            print("FATAL: No NRE mesh found under /World/nerf. Exiting...")
            sys.exit(1)

    def _add_physics_scene(self) -> None:
        """Add physics scene to the stage."""
        UsdPhysics.Scene.Define(self._stage, "/physicsScene")
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(self._stage.GetPrimAtPath("/physicsScene"))
        physx_scene.GetEnableCCDAttr().Set(True)
        physx_scene.GetEnableGPUDynamicsAttr().Set(False)
        physx_scene.GetBroadphaseTypeAttr().Set("MBP")

    def _get_scene_root_prim(self) -> str:
        """Get the root prim path for the scene.

        Returns:
            Path to the root prim (default prim path, '/World', or '/Root')
        """
        default_prim = self._stage.GetDefaultPrim()
        return str(default_prim.GetPath())

    async def load_env(self) -> tuple[OccupancyMap, OccupancyMap]:
        """Load environment and generate base occupancy map.

        Returns:
            Tuple containing:
                - Original occupancy map
                - Processed occupancy map with masked regions

        Note:
            Ground plane is automatically added for NRE environments.
            Collision is always enabled for NRE mesh.
        """
        app = omni.kit.app.get_app()

        print(f"Opening stage {self._env_url}...")
        await omni.usd.get_context().open_stage_async(self._env_url)
        self._stage = omni.usd.get_context().get_stage()
        self._add_physics_scene()

        print("Loading stage...")
        while is_stage_loading():
            await app.next_update_async()
        print("Stage loaded.")

        # Detect environment type
        self._env_type = detect_env_type(self._stage)

        # Get NRE mesh dimensions and add ground plane if NRE
        nre_bbox = None
        if self._env_type == "nre":
            self._set_nre_mesh_collision()
            if self._nre_mesh:
                nre_bbox = get_mesh_bounding_box(self._stage, self._nre_mesh)
                print("\nNRE mesh dimensions:")
                print(f"  Min: {nre_bbox['min']}")
                print(f"  Max: {nre_bbox['max']}")
                print(f"  Size: {nre_bbox['size']}")

                # Add ground plane for NRE environment
                print("Adding ground plane (NRE environment detected)...")
                GroundPlane("/World/defaultGroundPlane",
                           z_position=self.ground_plane_args.get('z_position', 0),
                           visible=self.ground_plane_args.get('is_visible', True))

            await app.next_update_async()

        await app.next_update_async()
        print("Loading environment complete.")

        # Generate base occupancy map without obstacles
        print("Generating base occupancy map without obstacles")
        omap_root = self._get_scene_root_prim()
        base_occupancy_map = await occupancy_map_generate_from_prim_async(
            omap_root,
            cell_size=self.CONFIG.get('cell_size', 0.05),
            z_min=self.CONFIG.get('z_min', 0.1),
            z_max=self.CONFIG.get('z_max', 0.62)
        )

        print("\nBase occupancy map info:")
        print(f"  Resolution: {base_occupancy_map.resolution}")
        print(f"  Origin: {base_occupancy_map.origin}")
        print(f"  Shape: {base_occupancy_map.data.shape}")
        print(f"  Data range: [{np.min(base_occupancy_map.data)}, {np.max(base_occupancy_map.data)}]")

        base_processed_map = process_occupancy_map(base_occupancy_map, nre_bbox)

        print("\nProcessed occupancy map info:")
        print(f"  Resolution: {base_processed_map.resolution}")
        print(f"  Origin: {base_processed_map.origin}")
        print(f"  Shape: {base_processed_map.data.shape}")
        print(f"  Data range: [{np.min(base_processed_map.data)}, {np.max(base_processed_map.data)}]")

        return base_occupancy_map, base_processed_map

    async def generate_scenes(
        self,
        base_occupancy_map: OccupancyMap,
        base_processed_map: OccupancyMap
    ) -> None:
        """Generate multiple scenes with different obstacle configurations.

        Args:
            base_occupancy_map: Original occupancy map
            base_processed_map: Processed occupancy map with masked regions
        """
        # Create base directory for saving scenes
        base_save_dir = os.path.join(self._out_dir, "generated_scenes")
        os.makedirs(base_save_dir, exist_ok=True)

        # Save base occupancy map
        base_omap_dir = os.path.join(base_save_dir, "base_omap", "occupancy_map")
        os.makedirs(base_omap_dir, exist_ok=True)

        # Save the occupancy map using the built-in save_ros method
        base_processed_map.save_ros(base_omap_dir)

        # Generate scenes
        num_scenes = self.CONFIG['scene_generation'].get('num_scenes', 1)
        for scene_idx in range(num_scenes):
            print(f"\nGenerating scene {scene_idx + 1}/{num_scenes}")

            scene_dir = os.path.join(base_save_dir, f"scene_{scene_idx:03d}")
            os.makedirs(scene_dir, exist_ok=True)

            # Generate scene with obstacles
            if self.add_obstacles:
                processed_map_copy = OccupancyMap(
                    resolution=base_processed_map.resolution,
                    origin=base_processed_map.origin,
                    data=base_processed_map.data.copy()  # Copy the raw data
                )

                await self._generate_single_scene(
                    scene_idx,
                    scene_dir,
                    base_occupancy_map,
                    processed_map_copy
                )

    async def _generate_single_scene(
        self,
        scene_idx: int,
        scene_dir: str,
        base_occupancy_map: OccupancyMap,
        processed_map: OccupancyMap
    ) -> None:
        """Generate a single scene with obstacles.

        Args:
            scene_idx: Index of the scene being generated
            scene_dir: Directory to save scene files
            base_occupancy_map: Original occupancy map
            processed_map: Processed occupancy map for obstacle placement
        """
        app = omni.kit.app.get_app()

        # Remove any existing obstacles
        await self._remove_obstacles()

        # Create a new OccupancyMap with the same data
        processed_map_copy = OccupancyMap(
            resolution=processed_map.resolution,
            origin=processed_map.origin,
            data=np.array(processed_map.ros_image(), dtype=np.uint8)
        )

        # Add obstacles directly with occupancy map
        print("Adding obstacles...")
        processed_map_copy = await self._add_random_obstacles(base_occupancy_map, processed_map_copy)
        print("Obstacles added.")

        # Generate and save new occupancy map
        print("Generating occupancy map with obstacles")
        omap_root = self._get_scene_root_prim()
        obstacle_occupancy_map = await occupancy_map_generate_from_prim_async(
            omap_root,
            cell_size=self.CONFIG.get('cell_size', 0.05),
            z_min=self.CONFIG.get('z_min', 0.1),
            z_max=self.CONFIG.get('z_max', 0.62)
        )

        # Save occupancy map using save_ros method
        omap_dir = os.path.join(scene_dir, "occupancy_map")
        os.makedirs(omap_dir, exist_ok=True)
        obstacle_occupancy_map.save_ros(omap_dir)

        # Save stage
        stage_path = os.path.join(scene_dir, "stage.usd")
        self._stage.Export(stage_path)

        print(f"Scene {scene_idx + 1} saved to {scene_dir}")
        await app.next_update_async()

    def _get_obstacle_size(self, usd_url: str) -> float:
        """Calculate radius of an obstacle from its USD file.

        Args:
            usd_url: Path to the USD file

        Returns:
            Radius of the obstacle in meters (half of maximum dimension)
        """
        try:
            temp_stage = Usd.Stage.Open(usd_url)
            if temp_stage:
                for prim in temp_stage.TraverseAll():
                    if prim.IsA(UsdGeom.Mesh):
                        bbox_info = get_mesh_bounding_box(temp_stage, prim.GetPath())
                        size = bbox_info['size']
                        return max(abs(float(size[0])), abs(float(size[1]))) / 2.0
        except Exception as e:
            print(f"Warning: Failed to get size for {os.path.basename(usd_url)}: {e}")
        return 0.5  # Default radius

    async def _add_random_obstacles(
        self,
        occupancy_map: OccupancyMap,
        processed_map: OccupancyMap
    ) -> OccupancyMap:
        """Add random obstacles to navigable regions.

        Args:
            occupancy_map: Original map for coordinate conversion
            processed_map: Processed map for obstacle placement

        Returns:
            Updated occupancy map with obstacles marked
        """
        processed_map_array = self._prepare_map_array(processed_map)
        height, width = processed_map_array.shape
        obstacle_sizes = self._calculate_obstacle_sizes(occupancy_map.resolution)

        obstacles_added = 0
        attempts = 0
        max_attempts = self.CONFIG['obstacles_config']['num_obstacles'] * 100

        while obstacles_added < self.CONFIG['obstacles_config']['num_obstacles'] and attempts < max_attempts:
            if await self._try_add_obstacle(
                occupancy_map,
                processed_map_array,
                obstacle_sizes,
                obstacles_added,
                width,
                height
            ):
                obstacles_added += 1
            attempts += 1

        if obstacles_added < self.CONFIG['obstacles_config']['num_obstacles']:
            print(f"Warning: Could only place {obstacles_added} obstacles out of {self.CONFIG['obstacles_config']['num_obstacles']} requested")

        return OccupancyMap(
            data=processed_map_array,
            resolution=processed_map.resolution,
            origin=processed_map.origin
        )

    def _prepare_map_array(self, processed_map):
        """Convert processed map to OccupancyMapDataValue format."""
        map_array = processed_map.data.copy()
        navigable_space = (map_array == 255)
        map_array[navigable_space] = OccupancyMapDataValue.FREESPACE
        map_array[~navigable_space] = OccupancyMapDataValue.OCCUPIED
        return map_array

    async def _try_add_obstacle(
        self,
        occupancy_map: OccupancyMap,
        map_array: np.ndarray,
        obstacle_sizes: dict,
        obstacle_idx: int,
        width: int,
        height: int
    ) -> bool:
        """Try to add a single obstacle at a random location.

        Args:
            occupancy_map: Map for coordinate conversion
            map_array: Current occupancy data
            obstacle_sizes: Size information for obstacles
            obstacle_idx: Index for naming obstacle
            width: Map width in pixels
            height: Map height in pixels

        Returns:
            True if obstacle was successfully placed
        """
        usd_url = self._get_random_obstacle_url()
        check_radius = obstacle_sizes[usd_url]['check_radius']

        x = int(np.random.uniform(0, width))
        y = int(np.random.uniform(0, height))

        if not self._is_region_in_bounds(x, y, check_radius, width, height):
            return False

        region = map_array[y-check_radius:y+check_radius+1, x-check_radius:x+check_radius+1]
        if not np.all(region == OccupancyMapDataValue.FREESPACE):
            return False

        await self._place_obstacle(x, y, usd_url, obstacle_idx, obstacle_sizes[usd_url], occupancy_map)
        map_array[y-check_radius:y+check_radius+1, x-check_radius:x+check_radius+1] = OccupancyMapDataValue.OCCUPIED
        return True

    async def _remove_obstacles(self) -> None:
        """Remove all obstacles from the scene."""
        app = omni.kit.app.get_app()

        print("Removing obstacles...")
        for obstacle in self._obstacles:
            prim_path = obstacle['prim_path']
            if self._stage.GetPrimAtPath(prim_path):
                self._stage.RemovePrim(prim_path)
        self._obstacles = []
        await app.next_update_async()
        print("Obstacles removed.")

    def _get_random_obstacle_url(self) -> str:
        """Get a random obstacle URL.

        Returns:
            URL to a randomly selected obstacle asset
        """
        return get_random_obstacle_url()

    def _calculate_obstacle_sizes(self, map_resolution: float) -> dict:
        """Calculate pixel sizes for all obstacles.

        Args:
            map_resolution: Resolution of the occupancy map in meters/pixel

        Returns:
            Dictionary mapping obstacle URLs to their size information
        """
        obstacle_sizes = {}
        for url, radius_meters in self._obstacle_sizes.items():
            radius_pixels = int(radius_meters / map_resolution)
            safety_padding = int(0.2 / map_resolution)  # 20cm padding
            check_radius = radius_pixels + safety_padding

            obstacle_sizes[url] = {
                'radius_meters': radius_meters,
                'radius_pixels': radius_pixels,
                'check_radius': check_radius,
                'area_pixels': np.pi * check_radius * check_radius,
                'area_meters': np.pi * (radius_meters + 0.2) * (radius_meters + 0.2)
            }
        return obstacle_sizes

    def _is_region_in_bounds(
        self,
        x: int,
        y: int,
        radius: int,
        width: int,
        height: int
    ) -> bool:
        """Check if a circular region is within image bounds.

        Args:
            x: Center x-coordinate
            y: Center y-coordinate
            radius: Region radius in pixels
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            True if region is within bounds
        """
        return not (y - radius < 0 or y + radius >= height or
                   x - radius < 0 or x + radius >= width)

    async def _place_obstacle(
        self,
        x: int,
        y: int,
        usd_url: str,
        obstacle_idx: int,
        obstacle_info: dict,
        occupancy_map: OccupancyMap
    ) -> None:
        """Place a single obstacle in the scene.

        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            usd_url: URL of obstacle asset
            obstacle_idx: Index for naming
            obstacle_info: Size information for obstacle
            occupancy_map: Map for coordinate conversion
        """
        world_point = occupancy_map.pixel_to_world_numpy(np.array([[x, y]]))
        pose_x = float(world_point[0][0])
        pose_y = float(world_point[0][1])
        pose_theta = random.uniform(0, 360)

        prim_name = f"obstacle_{obstacle_idx:02d}"
        prim_path = f"/World/obstacles/{prim_name}"

        stage_utils.add_reference_to_stage(usd_path=usd_url, prim_path=prim_path)
        prim = RigidPrim(prim_path=prim_path, name=prim_name)

        orientation = euler_angles_to_quat(np.array([0, 0, pose_theta]), degrees=True)
        prim.set_local_pose(translation=[pose_x, pose_y, 0.0], orientation=orientation)

        self._obstacles.append({
            'prim': prim,
            'prim_name': prim_name,
            'prim_path': prim_path,
            'usd_url': usd_url,
            'pose': [pose_x, pose_y, pose_theta],
            'radius': obstacle_info['radius_meters'],
        })

        await omni.kit.app.get_app().next_update_async()

    def _precalculate_obstacle_sizes(self) -> dict:
        """Pre-calculate obstacle sizes for all available assets.

        Returns:
            Dictionary mapping obstacle URLs to their radii in meters
        """
        obstacle_sizes = {}
        print("\nPre-calculating obstacle sizes...")

        for url in OBSTACLE_ASSET_URLS:
            radius_meters = self._get_obstacle_size(url)
            obstacle_sizes[url] = radius_meters
            print(f"Asset: {os.path.basename(url)}")
            print(f"  Radius: {radius_meters:.2f}m")
        return obstacle_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_url", type=str, required=True, help="Path to environment USD file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated scenes")
    parser.add_argument("--num_scenes", type=int, default=1, help="Number of scenes to generate")
    parser.add_argument("--num_obstacles", type=int, default=20, help="Number of obstacles per scene")

    args, unknown = parser.parse_known_args()

    # Default configuration
    config = {
        "env_url": args.env_url,
        "add_obstacles": True,
        "ground_plane_args": {
            "z_position": 0.05,
            "is_visible": False
        },
        "scene_generation": {
            "num_scenes": args.num_scenes
        },
        "obstacles_config": {
            "num_obstacles": args.num_obstacles
        }
    }

    scene_generator = GenerateAugmentedScene(config=config, out_dir=args.output_dir)

    async def main_async():
        try:
            base_occupancy_map, base_processed_map = await scene_generator.load_env()
            await scene_generator.generate_scenes(base_occupancy_map, base_processed_map)
        except Exception as e:
            print(f"Error in main_async: {e}")
            raise

    future = asyncio.ensure_future(main_async())

    while not future.done():
        simulation_app.update()

    if future.exception():
        raise future.exception()

    simulation_app.close()