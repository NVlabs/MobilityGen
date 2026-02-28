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


import enum
import numpy as np
import PIL.Image
import os

import tempfile
import omni.kit.usd
import typing as tp
from isaacsim.asset.gen.omap.bindings import _omap as _occupancy_map
from isaacsim.asset.gen.omap.utils import compute_coordinates,  update_location
from pxr import Sdf, UsdGeom, UsdPhysics, Usd, UsdShade, Kind

from ..types import Point2d
from omni.ext.mobility_gen.utils.global_utils import (
    get_app,
    get_stage
)
from omni.ext.mobility_gen.utils.prim_utils import prim_compute_bbox
from omni.ext.mobility_gen.occupancy_map import (
    OccupancyMap, 
    OccupancyMapDataValue, 
    ROS_FREESPACE_THRESH_DEFAULT, 
    ROS_OCCUPIED_THRESH_DEFAULT,
    OCCUPANCY_MAP_DEFAULT_CELL_SIZE,
    OCCUPANCY_MAP_DEFAULT_Z_MAX,
    OCCUPANCY_MAP_DEFAULT_Z_MIN
)


class OccupancyMapGenerateRotation(enum.Enum):

    ROTATE_0 = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3

    def degrees(self):
        if self == OccupancyMapGenerateRotation.ROTATE_0:
            return 0
        elif self == OccupancyMapGenerateRotation.ROTATE_90:
            return 90
        elif self == OccupancyMapGenerateRotation.ROTATE_180:
            return 180
        elif self == OccupancyMapGenerateRotation.ROTATE_270:
            return -90
        else:
            raise RuntimeError(f"Invalid rotation {self}.")



def compute_occupancy_bounds_from_prim_path(
        prim_path: str, 
        z_min: float, 
        z_max: float, 
        cell_size: float
    ):

    stage = get_stage()
    
    prim_path = os.path.join(prim_path)
    prim_path = stage.GetPrimAtPath(prim_path)

    lower_bound, upper_bound, midpoint = prim_compute_bbox(prim_path)
    lower_bound = (lower_bound[0], lower_bound[1], z_min)
    upper_bound = (upper_bound[0], upper_bound[1], z_max)

    width = upper_bound[0] - lower_bound[0]
    height = upper_bound[1] - lower_bound[1]
    origin = (lower_bound[0] - cell_size, lower_bound[1] - cell_size, 0)
    lower_bound = (0, 0, z_min)
    upper_bound = (width + cell_size, height + cell_size, z_max)
    return origin, lower_bound, upper_bound


async def occupancy_map_generate_from_prim_async(
        origin: tp.Tuple[float, float, float],
        lower_bound: tp.Tuple[float, float, float],
        upper_bound: tp.Tuple[float, float, float],
        cell_size: float = OCCUPANCY_MAP_DEFAULT_CELL_SIZE,
        rotation: OccupancyMapGenerateRotation = OccupancyMapGenerateRotation.ROTATE_180,
        unknown_as_freespace: bool = True
    ) -> OccupancyMap:

    app = get_app()

    om = _occupancy_map.acquire_omap_interface()
    
    timeline = omni.timeline.get_timeline_interface()

    await app.next_update_async()
    
    stage = get_stage()
    stage_scale = UsdGeom.GetStageMetersPerUnit(stage)

    if stage_scale != 1.0:
        raise RuntimeError("Stage unit must be 1 meter.")
    
    # Apply physics
    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    
    await app.next_update_async()
    

    # Compute bounds for occupancy map calculation using specified prim

    update_location(
        om,
        origin,
        lower_bound,
        upper_bound
    )
    
    await app.next_update_async()
    

    # Set cell size
    om.set_cell_size(cell_size)
    
    await app.next_update_async()
    

    # Generate occupancy map
    timeline.stop()
    
    await app.next_update_async()
    
    timeline.play()
    
    await app.next_update_async()
    
    om.generate()
    
    await app.next_update_async()
    
    timeline.stop()
    
    await app.next_update_async()
    

    # Format Image
    buffer = om.get_buffer()
    dims = om.get_dimensions()
    buffer = np.array(buffer)
    buffer = np.reshape(buffer, (dims[1], dims[0]))
    occupied_mask = buffer == 1.0
    freespace_mask = buffer == 0.0
    unknown_mask = ~(occupied_mask | freespace_mask)

    if unknown_as_freespace:
        freespace_mask[unknown_mask] = True
        unknown_mask = np.zeros_like(unknown_mask)

    image = np.zeros(occupied_mask.shape, dtype=np.uint8)
    image[occupied_mask] = OccupancyMapDataValue.OCCUPIED.ros_image_value()
    image[unknown_mask] = OccupancyMapDataValue.UNKNOWN.ros_image_value()
    image[freespace_mask] = OccupancyMapDataValue.FREESPACE.ros_image_value()
    image = PIL.Image.fromarray(image)
    image = image.rotate(rotation.degrees())

    # Format Yaml
    if rotation == OccupancyMapGenerateRotation.ROTATE_0:
        top_left, top_right, bottom_left, bottom_right, image_coords = compute_coordinates(om, cell_size)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_270:  # -90 degrees
        top_right, bottom_right, top_left, bottom_left, image_coords = compute_coordinates(om, cell_size)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_90:  # 90 degrees
        bottom_left, top_left, bottom_right, top_right, image_coords = compute_coordinates(om, cell_size)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_180:  # 180 degrees
        bottom_right, bottom_left, top_right, top_left, image_coords = compute_coordinates(om, cell_size)

    occupancy_map = OccupancyMap.from_ros_image(
        ros_image=image,
        resolution=cell_size,
        origin=[
            float(bottom_left[0]),
            float(bottom_left[1]),
            0.0
        ],
        negate=False,
        free_thresh=ROS_FREESPACE_THRESH_DEFAULT,
        occupied_thresh=ROS_OCCUPIED_THRESH_DEFAULT
    )
    
    _occupancy_map.release_omap_interface(om)

    return occupancy_map


def occupancy_map_add_to_stage(
        occupancy_map: OccupancyMap,
        stage: Usd.Stage,
        path: str,
        z_offset: float = 0.0
    ) -> Usd.Prim:

    image_path = os.path.join(tempfile.mkdtemp(), "texture.png")
    image = occupancy_map.ros_image()

    # need to flip, ros uses inverted coordinates on y axis
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image.save(image_path)

    x0, y0 = occupancy_map.top_left_pixel_world_coords()
    x1, y1 = occupancy_map.bottom_right_pixel_world_coords()

    # Add model
    modelRoot = UsdGeom.Xform.Define(stage, path)
    Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

    # Add mesh
    mesh = UsdGeom.Mesh.Define(stage, os.path.join(path, "mesh"))
    mesh.CreatePointsAttr([(x0, y0, z_offset), (x1, y0, z_offset), (x1, y1, z_offset), (x0, y1, z_offset)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateExtentAttr([(x0, y0, z_offset), (x1, y1, z_offset)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.varying)
    
    texCoords.Set([(0, 0), (1, 0), (1,1), (0, 1)])

    # Add material
    material_path = os.path.join(path, "material")
    material = UsdShade.Material.Define(stage, material_path)
    pbrShader = UsdShade.Shader.Define(stage, os.path.join(material_path, "shader"))
    pbrShader.CreateIdAttr("UsdPreviewSurface")
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

    # Add texture to material
    stReader = UsdShade.Shader.Define(stage, os.path.join(material_path, "st_reader"))
    stReader.CreateIdAttr('UsdPrimvarReader_float2')
    diffuseTextureSampler = UsdShade.Shader.Define(stage, os.path.join(material_path, "diffuse_texture"))
    diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
    diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(image_path)
    diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
    diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

    stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
    stInput.Set('st')
    stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    return modelRoot

