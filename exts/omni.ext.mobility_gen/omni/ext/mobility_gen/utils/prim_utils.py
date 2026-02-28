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


from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Kind, UsdPhysics
import typing as tp
from typing import Tuple, Sequence
import numpy as np
import math


def prim_add_collision(prim: Usd.Prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def prim_compute_bbox(prim: Usd.Prim, nested: bool = False) -> \
        tp.Tuple[tp.Tuple[float, float, float], tp.Tuple[float, float, float]]:
    """Computes the axis-aligned bounding box for a USD prim.
    
    Args:
        prim (Usd.Prim):  The USD prim to compute the bounding box of.

    Returns:
        Tuple[Tuple[float, float, float], Tuple[float, float, float]] The ((min_x, min_y, min_z), (max_x, max_y, max_z)) values of the bounding box.
    """
    
    bbox_cache: UsdGeom.BBoxCache = UsdGeom.BBoxCache(
        time=Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
        useExtentsHint=True
    )

    total_bounds = Gf.BBox3d()

    prims = []

    if nested:
        for p in Usd.PrimRange(prim):
            prims.append(p)
    else:
            prims.append(prim)

    for p in prims:
        total_bounds = Gf.BBox3d.Combine(
            total_bounds, Gf.BBox3d(bbox_cache.ComputeWorldBound(p).ComputeAlignedRange())
        )

    box = total_bounds.GetBox()

    return (box.GetMin(), box.GetMax(), box.GetMidpoint())


def prim_add_semantics(prim: Usd.Prim, type: str, name: str):
    """Adds semantics to a USD prim.
    
    This function adds semantics to a USD prim.  This is useful for assigning
    classes to objects when generating synthetic data with Omniverse Replicator.

    For example:

    add_semantics(dog_prim, "class", "dog")
    add_semantics(cat_prim, "class", "cat")

    Args:
        prim (Usd.Prim):  The USD prim to modify.
        type (str):  The semantics type.  This depends on how the data is ingested.
            Typically, when using Omniverse replicator you will set this to "class".
        name (str):  The value of the semantic type.  Typically, this would 
            correspond to the class label.

    Returns:
        Usd.Prim:  The USD prim with added semantics.
    """

    prim.AddAppliedSchema(f"SemanticsAPI:{type}_{name}")

    prim.CreateAttribute(f"semantic:{type}_{name}:params:semanticType", 
        Sdf.ValueTypeNames.String).Set(type)

    prim.CreateAttribute(f"semantic:{type}_{name}:params:semanticData", 
        Sdf.ValueTypeNames.String).Set(name)

    return prim


def prim_bind_material(prim: Usd.Prim, material: UsdShade.Material):
    """Binds a USD material to a USD prim.
    
    Args:
        prim (Usd.Prim):  The USD prim to modify.
        material (UsdShade.Material):  The USD material to bind to the USD prim.

    Returns:
        Usd.Prim:  The modified USD prim with the specified material bound to it.
    """

    prim.ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(prim).Bind(material, 
        UsdShade.Tokens.strongerThanDescendants)

    return prim


def prim_collapse_xform(prim: Usd.Prim):
    """Collapses all xforms on a given USD prim.
    
    This method collapses all Xforms on a given prim.  For example,
    a series of rotations, translations would be combined into a single matrix
    operation.

    Args:
        prim (Usd.Prim):  The Usd.Prim to collapse the transforms of.
    
    Returns:
        Usd.Prim:  The Usd.Prim.
    """

    x = UsdGeom.Xformable(prim)
    local = x.GetLocalTransformation()
    prim.RemoveProperty("xformOp:translate")
    prim.RemoveProperty("xformOp:transform")
    prim.RemoveProperty("xformOp:rotateX")
    prim.RemoveProperty("xformOp:rotateY")
    prim.RemoveProperty("xformOp:rotateZ")
    var = x.MakeMatrixXform()
    var.Set(local)

    return prim


def prim_get_xform_op_order(prim: Usd.Prim):
    """Returns the order of Xform ops on a given prim."""
    x = UsdGeom.Xformable(prim)
    op_order = x.GetXformOpOrderAttr().Get()
    if op_order is not None:
        op_order = list(op_order)
        return op_order
    else:
        return []


def prim_set_xform_op_order(prim: Usd.Prim, op_order: Sequence[str]):
    """Sets the order of Xform ops on a given prim"""
    x = UsdGeom.Xformable(prim)
    x.GetXformOpOrderAttr().Set(op_order)
    return prim


def prim_xform_op_move_end_to_front(prim: Usd.Prim):
    """Pops the last xform op on a given prim and adds it to the front."""
    order = prim_get_xform_op_order(prim)
    end = order.pop(-1)
    order.insert(0, end)
    prim_set_xform_op_order(prim, order)
    return prim


def prim_get_num_xform_ops(prim: Usd.Prim) -> int:
    """Returns the number of xform ops on a given prim."""
    return len(prim_get_xform_op_order(prim))


def prim_apply_xform_matrix(prim: Usd.Prim, transform: np.ndarray):
    """Applies a homogeneous transformation matrix to the current prim's xform list.

    Args:
        prim (Usd.Prim):  The USD prim to transform.
        transform (np.ndarray):  The 4x4 homogeneous transform matrix to apply.

    Returns:
        Usd.Prim:  The modified USD prim with the provided transform applied after current transforms.
    """
    x = UsdGeom.Xformable(prim)
    x.AddTransformOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(
        Gf.Matrix4d(transform)
    )
    prim_xform_op_move_end_to_front(prim)
    return prim


def prim_scale(prim: Usd.Prim, scale: Tuple[float, float, float]):
    """Scales a prim along the (x, y, z) dimensions.
    
    Args:
        prim (Usd.Prim):  The USD prim to scale.
        scale (Tuple[float, float, float]):  The scaling factors for the (x, y, z) dimensions.

    Returns:
        Usd.Prim:  The scaled prim.
    """
    x = UsdGeom.Xformable(prim)
    x.AddScaleOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(scale)
    prim_xform_op_move_end_to_front(prim)
    return prim


def prim_translate(prim: Usd.Prim, offset: Tuple[float, float, float]):
    """Translates a prim along the (x, y, z) dimensions.
    
    Args:
        prim (Usd.Prim):  The USD prim to translate.
        offset (Tuple[float, float, float]):  The offsets for the (x, y, z) dimensions.

    Returns:
        Usd.Prim:  The translated prim.
    """
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(offset)
    prim_xform_op_move_end_to_front(prim)
    return prim


def prim_rotate_x(prim: Usd.Prim, angle: float):
    """Rotates a prim around the X axis.
    
    Args:
        prim (Usd.Prim):  The USD prim to rotate.
        angle (float):  The rotation angle in degrees.

    Returns:
        Usd.Prim:  The rotated prim.
    """
    x = UsdGeom.Xformable(prim)
    x.AddRotateXOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(angle)
    prim_xform_op_move_end_to_front(prim)
    return prim

    
def prim_rotate_y(prim: Usd.Prim, angle: float):
    """Rotates a prim around the Y axis.
    
    Args:
        prim (Usd.Prim):  The USD prim to rotate.
        angle (float):  The rotation angle in degrees.

    Returns:
        Usd.Prim:  The rotated prim.
    """
    x = UsdGeom.Xformable(prim)
    x.AddRotateYOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(angle)
    prim_xform_op_move_end_to_front(prim)
    return prim


def prim_rotate_z(prim: Usd.Prim, angle: float):
    """Rotates a prim around the Z axis.
    
    Args:
        prim (Usd.Prim):  The USD prim to rotate.
        angle (float):  The rotation angle in degrees.

    Returns:
        Usd.Prim:  The rotated prim.
    """
    x = UsdGeom.Xformable(prim)
    x.AddRotateZOp(opSuffix=f"num_{prim_get_num_xform_ops(prim)}").Set(angle)
    prim_xform_op_move_end_to_front(prim)
    return prim


def _translation_to_np(t: Gf.Vec3d):
    return np.array(t)


def _rotation_to_np_quat(r: Gf.Rotation):
    quat = r.GetQuaternion()
    real = quat.GetReal()
    imag: Gf.Vec3d = quat.GetImaginary()
    return np.array([real, imag[0], imag[1], imag[2]])


def prim_get_local_transform(prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    """
    From: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/get-local-transforms.html

    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the local transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
    translation: Gf.Vec3d = local_transformation.ExtractTranslation()
    rotation: Gf.Rotation = local_transformation.ExtractRotation()
    return _translation_to_np(translation), _rotation_to_np_quat(rotation)


def prim_get_world_transform(prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    """
    From: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/get-world-transforms.html

    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    return _translation_to_np(translation), _rotation_to_np_quat(rotation)
