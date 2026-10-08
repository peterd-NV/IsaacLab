# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test USD-to-URDF exports consumed by Pink IK controllers."""

from isaaclab_physx.physics import PhysxCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=PhysxCfg()), device="cpu")

import math
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.controllers.utils import convert_usd_to_urdf

pytestmark = pytest.mark.integration


@pytest.mark.isaacsim_ci
def test_export_fills_effort_limits_without_changing_source(tmp_path):
    """Incomplete effort metadata must export a valid IK model without changing the robot's drives."""
    usd_path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    joint_names = ["driven", "metadata", "missing", "mimic", "nonfinite", "prismatic"]
    for index in range(len(joint_names) + 1):
        link = UsdGeom.Xform.Define(stage, f"/Robot/link_{index}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        mass = UsdPhysics.MassAPI.Apply(link)
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
    for index, name in enumerate(joint_names):
        joint_type = UsdPhysics.PrismaticJoint if name == "prismatic" else UsdPhysics.RevoluteJoint
        joint = joint_type.Define(stage, f"/Robot/{name}")
        joint.CreateBody0Rel().SetTargets([f"/Robot/link_{index}"])
        joint.CreateBody1Rel().SetTargets([f"/Robot/link_{index + 1}"])
        joint.CreateLowerLimitAttr(-1.0 if name == "prismatic" else -90.0)
        joint.CreateUpperLimitAttr(1.0 if name == "prismatic" else 90.0)
        prim = joint.GetPrim()
        prim.CreateAttribute("newton:velocityLimit", Sdf.ValueTypeNames.Float).Set(
            2.0 if name == "prismatic" else 180.0
        )
        if name != "missing":
            drive_type = "linear" if name == "prismatic" else "angular"
            drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
            drive.CreateMaxForceAttr(3.0 if name == "driven" else math.inf)
        if name in {"driven", "nonfinite"}:
            prim.CreateAttribute("urdf:limit:effort", Sdf.ValueTypeNames.Double).Set(
                math.nan if name == "nonfinite" else 7.0
            )
        if name == "mimic":
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, "rotZ")
            mimic.CreateReferenceJointRel().SetTargets(["/Robot/driven"])
            mimic.CreateGearingAttr(-1.0)
    # The exporter selects an unselected physics variant during robot discovery.
    # A fallback must not hide a finite limit authored inside that variant.
    variants = root.GetVariantSets().AddVariantSet("Physics")
    variants.AddVariant("physx")
    variants.SetVariantSelection("physx")
    with variants.GetVariantEditContext():
        stage.GetPrimAtPath("/Robot/metadata").CreateAttribute("urdf:limit:effort", Sdf.ValueTypeNames.Double).Set(7.0)
    variants.ClearVariantSelection()
    stage.GetRootLayer().Save()
    original_file = usd_path.read_bytes()
    original_root = stage.GetRootLayer().ExportToString()
    original_session = stage.GetSessionLayer().ExportToString()

    urdf_path, _ = convert_usd_to_urdf(str(usd_path), str(tmp_path / "export"))
    model = pin.buildModelFromUrdf(urdf_path)
    expected_efforts = {
        "driven": 3.0,
        "metadata": 7.0,
        "missing": 0.0,
        "mimic": 0.0,
        "nonfinite": 0.0,
        "prismatic": 0.0,
    }
    for name, expected_effort in expected_efforts.items():
        joint_id = model.getJointId(name)
        assert model.existJointName(name)
        joint = model.joints[joint_id]
        assert model.effortLimit[joint.idx_v] == expected_effort
        bound = 1.0 if name == "prismatic" else math.pi / 2
        np.testing.assert_allclose(model.lowerPositionLimit[joint.idx_q], -bound)
        np.testing.assert_allclose(model.upperPositionLimit[joint.idx_q], bound)
        np.testing.assert_allclose(model.velocityLimit[joint.idx_v], 2.0 if name == "prismatic" else math.pi)
    mimic_xml = ET.parse(urdf_path).find("joint[@name='mimic']/mimic")
    assert mimic_xml is not None
    assert mimic_xml.attrib["joint"] == "driven"
    assert float(mimic_xml.attrib["multiplier"]) == -1.0
    assert usd_path.read_bytes() == original_file
    assert stage.GetRootLayer().ExportToString() == original_root
    assert stage.GetSessionLayer().ExportToString() == original_session
