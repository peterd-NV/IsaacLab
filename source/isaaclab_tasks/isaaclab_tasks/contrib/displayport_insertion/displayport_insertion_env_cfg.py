# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton environment for inserting a DisplayPort connector with a Flexiv Rizon 4s."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.schemas import (
    NewtonCollisionPropertiesCfg,
    NewtonMaterialPropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
)
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxCollisionPropertiesCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.displayport_insertion.mdp as mdp

from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG

_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_PLUG_USD_PATH = _ASSET_DIR / "display_port_plug_newton_sdf.usda"
_SOCKET_USD_PATH = _ASSET_DIR / "display_port_socket_newton_sdf.usda"

_INSERTION_TARGET_POS = (0.475, 0.125, 0.0375)
_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_PLUG_START_OFFSET = (0.0, 0.0, 0.015)
_SOCKET_INSERTION_OFFSET = (0.0375, 0.0, 0.0)
_PLUG_INSERTION_OFFSET = (0.0, 0.0, 0.0221)
_PLUG_GOAL_ROT = (0.0, -0.70711, 0.0, 0.70711)
_GRASP_OFFSET = (0.0025, 0.0, -0.1875)

_GRAV_GRIPPER_MIMIC_GEARING = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": -1.0,
    "right_outer_finger_joint": -1.0,
}
_GRIPPER_OPEN_POSITION = 0.5
_GRIPPER_CLOSE_POSITION = -0.1


def _quat_rotate_vec(
    quat_xyzw: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply an XYZW quaternion rotation to a vector."""
    qx, qy, qz, qw = quat_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _quat_mul(
    quat_1_xyzw: tuple[float, float, float, float], quat_2_xyzw: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Multiply two XYZW quaternions."""
    x1, y1, z1, w1 = quat_1_xyzw
    x2, y2, z2, w2 = quat_2_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _compute_socket_root() -> tuple[float, float, float]:
    """Compute the socket root position from its calibrated mating point."""
    rotated = _quat_rotate_vec(_SOCKET_ROT, _SOCKET_INSERTION_OFFSET)
    return tuple(_INSERTION_TARGET_POS[index] - rotated[index] for index in range(3))


def _compute_plug_pose() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compute the initial plug root pose from its calibrated mating point."""
    plug_rot = _quat_mul(_SOCKET_ROT, _PLUG_GOAL_ROT)
    plug_offset_world = _quat_rotate_vec(plug_rot, _PLUG_INSERTION_OFFSET)
    plug_root = tuple(
        _INSERTION_TARGET_POS[index] - plug_offset_world[index] + _PLUG_START_OFFSET[index] for index in range(3)
    )
    return plug_root, plug_rot


_SOCKET_ROOT = _compute_socket_root()
_PLUG_ROOT, _PLUG_ROT = _compute_plug_pose()


def _create_robot_cfg() -> ArticulationCfg:
    """Create the calibrated Rizon 4s and Grav gripper configuration."""
    robot_cfg = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
            joint_drive_props=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": math.radians(32.44),
                "joint2": math.radians(-16.71),
                "joint3": math.radians(-5.69),
                "joint4": math.radians(128.38),
                "joint5": math.radians(6.74),
                "joint6": math.radians(55.95),
                "joint7": math.radians(111.54),
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    robot_cfg.actuators["shoulder"].effort_limit_sim = 123.0
    robot_cfg.actuators["shoulder"].stiffness = 6000.0
    robot_cfg.actuators["shoulder"].damping = 108.5
    robot_cfg.actuators["elbow"].effort_limit_sim = 64.0
    robot_cfg.actuators["elbow"].stiffness = 4200.0
    robot_cfg.actuators["elbow"].damping = 90.7
    robot_cfg.actuators["wrist"].effort_limit_sim = 39.0
    robot_cfg.actuators["wrist"].stiffness = 1500.0
    robot_cfg.actuators["wrist"].damping = 54.2
    robot_cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=200.0,
        velocity_limit_sim=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.1,
    )
    robot_cfg.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
        effort_limit_sim=20.0,
        velocity_limit_sim=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.05,
    )
    return robot_cfg


@configclass
class DisplayPortInsertionSceneCfg(InteractiveSceneCfg):
    """Scene configuration for DisplayPort insertion."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.707, 0.707),
        ),
    )

    dp_plug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dp_plug",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_PLUG_USD_PATH),
            rigid_props=NewtonRigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=NewtonCollisionPropertiesCfg(contact_offset=0.00001, rest_offset=-0.00005),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_PLUG_ROOT, rot=_PLUG_ROT),
    )

    dp_socket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dp_socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_SOCKET_USD_PATH),
            rigid_props=NewtonRigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(contact_offset=0.0001, rest_offset=-0.0001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_SOCKET_ROOT, rot=_SOCKET_ROT),
    )

    robot: ArticulationCfg = _create_robot_cfg()

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications matching the recorded Arena demonstrations."""

    arm_action: ActionTerm = mdp.PersistentTargetDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        body_name="flange",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.01,
    )
    gripper_action: ActionTerm = mdp.RateLimitedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_GRAV_GRIPPER_MIMIC_GEARING),
        open_command_expr={
            name: gearing * _GRIPPER_OPEN_POSITION for name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items()
        },
        close_command_expr={
            name: gearing * _GRIPPER_CLOSE_POSITION for name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items()
        },
        max_target_step=0.03,
    )


@configclass
class ObservationsCfg:
    """State observations matching the recorded Arena demonstrations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Non-concatenated policy observation group."""

        actions = ObsTerm(func=base_mdp.last_action)
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        plug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("dp_plug")})
        plug_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("dp_plug")})
        socket_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("dp_socket")})
        socket_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("dp_socket")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset events for the calibrated pre-grasp state."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    set_robot_to_grasp_pose = EventTerm(
        func=mdp.SetRobotToObjectGraspPose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "target_object_name": "dp_plug",
            "end_effector_body_name": "flange",
            "num_arm_joints": 7,
            "grasp_offset": list(_GRASP_OFFSET),
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper_joint_gearing": _GRAV_GRIPPER_MIMIC_GEARING,
            "gripper_close_position": _GRIPPER_CLOSE_POSITION,
            "max_iterations": 150,
            "position_threshold": 1.0e-6,
            "rotation_threshold": 1.0e-6,
        },
    )


@configclass
class TerminationsCfg:
    """Episode terminations for DisplayPort insertion."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.insertion_success,
        params={
            "socket_insertion_offset": _SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": _PLUG_INSERTION_OFFSET,
            "position_threshold": 0.003,
        },
    )


@configclass
class DisplayPortInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """DisplayPort insertion task using Newton's MJWarp solver and authored SDF colliders."""

    scene: DisplayPortInsertionSceneCfg = DisplayPortInsertionSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=7,
        physics_material=NewtonMaterialPropertiesCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=NewtonCfg(
            # Preserve the authored connector concavities before the replicated sources build their SDFs.
            simplify_meshes=False,
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=4096,
                nconmax=4096,
                impratio=10.0,
                cone="elliptic",
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                reduce_contacts=True,
                max_triangle_pairs=2**25,
            ),
            num_substeps=10,
            debug_mode=False,
        ),
        default_visualizer_cfg=VisualizerCfg(
            eye=(0.7, -0.45, 0.2975),
            lookat=(0.475, 0.125, 0.0675),
        ),
    )

    decimation = 7
    episode_length_s = 120.0
    wait_for_textures = False
    reset_sim_buffer_each_episode: bool = False
    """Preserve Newton simulation buffers between manual demonstration-recording episodes."""

    def __post_init__(self):
        """Configure task-calibrated teleoperation devices."""
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.15,
                    rot_sensitivity=0.3,
                    gripper_term=True,
                    sim_device=self.sim.device,
                    class_type=("isaaclab_tasks.contrib.displayport_insertion.teleop:InitiallyClosedKeyboard"),
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.15,
                    rot_sensitivity=0.3,
                    gripper_term=True,
                    sim_device=self.sim.device,
                    class_type=("isaaclab_tasks.contrib.displayport_insertion.teleop:InitiallyClosedSpaceMouse"),
                ),
            }
        )
