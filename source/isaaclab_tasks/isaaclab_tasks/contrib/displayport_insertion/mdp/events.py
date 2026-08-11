# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for DisplayPort connector insertion."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


class SetRobotToObjectGraspPose(ManagerTermBase):
    """Set a robot to a calibrated object grasp using reset-time inverse kinematics."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        robot_asset_cfg: SceneEntityCfg = cfg.params["robot_asset_cfg"]
        self.robot: Articulation = env.scene[robot_asset_cfg.name]
        self.target_object: RigidObject = env.scene[cfg.params["target_object_name"]]
        self.num_arm_joints: int = cfg.params["num_arm_joints"]
        self.gripper_close_position: float = cfg.params["gripper_close_position"]

        self.grasp_offset = torch.tensor(cfg.params["grasp_offset"], device=env.device, dtype=torch.float32).unsqueeze(
            0
        )
        self.grasp_rot_offset = torch.tensor(
            cfg.params["grasp_rot_offset"], device=env.device, dtype=torch.float32
        ).unsqueeze(0)

        eef_indices, _ = self.robot.find_bodies([cfg.params["end_effector_body_name"]])
        if len(eef_indices) != 1:
            raise ValueError(
                f"Expected one '{cfg.params['end_effector_body_name']}' body, found indices {eef_indices}."
            )
        self.eef_idx = eef_indices[0]
        self.jacobian_body_idx = self.eef_idx - 1

        self.all_joints, all_joint_names = self.robot.find_joints([".*"])
        joint_name_to_idx = {name: index for index, name in zip(self.all_joints, all_joint_names)}
        gripper_joint_gearing: dict[str, float] = cfg.params["gripper_joint_gearing"]
        missing_joints = [name for name in gripper_joint_gearing if name not in joint_name_to_idx]
        if missing_joints:
            raise ValueError(f"Gripper joints not found on the robot: {missing_joints}")
        self.gripper_joint_ids = torch.tensor(
            [joint_name_to_idx[name] for name in gripper_joint_gearing], device=env.device, dtype=torch.long
        )
        self.gripper_gearing = torch.tensor(
            list(gripper_joint_gearing.values()), device=env.device, dtype=torch.float32
        )
        self._ik_controllers: dict[int, DifferentialIKController] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg,
        target_object_name: str,
        end_effector_body_name: str,
        num_arm_joints: int,
        grasp_offset: list[float],
        grasp_rot_offset: list[float],
        gripper_joint_gearing: dict[str, float],
        gripper_close_position: float,
        max_iterations: int = 150,
        position_threshold: float = 1.0e-6,
        rotation_threshold: float = 1.0e-6,
    ) -> None:
        """Apply the calibrated grasp reset to the selected environments.

        Args:
            env: Environment containing the robot and object.
            env_ids: Environment indices to reset.
            robot_asset_cfg: Robot scene entity configuration.
            target_object_name: Name of the grasped rigid object.
            end_effector_body_name: Robot body used as the grasp frame.
            num_arm_joints: Number of arm joints preceding the gripper joints.
            grasp_offset: Object-to-grasp translation [m].
            grasp_rot_offset: Object-to-grasp quaternion in ``(x, y, z, w)`` order.
            gripper_joint_gearing: Gripper joint names and mimic gearing.
            gripper_close_position: Closed position of the gripper drive joint [rad].
            max_iterations: Maximum reset-time inverse-kinematics iterations.
            position_threshold: Position convergence threshold [m].
            rotation_threshold: Axis-angle convergence threshold [rad].
        """
        del env, robot_asset_cfg, target_object_name, end_effector_body_name
        del num_arm_joints, grasp_offset, grasp_rot_offset, gripper_joint_gearing, gripper_close_position

        target_pos, target_quat, grasp_offset_batch, grasp_rot_offset_batch = self._compute_target_grasp_pose(env_ids)
        self._solve_arm_ik(
            env_ids,
            target_pos,
            target_quat,
            max_iterations=max_iterations,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
        )
        self._align_object_to_gripper(env_ids, grasp_offset_batch, grasp_rot_offset_batch)
        self._set_closed_gripper_state(env_ids)

    def _compute_target_grasp_pose(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the target hand pose and batched grasp transform."""
        num_reset_envs = len(env_ids)
        grasp_offset = self.grasp_offset.expand(num_reset_envs, -1)
        grasp_rot_offset = self.grasp_rot_offset.expand(num_reset_envs, -1)
        object_pos = self.target_object.data.root_link_pos_w.torch[env_ids]
        object_quat = self.target_object.data.root_link_quat_w.torch[env_ids]
        target_quat = math_utils.quat_mul(object_quat, grasp_rot_offset)
        target_pos = object_pos + math_utils.quat_apply(target_quat, grasp_offset)
        return target_pos, target_quat, grasp_offset, grasp_rot_offset

    def _solve_arm_ik(
        self,
        env_ids: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        max_iterations: int,
        position_threshold: float,
        rotation_threshold: float,
    ) -> None:
        """Iteratively move the arm to the target hand pose."""
        ik_controller = self._get_ik_controller(len(env_ids))
        ik_controller.set_command(torch.cat((target_pos, target_quat), dim=-1))

        joint_limits = self.robot.data.joint_pos_limits.torch[env_ids, : self.num_arm_joints]
        joint_min = joint_limits[:, :, 0]
        joint_range = joint_limits[:, :, 1] - joint_min
        finite_limits = torch.isfinite(joint_min) & torch.isfinite(joint_range)
        wrap_mask = finite_limits & (joint_range > 0)
        safe_joint_min = torch.where(wrap_mask, joint_min, torch.zeros_like(joint_min))
        safe_joint_range = torch.where(wrap_mask, joint_range, torch.ones_like(joint_range))
        zero_joint_vel = torch.zeros_like(self.robot.data.joint_vel.torch[env_ids])

        for _ in range(max_iterations):
            joint_pos = self.robot.data.joint_pos.torch[env_ids].clone()
            eef_pos = self.robot.data.body_pos_w.torch[env_ids, self.eef_idx]
            eef_quat = self.robot.data.body_quat_w.torch[env_ids, self.eef_idx]
            pos_error, axis_angle_error = math_utils.compute_pose_error(
                eef_pos, eef_quat, target_pos, target_quat, rot_error_type="axis_angle"
            )
            if torch.all(torch.linalg.vector_norm(pos_error, dim=-1) < position_threshold) and torch.all(
                torch.linalg.vector_norm(axis_angle_error, dim=-1) < rotation_threshold
            ):
                break

            jacobians = self.robot.data.body_link_jacobian_w.torch
            jacobian = jacobians[env_ids, self.jacobian_body_idx, :, self.robot.num_base_dofs :]
            joint_pos = ik_controller.compute(eef_pos, eef_quat, jacobian, joint_pos)
            self._wrap_arm_joint_positions(joint_pos, safe_joint_min, safe_joint_range, wrap_mask)

            self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot.set_joint_velocity_target_index(target=zero_joint_vel, env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=zero_joint_vel, env_ids=env_ids)

    def _get_ik_controller(self, num_envs: int) -> DifferentialIKController:
        """Return a damped-least-squares controller sized for the reset batch."""
        controller = self._ik_controllers.get(num_envs)
        if controller is None:
            controller = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.1},
                ),
                num_envs=num_envs,
                device=self.device,
            )
            self._ik_controllers[num_envs] = controller
        return controller

    def _wrap_arm_joint_positions(
        self,
        joint_pos: torch.Tensor,
        safe_joint_min: torch.Tensor,
        safe_joint_range: torch.Tensor,
        wrap_mask: torch.Tensor,
    ) -> None:
        """Apply finite-range wrapping to arm joint positions."""
        arm_joint_pos = joint_pos[:, : self.num_arm_joints]
        wrapped_arm_joint_pos = safe_joint_min + torch.remainder(arm_joint_pos - safe_joint_min, safe_joint_range)
        joint_pos[:, : self.num_arm_joints] = torch.where(wrap_mask, wrapped_arm_joint_pos, arm_joint_pos)

    def _align_object_to_gripper(
        self,
        env_ids: torch.Tensor,
        grasp_offset: torch.Tensor,
        grasp_rot_offset: torch.Tensor,
    ) -> None:
        """Place the object at the grasp transform achieved by inverse kinematics."""
        num_reset_envs = len(env_ids)
        achieved_hand_pos = self.robot.data.body_pos_w.torch[env_ids, self.eef_idx].clone()
        achieved_hand_quat = self.robot.data.body_quat_w.torch[env_ids, self.eef_idx].clone()
        aligned_object_quat = math_utils.quat_mul(achieved_hand_quat, math_utils.quat_conjugate(grasp_rot_offset))
        aligned_object_pos = achieved_hand_pos - math_utils.quat_apply(achieved_hand_quat, grasp_offset)
        self.target_object.write_root_pose_to_sim_index(
            root_pose=torch.cat((aligned_object_pos, aligned_object_quat), dim=-1), env_ids=env_ids
        )
        self.target_object.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((num_reset_envs, 6), device=self.device), env_ids=env_ids
        )

    def _set_closed_gripper_state(self, env_ids: torch.Tensor) -> None:
        """Set the gripper state and actuator targets directly to the closed grasp."""
        joint_pos = self.robot.data.joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.gripper_joint_ids] = self.gripper_gearing * self.gripper_close_position
        self.robot.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)
        self.robot.set_joint_velocity_target_index(target=joint_vel, joint_ids=self.all_joints, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
