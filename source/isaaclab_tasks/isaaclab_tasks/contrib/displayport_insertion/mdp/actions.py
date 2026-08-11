# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for DisplayPort connector insertion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PersistentTargetDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Accumulate relative commands from the last target so zero input holds the flange pose."""

    cfg: PersistentTargetDifferentialIKActionCfg

    def __init__(self, cfg: PersistentTargetDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        use_target = self._target_initialized.unsqueeze(-1)
        command_base_pos = torch.where(use_target, self._ik_controller.ee_pos_des, ee_pos_curr)
        command_base_quat = torch.where(use_target, self._ik_controller.ee_quat_des, ee_quat_curr)
        self._ik_controller.set_command(self._processed_actions, command_base_pos, command_base_quat)
        self._target_initialized[:] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._target_initialized[env_ids] = False


@configclass
class PersistentTargetDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for persistent-target differential inverse kinematics."""

    class_type: type[PersistentTargetDifferentialIKAction] = PersistentTargetDifferentialIKAction


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Ramp binary targets to avoid oscillating the lightweight gripper finger joints."""

    cfg: RateLimitedBinaryJointPositionActionCfg

    def __init__(self, cfg: RateLimitedBinaryJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._smoothed_target = self._close_command.unsqueeze(0).repeat(self.num_envs, 1)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        target_delta = torch.clamp(
            self._processed_actions - self._smoothed_target,
            min=-self.cfg.max_target_step,
            max=self.cfg.max_target_step,
        )
        self._smoothed_target += target_delta
        self._processed_actions = self._smoothed_target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._smoothed_target[env_ids] = self._close_command


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a rate-limited binary joint-position action."""

    class_type: type[RateLimitedBinaryJointPositionAction] = RateLimitedBinaryJointPositionAction

    max_target_step: float = 0.03
    """Maximum change in commanded joint position per environment step [rad]."""
