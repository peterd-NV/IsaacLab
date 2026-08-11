# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for DisplayPort connector insertion."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv


def insertion_success(
    env: ManagerBasedRLEnv,
    socket_insertion_offset: tuple[float, float, float],
    plug_insertion_offset: tuple[float, float, float],
    position_threshold: float = 0.003,
) -> torch.Tensor:
    """Return whether the plug and socket mating points are aligned.

    Args:
        env: DisplayPort insertion environment.
        socket_insertion_offset: Socket-root to mating-point translation [m].
        plug_insertion_offset: Plug-root to mating-point translation [m].
        position_threshold: Maximum mating-point separation for success [m].

    Returns:
        Boolean success mask, shape ``(num_envs,)``.
    """
    socket = env.scene["dp_socket"]
    plug = env.scene["dp_plug"]

    socket_pos = socket.data.root_link_pos_w.torch
    socket_quat = socket.data.root_link_quat_w.torch
    plug_pos = plug.data.root_link_pos_w.torch
    plug_quat = plug.data.root_link_quat_w.torch

    socket_offset = torch.tensor(socket_insertion_offset, device=env.device).repeat(env.num_envs, 1)
    plug_offset = torch.tensor(plug_insertion_offset, device=env.device).repeat(env.num_envs, 1)
    socket_mate_pos, _ = math_utils.combine_frame_transforms(socket_pos, socket_quat, socket_offset)
    plug_mate_pos, _ = math_utils.combine_frame_transforms(plug_pos, plug_quat, plug_offset)

    return torch.linalg.vector_norm(socket_mate_pos - plug_mate_pos, dim=-1) < position_threshold
