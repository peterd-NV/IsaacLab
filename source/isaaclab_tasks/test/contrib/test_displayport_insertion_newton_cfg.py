# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration contract tests for the Newton DisplayPort insertion environment."""

from pathlib import Path

import pytest

from isaaclab.managers import ObservationTermCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_displayport_insertion_replay_contract():
    """The Lab task should preserve the Arena demonstration interface and Newton parameters."""
    cfg = parse_env_cfg("IsaacContrib-DisplayPort-Insertion-Rizon4s-Newton", num_envs=2)

    assert cfg.scene.num_envs == 2
    assert cfg.scene.replicate_physics
    assert cfg.sim.dt == pytest.approx(1.0 / 200.0)
    assert cfg.sim.render_interval == 7
    assert cfg.sim.physics.num_substeps == 10
    assert cfg.sim.physics.simplify_meshes is False
    assert cfg.decimation == 7
    assert cfg.episode_length_s == 120.0
    assert cfg.reset_sim_buffer_each_episode is False
    assert cfg.observations.policy.concatenate_terms is False
    observation_terms = [
        name for name, value in cfg.observations.policy.__dict__.items() if isinstance(value, ObservationTermCfg)
    ]
    assert observation_terms == [
        "actions",
        "joint_pos",
        "joint_vel",
        "plug_pos",
        "plug_quat",
        "socket_pos",
        "socket_quat",
    ]

    assert len(cfg.actions.arm_action.joint_names) == 7
    assert cfg.actions.arm_action.controller.command_type == "pose"
    assert cfg.actions.arm_action.controller.use_relative_mode
    assert cfg.actions.arm_action.scale == pytest.approx(0.01)
    assert len(cfg.actions.gripper_action.joint_names) == 6
    assert cfg.actions.gripper_action.max_target_step == pytest.approx(0.03)

    teleop_devices = cfg.teleop_devices.devices
    assert list(teleop_devices) == ["keyboard", "spacemouse"]
    for device_cfg in teleop_devices.values():
        assert device_cfg.pos_sensitivity == pytest.approx(0.15)
        assert device_cfg.rot_sensitivity == pytest.approx(0.3)
        assert device_cfg.gripper_term
        assert device_cfg.sim_device == cfg.sim.device
    assert str(teleop_devices["keyboard"].class_type).endswith(":InitiallyClosedKeyboard")
    assert str(teleop_devices["spacemouse"].class_type).endswith(":InitiallyClosedSpaceMouse")

    for asset_cfg in (cfg.scene.dp_plug, cfg.scene.dp_socket):
        assert Path(asset_cfg.spawn.usd_path).is_file()
