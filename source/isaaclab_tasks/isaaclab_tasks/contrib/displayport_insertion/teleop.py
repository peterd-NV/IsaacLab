# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific teleoperation devices for DisplayPort insertion."""

from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg


class InitiallyClosedKeyboard(Se3Keyboard):
    """SE(3) keyboard that starts with the pre-grasp gripper closed."""

    def __init__(self, cfg: Se3KeyboardCfg):
        """Initialize the keyboard and preserve the closed pre-grasp state."""
        super().__init__(cfg)
        self._close_gripper = True

    def reset(self):
        """Reset motion commands while keeping the gripper closed."""
        super().reset()
        self._close_gripper = True


class InitiallyClosedSpaceMouse(Se3SpaceMouse):
    """SE(3) SpaceMouse that starts with the pre-grasp gripper closed."""

    def __init__(self, cfg: Se3SpaceMouseCfg):
        """Initialize the SpaceMouse and preserve the closed pre-grasp state."""
        super().__init__(cfg)
        self._close_gripper = True

    def reset(self):
        """Reset motion commands while keeping the gripper closed."""
        super().reset()
        self._close_gripper = True
