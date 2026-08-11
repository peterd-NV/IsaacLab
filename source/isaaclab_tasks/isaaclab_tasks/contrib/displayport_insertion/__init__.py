# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DisplayPort connector insertion environments."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-DisplayPort-Insertion-Rizon4s-Newton",
    entry_point=f"{__name__}.displayport_insertion_env:DisplayPortInsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.displayport_insertion_env_cfg:DisplayPortInsertionEnvCfg",
    },
)
