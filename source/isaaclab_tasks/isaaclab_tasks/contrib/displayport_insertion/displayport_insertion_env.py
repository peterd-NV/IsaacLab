# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based DisplayPort insertion environment with Newton asset preparation."""

from __future__ import annotations

from typing import Any

from isaaclab_newton.physics import NewtonManager
from newton import GeoType, ShapeFlags

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.physics import PhysicsEvent

_SDF_MAX_RESOLUTION = 256
_SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
_SDF_PADDING = 0.005
_SDF_COLLIDERS_PER_ENV = 6


def _prepare_displayport_newton_builder(_event_payload: Any = None) -> None:
    """Build each replicated world's six connector SDF collision sources."""
    builder = NewtonManager._builder
    if builder is None:
        raise RuntimeError("Newton MODEL_INIT fired before the model builder was created.")
    num_envs = NewtonManager.get_num_envs()
    if num_envs is None or num_envs < 1:
        raise RuntimeError(f"Expected at least one replicated Newton world, found {num_envs}.")

    collider_counts = [0] * num_envs
    for index, (label, source) in enumerate(zip(builder.shape_label, builder.shape_source)):
        is_plug_collider = label.endswith("/dp_plug/collision_mesh")
        is_socket_collider = "/dp_socket/tn__2584N111_DisplayportCord_jP/Body" in label and label.endswith("/Mesh")
        if not (is_plug_collider or is_socket_collider):
            continue
        if not builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES:
            continue

        if builder.shape_type[index] != GeoType.MESH:
            raise RuntimeError(
                f"DisplayPort SDF collider was converted to geometry type {builder.shape_type[index]}: {label}"
            )
        if source is None:
            raise RuntimeError(f"DisplayPort SDF collider has no mesh source: {label}")
        world_index = builder.shape_world[index]
        if world_index is None or not 0 <= world_index < num_envs:
            raise RuntimeError(f"DisplayPort SDF collider has invalid Newton world index {world_index}: {label}")
        collider_counts[world_index] += 1

        if source.sdf is None:
            scale = tuple(float(builder.shape_scale[index][axis]) for axis in range(3))
            source.build_sdf(
                device=NewtonManager.get_device(),
                narrow_band_range=_SDF_NARROW_BAND_RANGE,
                max_resolution=_SDF_MAX_RESOLUTION,
                margin=_SDF_PADDING,
                shape_margin=0.0,
                scale=scale,
                texture_format="uint16",
            )

    invalid_worlds = {
        world_index: count for world_index, count in enumerate(collider_counts) if count != _SDF_COLLIDERS_PER_ENV
    }
    if invalid_worlds:
        raise RuntimeError(
            f"Expected {_SDF_COLLIDERS_PER_ENV} DisplayPort SDF colliders per Newton world, found {invalid_worlds}."
        )


class DisplayPortInsertionEnv(ManagerBasedRLEnv):
    """DisplayPort insertion environment that prepares local SDF assets before Newton model finalization."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        self._newton_builder_callback_handle = NewtonManager.register_callback(
            _prepare_displayport_newton_builder,
            PhysicsEvent.MODEL_INIT,
            name="displayport_newton_builder_preparation",
            wrap_weak_ref=False,
        )
        try:
            super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        except Exception:
            self._newton_builder_callback_handle.deregister()
            self._newton_builder_callback_handle = None
            raise

    def close(self) -> None:
        """Release the Newton model-builder callback and environment resources."""
        callback_handle = getattr(self, "_newton_builder_callback_handle", None)
        if callback_handle is not None:
            callback_handle.deregister()
            self._newton_builder_callback_handle = None
        super().close()
