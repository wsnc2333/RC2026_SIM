# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def base_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.2
) -> torch.Tensor:
    """Penalize the robot for having its base too close to the ground (crawling behavior).

    This function penalizes the agent when the robot's base height is below a threshold,
    which helps prevent the robot from learning to crawl on its belly instead of walking.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        min_height: Minimum allowed height for the robot base.
        penalty_weight: Scaling factor for the penalty.

    Returns:
        A tensor of penalty values for each environment.
    """
    # Extract the robot asset
    asset = env.scene[asset_cfg.name]

    # Get the base height (z-coordinate of the base)
    base_height = asset.data.root_pos_w[:, 2]

    # Calculate penalty for heights below the minimum
    # Penalty is positive when height is below threshold, zero otherwise
    penalty = torch.clamp(min_height - base_height, min=0.0)

    return penalty


def base_height_above_terrain(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.20,
) -> torch.Tensor:
    """Penalize the robot for having its base too close to the local terrain height.

    This function penalizes the agent when the robot's base height is below a threshold,
    which helps prevent the robot from learning to crawl on its belly instead of walking.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the height scanner sensor.
        asset_cfg: Configuration for the robot asset.
        min_height: Minimum allowed height for the robot base.
        penalty_weight: Scaling factor for the penalty.

    Returns:
        A tensor of penalty values for each environment.
    """
    asset = env.scene[asset_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]

    # 获取 height scanner 返回的地面命中点 z
    try:
        scanner = env.scene.sensors[sensor_cfg.name]
        # ray_hits_w shape: [num_envs, num_rays, 3]
        hits_z = scanner.data.ray_hits_w[..., 2]  # [num_envs, num_rays]

        if hits_z.ndim == 2:
            # 选中心 ray 作为局部地形代表
            center_idx = hits_z.shape[1] // 2
            terrain_h = hits_z[:, center_idx]

            # 若中心 ray 无效（inf / nan），尝试使用有效 ray 的均值
            invalid_mask = ~torch.isfinite(terrain_h)
            if invalid_mask.any():
                # 使用 per-env 有效 hits 的均值作为回退值
                valid_hits = torch.where(torch.isfinite(hits_z), hits_z, torch.nan)
                mean_hits = torch.nanmean(valid_hits, dim=1)
                terrain_h[invalid_mask] = mean_hits[invalid_mask]

            # 如仍有无效值（极端情况），回退到 sensor 高度 - offset
            still_invalid = ~torch.isfinite(terrain_h)
            if still_invalid.any():
                sensor_z = scanner.data.pos_w[:, 2]
                # offset: 与 observations.height_scan 一致（默认 0.5）
                offset = (
                    getattr(sensor_cfg, "params", {}).get("offset", 0.5)
                    if isinstance(sensor_cfg, dict)
                    else 0.5
                )
                terrain_h[still_invalid] = sensor_z[still_invalid] - offset
        else:
            # 若只有单个 ray，直接使用它
            terrain_h = hits_z.reshape(-1)
            terrain_h[~torch.isfinite(terrain_h)] = 0.0

    except Exception:
        # 任何异常回退为平地（不惩罚）
        terrain_h = torch.zeros_like(base_z)

    height_above = base_z - terrain_h
    penalty = torch.clamp(min_height - height_above, min=0.0)
    return penalty
