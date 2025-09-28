# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
from Robocon2026.robots.armdog import ARMDOG_CFG
from isaaclab.sensors import ImuCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Scene definition
##
@configclass
class BasicControlSceneCfg(InteractiveSceneCfg):
    """Configuration for the Armdog walking/basic_control scene."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
        debug_vis=False,
    )
    # Distant light
    Distantlight = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
            angle=20.0,
        ),
    )
    # ArmDog articulation
    robot = ARMDOG_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ArmDog",
    )
    Imu = ImuCfg(prim_path="{ENV_REGEX_NS}/ArmDog/go2/imu", debug_vis=True)
    Camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ArmDog/go2/base/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # TODO: 添加动作项
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            # Go2腿部
            ".*_hip_joint": 0.25,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        },
        use_default_offset=True,
        preserve_order=True,
    )

    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_shoulder_pan",
            ".*_shoulder_lift",
            ".*_elbow_flex",
            ".*_wrist_flex",
            ".*_wrist_roll",
            ".*_gripper",
        ],
        scale={
            ".*_shoulder_pan": 0.5,
            ".*_shoulder_lift": 0.5,
            ".*_elbow_flex": 0.5,
            ".*_wrist_flex": 0.5,
            ".*_wrist_roll": 0.5,
            ".*_gripper": 0.5,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # TODO: 添加观察项

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (resets)."""
    # TODO: 添加 reset 项
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # TODO: 添加更多奖励项


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # TODO: 添加 curricula 项
    pass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # TODO: 添加 command 项
    pass


##
# Environment configuration
##
@configclass
class BasicControlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BasicControlSceneCfg = BasicControlSceneCfg(
        num_envs=4096, env_spacing=4.0#, clone_in_fabric=True
    )

    # Basic MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization and simulation tuning for walking."""
        # control settings
        self.decimation = 2  # 控制频率 = sim.dt * decimation
        self.episode_length_s = 20  # 稍长的 episode 以便学习步态
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 3.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
