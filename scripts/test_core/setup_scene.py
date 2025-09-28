import argparse
from shlex import join
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create an empty scene.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=9, help="number of environments to create")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils 
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from Robocon2026.robots.dofbot import DOFBOT_CONFIG
from Robocon2026.robots.go2 import UNITREE_GO2_CFG
from Robocon2026.robots.jetbot import JETBOT_CFG
from Robocon2026.robots.armdog import ARMDOG_CFG
from Robocon2026.utils.utils import euler2quaternion
from isaaclab.sensors import ImuCfg
from isaaclab.sensors import CameraCfg

import numpy as np
import torch

def create_KFS(color):
    kfs = []
    if color == "red":
        coords = np.array(
            [
                [-4.2, 2.2, 0.375],
                [-3.0, 2.2, 0.575],
                [-1.8, 2.2, 0.375],
                [-4.2, 1.0, 0.575],
                [-3.0, 1.0, 0.775],
                [-1.8, 1.0, 0.575],
                [-4.2, -0.2, 0.375],
                [-3.0, -0.2, 0.575],
                [-1.8, -0.2, 0.775],
                [-4.2, -1.4, 0.575],
                [-3.0, -1.4, 0.375],
                [-1.8, -1.4, 0.575],
            ]
        )
    elif color == "blue":
        coords = np.array(
            [
                [4.2, 2.2, 0.375],
                [3.0, 2.2, 0.575],
                [1.8, 2.2, 0.375],
                [4.2, 1.0, 0.575],
                [3.0, 1.0, 0.775],
                [1.8, 1.0, 0.575],
                [4.2, -0.2, 0.375],
                [3.0, -0.2, 0.575],
                [1.8, -0.2, 0.775],
                [4.2, -1.4, 0.575],
                [3.0, -1.4, 0.375],
                [1.8, -1.4, 0.575],
            ]
        )
    # 任选
    # selected_indices = np.random.choice(12, 8, replace=False)
    # r1_indices = selected_indices[:3]
    # r1 排除中间位置
    while True:
        selected_indices = np.random.choice(12, 8, replace=False)
        r1_indices = selected_indices[:3]
        if 4 not in r1_indices and 7 not in r1_indices:
            break
    fake_index = selected_indices[3]
    r2_indices = selected_indices[4:]

    for idx in r1_indices:
        pos = coords[idx]
        kfs.append(
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}" + f"/r1_{color}_{idx}",
                spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/r1_{color}.usd"),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )
        )

    pos = coords[fake_index]
    kfs.append(
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}" + f"/fake_{color}",
            spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/fake_{color}.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )
    )

    used_x = set()
    for idx in r2_indices:
        pos = coords[idx]
        while True:
            x = np.random.randint(1, 16)
            if x not in used_x:
                used_x.add(x)
                break
        kfs.append(
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}" + f"/r1_{color}_{idx}",
                spawn=sim_utils.UsdFileCfg(usd_path=f"assets/KFS/r2_{color}_{x}.usd"),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )
        )

    return kfs

@configclass
class SceneCfg(InteractiveSceneCfg):
    # 创建穹顶灯光
    # Domelight = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=3000.0,
    #         color=(0.75, 0.75, 0.75),
    #     ),
    # )
    # 创建远光灯
    # Distantlight = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DistantLightCfg(
    #         intensity=3000.0,
    #         color=(0.75, 0.75, 0.75),
    #         angle=20.0,
    #     ),
    # )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file="assets/Matrials/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # 创建 Robocon 2026 地图
    Robocon2026map = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robocon2026Map",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/Map/robocon2026.usd"),
    )
    # 创建 Dofbot
    # Dofbot = DOFBOT_CONFIG.replace(
    #     prim_path="{ENV_REGEX_NS}/Dofbot"
    # )
    # 创建 Go2
    # Go2 = UNITREE_GO2_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Go2"
    # )
    # 创建 Jetbot
    # Jetbot = JETBOT_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/Jetbot"
    # )

    # 创建ArmDog
    Armdog = ARMDOG_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ArmDog",
    )
    # Imu = ImuCfg(prim_path="{ENV_REGEX_NS}/ArmDog/imu", debug_vis=True)
    # Camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/ArmDog/base/camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 1.0e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
    #     ),
    # )
    # 创建 KFS
    RedKFS1, RedKFS2, RedKFS3, RedKFS4, RedKFS5, RedKFS6, RedKFS7, RedKFS8 = create_KFS("red")
    BlueKFS1, BlueKFS2, BlueKFS3, BlueKFS4, BlueKFS5, BlueKFS6, BlueKFS7, BlueKFS8 = create_KFS("blue")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        if count % 1000 == 0:
            count = 0

            root_armdog_state = scene["Armdog"].data.default_root_state.clone()
            root_armdog_state[:, :3] += scene.env_origins
            scene["Armdog"].write_root_pose_to_sim(root_armdog_state[:, :7])
            scene["Armdog"].write_root_velocity_to_sim(root_armdog_state[:, 7:])
            joint_pos, joint_vel = (
                scene["Armdog"].data.default_joint_pos.clone(),
                scene["Armdog"].data.default_joint_vel.clone(),
            )
            scene["Armdog"].write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state ...")

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        # print("-------------------------------")
        # print(scene["Imu"])
        # print("Received linear velocity: ", scene["Imu"].data.lin_vel_b)
        # print("Received angular velocity: ", scene["Imu"].data.ang_vel_b)
        # print("Received linear acceleration: ", scene["Imu"].data.lin_acc_b)
        # print("Received angular acceleration: ", scene["Imu"].data.ang_acc_b)
        # print("-------------------------------")
        # print(scene["Camera"])
        # print("Received shape of rgb   image: ", scene["Camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["Camera"].data.output["distance_to_image_plane"].shape)


def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([15.0, 15.0, 10.0], [0.0, 0.0, 0.0])

    scene_cfg = SceneCfg(args_cli.num_envs, env_spacing=20.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete ...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
